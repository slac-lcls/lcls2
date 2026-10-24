/**
 * @file
 * @brief Declares psalg::AllocArray and AllocArray1D, reference-counted Arrays whose storage comes from an Allocator.
 */
#ifndef ALLOCARRAY__H
#define ALLOCARRAY__H

#include "psalg/utils/SysLog.hh"
#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/DescData.hh"
#include "Allocator.hh"

using namespace XtcData; // Array

namespace psalg {

/**
 * Array<T> whose shape (MaxRank words), a reference count and up to maxElem elements live in one block from an Allocator.
 * Copies share the block and increment the count; when the count reaches 0 the elements are destroyed and the block is freed.
 */
template <typename T>
class AllocArray:public Array<T>{
public:

    /** Allocate one block for the shape, the reference count and maxElem elements, and set the rank, every shape entry to 0 and the count to 1. The elements are not constructed. */
    AllocArray(Allocator& allocator, size_t maxElem, uint32_t rank):_allocator(allocator){
        void *ptr = _allocator.malloc(sizeof(Shape) +
                                      sizeof(*_refCntPtr) +
                                      maxElem*sizeof(T)); // shape + refcnt + data
        Array<T>::_shape = reinterpret_cast<uint32_t*>(ptr);
        _refCntPtr = reinterpret_cast<uint32_t*>(Array<T>::_shape+MaxRank);
        Array<T>::_data = reinterpret_cast<T*>(_refCntPtr+1);
        Array<T>::_rank = rank;
        *_refCntPtr = 1;
        for(int i = 0; i < MaxRank; i++) Array<T>::_shape[i] = 0;
    }

    /** Share other's block and increment the reference count. */
    AllocArray(const AllocArray<T>& other):_allocator(other._allocator){ // copy constructor
        if (this != &other) {
            this->_shape = other._shape;
            this->_data = other._data;
            this->_rank = other._rank;
            this->_refCntPtr = other._refCntPtr;
            incRefCnt(); // increment reference count in the original object
        }
    }

    /**
     * Decrement this object's count, freeing its block with its allocator when the count reaches 0, then share other's block and increment that count. Self-assignment does nothing.
     * @return *this.
     */
    AllocArray<T>& operator=(const AllocArray<T>& other){ // assignment operator
        if (this != &other) {
            decRefCnt();
            if (_refCnt() == 0) {
                _allocator.free(this->_shape);
            }

            this->_shape = other._shape;
            this->_data = other._data;
            this->_rank = other._rank;
            this->_refCntPtr = other._refCntPtr;
            this->_allocator = other._allocator;
            incRefCnt(); // increment reference count in the original object
        }
        return *this;
    }

    /** Log a syslog error if the count is already <= 0, decrement it, and when it reaches 0 call the destructor of each of the num_elem() elements and free the block. */
    virtual ~AllocArray(){
        if (_refCnt()<=0) {
            psalg::SysLog::error("AllocArray::~AllocArray: reference count <= 0: %d",_refCnt());
        }
        decRefCnt();
        if(_refCnt()==0) {
            for(unsigned i = 0; i < Array<T>::num_elem(); i++) {
            // call the destructor of everything we contain
            // this could include decrementing reference counts if we are
            // are array-of-arrays.
            Array<T>::_data[i].~T();
            }
            _allocator.free(Array<T>::_shape); // free the memory
        }
    }

    /** Increment the shared reference count. */
    void incRefCnt(){
        _refCnt()++;
    }

    /** Decrement the shared reference count. */
    void decRefCnt(){
        _refCnt()--;
    }

    // unfortunately need to make this public so cython can access it
    // in peakFinder.pyx:PyAllocArray1D
    /** Return a reference to the shared reference count; public so cython can reach it (per the comment above). */
    uint32_t& _refCnt(){
        return *_refCntPtr;
    }

protected:
    uint32_t *_refCntPtr;
    Allocator& _allocator;
};


/** Rank-1 AllocArray with std::vector-like push_back(), clear(), size() and capacity(). */
template <typename T>
class AllocArray1D:public AllocArray<T>{
public:

    /** Rank-1 AllocArray from *allocator with capacity maxElem and size 0. */
    AllocArray1D(Allocator *allocator, size_t maxElem):AllocArray<T>(*allocator, maxElem, 1){
        _maxShape = maxElem;
    }

    /** Rank-1 AllocArray from allocator with capacity maxElem and size 0. */
    AllocArray1D(Allocator& allocator, size_t maxElem):AllocArray<T>(allocator, maxElem, 1){
        _maxShape = maxElem;
    }

    /** Share other's block (AllocArray copy constructor) and copy its capacity. */
    AllocArray1D(const AllocArray1D<T>& other):AllocArray<T>(other){ // copy constructor
        if (this != &other) {
            this->_maxShape = other._maxShape;
        }
    }

    /**
     * Assign other as AllocArray::operator= does and copy its capacity.
     * @return *this.
     */
    AllocArray1D<T>& operator=(const AllocArray1D<T>& other){ // assignment operator
        if (this != &other) {
            AllocArray<T>::operator=(other);
            this->_maxShape = other._maxShape;
        }
        return *this;
    }

    // ----- std::vector-like methods

    /** Copy-construct i at index size() and increment the size. When the array is full a syslog error is logged, but the element is still written past the capacity. */
    void push_back(const T& i){
        if (AllocArray<T>::_shape[0] >= _maxShape) {
            psalg::SysLog::error("AllocArray: maxShape exceeded: %d >= %d\n",
                                 AllocArray<T>::_shape[0],_maxShape);
        }
        new(AllocArray<T>::_data+AllocArray<T>::_shape[0]) T(i);
        AllocArray<T>::_shape[0]++;
    }

    /** Call the destructor of each element and set the size to 0. */
    void clear(){
        for(unsigned i = 0; i < Array<T>::num_elem(); i++) {
            // call the destructor of everything we contain
            // this could include decrementing reference counts if we are
            // are array-of-arrays.
            Array<T>::_data[i].~T();
        }
        AllocArray<T>::_shape[0] = 0;
    }

    /** Return the capacity given to the constructor. */
    uint32_t capacity(){
        return _maxShape;
    }

    /** Return the number of elements (shape entry 0). */
    uint32_t size(){
        return AllocArray<T>::_shape[0];
    }

private:
    uint32_t  _maxShape;

};


} // namespace

#endif // ALLOCARRAY__H
