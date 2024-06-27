#ifndef ALLOCARRAY__H
#define ALLOCARRAY__H

#include "psalg/utils/SysLog.hh"
#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/DescData.hh"
#include "Allocator.hh"

using namespace XtcData; // Array

namespace psalg {

template <typename T>
class AllocArray:public Array<T>{
public:

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

    AllocArray(const AllocArray<T>& other):_allocator(other._allocator){ // copy constructor
        if (this != &other) {
            this->_shape = other._shape;
            this->_data = other._data;
            this->_rank = other._rank;
            this->_refCntPtr = other._refCntPtr;
            incRefCnt(); // increment reference count in the original object
        }
    }

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

    void incRefCnt(){
        _refCnt()++;
    }

    void decRefCnt(){
        _refCnt()--;
    }

    // unfortunately need to make this public so cython can access it
    // in peakFinder.pyx:PyAllocArray1D
    uint32_t& _refCnt(){
        return *_refCntPtr;
    }

protected:
    uint32_t *_refCntPtr;
    Allocator& _allocator;
};


template <typename T>
class AllocArray1D:public AllocArray<T>{
public:

    AllocArray1D(Allocator *allocator, size_t maxElem):AllocArray<T>(*allocator, maxElem, 1){
        _maxShape = maxElem;
    }

    AllocArray1D(Allocator& allocator, size_t maxElem):AllocArray<T>(allocator, maxElem, 1){
        _maxShape = maxElem;
    }

    AllocArray1D(const AllocArray1D<T>& other):AllocArray<T>(other){ // copy constructor
        if (this != &other) {
            this->_maxShape = other._maxShape;
        }
    }

    AllocArray1D<T>& operator=(const AllocArray1D<T>& other){ // assignment operator
        if (this != &other) {
            AllocArray<T>::operator=(other);
            this->_maxShape = other._maxShape;
        }
        return *this;
    }

    // ----- std::vector-like methods

    void push_back(const T& i){
        if (AllocArray<T>::_shape[0] >= _maxShape) {
            psalg::SysLog::error("AllocArray: maxShape exceeded: %d >= %d\n",
                                 AllocArray<T>::_shape[0],_maxShape);
        }
        new(AllocArray<T>::_data+AllocArray<T>::_shape[0]) T(i);
        AllocArray<T>::_shape[0]++;
    }

    void clear(){
        for(unsigned i = 0; i < Array<T>::num_elem(); i++) {
            // call the destructor of everything we contain
            // this could include decrementing reference counts if we are
            // are array-of-arrays.
            Array<T>::_data[i].~T();
        }
        AllocArray<T>::_shape[0] = 0;
    }

    uint32_t capacity(){
        return _maxShape;
    }

    uint32_t size(){
        return AllocArray<T>::_shape[0];
    }

private:
    uint32_t  _maxShape;

};


} // namespace

#endif // ALLOCARRAY__H
