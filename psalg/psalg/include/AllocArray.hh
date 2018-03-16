#ifndef ARRAY__H
#define ARRAY__H

#include <assert.h>
#include "xtcdata/xtc/ShapesData.hh" // Shape
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
        _refCntPtr = reinterpret_cast<uint32_t*>(Array<T>::_shape+Name::MaxRank);
        Array<T>::_data = reinterpret_cast<T*>(_refCntPtr+1);
        Array<T>::_rank = rank;
        *_refCntPtr = 1;
        for(int i = 0; i < Name::MaxRank; i++) Array<T>::_shape[i] = 0;
    }

    AllocArray(const AllocArray<T>& other):_allocator(other._allocator){ // copy constructor
        if (this != &other) {
            this->_shape = other._shape;
            this->_data = other._data;
            this->_rank = other._rank;
            this->_refCntPtr = other._refCntPtr;
            refCnt()++; // increment reference count in the original object
        }
    }

    AllocArray<T>& operator=(const AllocArray<T>& other){ // assignment operator
        if (this != &other) {
            refCnt()--;
            if (refCnt() == 0) _allocator.free(this->_shape);

            this->_shape = other._shape;
            this->_data = other._data;
            this->_rank = other._rank;
            this->_refCntPtr = other._refCntPtr;
            this->_allocator = other._allocator;
            refCnt()++; // increment reference count in the original object
        }
        return *this;
    }

    virtual ~AllocArray(){
        assert(refCnt()>0);
        refCnt()--;
        if(refCnt()==0) {
            for(unsigned i = 0; i < Array<T>::num_elem(); i++) {
            // call the destructor of everything we contain
            // this could include decrementing reference counts if we are
            // are array-of-arrays.
            Array<T>::_data[i].~T();
            }
            _allocator.free(Array<T>::_shape); // free the memory
        }
    }

    uint32_t& refCnt(){
        return *_refCntPtr;
    }

    void incRefCnt(){
        refCnt()++;
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
        assert(AllocArray<T>::_shape[0] < _maxShape);
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

#endif // ARRAY__H
