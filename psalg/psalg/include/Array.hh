#ifndef ARRAY__H
#define ARRAY__H

#include <assert.h>
#include "xtcdata/xtc/ShapesData.hh"
#include "Allocator.hh"

namespace temp {

template <typename T>
class Array {
public:

    Array(void *data=NULL, uint32_t *shape=NULL, uint32_t rank=0){
        _shape = shape;
        _data = reinterpret_cast<T*>(data);
        _rank = rank;
    }
    T& operator()(int i){
        assert(i < (int)_shape[0]);
        return _data[i];
    }
    T& operator()(int i, int j){
        assert(i<(int)_shape[0]);assert(j<(int)_shape[1]);
        return _data[i * _shape[1] + j];
    }
    const T& operator()(int i, int j) const{
        assert(i< _shape[0]);assert(j<_shape[1]);
        return _data[i * _shape[1] + j];
    }
    T& operator()(int i, int j, int k){
        assert(i< _shape[0]);assert(j<_shape[1]);assert(k<_shape[3]);
        return _data[(i * _shape[1] + j) * _shape[2] + k];
    }
    const T& operator()(int i, int j, int k) const
    {
        assert(i< _shape[0]);assert(j<_shape[1]);assert(k<_shape[3]);
        return _data[(i * _shape[1] + j) * _shape[2] + k];
    }
    uint32_t rank(){
        return _rank;
    }
    uint32_t* shape(){
        return _shape;
    }
    T* data(){
        return _data;
    }
    uint64_t num_elem(){
        uint64_t _num_elem = _shape[0];
        for(uint32_t i=1; i<_rank;i++){_num_elem*=_shape[i];};
        return _num_elem;
    }
    void shape(uint32_t a, uint32_t b=0, uint32_t c=0, uint32_t d=0, uint32_t e=0){
        assert(_rank > 0);
        assert(XtcData::Name::MaxRank == 5);
        _shape[0] = a;
        _shape[1] = b;
        _shape[2] = c;
        _shape[3] = d;
        _shape[4] = e;
    }

protected:
    uint32_t *_shape;
    T        *_data;
    uint32_t  _rank;
};

template <typename T>
class AllocArray:public Array<T>{
public:

    AllocArray(Allocator& allocator, size_t maxElem, uint32_t rank):_allocator(allocator){
        void *ptr = _allocator.malloc(sizeof(XtcData::Shape) +
                                            sizeof(*_refCntPtr) +
                                            maxElem*sizeof(T)); // shape + refcnt + data
        Array<T>::_shape = reinterpret_cast<uint32_t*>(ptr);
        _refCntPtr = reinterpret_cast<uint32_t*>(Array<T>::_shape+XtcData::Name::MaxRank);
        Array<T>::_data = reinterpret_cast<T*>(_refCntPtr+1);
        Array<T>::_rank = rank;
        *_refCntPtr = 1;
        for(int i = 0; i < XtcData::Name::MaxRank; i++) Array<T>::_shape[i] = 0;
    }

    AllocArray(const AllocArray<T>& other):_allocator(other._allocator){ // copy constructor
        if (this != &other) {
            //std::cout << "# copy" << std::endl;
            this->_shape = other._shape;
            this->_data = other._data;
            this->_rank = other._rank;
            this->_refCntPtr = other._refCntPtr;
            refCnt()++; // increment reference count in the original object
        }
    }

    AllocArray<T>& operator=(const AllocArray<T>& other){ // assignment operator
        if (this != &other) {
            //std::cout << "# assign" << std::endl;
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

    // std::vector-like methods

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
        } // TODO: look at assembly code when T is a float, gcc -S
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
