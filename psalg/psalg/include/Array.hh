#ifndef ARRAY__H
#define ARRAY__H

#include <assert.h>
#include "xtcdata/xtc/ShapesData.hh"
#include "Heap.hh"

namespace temp {

template <typename T>
class Array {
public:
  Array() {
        _rank=0;
  }

  Array(void *data, uint32_t *shape, uint32_t rank){
        _data = reinterpret_cast<T*>(data);
        _shape = shape;
        _rank = rank;
    }

  Array(void *ptr, uint32_t rank){
        _shape = reinterpret_cast<uint32_t*>(ptr);
        _data = reinterpret_cast<T*>(_shape+XtcData::Name::MaxRank);
        _rank = rank;
        for(int i = 0; i < XtcData::Name::MaxRank; i++) _shape[i] = 0;
    }

  Array(Heap& heap, size_t size, uint32_t rank){
        void *ptr = heap.malloc_array(size);
        _shape = reinterpret_cast<uint32_t*>(ptr);
        _data = reinterpret_cast<T*>(_shape+XtcData::Name::MaxRank);
        _rank = rank;
        for(int i = 0; i < XtcData::Name::MaxRank; i++) _shape[i] = 0;
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

    void push_back(T i){
        assert(_rank==1);
        _data[_shape[0]++] = i;
    }

    T& operator()(int i){
        assert(i < (int) num_elem());
        return _data[i];
    }
    T& operator()(int i, int j){
        assert(i<_shape[0]);assert(j<_shape[1]);
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

private:
    uint32_t *_shape;
    T *_data;
    uint32_t _rank;
    //TODO: bounds check
};

}

#endif // ARRAY__H