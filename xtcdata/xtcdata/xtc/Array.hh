#ifndef XTCDATA_ARRAY__H
#define XTCDATA_ARRAY__H

#include <stdint.h>
#include <assert.h>

namespace XtcData
{

// this enum is outside the Array class so people don't have to add a template
// type to access the enum.  Unfortunate, but it seems to be the way
// C++ works.

enum {MaxRank=5};

template <typename T>
class Array {
public:

    typedef uint32_t shape_t;
    typedef uint32_t size_t;

    Array(void *data, uint32_t *shape, uint32_t rank){
        _shape = shape;
        _data = reinterpret_cast<T*>(data);
        _rank = rank;
    }
    T& operator()(unsigned i){
        assert(i < _shape[0]);
        return _data[i];
    }
    T& operator()(unsigned i, unsigned j){
        assert(i<_shape[0]);assert(j<_shape[1]);
        return _data[i * _shape[1] + j];
    }
    const T& operator()(unsigned i, unsigned j) const{
        assert(i< _shape[0]);assert(j<_shape[1]);
        return _data[i * _shape[1] + j];
    }
    T& operator()(unsigned i, unsigned j, unsigned k){
        assert(i< _shape[0]);assert(j<_shape[1]);assert(k<_shape[3]);
        return _data[(i * _shape[1] + j) * _shape[2] + k];
    }
    const T& operator()(unsigned i, unsigned j, unsigned k) const
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
        assert(_rank < MaxRank);
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
    Array(){}
};

}; // namespace XtcData

#endif // XTCDATA_ARRAY__H
