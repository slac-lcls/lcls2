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

    Array(void *data, uint32_t *shape, uint32_t rank) : 
        _shape(shape), _data(reinterpret_cast<T*>(data)), _rank(rank) {}
    Array() : _shape(0), _data(0), _rank(0) {}
    Array(const Array& a) : _shape(a._shape), _data(a._data), _rank(a._rank) {}
    Array& operator=(Array& o){
        if(&o == this) return *this;
        _shape = o.shape;
        _rank = o.rank;
        _data = o._data;
        return *this;
    }
    T& operator()(unsigned i){
        assert(i<_shape[0]);
        return _data[i];
    }
    T& operator()(unsigned i, unsigned j){
        assert(i<_shape[0]);assert(j<_shape[1]);
        return _data[i * _shape[1] + j];
    }
    T& operator()(unsigned i, unsigned j, unsigned k){
        assert(i<_shape[0]);assert(j<_shape[1]);assert(k<_shape[2]);
        return _data[(i * _shape[1] + j) * _shape[2] + k];
    }
    T& operator()(unsigned i, unsigned j, unsigned k, unsigned l){
        assert(i<_shape[0]);assert(j<_shape[1]);assert(k<_shape[2]);assert(l<_shape[3]);
        return _data[((i * _shape[1] + j) * _shape[2] + k) * _shape[3] + l];
    }
    T& operator()(unsigned i, unsigned j, unsigned k, unsigned l, unsigned m){
        assert(i<_shape[0]);assert(j<_shape[1]);assert(k<_shape[2]);assert(l<_shape[3]);assert(m<_shape[4]);
        return _data[(((i * _shape[1] + j) * _shape[2] + k) * _shape[3] + l) * _shape[4] + m];
    }
    inline uint32_t rank() const {
        return _rank;
    }
    inline uint32_t* shape() const {
        return _shape;
    }
    inline T* data(){
        return _data;
    }
    inline const T* const_data() const {
        return _data;
    }
    uint64_t num_elem(){
      if(!_shape) return 0;
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
    inline void set_rank(uint32_t rank) {_rank = rank;}
    inline void set_shape(uint32_t *shape) {_shape = shape;}
    inline void set_data(void *data) {_data = reinterpret_cast<T*>(data);}

protected:
    uint32_t *_shape;
    T        *_data;
    uint32_t  _rank;
};

}; // namespace XtcData

#endif // XTCDATA_ARRAY__H
