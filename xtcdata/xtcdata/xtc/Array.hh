/**
 * @file
 * @brief Declares XtcData::Array, a non-owning view of a C-order (row-major) array, and the MaxRank constant.
 */
#ifndef XTCDATA_ARRAY__H
#define XTCDATA_ARRAY__H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace XtcData
{

// this enum is outside the Array class so people don't have to add a template
// type to access the enum.  Unfortunate, but it seems to be the way
// C++ works.

/** Maximum-rank constant, kept outside Array so it can be used without a template argument (per the comment above). */
enum {MaxRank=5 /**< Value 5. Name constructors abort for rank >= MaxRank; Array::shape(a, b, c, d, e) aborts when the rank is >= MaxRank. */ };

template <typename T>
/**
 * Non-owning view of an array of T: a data pointer, a pointer to a shape array and a rank.
 * Copies are shallow (pointers only). operator() computes C-order (row-major) offsets and aborts with a message if an index is out of range.
 */
class Array {
public:

    /** Wrap data (reinterpreted as T*), shape and rank; nothing is copied. */
    Array(void *data, uint32_t *shape, uint32_t rank) : 
        _shape(shape), _data(reinterpret_cast<T*>(data)), _rank(rank) {}
    /** Construct an empty view: null data and shape pointers, rank 0. */
    Array() : _shape(0), _data(0), _rank(0) {}
    /** Copy the data pointer, shape pointer and rank of a (shallow copy). */
    Array(const Array& a) : _shape(a._shape), _data(a._data), _rank(a._rank) {}
    /**
     * Copy the data pointer, shape pointer and rank of o (shallow); self-assignment does nothing.
     * @return *this.
     */
    Array& operator=(const Array& o){
        if(&o == this) return *this;
        _shape = o._shape;
        _rank = o._rank;
        _data = o._data;
        return *this;
    }
    /** Return a reference to element i. Aborts with a message if i >= shape[0]; the rank is not checked. */
    T& operator()(unsigned i){
        _checkOutOfBounds(i,_shape[0]);
        return _data[i];
    }
    /** Return a reference to element (i, j) at offset i*shape[1]+j. Aborts with a message if an index is out of range; the rank is not checked. */
    T& operator()(unsigned i, unsigned j){
        _checkOutOfBounds(i,_shape[0]);_checkOutOfBounds(j,_shape[1]);
        return _data[i * _shape[1] + j];
    }
    /** Return a reference to element (i, j, k) at the row-major offset. Aborts with a message if an index is out of range; the rank is not checked. */
    T& operator()(unsigned i, unsigned j, unsigned k){
        _checkOutOfBounds(i,_shape[0]);_checkOutOfBounds(j,_shape[1]);_checkOutOfBounds(k,_shape[2]);
        return _data[(i * _shape[1] + j) * _shape[2] + k];
    }
    /** Return a reference to element (i, j, k, l) at the row-major offset. Aborts with a message if an index is out of range; the rank is not checked. */
    T& operator()(unsigned i, unsigned j, unsigned k, unsigned l){
        _checkOutOfBounds(i,_shape[0]);_checkOutOfBounds(j,_shape[1]);_checkOutOfBounds(k,_shape[2]);_checkOutOfBounds(l,_shape[3]);
        return _data[((i * _shape[1] + j) * _shape[2] + k) * _shape[3] + l];
    }
    /** Return a reference to element (i, j, k, l, m) at the row-major offset. Aborts with a message if an index is out of range; the rank is not checked. */
    T& operator()(unsigned i, unsigned j, unsigned k, unsigned l, unsigned m){
        _checkOutOfBounds(i,_shape[0]);_checkOutOfBounds(j,_shape[1]);_checkOutOfBounds(k,_shape[2]);_checkOutOfBounds(l,_shape[3]);_checkOutOfBounds(m,_shape[4]);
        return _data[(((i * _shape[1] + j) * _shape[2] + k) * _shape[3] + l) * _shape[4] + m];
    }
    /** Return the rank. */
    inline uint32_t rank() const {
        return _rank;
    }
    /** Return the shape pointer (not a copy). */
    inline uint32_t* shape() const {
        return _shape;
    }
    /** Return the data pointer. */
    inline T* data(){
        return _data;
    }
    /** Return the data pointer as const T*. */
    inline const T* const_data() const {
        return _data;
    }
    /** Return the product of shape[0] .. shape[rank-1], or 0 if the shape pointer is null. With rank 0 and a non-null shape it returns shape[0]. */
    uint64_t num_elem() const {
      if(!_shape) return 0;
        uint64_t _num_elem = _shape[0];
        for(uint32_t i=1; i<_rank;i++){_num_elem*=_shape[i];};
        return _num_elem;
    }
    /** Write a, b, c, d, e into shape[0..4] (always five entries). Aborts with a message if the rank is 0 or >= MaxRank. */
    void shape(uint32_t a, uint32_t b=0, uint32_t c=0, uint32_t d=0, uint32_t e=0){
        if (_rank <= 0) {
            printf("*** %s:%d: rank %d too small for array\n",__FILE__,__LINE__,_rank);
            abort();
        }
        if (_rank >= MaxRank) {
            printf("*** %s:%d: rank %d too large for array\n",__FILE__,__LINE__,_rank);
            abort();
        }
        _shape[0] = a;
        _shape[1] = b;
        _shape[2] = c;
        _shape[3] = d;
        _shape[4] = e;
    }
    /** Set the rank. */
    inline void set_rank(uint32_t rank) {_rank = rank;}
    /** Set the shape pointer. */
    inline void set_shape(uint32_t *shape) {_shape = shape;}
    /** Set the data pointer (reinterpreted as T*). */
    inline void set_data(void *data) {_data = reinterpret_cast<T*>(data);}

protected:
    uint32_t *_shape;
    T        *_data;
    uint32_t  _rank;

private:
    void _checkOutOfBounds(unsigned index, uint32_t shape) {
        if (index>=shape) {
            printf("*** %s:%d: index %d out of range for shape %d\n",__FILE__,__LINE__,index,shape);
            abort();
        }
    }

};

}; // namespace XtcData

#endif // XTCDATA_ARRAY__H
