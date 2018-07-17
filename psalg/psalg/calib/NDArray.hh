#ifndef PSALG_NDARRAY_H
#define PSALG_NDARRAY_H

//---------------------------------------------------
// Created on 2018-07-06 by Mikhail Dubrovin
//---------------------------------------------------

/** Usage example
 *
 *  #include "psalg/calib/NDArray.hh"
 *
 *  typedef psalg::types::shape_t shape_t; // uint32_t
 *  typedef psalg::types::size_t  size_t;  // uint32_t
 *
 *  float data[] = {1,2,3,4,5,6,7,8,9,10,11,12};
 *  uint32_t sh[2] = {3,4};
 *  uint32_t ndim = 2;
 *
 *  // 1: use external data buffer:
 *  NDArray<float> a(sh, ndim, data);
 *
 *  // 2: use internal data buffer:
 *  NDArray<float> b(sh, ndim);
 *  // or set it later
 *  b.set_data_buffer(data) 
 *
 *  // 3: instatiate empty object and initialize it later
 *  NDArray<float> c();
 *  c.set_shape(sh, ndim);    // or alias reshape(sh, ndim)
 *  c.set_data_buffer(data);
 *
 *  size_t   ndim = a.ndim();
 *  size_t   size = a.size();
 *  shape_t* sh   = a.shape();
 *  T* data       = a.data();
 *
 *  T value = a(1,2);
 *
 *  std::cout << "ostream array: " << a << '\n';
 */

#include <iostream> //ostream
#include "xtcdata/xtc/Array.hh" // Array
#include "psalg/calib/Types.hh"
#include "psalg/utils/Logger.hh" // for MSG
#include <typeinfo>       // typeid

using namespace std;
//using namespace XtcData; // XtcData::Array

namespace psalg {

//-------------------

template <typename T>
class NDArray : public XtcData::Array<T> {

public:

  typedef psalg::types::shape_t shape_t; // uint32_t
  typedef psalg::types::size_t  size_t;  // uint32_t
  typedef XtcData::Array<T> base;  // base class scope

  //const static size_t MAXNDIM = 10; 
  enum {MAXNDIM = XtcData::MaxRank};

  NDArray(shape_t* sh, const size_t ndim, void *buf=0) :
    base(), _buf_ext(0), _buf_own(0)
  {
     set_shape(sh, ndim);
     set_data_buffer(buf);
  }

  NDArray() : base(), _buf_ext(0), _buf_own(0) {}

  ~NDArray(){if(_buf_own) delete _buf_own;}

  inline void set_shape(shape_t* shape, const size_t ndim) {
    MSG(TRACE, "set_shape for ndim="<<ndim);
    assert(ndim<MAXNDIM);
    base::_rank=ndim;
    std::memcpy(_shape, shape, sizeof(shape_t)*ndim);
    base::_shape = shape;
  }

  inline void reshape(shape_t* shape, const size_t ndim) {set_shape(shape, ndim);}

  inline T* data() {return base::data();}
  inline shape_t* shape() {return base::shape();}
  inline size_t ndim() {return base::rank();}
  inline size_t size() {
    size_t s=1; for(size_t i=0; i<ndim(); i++) s*=shape()[i]; 
    return s;
  }


  inline void set_data_buffer(void *buf=0) { // base::_data=reinterpret_cast<T*>(buf);}
    if(_buf_own) delete _buf_own;  
    if(buf) {
      _buf = _buf_ext = reinterpret_cast<T*>(buf);
    }
    else {
      _buf = _buf_own = new T[size()];
    }
    base::_data = _buf; //reinterpret_cast<T*>(_buf);
  }


  friend std::ostream& 
  operator << (std::ostream& os, NDArray& o) 
  {
    size_t nd = o.ndim();
    if(nd) {
      shape_t* sh = o.shape(); 
      os << "typeid=" << typeid(T).name() << " ndim=" << nd << " size=" << o.size() << " shape=(";
      for(size_t i=0; i<nd; i++) {os << sh[i]; if(i != (nd-1)) os << ",";}
      os << ")";
    }
    return os;
  }


private:
  shape_t _shape[MAXNDIM];
  T* _buf_ext;
  T* _buf_own;
  T* _buf;
  /// Copy constructor and assignment are disabled by default
  NDArray(const NDArray&) ;
  NDArray& operator = (const NDArray&) ;
};

} // namespace psalg

#endif // PSALG_NDARRAY_H
