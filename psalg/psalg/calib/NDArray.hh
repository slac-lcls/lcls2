#ifndef PSALG_NDARRAY_H
#define PSALG_NDARRAY_H

//---------------------------------------------------
// Created on 2018-07-06 by Mikhail Dubrovin
//---------------------------------------------------

#include <iostream> //ostream
#include "xtcdata/xtc/Array.hh" // Array
//#include "psalg/utils/Logger.hh" // for MSG

using namespace std;
//using namespace XtcData; // XtcData::Array

namespace psalg {

//-------------------

template <typename T>
class NDArray : public XtcData::Array<T> {

public:

  //const static size_t MAXNDIM = 10; 
  enum {MAXNDIM = XtcData::MaxRank};
  typedef uint32_t shape_t; // uint32_t
  typedef uint32_t size_t;  // uint32_t
  //typedef XtcData::Array<T>::shape_t shape_t;
  //typedef XtcData::Array<T>::size_t  size_t;

  NDArray(T *data, shape_t* sh=0, const size_t ndim=0) :
    XtcData::Array<T>(data, sh, ndim) {
      assert(ndim<MAXNDIM);
  }

  ~NDArray(){};

  inline void reshape(const size_t* shape, const size_t ndim) {
    assert(ndim<MAXNDIM);
    XtcData::Array<T>::_rank=ndim;
    std::memcpy(XtcData::Array<T>::_shape, shape, sizeof(shape_t)*ndim);
  }

  inline size_t ndim() {return XtcData::Array<T>::rank();}
  inline shape_t* shape() {return XtcData::Array<T>::shape();}
  inline size_t size() {
    size_t s=1; for(size_t i=0; i<ndim(); i++) s*=shape()[i]; 
    return s;
  }

  friend std::ostream& 
  operator << (std::ostream& os, NDArray& o) 
  {
    size_t nd = o.ndim();
    shape_t* sh = o.shape(); 
    os << "ndim=" << nd << " size=" << o.size() << " shape=(";
    for(size_t i=0; i<nd; i++) {os << sh[i]; if(i != (nd-1)) os << ",";}
    os << ")";
    return os;
  }

private:

  /// Copy constructor and assignment are disabled by default
  NDArray(const NDArray&) ;
  NDArray& operator = (const NDArray&) ;
};

} // namespace psalg

#endif // PSALG_NDARRAY_H
