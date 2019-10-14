#ifndef PSALG_NDARRAY_H
#define PSALG_NDARRAY_H

//---------------------------------------------------
// Created on 2018-07-06 by Mikhail Dubrovin
//---------------------------------------------------

/** Usage
 *
 *  #include "psalg/calib/NDArray.hh"
 *
 *  typedef psalg::types::shape_t shape_t; // uint32_t
 *  typedef psalg::types::size_t  size_t;  // uint32_t
 *
 *  float data[] = {1,2,3,4,5,6,7,8,9,10,11,12};
 *  shape_t sh[2] = {3,4};
 *  size_t ndim = 2;
 *
 *  // use external data buffer:
 *  NDArray<float> a(sh, ndim, data);
 *
 *  // set from NDArray or XtcData::Array a
 *  NDArray(a);
 *  
 *  // use internal data buffer:
 *  NDArray<float> b(sh, ndim);
 *  // or set it later
 *  b.set_data_buffer(data) 
 *
 *  // 3: instatiate empty object and initialize it later
 *  NDArray<float> c;
 *  c.set_shape(sh, ndim);    // or alias reshape(sh, ndim)
 *  c.set_shape(str_shape);   // sets _shape and _rank from string
 *  c.set_data_buffer(data);
 *  c.set_ndarray(a);         // where a is NDArray or XtcData::Array
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

#include <algorithm>    // std::sort, reverse, replace
#include <iostream> //ostream
#include "xtcdata/xtc/Array.hh" // Array
#include "psalg/calib/Types.hh"
#include "psalg/utils/Logger.hh" // for MSG
#include <typeinfo> // typeid
#include <cstring>  // memcpy
#include <type_traits> //std::remove_const, is_const

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

  using NON_CONST_T = typename remove_const<T>::type;  // non-const T

  bool T_IS_CONST = std::is_const<T>::value;

  //const static size_t MAXNDIM = 10; 
  enum {MAXNDIM = XtcData::MaxRank};

//-------------------

  NDArray(const shape_t* sh, const size_t ndim, void *buf_ext=0) :
    base(), _buf_ext(0), _buf_own(0)
  {
     set_shape(sh, ndim);
     set_data_buffer(buf_ext);
  }

//-------------------

  NDArray(const shape_t* sh, const size_t ndim, const void *buf_ext) :
    base(), _buf_ext(0), _buf_own(0)
  {
     set_shape(sh, ndim);
     set_const_data_buffer(buf_ext);
  }

//-------------------

  NDArray() : base(), _buf_ext(0), _buf_own(0) {}

//-------------------

  ~NDArray(){if(_buf_own) delete _buf_own;} // delete &_shape;}

//-------------------

// copy constructor from XtcData::Array<T>&
  NDArray(const base& o) :
    base(), _buf_ext(0), _buf_own(0)
  {
    //set_ndarray(o);
    set_shape(o.shape(), o.rank());
    set_data_copy(o.const_data());
  }

//-------------------
// copy constructor from NDArray<T>&
  NDArray(const NDArray<T>& o) :
    base(), _buf_ext(0), _buf_own(0)
  {
    set_shape(o.shape(), o.rank());
    set_data_copy(o.const_data()); // protected o._data - also works for derived class
  }

//-------------------

  //NDArray(const NDArray<T>&) = delete;
  //NDArray<T>& operator = (const NDArray<T>&) = delete;

//-------------------

  NDArray<T>& operator=(const NDArray<T>& o)
  {
    if(&o == this) return *this;
    base::operator=(o);
    set_shape(o.shape(), o.rank());
    set_data_copy(o.const_data());
    return *this;
  }

//-------------------

  inline void set_shape(const shape_t* shape=NULL, const size_t ndim=0) {
    //MSG(TRACE, "set_shape for ndim="<<ndim);
    assert(ndim<MAXNDIM);
    if(shape) std::memcpy(_shape, shape, sizeof(shape_t)*ndim);
    base::_rank = ndim;
    base::_shape = _shape;
  }

//-------------------
/// Converts string like "(5920, 388)" to array for _shape[]={5920, 388}

  inline void set_shape_string(const std::string& str) {
    //MSG(DEBUG, "set_shape for " << str);

    std::string s(str.substr(1, str.size()-2)); // remove '(' and ')'
    std::replace(s.begin(), s.end(), ',', ' '); // remove comma ',' separators

    std::stringstream ss(s);
    std::string fld;
    size_t ndim(0);
    do {ss >> fld; 
      shape_t v = (shape_t)atoi(fld.c_str());
      //cout << "   " << v << endl;
      _shape[ndim++]=v;
    } while(ss.good() && ndim < MAXNDIM);

    assert(ndim<MAXNDIM);
    base::_rank = ndim;
    base::_shape = _shape;
  }

//-------------------

  inline void reshape(const shape_t* shape, const size_t ndim) {set_shape(shape, ndim);}

  inline T* data() {return base::data();}

  inline const T* const_data() const {return base::const_data();}

  inline shape_t* shape() const {return base::shape();}

  inline size_t ndim() const {return base::rank();}

  inline size_t rank() const {return base::rank();}

//-------------------
// the same as uint64_t Array::num_elem()
//  inline uint64_t size() const {return base::num_elem();}
  inline size_t size() const {
    size_t s=1; for(size_t i=0; i<ndim(); i++) s*=shape()[i]; 
    return s;
  }

//-------------------

  inline void set_ndarray(base& a) {
     set_shape(a.shape(), a.rank());
     set_data_buffer(a.data()); // (void*)a.data()
  }

//-------------------

  inline void set_ndarray(NDArray<T>& a) {
     set_shape(a.shape(), a.rank());
     set_data_buffer(a.data());
  }


//-------------------
/// CONST !!! *buf
/// sets pointer to external data buffer
/// WARNING shape needs to be set first, othervice size() is undefined!

  inline void set_const_data_buffer(const void *buf_ext=0) {
    //MSG(TRACE, "In set_data_buffer *buf=" << buf_ext);
    if(_buf_own) {
       delete _buf_own;
       _buf_own = 0;
    }
    if(buf_ext) {
      _buf_ext = base::_data = reinterpret_cast<T*>(buf_ext);
    }
  }

  //  T_IS_CONST

//-------------------
/// sets pointer to data
/// WARNING shape needs to be set first, othervice size() is undefined!

  inline void set_data_buffer(void *buf_ext=0) { // base::_data=reinterpret_cast<T*>(buf_ext);}
    //MSG(TRACE, "In set_data_buffer *buf=" << buf_ext);
    if(_buf_own) {
       delete _buf_own;  
      _buf_own = 0;
    }
    if(buf_ext) {
      _buf_ext = base::_data = reinterpret_cast<T*>(buf_ext);
    }
    else {
      _buf_own = base::_data = new NON_CONST_T [size()];
      _buf_ext = 0;
    }
  }

//-------------------
/// copy content of external buffer in object memory buffer 
/// WARNING shape needs to be set first, othervice size() is undefined!

  inline void set_data_copy(const void *buf=0) {
    //MSG(TRACE, "In set_data_copy *buf=" << buf);
    if(_buf_own) delete _buf_own;  
    _buf_own = new NON_CONST_T [size()];
    std::memcpy(_buf_own, buf, sizeof(NON_CONST_T)*size());
    base::_data = _buf_own;
    _buf_ext = 0;
  }

//-------------------
/// Reserves internal data buffer for array of requested size.
/// For externally defined data buffer do not do anything,
/// othervise reserves memory on heap.
/// size is a number of values

  inline void reserve_data_buffer(const size_t size) {
    //MSG(TRACE, "In get_data_buffer size=" << size);
    if(_buf_ext) return;
    if(_buf_own) delete _buf_own;  
    _buf_own = new NON_CONST_T [size];
    base::_data = _buf_own;
  }

//-------------------

  friend std::ostream& 
  operator << (std::ostream& os, const NDArray<T>& o) 
  {
    size_t nd = o.ndim();
    if(nd) {
      shape_t* sh = o.shape();
      os << "typeid=" << typeid(T).name() << " ndim=" << nd << " size=" << o.size() << " shape=(";
      for(size_t i=0; i<nd; i++) {os << sh[i]; if(i != (nd-1)) os << ", ";}
      os << ") data=";
      size_t nvals = min((size_t)4, o.size());
      const T* d = o.const_data();
      for(size_t i=0; i<nvals; i++, d++) {os << *d; if(i<nvals) os << ", ";} if(nvals>1) os << "...";
    }
    return os;
  }

//-------------------

private:
  shape_t _shape[MAXNDIM];
  T* _buf_ext;
  NON_CONST_T* _buf_own;

};

} // namespace psalg

#endif // PSALG_NDARRAY_H
