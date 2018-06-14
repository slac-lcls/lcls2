
#ifndef PSALG_ARRAYMETADATA_H
#define PSALG_ARRAYMETADATA_H

//---------------------------------------------------
// Created on 2018-06-12 by Mikhail Dubrovin
//---------------------------------------------------

//#include <string>
//#include <iostream> // cout, puts etc.
//#include <fstream>
//#include <stdint.h>   // uint8_t, uint32_t, etc.
//#include <cstring>  // memcpy
// #include <iosfwd> // 
//#include <cstddef>  // size_t 
#include <iostream> //ostream
//#include "xtcdata/xtc/DescData.hh" // for Array
//#include "psalg/include/Logger.h" // for MsgLog

using namespace std;

namespace psalg {

//-------------------

class ArrayMetadata {

public:

  //const static size_t MAXNDIM = 10; 
  enum {MAXNDIM = 10};
  //typedef std::size_t shape_t;
  typedef uint32_t shape_t;
  typedef uint32_t size_t;

  ArrayMetadata(const shape_t* shape=0, const size_t ndim=0) : _shape(shape), _ndim(ndim) {
    //std::memcpy(_shape, shape, sizeof(ndim_t)*ndim);
    //set(shape, ndim);
  }

  ~ArrayMetadata(){};

  inline void set(const size_t* shape, const size_t ndim) {_shape=shape; _ndim=ndim;}

  //inline const char* __name__() {return (const char*)"ArrayMetadata";}
  inline const size_t   size()  const {size_t s=1; for(size_t i=0; i<_ndim; i++) s*=_shape[i]; return s;}
  inline const size_t   ndim()  const {return _ndim;}
  inline const shape_t* shape() const {return _shape;}

  friend std::ostream& 
  operator << (std::ostream& os, const ArrayMetadata& o) 
  {
    const size_t   nd = o.ndim();
    const shape_t* sh = o.shape(); 
    //const size_t size = o.size(); 
    os << "ndim=" << nd << " size=" << o.size() << " shape=(";
    for(size_t i=0; i<nd; i++) {os << sh[i]; if(i != (nd-1)) os << ",";}
    os << ")";
    return os;
  }

private:
  const shape_t* _shape;
  size_t _ndim;

  /// Copy constructor and assignment are disabled by default
  ArrayMetadata(const ArrayMetadata&) ;
  ArrayMetadata& operator = (const ArrayMetadata&) ;
};

} // namespace psalg

#endif // PSALG_ARRAYMETADATA_H
