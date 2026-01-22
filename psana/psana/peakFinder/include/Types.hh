
#ifndef PSALGOS_TYPES_H
#define PSALGOS_TYPES_H

//-----------------------------
// Types.h 2017-08-05
//-----------------------------

#include <cstddef>  // for size_t
#include <stdint.h> // for uint8_t, uint16_t etc.

/*
template <typename T>
class Vector {
  public:
    Vector(){}
    unsigned int len, capacity = 0;
    T* data[4500]; // TODO: fixed size (larger size seems to slows down code)
};*/

namespace LOG {
  enum {NONE=0, DEBUG=1, INFO=2, WARNING=4, ERROR=8, CRITICAL=16};
} //namespace LOG 

//-----------------------------
//-----------------------------

namespace types {

//-----------------------------
  typedef unsigned shape_t;
  //typedef float    pixel_nrms_t;
  //typedef float    pixel_bkgd_t;
  //typedef uint16_t pixel_mask_t;
  //typedef uint16_t pixel_status_t;
  //typedef double   common_mode_t;
  //typedef float    pedestals_t;
  //typedef float    pixel_gain_t;
  //typedef float    pixel_rms_t;

  typedef uint16_t mask_t;
  typedef uint16_t extrim_t;
  typedef uint16_t pixstatus_t;
  typedef uint32_t conmap_t;

//-----------------------------

struct TwoIndexes {
  int i;
  int j;

  TwoIndexes(const int& ii=0, const int& jj=0) : i(ii), j(jj) {}

  TwoIndexes& operator=(const TwoIndexes& rhs) {
    i = rhs.i;
    j = rhs.j;
    return *this;
  }
};

//-----------------------------
} // namespace types
//-----------------------------
#endif // PSALGOS_TYPES_H
//-----------------------------
