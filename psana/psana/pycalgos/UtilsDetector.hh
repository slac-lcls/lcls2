#ifndef PYCALGOS_UTILSDETECTOR_H
#define PYCALGOS_UTILSDETECTOR_H

#include <cstddef>  // for size_t
#include <stdint.h> // for uint8_t, uint16_t etc.

//#include <sstream>   // for stringstream
//#include <string>
//#include <vector>
//#include <iostream> // for cout, ostream
//#include <cstring>  // for memcpy
//#include <cmath>    // for sqrt
//#include <cstddef>  // for size_t
//#include "Types.hh"
//#include "psalg/alloc/AllocArray.hh"
//#include "psalg/alloc/Allocator.hh"

//using namespace psalg; // Array

using namespace std;

namespace utilsdetector {

//typedef long unsigned int size_t;
typedef double   time_t;
typedef uint16_t rawd_t;
typedef float    peds_t;
typedef float    gain_t;
typedef float    out_t;
typedef float    cc_t; 
typedef uint8_t  mask_t;

  struct ccstruct{
    peds_t pedestal;
    gain_t gain;
  };

  void  calib_jungrfau_blk_v1(const rawd_t *raw, const cc_t *cc, const size_t& size_blk, out_t *out);

  time_t calib_std        (const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const size_t& size, const rawd_t databits, out_t *out);
  time_t calib_jungfrau_v0(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const size_t& size, out_t *out);
  time_t calib_jungfrau_v1(const rawd_t *raw, const cc_t *cc, const size_t& size, const size_t& size_blk, out_t *out);

}; // namespace utilsdetector

#endif // PYCALGOS_UTILSDETECTOR_H
// EOF
