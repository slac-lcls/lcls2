#ifndef PYCALGOS_UTILSDETECTOR_H
#define PYCALGOS_UTILSDETECTOR_H

#include <cstddef>  // for size_t
#include <stdint.h> // for uint8_t, uint16_t etc.

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

//namespace utilsdetector {

typedef unsigned size_t;
typedef float    peds_t;
typedef float    fraw_t;
typedef float    gain_t;
typedef uint8_t  mask_t;

void calib_epix10ka(fraw_t *raw, peds_t *peds, gain_t *gain, mask_t *mask, const size_t& size, fraw_t *out);
//void calib_epix10ka(const fraw_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const size_t& size, fraw_t *out);

//}; // namespace utilsdetector

#endif // PYCALGOS_UTILSDETECTOR_H
// EOF
