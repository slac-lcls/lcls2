/**
 * @file
 * @brief Declares the LOG verbosity bit flags and the basic types of the peakFinder algorithms (namespace types).
 */

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

/** Verbosity bit flags, tested as e.g. m_pbits & LOG::DEBUG in the peak finders. */
namespace LOG {
  /** Verbosity bits. */
  enum {NONE=0, /**< Value 0. */ DEBUG=1, /**< Value 1. */ INFO=2, /**< Value 2. */ WARNING=4, /**< Value 4. */ ERROR=8, /**< Value 8. */ CRITICAL=16 /**< Value 16. */ };
} //namespace LOG 

//-----------------------------
//-----------------------------

/** Basic types of the peakFinder algorithms. */
namespace types {

//-----------------------------
  /** unsigned; shape element type. */
  typedef unsigned shape_t;
  //typedef float    pixel_nrms_t;
  //typedef float    pixel_bkgd_t;
  //typedef uint16_t pixel_mask_t;
  //typedef uint16_t pixel_status_t;
  //typedef double   common_mode_t;
  //typedef float    pedestals_t;
  //typedef float    pixel_gain_t;
  //typedef float    pixel_rms_t;

  /** uint16_t; mask element type. */
  typedef uint16_t mask_t;
  /** uint16_t; element type of the local-extrema maps (e.g. PeakFinderAlgos::localMaxima()). */
  typedef uint16_t extrim_t;
  /** uint16_t; not used by the other peakFinder headers or sources. */
  typedef uint16_t pixstatus_t;
  /** uint32_t; element type of the connected-pixel map (e.g. PeakFinderAlgos::connectedPixels()). */
  typedef uint32_t conmap_t;

//-----------------------------

/** Pair of int indexes i and j. */
struct TwoIndexes {
  int i;  ///< First index.
  int j;  ///< Second index.

  /** Set i to ii and j to jj (both default 0). */
  TwoIndexes(const int& ii=0, const int& jj=0) : i(ii), j(jj) {}

  /**
   * Copy i and j from rhs.
   * @return *this.
   */
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
