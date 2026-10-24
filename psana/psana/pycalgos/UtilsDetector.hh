/**
 * @file
 * @brief Declares utilsdetector pixel-calibration functions (pedestal subtraction and gain, several Jungfrau constant layouts) and a test singleton.
 */
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

/** Pixel calibration helpers (pedestal subtraction and gain) for raw detector data, and a test singleton class. */
namespace utilsdetector {

//typedef long unsigned int size_t;
/** double; type of the execution times in microseconds returned by the calib_* functions. Inside utilsdetector it hides the standard time_t. */
typedef double   time_t;
/** uint16_t; raw data element type. */
typedef uint16_t rawd_t;
/** float; pedestal element type. */
typedef float    peds_t;
/** float; gain element type. */
typedef float    gain_t;
/** float; output element type. */
typedef float    out_t;
/** float; element type of the combined calibration-constant arrays (pedestals and gains). */
typedef float    cc_t;
/** uint8_t; mask element type (multiplied into the result by calib_std() and calib_jungfrau_v0()). */
typedef uint8_t  mask_t;
/** uint32_t; size and index type. */
typedef uint32_t sizeb_t;

/** 4: number of gain-bit combinations, the first dimension of the static table filled by fill_CCSV3(). */
#define NGRINDS 4
/** 16777216: number of pixels per gain combination in the static table filled by fill_CCSV3(). */
#define NPIXELS 16777216
/** 2; not used anywhere else in psana. */
#define NCTYPES 2

  /** Pedestal and gain of one pixel for one gain-bit combination, as stored in the static table of fill_CCSV3(). */
  struct ccstruct{
    peds_t pedestal;  ///< Pedestal value.
    gain_t gain;  ///< Gain value.
  };

  /** On the first call only, copy cc, laid out as [NGRINDS][NPIXELS][2] (pedestal, gain), into the static table used by calib_jungfrau_v3_struct(), printing progress and the first five pixels of each combination to stdout. Later calls return at once. */
  void fill_CCSV3(const cc_t *cc);

  /** For each of size_blk pixels compute out[i] = ((raw[i] & 0x3fff) - cc[8*i+g]) * cc[8*i+g+4], where g = raw[i] >> 14; cc holds 4 pedestals then 4 gains per pixel. */
  void  calib_jungrfau_blk_v1(const rawd_t *raw, const cc_t *cc, const sizeb_t& size_blk, out_t *out);

  /**
   * Compute out[i] = ((raw[i] & databits) - peds[i]) * gain[i] * mask[i] for size elements.
   * @return Execution time in microseconds (steady_clock).
   */
  time_t calib_std(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const sizeb_t& size, const rawd_t databits, out_t *out);
  /**
   * For each pixel take g = raw[i] >> 14 (values above 1 become 2) and compute out[i] = ((raw[i] & 0x3fff) - peds[g*size+i]) * gain[g*size+i] * mask[i].
   * @return Execution time in microseconds.
   */
  time_t calib_jungfrau_v0(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const sizeb_t& size, out_t *out);
  /**
   * Apply calib_jungrfau_blk_v1() to consecutive blocks of size_blk pixels, with cc holding 8 values per pixel; the last size % size_blk pixels are not processed.
   * @return Execution time in microseconds.
   */
  time_t calib_jungfrau_v1(const rawd_t *raw, const cc_t *cc, const sizeb_t& size, const sizeb_t& size_blk, out_t *out);
  /**
   * Compute out[i] = ((raw[i] & 0x3fff) - cc[i+size*g]) * cc[i+size*g+4*size] with g = raw[i] >> 14, i.e. cc laid out as [2][4][size] (pedestals, then gains). size_blk is not used.
   * @return Execution time in microseconds.
   */
  time_t calib_jungfrau_v2(const rawd_t *raw, const cc_t *cc, const sizeb_t& size, const sizeb_t& size_blk, out_t *out);
  /**
   * Compute out[i] = ((raw[i] & 0x3fff) - cc[2*(i+npix*g)]) * cc[2*(i+npix*g)+1] with g = raw[i] >> 14, i.e. cc laid out as [4][npix][2]. size_blk is not used.
   * @return Execution time in microseconds.
   */
  time_t calib_jungfrau_v3(const rawd_t *raw, const cc_t *cc, const sizeb_t& npix, const sizeb_t& size_blk, out_t *out);
  /**
   * Fill the static table from cc on the first call (fill_CCSV3(), which always reads NPIXELS pixels per combination), then compute out[i] = ((raw[i] & 0x3fff) - pedestal) * gain from table entry [raw[i] >> 14][i]. size_blk is not used.
   * @return Execution time in microseconds, including the table fill.
   */
  time_t calib_jungfrau_v3_struct(const rawd_t *raw, const cc_t *cc, const sizeb_t& size, const sizeb_t& size_blk, out_t *out);
  /** Do nothing and return the time between two steady_clock reads, in microseconds. */
  time_t calib_jungfrau_v4_empty();
  /** Ignore all arguments and return calib_jungfrau_v4_empty(). */
  time_t calib_jungfrau_v5_empty(const rawd_t *raw, const cc_t *cc, const sizeb_t& npix, const sizeb_t& size_blk, out_t *out);

  /** Test singleton: instance() creates the single object (the private constructor prints a message) and print() prints the class name. */
  class CalibConsSingleton{
  public:
    /** Return the single instance, creating it on the first call; creation prints a message to stdout. */
    static CalibConsSingleton* instance();
    /** Print "CalibConsSingleton::print()" to stdout. */
    void print();
  private:
    CalibConsSingleton();                 // !!!!! Private so that it can not be called from outside
    virtual ~CalibConsSingleton(){};
    static CalibConsSingleton* m_pInstance; // !!!!! Singleton instance
    // Copy constructor and assignment are disabled by default
    CalibConsSingleton(const CalibConsSingleton&);
    CalibConsSingleton& operator = (const CalibConsSingleton&);
  };

}; // namespace utilsdetector

#endif // PYCALGOS_UTILSDETECTOR_H
// EOF
