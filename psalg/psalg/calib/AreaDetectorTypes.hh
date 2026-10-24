/**
 * @file
 * @brief Declares detector-level types: detector::CALIB_TYPE, element-type aliases, the AREADETTYPE enum and the name-to-type map with its lookup functions.
 */
#ifndef PSALG_AREADETECTORTYPES_H
#define PSALG_AREADETECTORTYPES_H

/** Usage
 *
 * #include "psalg/calib/AreaDetectorTypes.hh"
 */

#include "psalg/calib/CalibParsTypes.hh"
#include <string>
#include <map>

//#include "psalg/utils/Logger.hh" // MSG, LOGGER

using namespace calib;

//-------------------

/** Namespace of the psalg detector classes and types (Detector, AreaDetector and subclasses, AREADETTYPE, ...). */
namespace detector {

  /** Calibration types (values 0..7). Same name as calib::CALIB_TYPE but a shorter list, so COMMON_MODE is 7 here. */
  enum CALIB_TYPE {PEDESTALS=0, /**< Value 0. */ PIXEL_RMS, /**< Value 1. */ PIXEL_STATUS, /**< Value 2. */ PIXEL_GAIN, /**< Value 3. */ PIXEL_OFFSET, /**< Value 4. */ PIXEL_MASK, /**< Value 5. */ PIXEL_BKGD, /**< Value 6. */ COMMON_MODE /**< Value 7. */ };

  //typedef psalg::types::shape_t shape_t; // uint32_t
  //typedef psalg::types::size_t  size_t;  // uint32_t

  /** uint32_t; element type of shape arrays (e.g. AreaDetector::shape()). */
  typedef uint32_t shape_t;
  /** uint32_t. Inside namespace detector it hides the standard size_t (e.g. AreaDetector::size() returns this type). */
  typedef uint32_t size_t;

  /*
  typedef float    pixel_rms_t;
  typedef float    pixel_bkgd_t;
  typedef uint16_t pixel_mask_t;
  typedef uint16_t pixel_status_t;
  typedef double   common_mode_t;
  typedef float    pedestals_t;
  typedef float    pixel_gain_t;
  typedef float    pixel_offset_t;
  typedef float    pixel_rms_t;
  typedef uint32_t pixel_idx_t;
  typedef float    pixel_coord_t;
  typedef float    pixel_size_t;
  typedef float    tilt_angle_t;
  typedef float    geometry_t; // ??? TEMPORARY
  */

  /** uint16_t; element type of AreaDetector::raw(const event_t&). */
  typedef uint16_t raw_t;
  /** float; the type of the event argument of the AreaDetector accessors. The trailing comment relates it to query_t. */
  typedef float    event_t; // query_t
  /** float; element type of AreaDetector::calib(). */
  typedef float    calib_t;
  /** float; element type of AreaDetector::image(). */
  typedef float    image_t;

  /** Area detector types (0..36). find_area_dettype() maps a detector name to one of them using map_area_detname_to_dettype. */
  enum AREADETTYPE {
    UNDEFINED = 0,  ///< Value 0; map key "undefined".
    CSPAD,  ///< Value 1; map key "cspad-".
    CSPAD2X2,  ///< Value 2; map key "cspad2x2".
    PRINCETON,  ///< Value 3; map key "princeton".
    PNCCD,  ///< Value 4; map key "pnccd".
    TM6740,  ///< Value 5; map key "tm6740".
    OPAL1000,  ///< Value 6; map key "opal1000".
    OPAL2000,  ///< Value 7; map key "opal2000".
    OPAL4000,  ///< Value 8; map key "opal4000".
    OPAL8000,  ///< Value 9; map key "opal8000".
    ORCAFL40,  ///< Value 10; map key "orcafl40".
    EPIX,  ///< Value 11; has no key in map_area_detname_to_dettype, so find_area_dettype() never returns it.
    EPIX10KA,  ///< Value 12; map key "epix10ka".
    EPIX10KA2M,  ///< Value 13; map key "epix10ka2m", but find_area_dettype() matches the shorter key "epix10ka" first, so it never returns this value.
    EPIX100A,  ///< Value 14; map key "epix100a".
    EPIXS,  ///< Value 15; map key "epixs".
    EPIXUHR,  ///< Value 16; map key "epixuhr".
    FCCD960,  ///< Value 17; map key "fccd960", but find_area_dettype() matches the shorter key "fccd" first, so it never returns this value.
    FCCD,  ///< Value 18; map key "fccd".
    ANDOR3D,  ///< Value 19; map key "andor3d", but find_area_dettype() matches the shorter key "andor" first, so it never returns this value.
    ANDOR,  ///< Value 20; map key "andor".
    DUALANDOR,  ///< Value 21; map key "dualandor", but find_area_dettype() matches the shorter key "andor" first, so it never returns this value.
    ACQIRIS,  ///< Value 22; map key "acqiris".
    IMP,  ///< Value 23; map key "imp".
    QUARTZ4A150,  ///< Value 24; map key "quartz4a150".
    RAYONIX,  ///< Value 25; map key "rayonix".
    EVR,  ///< Value 26; map key "evr".
    TIMEPIX,  ///< Value 27; map key "timepix".
    FLI,  ///< Value 28; map key "fli".
    PIMAX,  ///< Value 29; map key "pimax".
    JUNGFRAU,  ///< Value 30; map key "jungfrau".
    ZYLA,  ///< Value 31; map key "zyla".
    EPICSCAM,  ///< Value 32; map key "controlscamera".
    PIXIS,  ///< Value 33; map key "pixis".
    UXI,  ///< Value 34; map key "uxi".
    STREAKC7700,  ///< Value 35; map key "streakc7700".
    ARCHON  ///< Value 36; map key "archon".
  };

  /** Map from a lower-case name fragment to AREADETTYPE, searched by find_area_dettype(). Declared static in the header, so each translation unit has its own copy. */
  static std::map<std::string, AREADETTYPE> map_area_detname_to_dettype = {
    {"undefined"     , UNDEFINED},
    {"cspad2x2"      , CSPAD2X2},
    {"cspad-"        , CSPAD},
    {"princeton"     , PRINCETON},
    {"pnccd"         , PNCCD},
    {"tm6740"        , TM6740},
    {"opal1000"      , OPAL1000},
    {"opal2000"      , OPAL2000},
    {"opal4000"      , OPAL4000},
    {"opal8000"      , OPAL8000},
    {"orcafl40"      , ORCAFL40},
    {"epix10ka"      , EPIX10KA},
    {"epix10ka2m"    , EPIX10KA2M},
    {"epix100a"      , EPIX100A},
    {"epixs"         , EPIXS},
    {"epixuhr"       , EPIXUHR},
    {"fccd960"       , FCCD960},
    {"fccd"          , FCCD},
    {"andor3d"       , ANDOR3D},
    {"andor"         , ANDOR},
    {"dualandor"     , DUALANDOR},
    {"acqiris"       , ACQIRIS},
    {"imp"           , IMP},
    {"quartz4a150"   , QUARTZ4A150},
    {"rayonix"       , RAYONIX},
    {"evr"           , EVR},
    {"timepix"       , TIMEPIX},
    {"fli"           , FLI},
    {"pimax"         , PIMAX},
    {"jungfrau"      , JUNGFRAU},
    {"zyla"          , ZYLA},
    {"controlscamera", EPICSCAM},
    {"pixis"         , PIXIS},
    {"uxi"           , UXI},
    {"streakc7700"   , STREAKC7700},
    {"archon"        , ARCHON}
  };

  /**
   * Return the type of the first key of map_area_detname_to_dettype, in alphabetical key order, that occurs (case-sensitively) in detname, or UNDEFINED if none does.
   * Because of this order a shorter key wins over a longer one that contains it (e.g. "epix10ka2m..." gives EPIX10KA).
   */
  const AREADETTYPE find_area_dettype(const std::string& detname);
  /** Log all entries of map_area_detname_to_dettype as one INFO message, one "key => value" line each. */
  void print_map_area_detname_to_dettype();

} // namespace Detector

//-------------------

#endif // PSALG_AREADETECTORTYPES_H

