#ifndef PSALG_AREADETECTORTYPES_H
#define PSALG_AREADETECTORTYPES_H

/** Usage
 *
 * #include "psalg/detector/AreaDetectorTypes.hh"
 */

#include "psalg/calib/CalibParsTypes.hh"
#include <string>
#include <map>

//#include "psalg/utils/Logger.hh" // MSG, LOGGER

using namespace calib;

//-------------------

namespace detector {

  enum CALIB_TYPE {PEDESTALS=0, PIXEL_RMS, PIXEL_STATUS, PIXEL_GAIN, PIXEL_OFFSET, PIXEL_MASK, PIXEL_BKGD, COMMON_MODE};

  //typedef psalg::types::shape_t shape_t; // uint32_t
  //typedef psalg::types::size_t  size_t;  // uint32_t

  typedef uint32_t shape_t;
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

  typedef float    event_t; // query_t
  typedef float    raw_t;
  typedef float    calib_t;
  typedef float    image_t;

  enum AREADETTYPE {
    UNDEFINED = 0,
    CSPAD,
    CSPAD2X2,
    PRINCETON,
    PNCCD,
    TM6740,
    OPAL1000,
    OPAL2000,
    OPAL4000,
    OPAL8000,
    ORCAFL40,
    EPIX,
    EPIX10KA,
    EPIX100A,
    FCCD960,
    FCCD,
    ANDOR3D,
    ANDOR,
    ACQIRIS,
    IMP,
    QUARTZ4A150,
    RAYONIX,
    EVR,
    TIMEPIX,
    FLI,
    PIMAX,
    JUNGFRAU,
    ZYLA,
    EPICSCAM,   
  };

  static std::map<std::string, AREADETTYPE> map_area_detname_to_dettype = {
    {"UNDEFINED"     , UNDEFINED},
    {"Cspad2x2"      , CSPAD2X2},
    {"Cspad"         , CSPAD},
    {"Princeton"     , PRINCETON},
    {"pnCCD"         , PNCCD},
    {"Tm6740"        , TM6740},
    {"Opal1000"      , OPAL1000},
    {"Opal2000"      , OPAL2000},
    {"Opal4000"      , OPAL4000},
    {"Opal8000"      , OPAL8000},
    {"OrcaFl40"      , ORCAFL40},
    {"Epix10ka"      , EPIX10KA},
    {"Epix100a"      , EPIX100A},
    {"Fccd960"       , FCCD960},
    {"Fccd"          , FCCD},
    {"Andor3d"       , ANDOR3D},
    {"Andor"         , ANDOR},
    {"Acqiris"       , ACQIRIS},
    {"Imp"           , IMP},
    {"Quartz4A150"   , QUARTZ4A150},
    {"Rayonix"       , RAYONIX},
    {"Evr"           , EVR},
    {"Timepix"       , TIMEPIX},
    {"Fli"           , FLI},
    {"Pimax"         , PIMAX},
    {"Jungfrau"      , JUNGFRAU},
    {"Zyla"          , ZYLA},
    {"ControlsCamera", EPICSCAM}
  };

  const AREADETTYPE find_area_dettype(const std::string& detname);
  void print_map_area_detname_to_dettype();

} // namespace Detector

//-------------------

#endif // PSALG_AREADETECTORTYPES_H

