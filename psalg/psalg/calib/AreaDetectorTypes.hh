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

  typedef uint16_t raw_t;
  typedef float    event_t; // query_t
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
    EPIX10KA2M,
    EPIX100A,
    EPIXS,
    FCCD960,
    FCCD,
    ANDOR3D,
    ANDOR,
    DUALANDOR,
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
    PIXIS,
    UXI,
    STREAKC7700,
    ARCHON
  };

  static std::map<std::string, AREADETTYPE> map_area_detname_to_dettype = {
    {"undefined-"     , UNDEFINED},
    {"cspad2x2-"      , CSPAD2X2},
    {"cspad-"         , CSPAD},
    {"princeton-"     , PRINCETON},
    {"pnccd-"         , PNCCD},
    {"tm6740-"        , TM6740},
    {"opal1000-"      , OPAL1000},
    {"opal2000-"      , OPAL2000},
    {"opal4000-"      , OPAL4000},
    {"opal8000-"      , OPAL8000},
    {"orcafl40-"      , ORCAFL40},
    {"epix10ka-"      , EPIX10KA},
    {"epix10ka2m-"    , EPIX10KA2M},
    {"epix100a-"      , EPIX100A},
    {"epixs-"         , EPIXS},
    {"fccd960-"       , FCCD960},
    {"fccd-"          , FCCD},
    {"andor3d-"       , ANDOR3D},
    {"andor-"         , ANDOR},
    {"dualandor-"     , DUALANDOR},
    {"acqiris-"       , ACQIRIS},
    {"imp-"           , IMP},
    {"quartz4a150-"   , QUARTZ4A150},
    {"rayonix-"       , RAYONIX},
    {"evr-"           , EVR},
    {"timepix-"       , TIMEPIX},
    {"fli-"           , FLI},
    {"pimax-"         , PIMAX},
    {"jungfrau"       , JUNGFRAU},
    {"zyla-"          , ZYLA},
    {"controlscamera-", EPICSCAM},
    {"pixis-"         , PIXIS},
    {"uxi-"           , UXI},
    {"streakc7700-"   , STREAKC7700},
    {"archon-"        , ARCHON}
  };

  const AREADETTYPE find_area_dettype(const std::string& detname);
  void print_map_area_detname_to_dettype();

} // namespace Detector

//-------------------

#endif // PSALG_AREADETECTORTYPES_H

