/**
 * @file
 * @brief Declares calib::CALIB_TYPE, element-type aliases for calibration arrays, and name_of_calibtype().
 */
#ifndef PSALG_CALIBPARSTYPES_H
#define PSALG_CALIBPARSTYPES_H

/** Usage
 *
 * #include "psalg/calib/CalibParsTypes.hh"
 */

#include <string>
#include <cstdint>
//#include <map>
//#include "psalg/utils/Logger.hh" // MSG, LOGGER

//-------------------

namespace calib {

  /** Calibration constant types. name_of_calibtype() maps them to names, which Query::set_calibtype() stores as the CALIBTYPE parameter. */
  enum CALIB_TYPE {PEDESTALS=0, /**< Value 0; name "pedestals". */ 
                   PIXEL_RMS,  ///< Value 1; name "pixel_rms".
                   PIXEL_STATUS,  ///< Value 2; name "pixel_status".
                   PIXEL_GAIN,  ///< Value 3; name "pixel_gain".
                   PIXEL_OFFSET,  ///< Value 4; name "pixel_offset".
                   PIXEL_MASK,  ///< Value 5; name "pixel_mask".
                   PIXEL_BKGD,  ///< Value 6; name "pixel_bkgd".
                   PIXEL_IDX,  ///< Value 7; name "pixel_idx".
                   PIXEL_COORD,  ///< Value 8; name "pixel_coord".
                   PIXEL_SIZE,  ///< Value 9; name "pixel_size".
                   PIXEL_AREA,  ///< Value 10; name_of_calibtype() has no case for it and returns "undefined".
                   TILT_ANGLE,  ///< Value 11; name_of_calibtype() has no case for it and returns "undefined".
                   COMMON_MODE,  ///< Value 12; name "common_mode".
                   GEOMETRY,  ///< Value 13; name "geometry".
  };

  //typedef psalg::types::shape_t shape_t; // uint32_t
  //typedef psalg::types::size_t  size_t;  // uint32_t

  /** float; element type of CalibPars::pedestals(). */
  typedef float    pedestals_t;
  /** float; element type of CalibPars::rms(). */
  typedef float    pixel_rms_t;
  /** uint16_t; element type of CalibPars::status(). */
  typedef uint16_t pixel_status_t;
  /** float; element type of CalibPars::gain(). */
  typedef float    pixel_gain_t;
  /** float; element type of CalibPars::offset(). */
  typedef float    pixel_offset_t;
  /** uint16_t; element type of the CalibPars mask arrays (mask_calib(), mask(), mask_geo(), ...). */
  typedef uint16_t pixel_mask_t;
  /** float; element type of CalibPars::background(). */
  typedef float    pixel_bkgd_t;
  /** uint32_t; element type of CalibPars::indexes(). */
  typedef uint32_t pixel_idx_t;
  /** double; element type of CalibPars::coords(). */
  typedef double   pixel_coord_t;
  /** double; element type of CalibPars::pixel_size(), image_xaxis() and image_yaxis(). */
  typedef double   pixel_size_t;
  /** double; element type of CalibPars::pixel_area(). */
  typedef double   pixel_area_t;
  /** double; argument type of AreaDetector::tilt_geo(). */
  typedef double   tilt_angle_t;
  /** double; element type of CalibPars::common_mode(). */
  typedef double   common_mode_t;

  /** std::string holding text from a file (per the trailing comment); returned by CalibPars::geometry_str(). */
  typedef std::string geometry_t; // text from file
  /** float; the trailing comment marks it as a temporary substitution for an object. */
  typedef float query_t;          // TEMPORARY substitution for object

  /** Return the lower-case name of ctype (e.g. "pedestals", "pixel_rms", "geometry"), or "undefined" for PIXEL_AREA, TILT_ANGLE and any value without a case. */
  const char* name_of_calibtype(const CALIB_TYPE& ctype);

} // namespace calib

//-------------------

#endif // PSALG_CALIBPARSTYPES_H

