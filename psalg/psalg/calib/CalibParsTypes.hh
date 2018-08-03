#ifndef PSALG_CALIBPARSTYPES_H
#define PSALG_CALIBPARSTYPES_H

/** Usage
 *
 * #include "psalg/calib/CalibParsTypes.hh"
 */

#include <string>
//#include <map>
//#include "psalg/utils/Logger.hh" // MSG, LOGGER

//-------------------

namespace calib {

  enum CALIB_TYPE {PEDESTALS=0,
                   PIXEL_RMS,
                   PIXEL_STATUS,
                   PIXEL_GAIN,
                   PIXEL_OFFSET,
                   PIXEL_MASK,
                   PIXEL_BKGD,
                   PIXEL_IDX,
                   PIXEL_COORD,
                   PIXEL_SIZE,
                   TILT_ANGLE,
                   COMMON_MODE
  };

  //typedef psalg::types::shape_t shape_t; // uint32_t
  //typedef psalg::types::size_t  size_t;  // uint32_t

  typedef float    pedestals_t;
  typedef float    pixel_rms_t;
  typedef uint16_t pixel_status_t;
  typedef float    pixel_gain_t;
  typedef float    pixel_offset_t;
  typedef uint16_t pixel_mask_t;
  typedef float    pixel_bkgd_t;
  typedef uint32_t pixel_idx_t;
  typedef float    pixel_coord_t;
  typedef float    pixel_size_t;
  typedef float    tilt_angle_t;
  typedef double   common_mode_t;

  typedef std::string geometry_t; // text from file
  typedef float query_t;          // TEMPORARY substitution for object
} // namespace calib

//-------------------

#endif // PSALG_CALIBPARSTYPES_H

