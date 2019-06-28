
#include "psalg/calib/CalibParsTypes.hh"

namespace calib {

//-------------------

  const char* name_of_calibtype(const CALIB_TYPE& ctype) { 
    switch(ctype)
    {
      case PEDESTALS   : return "pedestals";
      case PIXEL_RMS   : return "pixel_rms";
      case PIXEL_STATUS: return "pixel_status";
      case PIXEL_GAIN  : return "pixel_gain";
      case PIXEL_OFFSET: return "pixel_offset";
      case PIXEL_MASK  : return "pixel_mask";
      case PIXEL_BKGD  : return "pixel_bkgd";
      case PIXEL_IDX   : return "pixel_idx";
      case PIXEL_COORD : return "pixel_coord";
      case PIXEL_SIZE  : return "pixel_size";
      case COMMON_MODE : return "common_mode";
      case GEOMETRY    : return "geometry";
      default          : return "undefined"; 
    }
  }

//-------------------

} // namespace calib

//-------------------
