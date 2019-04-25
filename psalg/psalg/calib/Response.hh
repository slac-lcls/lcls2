#ifndef PSALG_RESPONSE_H
#define PSALG_RESPONSE_H
//-----------------------------

#include "psalg/calib/CalibParsTypes.hh"
#include "psalg/calib/NDArray.hh"

//using namespace std;
using namespace psalg; // for NDArray

namespace calib {

//-----------------------------

class Response {
public:

  Response(){}
  Response(const char* detname) : _detname(detname) {}
  Response(const std::string& detname) : _detname(detname) {}
  virtual ~Response(){}

  inline std::string detname() {return _detname;}

  Response(const Response&) = delete;
  Response& operator = (const Response&) = delete;

protected:

  NDArray<common_mode_t>  _common_mode;
  NDArray<pedestals_t>    _pedestals;

  NDArray<pixel_rms_t>    _pixel_rms;
  NDArray<pixel_status_t> _pixel_status;
  NDArray<pixel_gain_t>   _pixel_gain;
  NDArray<pixel_offset_t> _pixel_offset;
  NDArray<pixel_bkgd_t>   _pixel_bkgd;
  NDArray<pixel_mask_t>   _pixel_mask;

  NDArray<pixel_idx_t>    _pixel_idx;
  NDArray<pixel_coord_t>  _pixel_coord;
  NDArray<pixel_size_t>   _pixel_size;

  geometry_t              _geometry;

  const std::string       _detname;

}; // class

//-----------------------------

} // namespace calib

#endif // PSALG_RESPONSE_H
