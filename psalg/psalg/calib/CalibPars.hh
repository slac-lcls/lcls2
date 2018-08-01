#ifndef PSALG_CALIBPARS_H
#define PSALG_CALIBPARS_H
//-----------------------------

#include <string>
#include "psalg/calib/NDArray.hh" // NDArray
#include "psalg/calib/CalibParsTypes.hh"

//using namespace std;
using namespace psalg;

namespace calib {

//-----------------------------

class CalibPars {
public:

  CalibPars(const std::string& detname);
  virtual ~CalibPars() {}

  void _default_msg(const std::string& msg=std::string()) const;
  const std::string& detname() {return _detname;}

  /// access to calibration constants
  virtual const NDArray<common_mode_t>&   common_mode      (const request_t&);
  virtual const NDArray<pedestals_t>&     pedestals        (const request_t&);
  virtual const NDArray<pixel_rms_t>&     rms              (const request_t&);
  virtual const NDArray<pixel_status_t>&  status           (const request_t&);
  virtual const NDArray<pixel_gain_t>&    gain             (const request_t&);
  virtual const NDArray<pixel_offset_t>&  offset           (const request_t&);
  virtual const NDArray<pixel_bkgd_t>&    background       (const request_t&);
  virtual const NDArray<pixel_mask_t>&    mask_calib       (const request_t&);
  virtual const NDArray<pixel_mask_t>&    mask_from_status (const request_t&);
  virtual const NDArray<pixel_mask_t>&    mask_edges       (const request_t&, const size_t& nnbrs=8);
  virtual const NDArray<pixel_mask_t>&    mask_neighbors   (const request_t&, const size_t& nrows=1, const size_t& ncols=1);
  virtual const NDArray<pixel_mask_t>&    mask             (const request_t&, const size_t& mbits=0177777);
  virtual const NDArray<pixel_mask_t>&    mask             (const request_t&, const bool& calib=true,
							                      const bool& sataus=true,
                                                                              const bool& edges=true,
							                      const bool& neighbors=true);

  /// access to geometry
  virtual const geometry_t& geometry(const request_t&);
  virtual const NDArray<pixel_idx_t>&   indexes    (const request_t&, const size_t& axis=0);
  virtual const NDArray<pixel_coord_t>& coords     (const request_t&, const size_t& axis=0);
  virtual const NDArray<pixel_size_t>&  pixel_size (const request_t&, const size_t& axis=0);
  virtual const NDArray<pixel_size_t>&  image_xaxis(const request_t&);
  virtual const NDArray<pixel_size_t>&  image_yaxis(const request_t&);
  //virtual void move_geo(const request_t&, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz);
  //virtual void tilt_geo(const request_t&, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz);

  CalibPars(const CalibPars&) = delete;
  CalibPars& operator = (const CalibPars&) = delete;
  CalibPars(){}

private:

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

} // namespace calib

#endif // PSALG_CALIBPARS_H
//-----------------------------
