#ifndef PSALG_CALIBPARSDBWEB_H
#define PSALG_CALIBPARSDBWEB_H
//-----------------------------

#include "psalg/calib/CalibPars.hh"

//using namespace std;
//using namespace calib;
using namespace psalg;

namespace calib {

//-----------------------------

class CalibParsDBWeb : public CalibPars {
public:

  CalibParsDBWeb(const std::string& detname);
  virtual ~CalibParsDBWeb();

  void _default_msg(const std::string& msg=std::string()) const;

  /// access to calibration constants
  const NDArray<common_mode_t>&   common_mode      (const query_t&);

  /*
  const NDArray<pedestals_t>&     pedestals        (const query_t&);
  const NDArray<pixel_rms_t>&     rms              (const query_t&);
  const NDArray<pixel_status_t>&  status           (const query_t&);
  const NDArray<pixel_gain_t>&    gain             (const query_t&);
  const NDArray<pixel_offset_t>&  offset           (const query_t&);
  const NDArray<pixel_bkgd_t>&    background       (const query_t&);
  const NDArray<pixel_mask_t>&    mask_calib       (const query_t&);
  const NDArray<pixel_mask_t>&    mask_from_status (const query_t&);
  const NDArray<pixel_mask_t>&    mask_edges       (const query_t&, const size_t& nnbrs=8);
  const NDArray<pixel_mask_t>&    mask_neighbors   (const query_t&, const size_t& nrows=1, const size_t& ncols=1);
  const NDArray<pixel_mask_t>&    mask_bits        (const query_t&, const size_t& mbits=0177777);
  const NDArray<pixel_mask_t>&    mask             (const query_t&, const bool& calib=true,
  						                    const bool& sataus=true,
                                                                    const bool& edges=true,
  						                    const bool& neighbors=true);

  /// access to geometry
  const geometry_t& geometry(const query_t&);
  const NDArray<pixel_idx_t>&   indexes    (const query_t&, const size_t& axis=0);
  const NDArray<pixel_coord_t>& coords     (const query_t&, const size_t& axis=0);
  const NDArray<pixel_size_t>&  pixel_size (const query_t&, const size_t& axis=0);
  const NDArray<pixel_size_t>&  image_xaxis(const query_t&);
  const NDArray<pixel_size_t>&  image_yaxis(const query_t&);
  //virtual void move_geo(const query_t&, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz);
  //virtual void tilt_geo(const query_t&, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz);
  */

  CalibParsDBWeb(const CalibParsDBWeb&) = delete;
  CalibParsDBWeb& operator = (const CalibParsDBWeb&) = delete;
  CalibParsDBWeb(){}

private:
  NDArray<common_mode_t>  _common_mode;


  /*
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
  */

}; // class

//-----------------------------

} // namespace calib

#endif // PSALG_CALIBPARSDBWEB_H
