#ifndef PSALG_CALIBPARSEPIX100A_H
#define PSALG_CALIBPARSEPIX100A_H
//-----------------------------

#include "psalg/calib/CalibPars.hh"

//using namespace std;
//using namespace calib;
using namespace psalg;

namespace calib {

//-----------------------------

class CalibParsEpix100a : public CalibPars {
public:

  CalibParsEpix100a(const char* detname = "Epix100a");
  virtual ~CalibParsEpix100a();

  void _default_msg(const std::string& msg=std::string()) const;

  /// access to calibration constants
  const NDArray<common_mode_t>&   common_mode      (const Query&);

  /*
  const NDArray<pedestals_t>&     pedestals        (const Query&);
  const NDArray<pixel_rms_t>&     rms              (const Query&);
  const NDArray<pixel_status_t>&  status           (const Query&);
  const NDArray<pixel_gain_t>&    gain             (const Query&);
  const NDArray<pixel_offset_t>&  offset           (const Query&);
  const NDArray<pixel_bkgd_t>&    background       (const Query&);
  const NDArray<pixel_mask_t>&    mask_calib       (const Query&);
  const NDArray<pixel_mask_t>&    mask_from_status (const Query&);
  const NDArray<pixel_mask_t>&    mask_edges       (const Query&);
  const NDArray<pixel_mask_t>&    mask_neighbors   (const Query&);
  const NDArray<pixel_mask_t>&    mask_bits        (const Query&);
  const NDArray<pixel_mask_t>&    mask             (const Query&);

  /// access to geometry
  const geometry_t& geometry(const Query&);
  const NDArray<pixel_idx_t>&   indexes    (const Query&);
  const NDArray<pixel_coord_t>& coords     (const Query&);
  const NDArray<pixel_size_t>&  pixel_size (const Query&);
  const NDArray<pixel_size_t>&  image_xaxis(const Query&);
  const NDArray<pixel_size_t>&  image_yaxis(const Query&);
  //virtual void move_geo(const Query&);
  //virtual void tilt_geo(const Query&);
  */

  CalibParsEpix100a(const CalibParsEpix100a&) = delete;
  CalibParsEpix100a& operator = (const CalibParsEpix100a&) = delete;

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

#endif // PSALG_CALIBPARSEPIX100A_H
