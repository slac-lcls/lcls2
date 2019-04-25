#ifndef PSALG_CALIBPARS_H
#define PSALG_CALIBPARS_H
//-------------------

#include "psalg/calib/NDArray.hh"
#include "psalg/calib/CalibParsTypes.hh"

#include "psalg/calib/Query.hh"
#include "psalg/calib/Response.hh"

//using namespace std;
using namespace psalg;

namespace calib {

//-------------------

class CalibPars {
public:

  CalibPars(const char* detname = "Undefined detname");
  virtual ~CalibPars();

  const std::string& detname() {return _detname;}

  //-------------------

  virtual const Response& calib_constants(const Query&);

  //-------------------

  /// access to calibration constants
  virtual const NDArray<common_mode_t>&   common_mode      (const Query&);
  virtual const NDArray<pedestals_t>&     pedestals        (const Query&);
  virtual const NDArray<pixel_rms_t>&     rms              (const Query&);
  virtual const NDArray<pixel_status_t>&  status           (const Query&);
  virtual const NDArray<pixel_gain_t>&    gain             (const Query&);
  virtual const NDArray<pixel_offset_t>&  offset           (const Query&);
  virtual const NDArray<pixel_bkgd_t>&    background       (const Query&);
  virtual const NDArray<pixel_mask_t>&    mask_calib       (const Query&);
  virtual const NDArray<pixel_mask_t>&    mask_from_status (const Query&);
  virtual const NDArray<pixel_mask_t>&    mask_edges       (const Query&);//, const size_t& nnbrs=8);
  virtual const NDArray<pixel_mask_t>&    mask_neighbors   (const Query&);//, const size_t& nrows=1, const size_t& ncols=1);
  virtual const NDArray<pixel_mask_t>&    mask_bits        (const Query&);//, const size_t& mbits=0177777);
  virtual const NDArray<pixel_mask_t>&    mask             (const Query&);//, const bool& calib=true,
							                  //  const bool& sataus=true,
                                                                          //  const bool& edges=true,
							                  //  const bool& neighbors=true);

  /// access to geometry
  virtual const geometry_t& geometry(const Query&);
  virtual const NDArray<pixel_idx_t>&   indexes    (const Query&);//, const size_t& axis=0);
  virtual const NDArray<pixel_coord_t>& coords     (const Query&);//, const size_t& axis=0);
  virtual const NDArray<pixel_size_t>&  pixel_size (const Query&);//, const size_t& axis=0);
  virtual const NDArray<pixel_size_t>&  image_xaxis(const Query&);
  virtual const NDArray<pixel_size_t>&  image_yaxis(const Query&);
  //virtual void move_geo(const Query&, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz);
  //virtual void tilt_geo(const Query&, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz);

 //-------------------

  CalibPars(const CalibPars&) = delete;
  CalibPars& operator = (const CalibPars&) = delete;
  CalibPars(){}

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
  Response                _response;

private :

  void _default_msg(const std::string& msg=std::string()) const;

}; // class

//-------------------

} // namespace calib

#endif // PSALG_CALIBPARS_H
