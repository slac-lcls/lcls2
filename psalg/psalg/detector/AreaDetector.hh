#ifndef PSALG_AREADETECTOR_H
#define PSALG_AREADETECTOR_H
//-----------------------------

#include <string>
#include "psalg/calib/NDArray.hh" // NDArray
#include "psalg/detector/Detector.hh"
#include "psalg/detector/AreaDetectorTypes.hh"

using namespace std;
using namespace psalg;

namespace detector {

//-----------------------------

class AreaDetector : public Detector {
public:

  AreaDetector(const std::string& detname) : Detector(detname) {_shape = new shape_t[5]; std::fill_n(_shape, 5, 0); _shape[0]=11;}
    //AreaDetector(const std::string& detname) : _detname(detname) {_shape = new shape_t[5]; std::fill_n(_shape, 5, 0); _shape[0]=11;}
  virtual ~AreaDetector() {delete _shape;}

  //const std::string detname() {return _detname;};
  void _default_msg(const std::string& msg=std::string());

  /// shape, size, ndim of data from configuration object
  virtual const shape_t* shape(const event_t&);
  virtual const size_t   ndim (const event_t&);
  virtual const size_t   size (const event_t&);

  virtual const NDArray<raw_t>& raw(const event_t&);

  /*
  /// access to calibration constants
  virtual const NDArray<common_mode_t>&   common_mode      (const event_t&) = 0;
  virtual const NDArray<pedestals_t>&     pedestals        (const event_t&) = 0;
  virtual const NDArray<pixel_rms_t>&     rms              (const event_t&) = 0;
  virtual const NDArray<pixel_status_t>&  status           (const event_t&) = 0;
  virtual const NDArray<pixel_gain_t>&    gain             (const event_t&) = 0;
  virtual const NDArray<pixel_offset_t>&  offset           (const event_t&) = 0;
  virtual const NDArray<pixel_bkgd_t>&    background       (const event_t&) = 0;
  virtual const NDArray<pixel_mask_t>&    mask_calib       (const event_t&) = 0;
  virtual const NDArray<pixel_mask_t>&    mask_from_status (const event_t&) = 0;
  virtual const NDArray<pixel_mask_t>&    mask_edges       (const event_t&, const size_t& nnbrs=8) = 0;
  virtual const NDArray<pixel_mask_t>&    mask_neighbors   (const event_t&, const size_t& nrows=1, const size_t& ncols=1) = 0;
  virtual const NDArray<pixel_mask_t>&    mask             (const event_t&, const size_t& mbits=0177777) = 0;
  virtual const NDArray<pixel_mask_t>&    mask             (const event_t&, const bool& calib=true,
							                    const bool& sataus=true,
                                                                            const bool& edges=true,
							                    const bool& neighbors=true) = 0;

  /// access to raw, calibrated data, and image
  virtual const NDArray<calib_t>& calib(const event_t&) = 0;
  virtual const NDArray<image_t>& image(const event_t&) = 0;
  virtual const NDArray<image_t>& image(const event_t&, const NDArray<image_t>& nda) = 0;
  virtual const NDArray<image_t>& array_from_image(const event_t&, const NDArray<image_t>&) = 0;
  virtual void move_geo(const event_t&, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz) = 0;
  virtual void tilt_geo(const event_t&, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz) = 0;

  /// access to geometry
  virtual const geometry_t* geometry(const event_t&) = 0;
  virtual const NDArray<pixel_idx_t>&   indexes    (const event_t&, const size_t& axis=0) = 0;
  virtual const NDArray<pixel_coord_t>& coords     (const event_t&, const size_t& axis=0) = 0;
  virtual const NDArray<pixel_size_t>&  pixel_size (const event_t&, const size_t& axis=0) = 0;
  virtual const NDArray<pixel_size_t>&  image_xaxis(const event_t&) = 0;
  virtual const NDArray<pixel_size_t>&  image_yaxis(const event_t&) = 0;

  */

  protected:
    shape_t* _shape;

  private:
    //std::string _detname;
    NDArray<raw_t> _raw_nda;

}; // class

} // namespace detector

#endif // PSALG_AREADETECTOR_H
//-----------------------------
