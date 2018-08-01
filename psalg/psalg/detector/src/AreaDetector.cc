
#include "psalg/detector/AreaDetector.hh"
#include "psalg/utils/Logger.hh" // for MSG

//using namespace std;
using namespace psalg;

//-------------------

namespace detector {

AreaDetector::AreaDetector(const std::string& detname) : Detector(detname, AREA_DETECTOR) {
  _shape = new shape_t[5]; std::fill_n(_shape, 5, 0); _shape[0]=11;
}
 
void AreaDetector::_default_msg(const std::string& msg) const {
  MSG(WARNING, "DEFAULT METHOD "<< msg << " SHOULD BE RE-IMPLEMENTED IN THE DERIVED CLASS.");
}

const shape_t* AreaDetector::shape(const event_t&) {
  _default_msg("shape(...)");
  return &_shape[0];
}

const size_t AreaDetector::ndim(const event_t&) {
  _default_msg("ndim(...)");
  return 0;
}

const size_t AreaDetector::size(const event_t&) {
  _default_msg("size(...)");
  return 0;
}

const NDArray<raw_t>& AreaDetector::raw(const event_t&) {
  _default_msg("raw(...)");
  return _raw;
}

/// access to calibration constants
const NDArray<common_mode_t>& AreaDetector::common_mode(const event_t&) {
  _default_msg(std::string("common_mode(...)"));
  return _common_mode;
}

const NDArray<pedestals_t>& AreaDetector::pedestals(const event_t&) {
  _default_msg("pedestals(...)");
  return _pedestals;
}

const NDArray<pixel_rms_t>& AreaDetector::rms(const event_t&) {
  _default_msg("rms(...)");
  return _pixel_rms;
}

const NDArray<pixel_status_t>& AreaDetector::status(const event_t&) {
  _default_msg("status(...)");
  return _pixel_status;
}

const NDArray<pixel_gain_t>& AreaDetector::gain(const event_t&) {
  _default_msg("gain(...)");
  return _pixel_gain;
}

const NDArray<pixel_offset_t>& AreaDetector::offset(const event_t&) {
  _default_msg("offset(...)");
  return _pixel_offset;
}

const NDArray<pixel_bkgd_t>& AreaDetector::background(const event_t&) {
  _default_msg("background(...)");
  return _pixel_bkgd;
}

const NDArray<pixel_mask_t>& AreaDetector::mask_calib(const event_t&) {
  _default_msg("mask_calib(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& AreaDetector::mask_from_status(const event_t&) {
  _default_msg("mask_from_status(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& AreaDetector::mask_edges(const event_t&, const size_t& nnbrs) {
  _default_msg("mask_edges(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& AreaDetector::mask_neighbors(const event_t&, const size_t& nrows, const size_t& ncols) {
  _default_msg("mask_neighbors(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& AreaDetector::mask(const event_t&, const size_t& mbits) {
  _default_msg("mask(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& AreaDetector::mask(const event_t&, const bool& calib,
					                        const bool& sataus,
                                                                const bool& edges,
						                const bool& neighbors) {
  _default_msg("mask(...)");
  return _pixel_mask;
}

/// access to raw, calibrated data, and image
const NDArray<calib_t>& AreaDetector::calib(const event_t&) {
  _default_msg("calib(...)");
  return _calib;
}

const NDArray<image_t>& AreaDetector::image(const event_t&) {
  _default_msg("image(...)");
  return _image;
}

const NDArray<image_t>& AreaDetector::image(const event_t&, const NDArray<image_t>& nda) {
  _default_msg("image(...)");
  return _image;
}

const NDArray<image_t>& AreaDetector::array_from_image(const event_t&, const NDArray<image_t>&) {
  _default_msg("array_from_image(...)");
  return _image;
}

void AreaDetector::move_geo(const event_t&, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz) {
  _default_msg("move_geo(...)");
}

void AreaDetector::tilt_geo(const event_t&, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz) {
  _default_msg("tilt_geo(...)");
}


/// access to geometry
const geometry_t* AreaDetector::geometry(const event_t&) {
  _default_msg("geometry(...)");
  return _geometry;
}

const NDArray<pixel_idx_t>&   AreaDetector::indexes(const event_t&, const size_t& axis) {
  _default_msg("indexes(...)");
  return _pixel_idx;
}

const NDArray<pixel_coord_t>& AreaDetector::coords(const event_t&, const size_t& axis) {
  _default_msg("coords(...)");
  return _pixel_coord;
}

const NDArray<pixel_size_t>& AreaDetector::pixel_size(const event_t&, const size_t& axis) {
  _default_msg("pixel_size(...)");
  return _pixel_size;
}

const NDArray<pixel_size_t>& AreaDetector::image_xaxis(const event_t&) {
  _default_msg("image_xaxis(...)");
  return _pixel_size;
}

const NDArray<pixel_size_t>& AreaDetector::image_yaxis(const event_t&) {
  _default_msg("image_yaxis(...)");
  return _pixel_size;
}

} // namespace detector

//-------------------
