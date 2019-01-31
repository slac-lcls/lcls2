#include "psalg/calib/CalibParsDBWeb.hh"

//using namespace std;
using namespace psalg; // for NDArray

//-------------------

namespace calib {

CalibParsDBWeb::CalibParsDBWeb(const std::string& detname) : CalibPars(detname) {
  MSG(DEBUG, "In c-tor CalibParsDBWeb for " << detname);
}

CalibParsDBWeb::~CalibParsDBWeb() {
  MSG(DEBUG, "In d-tor CalibParsDBWeb for " << detname());
}
 
void CalibParsDBWeb::_default_msg(const std::string& msg) const {
  MSG(WARNING, "METHOD CalibParsDBWeb::"<< msg << " TBE");
}

/// access to calibration constants
const NDArray<common_mode_t>& CalibParsDBWeb::common_mode(const query_t&) {
  _default_msg(std::string("common_mode(...)"));
  return _common_mode;
}

  /*

const NDArray<pedestals_t>& CalibParsDBWeb::pedestals(const query_t&) {
  _default_msg("pedestals(...)");
  return _pedestals;
}

const NDArray<pixel_rms_t>& CalibParsDBWeb::rms(const query_t&) {
  _default_msg("rms(...)");
  return _pixel_rms;
}

const NDArray<pixel_status_t>& CalibParsDBWeb::status(const query_t&) {
  _default_msg("status(...)");
  return _pixel_status;
}

const NDArray<pixel_gain_t>& CalibParsDBWeb::gain(const query_t&) {
  _default_msg("gain(...)");
  return _pixel_gain;
}

const NDArray<pixel_offset_t>& CalibParsDBWeb::offset(const query_t&) {
  _default_msg("offset(...)");
  return _pixel_offset;
}

const NDArray<pixel_bkgd_t>& CalibParsDBWeb::background(const query_t&) {
  _default_msg("background(...)");
  return _pixel_bkgd;
}

const NDArray<pixel_mask_t>& CalibParsDBWeb::mask_calib(const query_t&) {
  _default_msg("mask_calib(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibParsDBWeb::mask_from_status(const query_t&) {
  _default_msg("mask_from_status(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibParsDBWeb::mask_edges(const query_t&, const size_t& nnbrs) {
  _default_msg("mask_edges(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibParsDBWeb::mask_neighbors(const query_t&, const size_t& nrows, const size_t& ncols) {
  _default_msg("mask_neighbors(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibParsDBWeb::mask_bits(const query_t&, const size_t& mbits) {
  _default_msg("mask(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibParsDBWeb::mask(const query_t&, const bool& calib,
					                     const bool& sataus,
                                                             const bool& edges,
						             const bool& neighbors) {
  _default_msg("mask(...)");
  return _pixel_mask;
}

/// access to geometry
const geometry_t& CalibParsDBWeb::geometry(const query_t&) {
  _default_msg("geometry(...)");
  return _geometry;
}

const NDArray<pixel_idx_t>&   CalibParsDBWeb::indexes(const query_t&, const size_t& axis) {
  _default_msg("indexes(...)");
  return _pixel_idx;
}

const NDArray<pixel_coord_t>& CalibParsDBWeb::coords(const query_t&, const size_t& axis) {
  _default_msg("coords(...)");
  return _pixel_coord;
}

const NDArray<pixel_size_t>& CalibParsDBWeb::pixel_size(const query_t&, const size_t& axis) {
  _default_msg("pixel_size(...)");
  return _pixel_size;
}

const NDArray<pixel_size_t>& CalibParsDBWeb::image_xaxis(const query_t&) {
  _default_msg("image_xaxis(...)");
  return _pixel_size;
}

const NDArray<pixel_size_t>& CalibParsDBWeb::image_yaxis(const query_t&) {
  _default_msg("image_yaxis(...)");
  return _pixel_size;
}

void CalibParsDBWeb::move_geo(const query_t&, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz) {
  _default_msg("move_geo(...)");
}

void CalibParsDBWeb::tilt_geo(const query_t&, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz) {
  _default_msg("tilt_geo(...)");
}
  */

} // namespace calib

//-------------------
