#include "psalg/calib/CalibParsEpix100a.hh"

//using namespace std;
using namespace psalg; // for NDArray

//-------------------

namespace calib {

CalibParsEpix100a::CalibParsEpix100a(const std::string& detname) : CalibPars(detname) {
  MSG(DEBUG, "In c-tor CalibParsEpix100a for " << detname);
}

CalibParsEpix100a::~CalibParsEpix100a() {
  MSG(DEBUG, "In d-tor CalibParsEpix100a for " << detname());
}
 
void CalibParsEpix100a::_default_msg(const std::string& msg) const {
  MSG(WARNING, "METHOD CalibParsEpix100a::"<< msg << " TBE");
}

/// access to calibration constants
const NDArray<common_mode_t>& CalibParsEpix100a::common_mode(const query_t&) {
  _default_msg(std::string("common_mode(...)"));
  return _common_mode;
}

  /*

const NDArray<pedestals_t>& CalibParsEpix100a::pedestals(const query_t&) {
  _default_msg("pedestals(...)");
  return _pedestals;
}

const NDArray<pixel_rms_t>& CalibParsEpix100a::rms(const query_t&) {
  _default_msg("rms(...)");
  return _pixel_rms;
}

const NDArray<pixel_status_t>& CalibParsEpix100a::status(const query_t&) {
  _default_msg("status(...)");
  return _pixel_status;
}

const NDArray<pixel_gain_t>& CalibParsEpix100a::gain(const query_t&) {
  _default_msg("gain(...)");
  return _pixel_gain;
}

const NDArray<pixel_offset_t>& CalibParsEpix100a::offset(const query_t&) {
  _default_msg("offset(...)");
  return _pixel_offset;
}

const NDArray<pixel_bkgd_t>& CalibParsEpix100a::background(const query_t&) {
  _default_msg("background(...)");
  return _pixel_bkgd;
}

const NDArray<pixel_mask_t>& CalibParsEpix100a::mask_calib(const query_t&) {
  _default_msg("mask_calib(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibParsEpix100a::mask_from_status(const query_t&) {
  _default_msg("mask_from_status(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibParsEpix100a::mask_edges(const query_t&, const size_t& nnbrs) {
  _default_msg("mask_edges(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibParsEpix100a::mask_neighbors(const query_t&, const size_t& nrows, const size_t& ncols) {
  _default_msg("mask_neighbors(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibParsEpix100a::mask_bits(const query_t&, const size_t& mbits) {
  _default_msg("mask(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibParsEpix100a::mask(const query_t&, const bool& calib,
					                     const bool& sataus,
                                                             const bool& edges,
						             const bool& neighbors) {
  _default_msg("mask(...)");
  return _pixel_mask;
}

/// access to geometry
const geometry_t& CalibParsEpix100a::geometry(const query_t&) {
  _default_msg("geometry(...)");
  return _geometry;
}

const NDArray<pixel_idx_t>&   CalibParsEpix100a::indexes(const query_t&, const size_t& axis) {
  _default_msg("indexes(...)");
  return _pixel_idx;
}

const NDArray<pixel_coord_t>& CalibParsEpix100a::coords(const query_t&, const size_t& axis) {
  _default_msg("coords(...)");
  return _pixel_coord;
}

const NDArray<pixel_size_t>& CalibParsEpix100a::pixel_size(const query_t&, const size_t& axis) {
  _default_msg("pixel_size(...)");
  return _pixel_size;
}

const NDArray<pixel_size_t>& CalibParsEpix100a::image_xaxis(const query_t&) {
  _default_msg("image_xaxis(...)");
  return _pixel_size;
}

const NDArray<pixel_size_t>& CalibParsEpix100a::image_yaxis(const query_t&) {
  _default_msg("image_yaxis(...)");
  return _pixel_size;
}

void CalibParsEpix100a::move_geo(const query_t&, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz) {
  _default_msg("move_geo(...)");
}

void CalibParsEpix100a::tilt_geo(const query_t&, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz) {
  _default_msg("tilt_geo(...)");
}
  */

} // namespace calib

//-------------------
