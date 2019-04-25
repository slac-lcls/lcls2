#include "psalg/calib/CalibPars.hh"
#include "psalg/utils/Logger.hh" // for MSG

//using namespace std;
using namespace psalg; // for NDArray

//-------------------

namespace calib {

CalibPars::CalibPars(const char* detname) : _detname(detname) {
  MSG(DEBUG, "In c-tor CalibPars for " << detname);
}

CalibPars::~CalibPars() {
  MSG(DEBUG, "In d-tor CalibPars for " << detname());
}
 
void CalibPars::_default_msg(const std::string& msg) const {
  MSG(WARNING, "DEFAULT METHOD CalibPars::"<< msg << " SHOULD BE RE-IMPLEMENTED IN THE DERIVED CLASS.");
}

  //const std::string& CalibPars::get_string(const query_t&) {
  //  _default_msg("pedestals(...)");
  //  return std::string;
  //}

//-------------------

/// access to calibration constants

const Response& CalibPars::calib_constants(const Query&) {
  _default_msg(std::string("const Response& calib_constants(const Query&)"));
  return _response;
}

//-------------------

const NDArray<common_mode_t>& CalibPars::common_mode(const Query&) {
  _default_msg(std::string("common_mode(...)"));
  return _common_mode;
}

const NDArray<pedestals_t>& CalibPars::pedestals(const Query&) {
  _default_msg("pedestals(...)");
  return _pedestals;
}

const NDArray<pixel_rms_t>& CalibPars::rms(const Query&) {
  _default_msg("rms(...)");
  return _pixel_rms;
}

const NDArray<pixel_status_t>& CalibPars::status(const Query&) {
  _default_msg("status(...)");
  return _pixel_status;
}

const NDArray<pixel_gain_t>& CalibPars::gain(const Query&) {
  _default_msg("gain(...)");
  return _pixel_gain;
}

const NDArray<pixel_offset_t>& CalibPars::offset(const Query&) {
  _default_msg("offset(...)");
  return _pixel_offset;
}

const NDArray<pixel_bkgd_t>& CalibPars::background(const Query&) {
  _default_msg("background(...)");
  return _pixel_bkgd;
}

const NDArray<pixel_mask_t>& CalibPars::mask_calib(const Query&) {
  _default_msg("mask_calib(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibPars::mask_from_status(const Query&) {
  _default_msg("mask_from_status(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibPars::mask_edges(const Query&) {
  _default_msg("mask_edges(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibPars::mask_neighbors(const Query&) {
  _default_msg("mask_neighbors(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibPars::mask_bits(const Query&) {
  _default_msg("mask(...)");
  return _pixel_mask;
}

const NDArray<pixel_mask_t>& CalibPars::mask(const Query&) {
  //, const bool& calib,
  //const bool& sataus,
  //const bool& edges,
  //const bool& neighbors) {
  _default_msg("mask(...)");
  return _pixel_mask;
}

/// access to geometry
const geometry_t& CalibPars::geometry(const Query&) {
  _default_msg("geometry(...)");
  return _geometry;
}

const NDArray<pixel_idx_t>&   CalibPars::indexes(const Query&) {
  _default_msg("indexes(...)");
  return _pixel_idx;
}

const NDArray<pixel_coord_t>& CalibPars::coords(const Query&) {
  _default_msg("coords(...)");
  return _pixel_coord;
}

const NDArray<pixel_size_t>& CalibPars::pixel_size(const Query&) {
  _default_msg("pixel_size(...)");
  return _pixel_size;
}

const NDArray<pixel_size_t>& CalibPars::image_xaxis(const Query&) {
  _default_msg("image_xaxis(...)");
  return _pixel_size;
}

const NDArray<pixel_size_t>& CalibPars::image_yaxis(const Query&) {
  _default_msg("image_yaxis(...)");
  return _pixel_size;
}

  /*
void CalibPars::move_geo(const Query&, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz) {
  _default_msg("move_geo(...)");
}

void CalibPars::tilt_geo(const Query&, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz) {
  _default_msg("tilt_geo(...)");
}
  */

} // namespace calib

//-------------------
//-------------------
