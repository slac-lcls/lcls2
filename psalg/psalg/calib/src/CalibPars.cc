#include "psalg/calib/CalibPars.hh"
#include "psalg/utils/Logger.hh" // for MSG

#include "psalg/calib/CalibParsDBStore.hh"

//using namespace std;
using namespace psalg; // for NDArray

//-------------------

namespace calib {

CalibPars::CalibPars(const char* detname, const DBTYPE& dbtype)
  : _detname(detname)
  , _calibparsdb(getCalibParsDB(dbtype))
  , _geometryaccess(NULL) {
  MSG(DEBUG, "In c-tor CalibPars for " << detname);

  //if(_calibparsdb) {delete _calibparsdb; _calibparsdb=NULL;}
}

CalibPars::~CalibPars() {
  MSG(DEBUG, "In d-tor CalibPars for " << detname());
  deleteGeometryAccess();
}
 
void CalibPars::_default_msg(const std::string& msg) const {
  MSG(WARNING, "DEFAULT METHOD CalibPars::"<< msg << " SHOULD BE RE-IMPLEMENTED IN THE DERIVED CLASS.");
}

  //const std::string& CalibPars::get_string(const query_t&) {
  //  _default_msg("pedestals(...)");
  //  return std::string;
  //}

//-------------------

/** REPLACE METHODS LIKE SHOWN BELOW WITH PARAMETRIC MACRO
const NDArray<pedestals_t>& CalibPars::pedestals(Query& q) {
  std::cout << "CalibPars::pedestals dbtypename: " << _calibparsdb->dbtypename() << '\n';
  return _calibparsdb->get_ndarray_float(q);
}
*/

#define GET_NDARRAY_DEF(T,N,D)\
const NDArray<T>& Calib##N(Query& q) {\
  MSG(DEBUG, std::string("==========> Calib"#N"(Query&) for ") << _calibparsdb->dbtypename());\
  return _calibparsdb->get_ndarray_##D(q);\
}

//-------------------

GET_NDARRAY_DEF(common_mode_t,  Pars::common_mode,      double)
GET_NDARRAY_DEF(pedestals_t,    Pars::pedestals,        float)
GET_NDARRAY_DEF(pixel_rms_t,    Pars::rms,              float)
GET_NDARRAY_DEF(pixel_status_t, Pars::status,           uint16)
GET_NDARRAY_DEF(pixel_gain_t,   Pars::gain,             float)
GET_NDARRAY_DEF(pixel_gain_t,   Pars::offset,           float)
GET_NDARRAY_DEF(pixel_bkgd_t,   Pars::background,       float)
GET_NDARRAY_DEF(pixel_mask_t,   Pars::mask_calib,       uint16)
GET_NDARRAY_DEF(pixel_mask_t,   Pars::mask_from_status, uint16)
GET_NDARRAY_DEF(pixel_mask_t,   Pars::mask_edges,       uint16)
GET_NDARRAY_DEF(pixel_mask_t,   Pars::mask_neighbors,   uint16)
GET_NDARRAY_DEF(pixel_mask_t,   Pars::mask_bits,        uint16)
GET_NDARRAY_DEF(pixel_mask_t,   Pars::mask,             uint16)

/// access to geometry
const geometry_t& CalibPars::geometry_str(Query& q) {
  //_default_msg("geometry(...)");
  MSG(DEBUG, std::string("==========> CalibPars::geometry_str(Query&) for ") << _calibparsdb->dbtypename());\
  return _calibparsdb->get_string(q);
}

/*
const geometry_t& CalibPars::geometry(Query& q) {
  _default_msg("geometry(...)");
  return _geometry;
}
*/

geometry::GeometryAccess* CalibPars::geometryAccess(Query& q) {
  if(!_geometryaccess) {
      std::stringstream ss(geometry_str(q));
      _geometryaccess = new geometry::GeometryAccess(ss);
  }
  return _geometryaccess;
}

void CalibPars::deleteGeometryAccess() {
  if(_geometryaccess) delete _geometryaccess;
}

const NDArray<pixel_idx_t>&   CalibPars::indexes(Query& q) {
  _default_msg("indexes(...)");
  return _pixel_idx;
}

const NDArray<pixel_coord_t>& CalibPars::coords(Query& q) {
  _default_msg("coords(...)");
  return _pixel_coord;
}

const NDArray<pixel_size_t>& CalibPars::pixel_size(Query& q) {
  _default_msg("pixel_size(...)");
  return _pixel_size;
}

const NDArray<pixel_size_t>& CalibPars::image_xaxis(Query& q) {
  _default_msg("image_xaxis(...)");
  return _pixel_size;
}

const NDArray<pixel_size_t>& CalibPars::image_yaxis(Query& q) {
  _default_msg("image_yaxis(...)");
  return _pixel_size;
}

  /*
void CalibPars::move_geo(Query& q, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz) {
  _default_msg("move_geo(...)");
}

void CalibPars::tilt_geo(Query& q, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz) {
  _default_msg("tilt_geo(...)");
}
  */

} // namespace calib

//-------------------
//-------------------
