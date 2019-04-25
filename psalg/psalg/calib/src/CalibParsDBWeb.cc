#include "psalg/calib/CalibParsDBWeb.hh"

//using namespace std;
using namespace psalg; // for NDArray

//-------------------

namespace calib {

CalibParsDBWeb::CalibParsDBWeb() : CalibParsDB("DBWeb") {
  MSG(DEBUG, "In c-tor CalibParsDBWeb for " << dbtypename());
}

CalibParsDBWeb::~CalibParsDBWeb() {
  MSG(DEBUG, "In d-tor CalibParsDBWeb for " << dbtypename());
}
 
void CalibParsDBWeb::_default_msg(const std::string& msg) const {
  MSG(WARNING, "METHOD CalibParsDBWeb::"<< msg << " TBE");
}

//-------------------

#define GET_NDARRAY(T,N)\
const NDArray<T>& CalibParsDBWeb::get_ndarray_##N(const Query& q){\
  _default_msg(std::string("get_ndarray_"#N"(Query)"));\
  return _ndarray_##N;\
}

//-------------------

GET_NDARRAY(double,   double)
GET_NDARRAY(float,    float)
GET_NDARRAY(uint16_t, uint16)
GET_NDARRAY(uint32_t, uint32)

//-------------------

/// access to calibration constants
/*
const NDArray<common_mode_t>& CalibParsDBWeb::common_mode(const query_t&) {
  _default_msg(std::string("common_mode(...)"));
  return _common_mode;
}
*/

} // namespace calib

//-------------------
