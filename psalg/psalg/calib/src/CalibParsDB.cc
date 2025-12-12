
#include "psalg/calib/CalibParsDB.hh"
#include "psalg/utils/Logger.hh" // for MSG

//using namespace std;
using namespace psalg; // for NDArray

//-------------------

namespace calib {

CalibParsDB::CalibParsDB(const char* dbtypename) :_dbtypename(dbtypename) {
  MSG(DEBUG, "In c-tor CalibParsDB for " << _dbtypename);
}

CalibParsDB::~CalibParsDB() {
  MSG(DEBUG, "In d-tor CalibParsDB for " << _dbtypename);
}
 
void CalibParsDB::_default_msg(const std::string& msg) const {
  MSG(WARNING, "DEFAULT METHOD CalibParsDB::"<< msg << " SHOULD BE RE-IMPLEMENTED IN THE DERIVED CLASS.");
}

//-------------------

#define GET_NDARRAY(T,N)\
const NDArray<T>& CalibParsDB::get_ndarray_##N(Query& q){\
  _default_msg(std::string("get_ndarray_"#N"(Query&)"));\
  return _ndarray_##N;\
}

//-------------------

GET_NDARRAY(double,   double)
GET_NDARRAY(float,    float)
GET_NDARRAY(uint16_t, uint16)
GET_NDARRAY(uint32_t, uint32)

//-------------------

const std::string& CalibParsDB::get_string(Query& q) {
  _default_msg(std::string("get_string(Query&)"));
  return _string;
}

//-------------------

const rapidjson::Document& CalibParsDB::get_data(Query&) {
  _default_msg(std::string("get_data(Query&)"));
  return _data;
}

//-------------------

const rapidjson::Document& CalibParsDB::get_metadata(Query&) {
  _default_msg(std::string("get_metadata(Query&)"));
  return _metadata;
}

//-------------------

//const NDArray<float>& CalibParsDB::get_ndarray_float(Query& q) {
//  _default_msg(std::string("get_ndarray_float(Query&)"));
//  return _ndarray_float;
//}
  
//-------------------

} // namespace calib

//-------------------
