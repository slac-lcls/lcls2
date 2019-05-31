#include "psalg/calib/CalibParsDBWeb.hh"

//using namespace std;
using namespace psalg; // for NDArray

//-------------------

namespace calib {

CalibParsDBWeb::CalibParsDBWeb() : CalibParsDB("DBWEB") {
  MSG(DEBUG, "In c-tor CalibParsDBWeb for " << dbtypename());
}

CalibParsDBWeb::~CalibParsDBWeb() {
  MSG(DEBUG, "In d-tor CalibParsDBWeb for " << dbtypename());
}
 
void CalibParsDBWeb::_default_msg(const std::string& msg) const {
  MSG(WARNING, "METHOD CalibParsDBWeb::"<< msg << " TBE");
}

//-------------------
/** 
#define GET_NDARRAY_TEST(T,N)\
const NDArray<T>& CalibParsDBWeb::get_ndarray_##N(Query& q){\
  _default_msg(std::string("get_ndarray_"#N"(Query)"));\
  return _ndarray_##N;\
}
*/

//GET_NDARRAY_TEST(float,    float)
//GET_NDARRAY_TEST(double,   double)
//GET_NDARRAY_TEST(uint16_t, uint16)
//GET_NDARRAY_TEST(uint32_t, uint32)

//-------------------

/** working code for parameterization in macro:

const NDArray<float>& CalibParsDBWeb::get_ndarray_float(Query& q) {
  _default_msg(std::string("get_ndarray_float(Query&)"));
  typedef float T;
  rapidjson::Document doc;
  Query::map_t& qmap = q.qmap();
  calib_constants_nda<T>(_ndarray_float, doc
			     ,  qmap[Query::DETECTOR].c_str()
			     , (qmap[Query::EXPERIMENT] == "NULL") ? NULL : qmap[Query::EXPERIMENT].c_str()
			     ,  qmap[Query::CALIBTYPE].c_str()
			     ,  stoi(qmap[Query::RUN])
			     ,  stoi(qmap[Query::TIME_SEC])
			     , (qmap[Query::VERSION] == "NULL") ? NULL : qmap[Query::VERSION].c_str()
			    );
  return _ndarray_float;
}
*/

//-------------------

#define QUERY_PARAMETERS(qmap)\
	   qmap[Query::DETECTOR].c_str()\
	, (qmap[Query::EXPERIMENT] == "NULL") ? NULL : qmap[Query::EXPERIMENT].c_str()\
	,  qmap[Query::CALIBTYPE].c_str()\
	,  stoi(qmap[Query::RUN])\
	,  stoi(qmap[Query::TIME_SEC])\
	, (qmap[Query::VERSION] == "NULL") ? NULL : qmap[Query::VERSION].c_str()
      //, URLWS

//-------------------

#define GET_NDARRAY(T,N)\
const NDArray<T>& CalibParsDBWeb::get_ndarray_##N(Query& q) {\
  _default_msg(std::string("get_ndarray_"#N"(Query&)"));\
  rapidjson::Document doc;\
  Query::map_t& qmap = q.qmap();\
  calib_constants_nda<T>(_ndarray_##N, doc, QUERY_PARAMETERS(qmap));\
  return _ndarray_##N;\
}

//-------------------

GET_NDARRAY(float,    float)
GET_NDARRAY(double,   double)
GET_NDARRAY(uint16_t, uint16)
GET_NDARRAY(uint32_t, uint32)

//-------------------

const std::string& CalibParsDBWeb::get_string(Query& q) {
  _default_msg(std::string("get_string(Query&)"));
  Query::map_t& qmap = q.qmap();
  calib_constants(_string, _metadata, QUERY_PARAMETERS(qmap));
  return _string;
}

//-------------------

const rapidjson::Document& CalibParsDBWeb::get_data(Query& q) {
  _default_msg(std::string("get_data(Query&)"));
  Query::map_t& qmap = q.qmap();
  calib_constants_doc(_data, _metadata, QUERY_PARAMETERS(qmap));
  return _data;
}

//-------------------

const rapidjson::Document& CalibParsDBWeb::get_metadata(Query& q) {
  _default_msg(std::string("get_metadata(Query&)"));
  Query::map_t& qmap = q.qmap();
  calib_doc(_metadata, QUERY_PARAMETERS(qmap));
  return _metadata;
}

//-------------------
//-------------------
//-------------------
//-------------------
//-------------------
//-------------------

} // namespace calib

//-------------------
