
#include "psalg/calib/Query.hh"
#include "psalg/utils/Logger.hh" // for MSG
#include <iostream> // to_string C++11

//using namespace std;
using namespace psalg; // for NDArray

//-------------------

namespace calib {

Query::Query()
  : _query(), _constr_type(DEFAULT) {_msg_init();}


Query::Query(const std::string& query)
  : _query(query), _constr_type(STRING_QUERY) {_msg_init();}


Query::Query(const map_t& qmap)
  : _qmap(qmap), _constr_type(MAP_QUERY) {_msg_init();}


Query::Query(const char* det, const char* exp, const char* ctype, const unsigned run, const unsigned time_sec, const char* version)
  : _constr_type(PARS_QUERY) {
  _msg_init(std::string(" detector ") + det);

  _qmap["det"]      = std::string(det);
  _qmap["exp"]      = std::string((exp != NULL) ? exp : "");
  _qmap["ctype"]    = std::string((ctype != NULL) ? ctype: "");
  _qmap["run"]      = std::to_string(run);
  _qmap["time_sec"] = std::to_string(time_sec);
  _qmap["version"]  = std::string((version!= NULL) ? version: "");
}

//-------------------

Query::~Query() {
  MSG(DEBUG, "In d-tor Query for c-tor #" << _constr_type);
}

//-------------------

void Query::_msg_init(const std::string& add) const {
  MSG(DEBUG, "In c-tor Query #" << _constr_type << add);
}

//-------------------

} // namespace calib

//-------------------
