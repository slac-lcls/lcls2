
#include "psalg/calib/Query.hh"
#include "psalg/utils/Logger.hh" // for MSG
#include <iostream> // to_string C++11, ostream

//using namespace std;
using namespace psalg; // for NDArray

//-------------------

namespace calib {

  Query::Query()
    : _query(), _constr_type(QUERY_DEFAULT) {set_qmap(); _msg_init();}


  Query::Query(const std::string& query)
    : _query(query), _constr_type(QUERY_STRING) {set_qmap(); _msg_init();}


  Query::Query(const map_t& qmap)
    : _qmap(qmap), _constr_type(QUERY_MAP) {_msg_init();}


  Query::Query(const char* det, const char* exp, const char* ctype,
               const unsigned run, const unsigned time_sec, const char* version)
    : _constr_type(QUERY_PARS) {
    _msg_init(std::string(" detector ") + det);

    _qmap[DETECTOR]   = std::string(det);
    _qmap[EXPERIMENT] = _string_from_char(exp);
    _qmap[CALIBTYPE]  = _string_from_char(ctype);
    _qmap[RUN]        = _string_from_uint(run);
    _qmap[TIME_SEC]   = _string_from_uint(time_sec);
    _qmap[VERSION]    = _string_from_char(version);
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

  std::string Query::_string_from_char(const char* p) {
     return std::string((p != NULL) ? p : "NONE");
  }

//-------------------

  std::string Query::_string_from_uint(const unsigned p) {
     return std::to_string(p);
  }

//-------------------

  std::string Query::string_members(const char* sep) {
     std::stringstream ss;
        ss     << "DETECTOR: "   << _qmap[DETECTOR]
        << sep << "EXPERIMENT: " << _qmap[EXPERIMENT]
        << sep << "CALIBTYPE: "  << _qmap[CALIBTYPE]
        << sep << "RUN: "        << _qmap[RUN]
        << sep << "TIME_SEC: "   << _qmap[TIME_SEC]
        << sep << "VERSION: "    << _qmap[VERSION];
     return ss.str();
  }

//-------------------

  void Query::set_paremeter(const QUERY_PAR t, const char* p) {
    _qmap[t] = std::string(p);
  } 

//-------------------

  void Query::set_qmap(const map_t* map) {
    if(map) {_qmap = *map; return;}

    // else set default values
    _qmap[DETECTOR]   = "NONE";
    _qmap[EXPERIMENT] = "NONE";
    _qmap[CALIBTYPE]  = "NONE";
    _qmap[RUN]        = "NONE";
    _qmap[TIME_SEC]   = "NONE";
    _qmap[VERSION]    = "NONE";
  } 

//-------------------

  std::string Query::query() {
    switch(_constr_type)
    {
      case QUERY_PARS : // the same as QUERY_MAP
      case QUERY_MAP  :
           _query = "XXXXXXXXXXXXXXXXXXXXXX";

      case QUERY_STRING  : // the same as QUERY_DEFAULT
      case QUERY_DEFAULT : // the same as default
      default: break;
    }
    return _query;
  }

//-------------------

} // namespace calib

//-------------------
