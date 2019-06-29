
#include "psalg/calib/Query.hh"
#include "psalg/utils/Logger.hh" // for MSG
#include <iostream> // to_string C++11, ostream

#include "psalg/calib/MDBWebUtils.hh"

//using namespace std;
using namespace psalg; // for NDArray

//-------------------

namespace calib {

  Query::Query()
    : _query(), _constr_type(QUERY_DEFAULT) {set_qmap(); _msg_init();}


  Query::Query(const std::string& query)
    : _query(query), _constr_type(QUERY_STRING) {set_qmap(); _msg_init();}


  Query::Query(const map_t& qmap)
    : _qmap(qmap), _constr_type(QUERY_MAP) {_msg_init(std::string(" detector ") + _qmap[DETECTOR]);}


  Query::Query(const char* det, const char* exp, const char* ctype,
               const unsigned run, const unsigned time_sec, const char* version)
    : _constr_type(QUERY_PARS) {
    _msg_init(std::string(" detector ") + det);

    set_paremeters(det, exp, ctype, run, time_sec, version);
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
     return std::string((p != NULL) ? p : "NULL");
  }

//-------------------

  std::string Query::_string_from_uint(const unsigned p) {
     return std::to_string(p);
  }

//-------------------

  std::string Query::string_members(const char* sep) {
     std::stringstream ss;
     ss          << "DETECTOR: "   << parameter(DETECTOR)
          << sep << "EXPERIMENT: " << parameter(EXPERIMENT)
          << sep << "CALIBTYPE: "  << parameter(CALIBTYPE)
          << sep << "RUN: "        << parameter(RUN)
          << sep << "TIME_SEC: "   << parameter(TIME_SEC)
	  << sep << "VERSION: "    << parameter(VERSION)
          << sep << "AXISNUM: "    << parameter(AXISNUM)
          << sep << "MASKBITS: "   << parameter(MASKBITS)
          << sep << "MASKBITSGEO: "<< parameter(MASKBITSGEO);
     return ss.str();
  }

//-------------------

  void Query::set_qmap(const map_t* map) {
    if(map) {
      _qmap = *map;
    }
    else { // set default values
      _qmap[DETECTOR]   = "NULL";
      _qmap[EXPERIMENT] = "NULL";
      _qmap[CALIBTYPE]  = "NULL";
      _qmap[RUN]        = "0";
      _qmap[TIME_SEC]   = "0";
      _qmap[VERSION]    = "NULL";
      _qmap[AXISNUM]    = "0";   
      _qmap[MASKBITS]   = "0377";   
      _qmap[MASKBITSGEO]= "0377";   
    }
  } 

//-------------------

  void Query::set_paremeters(const char* det, const char* exp, const char* ctype,
			     const unsigned run, const unsigned time_sec, const char* version,
                             const AXIS axis, const unsigned mbits, const unsigned mbitsgeo) {

    _qmap[DETECTOR]   = std::string(det);
    _qmap[EXPERIMENT] = _string_from_char(exp);
    _qmap[CALIBTYPE]  = _string_from_char(ctype);
    _qmap[RUN]        = _string_from_uint(run);
    _qmap[TIME_SEC]   = _string_from_uint(time_sec);
    _qmap[VERSION]    = _string_from_char(version);
    _qmap[AXISNUM]    = _string_from_uint(axis);
    _qmap[MASKBITS]   = _string_from_uint(mbits);
    _qmap[MASKBITSGEO]= _string_from_uint(mbitsgeo);

    _constr_type = QUERY_PARS;
  }

//-------------------

  void Query::set_paremeter(const QUERY_PAR t, const char* p) {
    _qmap[t] = std::string(p);
  } 

//-------------------

  void Query::set_calibtype(const CALIB_TYPE& ctype) {
    _qmap[CALIBTYPE] = std::string(name_of_calibtype(ctype));
  } 

//-------------------

  bool Query::is_set(const QUERY_PAR t) {return _qmap.find(t) != _qmap.end();}
  //bool Query::is_set(const QUERY_PAR t) {return _qmap.count(t) > 0;}

//-------------------

  std::string Query::parameter_default(const QUERY_PAR t) {    
    switch(t) {
    case DETECTOR    : return "NULL";
    case EXPERIMENT  : return "NULL";
    case CALIBTYPE   : return "NULL";
    case RUN         : return "0";   
    case TIME_SEC    : return "0";   
    case VERSION     : return "NULL";
    case AXISNUM     : return "0";   
    case MASKBITS    : return "0377";   
    case MASKBITSGEO : return "0377";   
    default          : return "NULL";
    }
  }
 
//-------------------

  std::string Query::parameter(const QUERY_PAR t) {
    return is_set(t) ? _qmap[t] : parameter_default(t);
  } 

//-------------------

  int Query::parameter_int(const QUERY_PAR t) {
    return stoi(parameter(t));
  } 

//-------------------

  unsigned Query::parameter_uint(const QUERY_PAR t) {
    return (unsigned)stoi(parameter(t));
  } 

//-------------------

  uint16_t Query::parameter_uint16(const QUERY_PAR t) {
    return (uint16_t)stoi(parameter(t));
  } 

//-------------------
/**
 * Should be implemented for partiqular DB. 
 * Shown implementation for WebDB/MongoDB
 */
  std::string Query::query() {
    switch(_constr_type)
    {
      case QUERY_PARS : // the same as QUERY_MAP
      case QUERY_MAP  : {
          _query = "XXXXXXXXXXXXXXXXXXXXXX";
          std::map<std::string, std::string> omap;
          dbnames_collection_query(omap
				   ,_qmap[DETECTOR].c_str()
				   ,_qmap[EXPERIMENT].c_str()
				   ,_qmap[CALIBTYPE].c_str()
				   , stoi(_qmap[RUN])
				   , stoi(_qmap[TIME_SEC])
				   ,_qmap[VERSION].c_str()
				  );

          _query = omap["query"]; // "{\"ctype\":\"pedestals\", \"run\":{\"$lte\": 87}}";
          //const std::string& db_det  = omap["db_det"];
          //const std::string& db_exp  = omap["db_exp"];
          //const std::string& colname = omap["colname"];
      }
      case QUERY_STRING  : // the same as QUERY_DEFAULT
      case QUERY_DEFAULT : // the same as default
      default: break;
    }
    return _query;
  }

//-------------------

  std::ostream& operator << (std::ostream& os, Query& o) {
    os << o.string_members(" ");
    return os;
  }

//-------------------

} // namespace calib

//-------------------
