/**
 * @file
 * @brief Declares calib::Query, a set of string parameters (detector, experiment, calib type, run, ...) used to look up calibration constants.
 */
#ifndef PSALG_QUERY_H
#define PSALG_QUERY_H
//-----------------------------

//#include <string>
//#include <vector>
//#include <map>
#include <iostream> //ostream

#include "psalg/calib/CalibParsTypes.hh"
#include "psalg/calib/MDBWebUtils.hh"

//using namespace std;
using namespace psalg;

namespace calib {

//-----------------------------

/** Calibration-database query: a map from QUERY_PAR keys to string values plus a stored query string. query() builds the web-DB JSON query from the map. */
class Query {
public:

  /** Axis selector; set_paremeters() stores it as the AXISNUM parameter. */
  enum AXIS {AXIS_X=0, /**< Value 0. */ AXIS_Y, /**< Value 1. */ AXIS_Z /**< Value 2. */ };
  /** Which constructor (or set_paremeters()) last set up the query; query() builds a query string only for QUERY_MAP and QUERY_PARS. */
  enum CONSTR_TYPE {QUERY_DEFAULT=0, /**< 0: set by Query(). */ QUERY_STRING, /**< 1: set by Query(const std::string&). */ QUERY_MAP, /**< 2: set by Query(const map_t&). */ QUERY_PARS /**< 3: set by the parameter constructor and by set_paremeters(). */ }; 
  /** Keys of the parameter map; parameter_default() gives each key's default string. */
  enum QUERY_PAR {DETECTOR=0, /**< Key 0; default "NULL". */ EXPERIMENT, /**< Key 1; default "NULL". */ CALIBTYPE, /**< Key 2; default "NULL". */ RUN, /**< Key 3; default "0". */ TIME_SEC, /**< Key 4; default "0". */ VERSION, /**< Key 5; default "NULL". */ AXISNUM, /**< Key 6; default "0". */ MASKBITS, /**< Key 7; default "0377". */ MASKBITSGEO /**< Key 8; default "0377". */ }; 

  /** Map from QUERY_PAR key to its string value. */
  typedef std::map<QUERY_PAR, std::string> map_t;

  /** Default query: empty query string and all nine parameters at their defaults (set_qmap()). */
  Query();
  /** Store query as the query string and set all parameters to their defaults; query() returns the string unchanged. */
  Query(const std::string& query);
  /** Copy qmap as the parameter map (no defaults are added). Reading DETECTOR for the debug message inserts an empty DETECTOR entry if it is missing. */
  Query(const map_t& qmap);
  /** Set the parameters with set_paremeters(det, exp, ctype, run, time_sec, version); axis and mask bits take their defaults (AXIS_X, 0377). */
  Query(const char* det, const char* exp=NULL, const char* ctype=NULL,
        const unsigned run=0, const unsigned time_sec=0, const char* version=NULL);

  /** Destructor; only logs a debug message. */
  virtual ~Query();

  /** Copy construction is disabled. */
  Query(const Query&) = delete;
  /** Copy assignment is disabled. */
  Query& operator = (const Query&) = delete;

  /** Return which constructor (or set_paremeters()) last set up the query. */
  const CONSTR_TYPE& constr_type() {return _constr_type;}

  /** Return "DETECTOR: v", "EXPERIMENT: v", ... "MASKBITSGEO: v" for all nine parameters (values from parameter()), joined by sep. */
  std::string string_members(const char* sep="\n");

  /** Replace the parameter map with *map, or, if map is NULL, set all nine parameters to their defaults. */
  void set_qmap(const map_t* map=NULL); // - pointer in order to use default

  /**
   * Set all nine parameters: NULL char pointers (exp, ctype, version) become "NULL" and numbers become decimal strings (so mbits 0377 is stored as "255").
   * Sets constr_type() to QUERY_PARS. det must not be NULL, as it is passed straight to std::string.
   */
  void set_paremeters(const char* det, const char* exp=NULL, const char* ctype=NULL,
		      const unsigned run=0, const unsigned time_sec=0, const char* version=NULL,
                      const AXIS axis=AXIS_X, const unsigned mbits=0377, const unsigned mbitsgeo=0377);

  /** Set parameter t to the string p. */
  void set_paremeter(const QUERY_PAR t, const char* p);

  /** Set CALIBTYPE to name_of_calibtype(ctype). */
  void set_calibtype(const CALIB_TYPE& ctype);

  /** Return true if the map has an entry for t. */
  bool is_set(const QUERY_PAR t);

  /** Return the default string for t: "NULL" for DETECTOR, EXPERIMENT, CALIBTYPE and VERSION; "0" for RUN, TIME_SEC and AXISNUM; "0377" for MASKBITS and MASKBITSGEO. */
  std::string parameter_default(const QUERY_PAR t);

  /** Return the value stored for t, or parameter_default(t) if t is not set. */
  std::string parameter(const QUERY_PAR t);

  /** Return std::stoi(parameter(t)); stoi throws std::invalid_argument for a non-numeric value such as "NULL". */
  int parameter_int(const QUERY_PAR t);

  /** Return (unsigned)std::stoi(parameter(t)). stoi reads decimal, so the default "0377" gives 377; it throws for a non-numeric value. */
  unsigned parameter_uint(const QUERY_PAR t);

  /** Return (uint16_t)std::stoi(parameter(t)). stoi reads decimal, so the default "0377" gives 377; it throws for a non-numeric value. */
  uint16_t parameter_uint16(const QUERY_PAR t);

  /** Return a reference to the parameter map. */
  map_t& qmap(){return _qmap;}

  /**
   * For QUERY_MAP and QUERY_PARS, build the JSON query with dbnames_collection_query() from DETECTOR, EXPERIMENT, CALIBTYPE, RUN, TIME_SEC and VERSION, store it and return it; otherwise return the stored string.
   * Values are passed as strings, so a VERSION left at "NULL" is added to the query as "version":"NULL".
   */
  std::string query();

  /** Write o.string_members(" ") to os and return os. */
  friend std::ostream& operator << (std::ostream& os, Query& o);

protected:

  std::string _query;
  map_t       _qmap;

private:

  CONSTR_TYPE _constr_type;
  void _msg_init(const std::string& add="") const;

  std::string _string_from_char(const char* p);
  std::string _string_from_uint(const unsigned p);
}; // class

//-----------------------------

} // namespace calib

#endif // PSALG_QUERY_H
