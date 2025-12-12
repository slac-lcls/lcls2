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

class Query {
public:

  enum AXIS {AXIS_X=0, AXIS_Y, AXIS_Z};
  enum CONSTR_TYPE {QUERY_DEFAULT=0, QUERY_STRING, QUERY_MAP, QUERY_PARS}; 
  enum QUERY_PAR {DETECTOR=0, EXPERIMENT, CALIBTYPE, RUN, TIME_SEC, VERSION, AXISNUM, MASKBITS, MASKBITSGEO}; 

  typedef std::map<QUERY_PAR, std::string> map_t;

  Query();
  Query(const std::string& query);
  Query(const map_t& qmap);
  Query(const char* det, const char* exp=NULL, const char* ctype=NULL,
        const unsigned run=0, const unsigned time_sec=0, const char* version=NULL);

  virtual ~Query();

  Query(const Query&) = delete;
  Query& operator = (const Query&) = delete;

  const CONSTR_TYPE& constr_type() {return _constr_type;}

  std::string string_members(const char* sep="\n");

  void set_qmap(const map_t* map=NULL); // - pointer in order to use default

  void set_paremeters(const char* det, const char* exp=NULL, const char* ctype=NULL,
		      const unsigned run=0, const unsigned time_sec=0, const char* version=NULL,
                      const AXIS axis=AXIS_X, const unsigned mbits=0377, const unsigned mbitsgeo=0377);

  void set_paremeter(const QUERY_PAR t, const char* p);

  void set_calibtype(const CALIB_TYPE& ctype);

  bool is_set(const QUERY_PAR t);

  std::string parameter_default(const QUERY_PAR t);

  std::string parameter(const QUERY_PAR t);

  int parameter_int(const QUERY_PAR t);

  unsigned parameter_uint(const QUERY_PAR t);

  uint16_t parameter_uint16(const QUERY_PAR t);

  map_t& qmap(){return _qmap;}

  std::string query();

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
