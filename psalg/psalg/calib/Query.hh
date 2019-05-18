#ifndef PSALG_QUERY_H
#define PSALG_QUERY_H
//-----------------------------

//#include <string>
//#include <vector>
//#include <map>

#include "psalg/calib/CalibParsTypes.hh"
#include "psalg/calib/MDBWebUtils.hh"

//using namespace std;
using namespace psalg;

namespace calib {

//-----------------------------
//typedef std::map<std::string, std::string> map_t;
//typedef std::map<const std::string, std::string> map_t;

class Query {
public:

  enum CONSTR_TYPE {QUERY_DEFAULT=0, QUERY_STRING, QUERY_MAP, QUERY_PARS}; 
  enum QUERY_PAR{DETECTOR=0, EXPERIMENT, CALIBTYPE, RUN, TIME_SEC, VERSION}; 

  //typedef std::map<const char*, std::string> map_t;
  typedef std::map<const QUERY_PAR, std::string> map_t;

  Query();
  Query(const std::string& query);
  Query(const map_t& qmap);
  Query(const char* det, const char* exp=NULL, const char* ctype=NULL,
        const unsigned run=0, const unsigned time_sec=0, const char* version=NULL);

  //Query(const char* dbname, const char* colname, const char* urlws=URLWS){}

  virtual ~Query();

  Query(const Query&) = delete;
  Query& operator = (const Query&) = delete;

  const CONSTR_TYPE& constr_type() {return _constr_type;}

  std::string string_members(const char* sep="\n");

  void set_paremeters(const char* det, const char* exp=NULL, const char* ctype=NULL,
		      const unsigned run=0, const unsigned time_sec=0, const char* version=NULL);

  void set_paremeter(const QUERY_PAR t, const char* p);

  void set_qmap(const map_t* map=NULL); // map_t* - pointer in order to use default

  map_t& qmap(){return _qmap;}

  std::string query();

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
