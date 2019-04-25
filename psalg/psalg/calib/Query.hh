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

class Query {
public:

  Query() : _query() {}
  Query(const char* query) : _query(query) {}
  Query(const std::string& query) : _query(query) {}
  Query(const std::map<std::string, std::string>& qmap) : _qmap(qmap) {}
  //Query(const char* dbname, const char* colname, const char* urlws=URLWS){}
  virtual ~Query(){}

  inline std::string query() {return _query;}

  Query(const Query&) = delete;
  Query& operator = (const Query&) = delete;

//  void
//  query(std::map<std::string, std::string>& omap, const char* det, const char* exp=NULL, const char* ctype=NULL, const unsigned run=0, const unsigned time_sec=0, const char* version=NULL);

protected:

  const std::string _query;
  const std::map<std::string, std::string> _qmap;
}; // class

//-----------------------------

} // namespace calib

#endif // PSALG_QUERY_H
