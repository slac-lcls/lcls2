#ifndef PSALG_RESPONSEDB_H
#define PSALG_RESPONSEDB_H
//-----------------------------

#include "psalg/calib/CalibParsDBTypes.hh"
#include "psalg/calib/NDArray.hh"

//using namespace std;
using namespace psalg; // for NDArray

namespace calib {

//-----------------------------

class ResponseDB {
public:

  ResponseDB(){}
  ResponseDB(const char* dbname) : _dbname(dbname) {}
  ResponseDB(const std::string& dbname) : _dbname(dbname) {}
  virtual ~ResponseDB(){}

  ResponseDB(const ResponseDB&) = delete;
  ResponseDB& operator = (const ResponseDB&) = delete;

  inline std::string dbname() {return _dbname;}

  NDArray<double>    _ndarray_double;
  NDArray<float>     _ndarray_float;
  NDArray<uint16_t>  _ndarray_uint16;
  NDArray<uint32_t>  _ndarray_uint32;
  std::string        _string;

protected:

  const std::string  _dbname;

}; // class

//-----------------------------

} // namespace calib

#endif // PSALG_RESPONSEDB_H
