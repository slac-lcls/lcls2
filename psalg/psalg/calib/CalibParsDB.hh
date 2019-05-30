#ifndef PSALG_CALIBPARSDB_H
#define PSALG_CALIBPARSDB_H

//-------------------

#include "psalg/calib/CalibParsDBTypes.hh"
#include "psalg/calib/NDArray.hh"
#include "psalg/calib/Query.hh"
#include <rapidjson/document.h>
//#include "psalg/calib/ResponseDB.hh"

using namespace psalg;

namespace calib {

//-------------------

class CalibParsDB {
public:

  CalibParsDB(const char* dbtypename = "Default-Base-NoDB");
  virtual ~CalibParsDB();

  const std::string& dbtypename(){return _dbtypename;}

  virtual const NDArray<float>&      get_ndarray_float (Query&);
  virtual const NDArray<double>&     get_ndarray_double(Query&);
  virtual const NDArray<uint16_t>&   get_ndarray_uint16(Query&);
  virtual const NDArray<uint32_t>&   get_ndarray_uint32(Query&);
  virtual const std::string&         get_string        (Query&);
  virtual const rapidjson::Document& get_data          (Query&);
  virtual const rapidjson::Document& get_metadata      (Query&);

  CalibParsDB(const CalibParsDB&) = delete;
  CalibParsDB& operator = (const CalibParsDB&) = delete;

protected:

  NDArray<double>     _ndarray_double;
  NDArray<float>      _ndarray_float;
  NDArray<uint16_t>   _ndarray_uint16;
  NDArray<uint32_t>   _ndarray_uint32;
  std::string         _string;
  rapidjson::Document _document;
  rapidjson::Document _data;
  rapidjson::Document _metadata;
  //ResponseDB          _response;

private:
  void _default_msg(const std::string& msg=std::string()) const;
  const std::string _dbtypename;
}; // class

//-------------------

} // namespace calib

#endif // PSALG_CALIBPARSDB_H
