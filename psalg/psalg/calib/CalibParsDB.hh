#ifndef PSALG_CALIBPARSDB_H
#define PSALG_CALIBPARSDB_H

//-------------------

#include "psalg/calib/CalibParsTypes.hh"
#include "psalg/calib/NDArray.hh"
#include "psalg/calib/Query.hh"
#include "psalg/calib/Response.hh"

using namespace psalg;

namespace calib {

//-------------------

class CalibParsDB {
public:

  CalibParsDB(const char* dbtypename = "Default-Base-NoDB");
  virtual ~CalibParsDB();

  const std::string& dbtypename(){return _dbtypename;}

  virtual const NDArray<double>&   get_ndarray_double(const Query&);
  virtual const NDArray<float>&    get_ndarray_float (const Query&);
  virtual const NDArray<uint16_t>& get_ndarray_uint16(const Query&);
  virtual const NDArray<uint32_t>& get_ndarray_uint32(const Query&);
  virtual const std::string&       get_string        (const Query&);

  virtual const Response&          get_responce      (const Query&);

  CalibParsDB(const CalibParsDB&) = delete;
  CalibParsDB& operator = (const CalibParsDB&) = delete;

protected:

  NDArray<double>    _ndarray_double;
  NDArray<float>     _ndarray_float;
  NDArray<uint16_t>  _ndarray_uint16;
  NDArray<uint32_t>  _ndarray_uint32;
  const std::string  _string;
  Response           _response;

private:
  void _default_msg(const std::string& msg=std::string()) const;
  const std::string _dbtypename;
}; // class

//-------------------

} // namespace calib

#endif // PSALG_CALIBPARSDB_H
