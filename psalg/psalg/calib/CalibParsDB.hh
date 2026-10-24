/**
 * @file
 * @brief Declares calib::CalibParsDB, the base class for calibration-constant backends.
 */
#ifndef PSALG_CALIBPARSDB_H
#define PSALG_CALIBPARSDB_H

//-------------------

#include "psalg/calib/CalibParsDBTypes.hh"
#include "psalg/calib/NDArray.hh"
#include "psalg/calib/Query.hh"
#include <rapidjson/document.h>

using namespace psalg;

namespace calib {

//-------------------

/** Base class for calibration-constant backends (see getCalibParsDB()). Every getter here is a default that logs a warning asking for re-implementation and returns an empty member; CalibParsDBWeb overrides them. */
class CalibParsDB {
public:

  /** Store dbtypename and log a debug message. */
  CalibParsDB(const char* dbtypename = "Default-Base-NoDB");
  /** Destructor; only logs a debug message. */
  virtual ~CalibParsDB();

  /** Return the type name given to the constructor. */
  const std::string& dbtypename(){return _dbtypename;}

  /** Default: logs a re-implement warning and returns an internal array that this class never fills. */
  virtual const NDArray<float>&      get_ndarray_float (Query&);
  /** Default: logs a re-implement warning and returns an internal array that this class never fills. */
  virtual const NDArray<double>&     get_ndarray_double(Query&);
  /** Default: logs a re-implement warning and returns an internal array that this class never fills. */
  virtual const NDArray<uint16_t>&   get_ndarray_uint16(Query&);
  /** Default: logs a re-implement warning and returns an internal array that this class never fills. */
  virtual const NDArray<uint32_t>&   get_ndarray_uint32(Query&);
  /** Default: logs a re-implement warning and returns an internal string that this class never fills. */
  virtual const std::string&         get_string        (Query&);
  /** Default: logs a re-implement warning and returns an internal document that this class never fills. */
  virtual const rapidjson::Document& get_data          (Query&);
  /** Default: logs a re-implement warning and returns an internal document that this class never fills. */
  virtual const rapidjson::Document& get_metadata      (Query&);

  /** Copy construction is disabled. */
  CalibParsDB(const CalibParsDB&) = delete;
  /** Copy assignment is disabled. */
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

private:
  void _default_msg(const std::string& msg=std::string()) const;
  const std::string _dbtypename;
}; // class

//-------------------

} // namespace calib

#endif // PSALG_CALIBPARSDB_H
