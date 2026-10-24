/**
 * @file
 * @brief Declares calib::CalibParsDBWeb, the CalibParsDB backend that reads calibration constants from the calibration web service.
 */
#ifndef PSALG_CALIBPARSDBWEB_H
#define PSALG_CALIBPARSDBWEB_H
//-----------------------------

#include "psalg/calib/CalibParsDB.hh" // #include "psalg/calib/Query.hh"
#include "psalg/calib/MDBWebUtils.hh"

using namespace psalg; // for NDArray

namespace calib {

//-----------------------------

/** CalibParsDB backend that fetches constants with the MDBWebUtils functions. Parameters come from the Query map: RUN and TIME_SEC are converted with std::stoi, and EXPERIMENT or VERSION equal to "NULL" are passed as NULL. */
class CalibParsDBWeb : public CalibParsDB {
public:

  /** Construct with the type name "DBWEB". */
  CalibParsDBWeb();
  /** Destructor; only logs a debug message. */
  virtual ~CalibParsDBWeb();

  /** Fetch the constants selected by q with calib_constants_nda<float>() into an internal array and return it; the next call overwrites it. */
  virtual const NDArray<float>&      get_ndarray_float (Query&);
  /** Fetch the constants selected by q with calib_constants_nda<double>() into an internal array and return it; the next call overwrites it. */
  virtual const NDArray<double>&     get_ndarray_double(Query&);
  /** Fetch the constants selected by q with calib_constants_nda<uint16_t>() into an internal array and return it; the next call overwrites it. */
  virtual const NDArray<uint16_t>&   get_ndarray_uint16(Query&);
  /** Fetch the constants selected by q with calib_constants_nda<uint32_t>() into an internal array and return it; the next call overwrites it. */
  virtual const NDArray<uint32_t>&   get_ndarray_uint32(Query&);
  /** Fetch the data selected by q with calib_constants() into an internal string (and its metadata into an internal document) and return the string. */
  virtual const std::string&         get_string        (Query&);
  /** Log a "TBE" warning, then fetch with calib_constants_doc(): the data is parsed as JSON into an internal document, which is returned (metadata goes into another internal document). */
  virtual const rapidjson::Document& get_data          (Query&);
  /** Log a "TBE" warning, then fetch only the metadata document selected by q with calib_doc() and return it. */
  virtual const rapidjson::Document& get_metadata      (Query&);
 
  /** Copy construction is disabled. */
  CalibParsDBWeb(const CalibParsDBWeb&) = delete;
  /** Copy assignment is disabled. */
  CalibParsDBWeb& operator = (const CalibParsDBWeb&) = delete;

private:

  void _default_msg(const std::string& msg=std::string()) const;

}; // class

//-----------------------------

} // namespace calib

#endif // PSALG_CALIBPARSDBWEB_H
