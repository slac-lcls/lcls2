/**
 * @file
 * @brief Declares calib::DBTYPE, the calibration database backend types, and name_of_dbtype().
 */
#ifndef PSALG_CALIBPARSDBTYPES_H
#define PSALG_CALIBPARSDBTYPES_H

/** Usage
 *
 * #include "psalg/calib/CalibParsDBTypes.hh"
 */

//#include <string>

//-------------------

namespace calib {

  /** Calibration database backend types, used by getCalibParsDB() and name_of_dbtype(). */
  enum DBTYPE {DBDEF, /**< Value 0; getCalibParsDB() returns the base CalibParsDB. */ DBWEB, /**< Value 1; getCalibParsDB() returns a CalibParsDBWeb. */ DBMONGO, /**< Value 2; name_of_dbtype() has no case for it and throws, and getCalibParsDB() does not implement it. */ DBCALIB, /**< Value 3; getCalibParsDB() does not implement it and returns NULL. */ DBHDF5 /**< Value 4; getCalibParsDB() does not implement it and returns NULL. */ };

  /** Return "DBDEF", "DBWEB", "DBCALIB" or "DBHDF5"; throws a const char* for any other value, including DBMONGO. */
  const char* name_of_dbtype(const DBTYPE);

  //typedef double      db_double_t;
  //typedef float       db_float_t;
  //typedef uint16_t    db_uint16_t;
  //typedef uint32_t    db_uint32_t;
  //typedef int16_t     db_int16_t;
  //typedef int32_t     db_int32_t;
  //typedef std::string db_string_t;

} // namespace calib

//-------------------

#endif // PSALG_CALIBPARSDBTYPES_H

