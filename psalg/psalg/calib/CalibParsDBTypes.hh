#ifndef PSALG_CALIBPARSDBTYPES_H
#define PSALG_CALIBPARSDBTYPES_H

/** Usage
 *
 * #include "psalg/calib/CalibParsDBTypes.hh"
 */

//#include <string>

//-------------------

namespace calib {

  enum DBTYPE {DBDEF, DBWEB, DBMONGO, DBCALIB, DBHDF5};

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

