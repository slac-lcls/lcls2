//-------------------

//#include "CalibParsDB.hh"
#include "CalibParsDBStore.hh"
#include "CalibParsDBWeb.hh"
//#include "CalibParsDBCalib.hh"
//#include "CalibParsDBHDF5.hh"

//-------------------

namespace calib {

  /**
   * Factory method for DB selection.
   *
   */

//-------------------

  CalibParsDB* getCalibParsDB(const DBTYPE dbtype) {
    MSG(DEBUG, "getCalibParsDB for dbtype name: " << name_of_dbtype(dbtype));

    if      (dbtype == calib::DBDEF) return new CalibParsDB("Default base - NoDB");
    else if (dbtype == calib::DBWEB) return new CalibParsDBWeb();
    //else if (dbtype == calib::DBCALIB) return new CalibParsDBCalib(detname);
    //else if (dbtype == calib::DBHDF5)  return new CalibParsDBHDF5(detname);
    else {
      MSG(WARNING, "Not implemented CalibParsDB for dbtype " << dbtype);
      return NULL;
    }

    return NULL;
  }

} // namespace calib

//-------------------
