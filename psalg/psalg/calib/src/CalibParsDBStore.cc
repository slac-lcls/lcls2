//-------------------

//#include "psalg/calib/CalibParsDB.hh"
#include "psalg/calib/CalibParsDBStore.hh"
#include "psalg/calib/CalibParsDBWeb.hh"
//#include "psalg/calib/CalibParsDBMongo.hh"
//#include "psalg/calib/CalibParsDBCalib.hh"
//#include "psalg/calib/CalibParsDBHDF5.hh"

//-------------------

namespace calib {

  /**
   * Factory method for DB selection.
   *
   *
   *
   *
   */
  /*
  CalibPars* getCalibParsDB(const std::string& detname, const DBTYPE& dbtype) {
    MSG(DEBUG, "getCalibParsDB for dbtype " << dbtype);

    if (dbtype == calib::DBWEB) return new CalibParsDBWeb(detname);
    //else if (dbtype == calib::DBMONGO) return new CalibParsDBMongo(detname);
    //else if (dbtype == calib::DBCALIB) return new CalibParsDBCalib(detname);
    //else if (dbtype == calib::DBHDF5)  return new CalibParsDBHDF5(detname);
    else {
      MSG(WARNING, "Not implemented CalibParsDB for dbtype " << dbtype);
      return NULL;
    }

    return NULL;
  }
  */

//-------------------

  CalibParsDB* getCalibParsDB(const DBTYPE& dbtype) {
    MSG(DEBUG, "getCalibParsDB for dbtype " << dbtype);

    if      (dbtype == calib::DBDEF) return new CalibParsDB("Default base - NoDB");
    else if (dbtype == calib::DBWEB) return new CalibParsDBWeb();
    //else if (dbtype == calib::DBMONGO) return new CalibParsDBMongo(detname);
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
