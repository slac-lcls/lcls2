//-------------------

#include "psalg/calib/CalibParsDBTypes.hh"

//-------------------

namespace calib {

  const char* name_of_dbtype(const DBTYPE dbtype) {
    switch(dbtype)
      {
      case DBDEF   : return "DBDEF";
      case DBWEB   : return "DBWEB";
      case DBCALIB : return "DBCALIB";
      case DBHDF5  : return "DBHDF5";
      default:
        //MSG(WARNING, "Unknown DBTYPE " << dbtype);
        throw "Requested unknown DBTYPE";
      }
  }

} // namespace calib

//-------------------
