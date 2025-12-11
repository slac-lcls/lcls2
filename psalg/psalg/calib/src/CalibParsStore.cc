//-------------------

#include "AreaDetectorTypes.hh"
#include "CalibParsStore.hh"

#include "CalibParsEpix100a.hh"
//#include "CalibParsCspad.hh"
//#include "CalibParsCspad2x2.hh"
//#include "CalibParsPnccd.hh"
//#include "CalibPars....hh"

//-------------------

namespace calib {

  CalibPars* getCalibPars(const char* detname, const DBTYPE& dbtype) {

    const detector::AREADETTYPE dettype = detector::find_area_dettype(detname);
    MSG(DEBUG, "getCalibPars for name " << detname << " AREADETTYPE=" << dettype << " dbtypename=" << name_of_dbtype(dbtype));

    if      (dettype == detector::UNDEFINED) return new CalibPars(detname, dbtype); // base class
    else if (dettype == detector::EPIX100A)  return new CalibParsEpix100a(detname, dbtype);
    else if (dettype == detector::PNCCD)     return new CalibPars(detname, dbtype);
    //else if (dettype == detector::CSPAD2X2) return new CalibParsCspad2x2(detname, dbtype);
    //else if (dettype == detector::CSPAD) return new CalibParsCspad(detname, dbtype);
    //else if (dettype == detector::...) return new CalibPars...(detname, dbtype);
    else {
      MSG(WARNING, "Not implemented CalibPars for detname " << detname);
      throw "Not implemented CalibPars for detname";
      //return NULL;
    }
  }

} // namespace calib

//-------------------
