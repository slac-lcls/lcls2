//-------------------

#include "psalg/calib/AreaDetectorTypes.hh"
#include "psalg/calib/CalibParsStore.hh"

#include "psalg/calib/CalibParsEpix100a.hh"
//#include "psalg/calib/CalibParsCspad.hh"
//#include "psalg/calib/CalibParsCspad2x2.hh"
//#include "psalg/calib/CalibParsPnccd.hh"
//#include "psalg/calib/CalibPars....hh"

//-------------------

namespace calib {

  CalibPars* getCalibPars(const char* detname) {

    const detector::AREADETTYPE dettype = detector::find_area_dettype(detname);
    MSG(INFO, "getCalibPars for name " << detname << " AREADETTYPE=" << dettype);

    if      (dettype == detector::UNDEFINED) return new CalibPars(detname); // base class
    else if (dettype == detector::EPIX100A)  return new CalibParsEpix100a(detname);
    //else if (dettype == detector::CSPAD2X2) return new CalibParsCspad2x2(detname);
    //else if (dettype == detector::CSPAD) return new CalibParsCspad(detname);
    //else if (dettype == detector::...) return new CalibPars...(detname);
    else {
      MSG(WARNING, "Not implemented CalibPars for detname " << detname);
      throw "Not implemented CalibPars for detname";
      //return NULL;
    }
  }

} // namespace calib

//-------------------
