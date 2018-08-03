//-------------------

#include "psalg/detector/AreaDetectorTypes.hh"
#include "psalg/calib/CalibParsStore.hh"

#include "psalg/calib/CalibParsEpix100a.hh"
//#include "psalg/calib/CalibParsCspad.hh"
//#include "psalg/calib/CalibParsCspad2x2.hh"
//#include "psalg/calib/CalibParsPnccd.hh"
//#include "psalg/calib/CalibPars....hh"

//-------------------

namespace calib {

  CalibPars* getCalibPars(const std::string& detname) {

    const detector::AREADETTYPE dettype = detector::find_area_dettype(detname);
    MSG(INFO, "getCalibPars for name " << detname << " AREADETTYPE=" << dettype);

    //if (dettype == detector::EPIX100A) return new CalibPars(detname); // for test only
    if (dettype == detector::EPIX100A) return new CalibParsEpix100a(detname);
    //else if (dettype == detector::CSPAD2X2) return new CalibParsCspad2x2(detname);
    //else if (dettype == detector::CSPAD) return new CalibParsCspad(detname);
    //else if (dettype == detector::...) return new CalibPars...(detname);
    else {
      MSG(WARNING, "Not implemented CalibPars for detname " << detname);
      return NULL;
    }

    return NULL;
  }

} // namespace calib

//-------------------
