//-------------------

#include "psalg/detector/AreaDetectorStore.hh"

#include "psalg/detector/AreaDetectorEpix100a.hh"
//#include "psalg/detector/AreaDetectorCspad.hh"
//#include "psalg/detector/AreaDetectorCspad2x2.hh"
//#include "psalg/detector/AreaDetectorPnccd.hh"
//#include "psalg/detector/AreaDetector....hh"

//-------------------

namespace detector {

  AreaDetector* getAreaDetector(const std::string& detname) {

    const AREADETTYPE dettype = find_area_dettype(detname);
    MSG(INFO, "getAreaDetector for name " << detname << " AREADETTYPE=" << dettype);

    if (dettype == EPIX100A) return new AreaDetectorEpix100a(detname);
    //else if (dettype == CSPAD2X2) return new AreaDetectorCspad2x2(detname);
    //else if (dettype == CSPAD) return new AreaDetectorCspad(detname);
    //else if (dettype == ...) return new AreaDetector...(detname);
    else {
      MSG(WARNING, "Not implemented detector for name " << detname);
      return NULL;
    }

    return NULL;
  }

} // namespace detector

//-------------------
