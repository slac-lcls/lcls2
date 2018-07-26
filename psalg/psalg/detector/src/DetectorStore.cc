//-------------------

#include "psalg/detector/DetectorStore.hh" // DETTYPE, AREA_DETECTOR, WF_DETECTOR,...

#include "psalg/detector/AreaDetectorStore.hh"
//#include "psalg/detector/WFDetectorStore.hh"
//#include "psalg/detector/BldDetectorStore.hh"
//#include "psalg/detector/DdlDetectorStore.hh"

//-------------------

namespace detector {

  Detector* getDetector(const std::string& detname) {

    const DETTYPE dettype = find_dettype(detname);

    MSG(INFO, "getDetector for name " << detname << " DETTYPE=" << dettype);

    if     (dettype == AREA_DETECTOR) return getAreaDetector(detname);
    //else if(dettype == WF_DETECTOR)   return getWFDetector(detname);
    //else if(dettype == BLD_DETECTOR)  return getBldDetector(detname);
    //else if(dettype == ..._DETECTOR)  return get...Detector(detname);
    else {
      MSG(WARNING, "Not implemented detector for name " << detname);
      return NULL;
    }

    return NULL;
  }

} // namespace detector

//-------------------
