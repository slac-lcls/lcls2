//-------------------

#include "DetectorStore.hh" // DETTYPE, AREA_DETECTOR, WF_DETECTOR,...

#include "AreaDetectorStore.hh"
//#include "WFDetectorStore.hh"
//#include "BldDetectorStore.hh"
//#include "...DetectorStore.hh"

//-------------------

namespace detector {

  //auto getDetector(const std::string& detname) { // for >= C++14
  Detector* getDetector(const std::string& detname) {

    const DETTYPE dettype = find_dettype(detname);

    MSG(INFO, "getDetector for name " << detname << " DETTYPE=" << dettype);

    //if     (dettype == AREA_DETECTOR) return dynamic_cast<AreaDetector*>(getAreaDetector(detname));
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
