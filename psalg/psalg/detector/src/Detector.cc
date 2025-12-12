//-------------------

#include "psalg/detector/Detector.hh"

//-------------------

namespace detector {

  Detector::Detector(const std::string& detname, const DETTYPE& dettype) : 
    _detname(detname), _dettype(dettype){
    MSG(DEBUG, "In c-tor Detector for " << detname)
    if(dettype == UNDEFINED_DETECTOR) _dettype = find_dettype(detname);
  }

  Detector::~Detector() {
    MSG(DEBUG, "In d-tor Detector for " << detname())
  }

} // namespace detector

//-------------------
