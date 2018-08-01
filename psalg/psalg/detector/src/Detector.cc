//-------------------

#include "psalg/detector/Detector.hh"

//-------------------

namespace detector {

  Detector::Detector(const std::string& detname, const DETTYPE& dettype) : 
    _detname(detname), _dettype(dettype){
    if(dettype == UNDEFINED_DETECTOR) _dettype = find_dettype(detname);
  }

} // namespace detector

//-------------------
