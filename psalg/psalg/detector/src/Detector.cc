
#include "psalg/detector/Detector.hh"
//#include "psalg/detector/DetectorTypes.hh" // find_dettype

//using namespace std;
//using namespace psalg;
//using namespace detector;

#include <iostream> //ostream, cout

namespace detector {

//-----------------------------
/*
void AreaDetector::_default_msg(const std::string& msg) {
  MSG(WARNING, "DEFAULT METHOD "<< msg << " SHOULD BE RE-IMPLEMENTED IN THE DERIVED CLASS.");
}
*/

  Detector::Detector(const std::string& detname, const DETTYPE& dettype) : 
    _detname(detname), _dettype(dettype){
    if(dettype == NONDEFINED_DETECTOR) _dettype = find_dettype(detname);
  }

} // namespace detector

//-----------------------------
