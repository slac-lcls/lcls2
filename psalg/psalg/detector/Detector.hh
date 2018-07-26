#ifndef PSALG_DETECTOR_H
#define PSALG_DETECTOR_H
//-----------------------------

#include "psalg/utils/Logger.hh" // for MSG, MSGSTREAM

#include <string>
#include "psalg/detector/DetectorTypes.hh"
#include <iostream> //ostream

//using namespace std;
//using namespace psalg;

namespace detector {

//-----------------------------
class Detector {
public:

  Detector(const std::string& detname="NoDevice", const DETTYPE& dettype=NONDEFINED_DETECTOR);
  virtual ~Detector() {}

  const std::string& detname() {return _detname;};
  const DETTYPE&     dettype() {return _dettype;};

  friend std::ostream& operator << (std::ostream& os, Detector& o) {
    os << "Detector name=" << o.detname() << " type=" << o.dettype(); return os;
  }

private:
    std::string _detname;
    DETTYPE     _dettype;
}; // class

} // namespace detector

#endif // PSALG_DETECTOR_H
//-----------------------------
