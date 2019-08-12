#ifndef PSALG_DETECTOR_H
#define PSALG_DETECTOR_H

//-------------------

#include <string>
#include <iostream> //ostream, cout
#include "psalg/utils/Logger.hh" // for MSG, MSGSTREAM
#include "psalg/detector/DetectorTypes.hh"

//using namespace std;

//-------------------

namespace detector {

class Detector {
public:

  Detector(const std::string& detname="NoDevice", const DETTYPE& dettype=UNDEFINED_DETECTOR);
  virtual ~Detector();

  const std::string& detname() {return _detname;}
  const DETTYPE&     dettype() {return _dettype;}

  friend std::ostream& operator << (std::ostream& os, Detector& o) {
    os << "Detector name=" << o.detname() << " type=" << o.dettype() << " typename=" << dettypename(o.dettype()); return os;
  }

  Detector(const Detector&) = delete;
  Detector& operator = (const Detector&) = delete;

private:
    std::string _detname;
    DETTYPE     _dettype;
}; // class

} // namespace detector

//-------------------

#endif // PSALG_DETECTOR_H
