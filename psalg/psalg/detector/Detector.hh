/**
 * @file
 * @brief Declares detector::Detector, the base class holding a detector name and type.
 */
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

/** Base detector class holding a name and a DETTYPE. */
class Detector {
public:

  /** Store detname and dettype; if dettype is UNDEFINED_DETECTOR it is looked up from detname with find_dettype(). */
  Detector(const std::string& detname="NoDevice", const DETTYPE& dettype=UNDEFINED_DETECTOR);
  /** Destructor; only logs a debug message. */
  virtual ~Detector();

  /** Return the detector name. */
  const std::string& detname() {return _detname;}
  /** Return the detector type. */
  const DETTYPE&     dettype() {return _dettype;}

  /** Write "Detector name=", the name, " type=", the numeric type and " typename=" with dettypename() of o to os, and return os. */
  friend std::ostream& operator << (std::ostream& os, Detector& o) {
    os << "Detector name=" << o.detname() 
       << " type="         << o.dettype() 
       << " typename="     << dettypename(o.dettype());
    return os;
  }

  /** Copy construction is disabled. */
  Detector(const Detector&) = delete;
  /** Copy assignment is disabled. */
  Detector& operator = (const Detector&) = delete;

private:
  std::string _detname;
  DETTYPE     _dettype;
}; // class

} // namespace detector

//-------------------

#endif // PSALG_DETECTOR_H
