#ifndef PSALG_DATASOURCESIMULATOR_H
#define PSALG_DATASOURCESIMULATOR_H
//-----------------------------

#include <string>
//#include "psalg/calib/NDArray.hh" // NDArray
//#include "psalg/detector/Types.hh"

#include "psalg/calib/ArrayIO.hh" // ArrayIO

#include "psalg/utils/DirFileIterator.hh"
//#include "psalg/detector/AreaDetectorStore.hh"
#include "psalg/detector/DetectorStore.hh"

//using namespace std;
//using namespace psalg;

namespace detector {

//typedef DirFileIterator EventIteratorSimulator;

//-----------------------------

class EventSimulator : public ArrayIO<float> {
public:
  EventSimulator(const std::string& fname="") : ArrayIO<float>(fname) {}
  virtual ~EventSimulator() {}

  NDArray<float>& nda(){return ndarray();}

private:
  EventSimulator(const EventSimulator&);
  EventSimulator& operator = (const EventSimulator&);
};

//-----------------------------

class EventIteratorSimulator : public DirFileIterator {
public:
  EventIteratorSimulator(const char* dirname, const char* pattern) : DirFileIterator(dirname, pattern), _pevent(0) {}
  virtual ~EventIteratorSimulator() {if(_pevent) delete _pevent;}

  inline EventSimulator& next() {
    if(_pevent) delete _pevent;
    const std::string& fname = DirFileIterator::next();
    _pevent = new EventSimulator(fname);
    return *_pevent;
  };

private:
  EventSimulator *_pevent;
};

//-----------------------------
class DataSourceSimulator {
public:

  DataSourceSimulator(const char* dirname="/reg/neh/home/dubrovin/LCLS/con-detector/work/",
                      const char* pattern=0) // pattern="nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1"
    : _event_iterator(dirname, pattern) {}

  virtual ~DataSourceSimulator() {}

  Detector* detector(const std::string& detname) {return getDetector(detname);};
  EventIteratorSimulator& events() {return _event_iterator;};

  DataSourceSimulator(const DataSourceSimulator&) = delete;
  DataSourceSimulator& operator=(DataSourceSimulator const&) = delete;
  //DataSourceSimulator() {}

private:
  //NDArray<raw_t> _raw_nda;
  EventIteratorSimulator _event_iterator;

}; // class

} // namespace detector

#endif // PSALG_DATASOURCESIMULATOR_H
//-----------------------------
