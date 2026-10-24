/**
 * @file
 * @brief Declares a file-based data source simulator: EventSimulator, EventIteratorSimulator and DataSourceSimulator.
 */
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

/** One simulated event: an ArrayIO<float> that loads a float array from a text file; nda() returns it. */
class EventSimulator : public ArrayIO<float> {
public:
  /** Load fname with ArrayIO<float>; if the file cannot be opened the status is UNREADABLE and the array stays empty. */
  EventSimulator(const std::string& fname="") : ArrayIO<float>(fname) {}
  /** Destructor; does nothing. */
  virtual ~EventSimulator() {}

  /** Return the loaded array (ArrayIO::ndarray()). */
  NDArray<float>& nda(){return ndarray();}

private:
  EventSimulator(const EventSimulator&);
  EventSimulator& operator = (const EventSimulator&);
};

//-----------------------------

/** DirFileIterator whose next() wraps the next matching file in an EventSimulator. */
class EventIteratorSimulator : public DirFileIterator {
public:
  /** Iterate over the files of dirname whose names contain pattern (all files if pattern is null). */
  EventIteratorSimulator(const char* dirname, const char* pattern) : DirFileIterator(dirname, pattern), _pevent(0) {}
  /** Delete the current EventSimulator, if any. */
  virtual ~EventIteratorSimulator() {if(_pevent) delete _pevent;}

  /** Delete the previous EventSimulator and return a new one for the next matching file name. When no file is left the name is empty, so the returned event's file cannot be opened. */
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
/** File-based data source for tests: detector() gets a Detector from getDetector() and events() iterates over array files in a directory. */
class DataSourceSimulator {
public:

  /** Create the event iterator over dirname and pattern. The default dirname is a hard-coded /reg/neh/home/dubrovin path. */
  DataSourceSimulator(const char* dirname="/reg/neh/home/dubrovin/LCLS/con-detector/work/",
                      const char* pattern=0) // pattern="nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1"
    : _event_iterator(dirname, pattern) {}

  /** Destructor; does nothing. */
  virtual ~DataSourceSimulator() {}

  /** Return getDetector(detname), which gives NULL for names that are not area detectors. */
  Detector* detector(const std::string& detname) {return getDetector(detname);};
  /** Return the event iterator. */
  EventIteratorSimulator& events() {return _event_iterator;};

  /** Copy construction is disabled. */
  DataSourceSimulator(const DataSourceSimulator&) = delete;
  /** Copy assignment is disabled. */
  DataSourceSimulator& operator=(DataSourceSimulator const&) = delete;
  //DataSourceSimulator() {}

private:
  //NDArray<raw_t> _raw_nda;
  EventIteratorSimulator _event_iterator;

}; // class

} // namespace detector

#endif // PSALG_DATASOURCESIMULATOR_H
//-----------------------------
