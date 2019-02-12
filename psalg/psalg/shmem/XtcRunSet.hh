#ifndef PsAlg_ShMem_XtcRunSet_hh
#define PsAlg_ShMem_XtcRunSet_hh

#include "xtcdata/xtc/Dgram.hh"

#include "xtcdata/xtc/XtcFileIterator.hh"

using XtcData::Dgram;
using XtcData::XtcFileIterator;

#include <string>
#include <list>

class XtcRunSet {
private:
  std::list<std::string> _paths;
//  XtcRun _run;
  bool _runIsValid;
  class MyMonitorServer* _server;
  XtcFileIterator* _iter;
  long long int _period;
  bool _verbose;
  bool _veryverbose;
  bool _skipToNextRun();
  bool _openFile(std::string fname);
  void _addPaths(std::list<std::string> newPaths);
  double timeDiff(struct timespec* end, struct timespec* start);
  Dgram* next();

public:
  XtcRunSet();
  ~XtcRunSet();
  void addSinglePath(std::string path);
  void addPathsFromDir(std::string dirPath, std::string matchString = "");
  void addPathsFromRunPrefix(std::string runPrefix);
  void addPathsFromListFile(std::string listFile);
  void connect(char* partitionTag, unsigned sizeOfBuffers, int numberOfBuffers, unsigned nclients, int rate, bool verbose = false, bool veryverbose = false);
  void run();
  void wait();
};

#endif
