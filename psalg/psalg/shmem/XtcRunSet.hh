/**
 * @file
 * @brief Declares XtcRunSet, which replays the datagrams of a list of xtc files into a shared-memory monitor server.
 */
#ifndef PsAlg_ShMem_XtcRunSet_hh
#define PsAlg_ShMem_XtcRunSet_hh

#include "xtcdata/xtc/Dgram.hh"

#include "xtcdata/xtc/XtcFileIterator.hh"

using XtcData::Dgram;
using XtcData::XtcFileIterator;

#include <string>
#include <list>

/** Replays the datagrams of a list of xtc files, file after file, into a shared-memory monitor server (an XtcMonitorServer subclass defined in XtcRunSet.cc). */
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
  bool _interactive;

public:
  /** Start with no paths, no server and no open file. */
  XtcRunSet();
  /** If connect() created a server, call its XtcMonitorServer::unlink(), which sets its terminate flag and closes and unlinks its message queues (the shared memory is not unlinked); the server object is not deleted. */
  ~XtcRunSet();
  /** Append path to the file list (not sorted). */
  void addSinglePath(std::string path);
  /**
   * Add every entry of dirPath ("." if empty) whose name contains ".xtc" and whose path contains matchString (any path if empty); the new paths are sorted and appended.
   * Prints the arguments and each added path; if the directory cannot be opened it calls perror() and returns.
   */
  void addPathsFromDir(std::string dirPath, std::string matchString = "");
  /** Split runPrefix at its last '/' into a directory and a name prefix and call addPathsFromDir(directory, prefix); without a '/' the current directory is searched. */
  void addPathsFromRunPrefix(std::string runPrefix);
  /** Read whitespace-separated file names from listFile and append them, sorted. Prints the file name; if it cannot be opened it calls perror() and returns. */
  void addPathsFromListFile(std::string listFile);
  /**
   * On the first call only: store the flags, set the period to 1e9/rate ns (0, unthrottled, if rate <= 0), and create the monitor server for partitionTag with the given buffer sizes and client count.
   * Prints the rate and the time taken to open shared memory. The server runs in distribute(true) mode with 4 spare buffers; later calls do nothing.
   */
  void connect(char* partitionTag, unsigned sizeOfBuffers, int numberOfBuffers, unsigned nclients, int rate,
               bool verbose = false, bool veryverbose = false, bool interactive = false);
  /**
   * Pass every datagram of the listed files, in order, to the server's events(); connect() must have been called first.
   * When verbose, non-L1Accept transitions are printed and L1Accepts show an average rate; a nonzero period makes it sleep (or, if interactive, wait for a carriage return) to hold the rate.
   * Returns when the files are used up or one fails to open; files are read with an XtcFileIterator of maximum datagram size 0x4000000.
   */
  void run();
  /** Call wait() on the server created by connect(). */
  void wait();
  /** Call the server's XtcMonitorServer::unlink() (sets its terminate flag and closes and unlinks its message queues; the shared memory is not unlinked) and print "Unlinked/exited server". */
  void exit();
};

#endif
