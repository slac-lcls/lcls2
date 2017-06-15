#ifndef Pds_XtcFileIterator_hh
#define Pds_XtcFileIterator_hh

#include "pdsdata/xtc/Dgram.hh"

#include <stdio.h>

namespace Pds {

class XtcFileIterator {
public:
  XtcFileIterator(int fd, size_t maxDgramSize);
  ~XtcFileIterator();
  Dgram* next();
private:
  int      _fd;
  size_t   _maxDgramSize;
  char*    _buf;
};

}

#endif
