
#include "pdsdata/xtc/XtcFileIterator.hh"
#include <new>
#include <stdio.h>
#include <unistd.h>

using namespace Pds;

XtcFileIterator::XtcFileIterator(int fd, size_t maxDgramSize) :
  _fd(fd),_maxDgramSize(maxDgramSize),_buf(new char[maxDgramSize]) {}

XtcFileIterator::~XtcFileIterator() {
  delete[] _buf;
}

Dgram* XtcFileIterator::next() {
  Dgram& dg = *(Dgram*)_buf;
  if (::read(_fd, &dg, sizeof(dg))==0) return 0;
  size_t payloadSize = dg.xtc.sizeofPayload();
  if ((payloadSize+sizeof(dg))>_maxDgramSize) {
    printf("Datagram size %zu larger than maximum: %zu\n", payloadSize+sizeof(dg), _maxDgramSize);
    return 0;
  }
  ssize_t sz = ::read(_fd, dg.xtc.payload(), payloadSize);
  if (sz != (ssize_t)payloadSize) {
    printf("XtcFileIterator::next read incomplete payload %d/%d\n",
     (int) sz, (int) payloadSize);
  }

  return sz!=(ssize_t)payloadSize ? 0: &dg;
}
