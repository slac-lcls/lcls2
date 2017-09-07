#ifndef XtcData_XtcFileIterator_hh
#define XtcData_XtcFileIterator_hh

#include "xtcdata/xtc/Dgram.hh"

#include <stdio.h>

namespace XtcData
{

class XtcFileIterator
{
    public:
    XtcFileIterator(int fd, size_t maxDgramSize);
    ~XtcFileIterator();
    Dgram* next();

    private:
    int _fd;
    size_t _maxDgramSize;
    char* _buf;
};
}

#endif
