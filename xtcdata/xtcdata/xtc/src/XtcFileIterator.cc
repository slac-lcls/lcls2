
#include "xtcdata/xtc/XtcFileIterator.hh"
#include <new>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

using namespace XtcData;

XtcFileIterator::XtcFileIterator(int fd, size_t maxDgramSize)
: _fd(fd), _maxDgramSize(maxDgramSize), _buf(new char[maxDgramSize])
{
}

XtcFileIterator::~XtcFileIterator()
{
    delete[] _buf;
}

Dgram* XtcFileIterator::next()
{
    Dgram& dg = *(Dgram*)_buf;
    if (::read(_fd, &dg, sizeof(dg)) == 0) return 0;
    size_t payloadSize = dg.xtc.sizeofPayload();
    if ((payloadSize + sizeof(dg)) > _maxDgramSize) {
        printf("Datagram size %zu larger than maximum: %zu\n", payloadSize + sizeof(dg), _maxDgramSize);
        return 0;
    }
    char* p = dg.xtc.payload();
    printf("Reading %zu into %p\n", payloadSize, p);
    if (payloadSize == 0)
        return &dg;

    ssize_t sz = ::read(_fd, p, payloadSize);
    while (1) {
        printf("  Read %zu into %p\n", sz, p);
        if (sz < 0) {
            printf("XtcFileIterator::next read error '%s' for payload of size %zu", strerror(errno), payloadSize);
            return 0;
        }
        else if (sz > 0) {
            p += sz;
            payloadSize -= sz;
            if (payloadSize ==0)
                break;
            sz = ::read(_fd, p, payloadSize);
        }
        else {
            printf("XtcFileIterator::next read incomplete payload %zd/%zd\n", p-dg.xtc.payload(), dg.xtc.sizeofPayload());
            return 0;
        }
    }

    return &dg;
}

void XtcFileIterator::rewind()
{
    lseek(_fd, 0, SEEK_SET);
}
