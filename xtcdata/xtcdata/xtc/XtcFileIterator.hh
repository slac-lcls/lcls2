/**
 * @file
 * @brief Declares XtcData::XtcFileIterator, which reads datagrams one by one from a file descriptor.
 */
#ifndef XtcData_XtcFileIterator_hh
#define XtcData_XtcFileIterator_hh

#include "xtcdata/xtc/Dgram.hh"

#include <stdio.h>

namespace XtcData
{

/** Reads datagrams sequentially from a file descriptor into a single internal buffer of maxDgramSize bytes. */
class XtcFileIterator
{
public:
    /** Store fd and allocate the internal buffer of maxDgramSize bytes. The file descriptor is not opened or closed by this class. */
    XtcFileIterator(int fd, size_t maxDgramSize);
    /** Free the internal buffer; fd is not closed. */
    ~XtcFileIterator();
    /**
     * Read the next Dgram header (sizeof(Dgram) bytes) and then its xtc payload into the internal buffer.
     * Only a header read returning 0 is treated as end of file; failed or short header reads are not checked.
     * @return Pointer to the Dgram in the internal buffer (overwritten by the next call), or 0 at end of file, if header plus payload exceed maxDgramSize, or if the payload read fails or ends early (these last cases print a message).
     */
    Dgram* next();
    /** Seek fd back to offset 0 with lseek(); the result is not checked. */
    void rewind();
    /** Return the buffer size maxDgramSize given to the constructor. */
    size_t size() const { return _maxDgramSize; }

private:
    int _fd;
    size_t _maxDgramSize;
    char* _buf;
};
}

#endif
