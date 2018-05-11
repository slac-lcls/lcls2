from libc.stdio cimport *
from posix.fcntl cimport *
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
from posix.unistd cimport pread

cdef struct Xtc:
    int junks[4]
    unsigned extent

cdef struct Sequence:
    int junks[2]
    unsigned low
    unsigned high

cdef struct Dgram:
    Sequence seq
    int junks[4]
    Xtc xtc

cdef class DgramChunk:
    cdef char* buf
    cdef int fd
    cdef ssize_t chunksize

    def __init__(self, int fd):
        self.fd = fd
        self.chunksize = 0x1000000
        self.buf = <char*> malloc(self.chunksize)

    def __dealloc__(self):
        free(self.buf)

    def get(self, ssize_t displacement, unsigned n_events):
        cdef ssize_t got = pread(self.fd, self.buf, self.chunksize, displacement)
        cdef ssize_t offset = 0
        cdef Dgram* d
        cdef ssize_t payload = 0
        
        if got == 0:
            return 0
        
        cdef unsigned long ts_value = 0
        cdef unsigned ts_seconds = 0
        cdef unsigned ts_nanoseconds = 0

        for i in range(n_events):
            if offset >= got: 
                break
            d = <Dgram *>(self.buf + offset)
            ts_value = <unsigned long>d.seq.high << 32 | d.seq.low
            ts_seconds = d.seq.high
            ts_nanoseconds = d.seq.low
            payload = d.xtc.extent - sizeof(Xtc)
            offset += sizeof(Dgram) + payload
        
        cdef char [:] view = <char [:offset]> self.buf
        return view




