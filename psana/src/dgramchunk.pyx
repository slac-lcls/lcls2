from libc.stdio cimport *
from posix.fcntl cimport *
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
from posix.unistd cimport pread

cdef struct Dgram:
    int junks[12]
    int extent

cdef struct Xtc:
    int junks[5]

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

        for i in range(n_events):
            if offset >= got: 
                break
            d = <Dgram *>(self.buf + offset)
            payload = d.extent - sizeof(Xtc)
            offset += sizeof(Dgram) + payload
        
        cdef char [:] view = <char [:offset]> self.buf
        return view




