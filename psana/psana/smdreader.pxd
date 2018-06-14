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

cdef struct Buffer:
    char* chunk
    size_t got
    size_t offset
    unsigned nevents
    unsigned long timestamp
    size_t block_offset
    size_t block_size

cdef class SmdReader:
    cdef int* fds
    cdef size_t chunksize
    cdef int maxretries
    cdef Buffer *bufs
    cdef int nfiles
    cdef unsigned got_events
