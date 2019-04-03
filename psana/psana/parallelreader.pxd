from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from posix.unistd cimport read, sleep
from cpython cimport array
import array

cdef struct Buffer:
    char* chunk
    size_t got
    size_t offset
    size_t prev_offset
    unsigned nevents
    unsigned long timestamp
    size_t block_offset

cdef class ParallelReader:
    cdef int[:] fds
    cdef size_t chunksize
    cdef int maxretries
    cdef int sleep_secs
    cdef int nfiles
    cdef Buffer *bufs

    cdef inline void _init_buffers(self)
    cdef inline void reset_buffers(self)
    cdef inline size_t _read_with_retries(self, int buf_id, size_t displacement, size_t count) nogil
    cdef inline void read(self)
    cdef inline void read_partial(self)
