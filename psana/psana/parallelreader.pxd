from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from posix.unistd cimport read, sleep
from cpython cimport array
import array
from libc.stdint cimport uint32_t, uint64_t

cdef struct Buffer:
    char* chunk
    uint64_t got
    uint64_t offset 
    int nevents             
    uint64_t timestamp                    # ts of the last dgram or of the dgram at max_events
    uint64_t ts_arr[0x1000000]            # dgram timestamps 
    uint64_t next_offset_arr[0x1000000]   # their offset + size of dgram and payload
    int needs_reread
    uint64_t lastget_offset

cdef class ParallelReader:
    cdef int[:] file_descriptors
    cdef size_t chunksize
    cdef int max_events
    cdef Py_ssize_t nfiles
    cdef int coarse_freq
    cdef Buffer *bufs
    cdef Buffer *step_bufs
    cdef unsigned L1Accept

    cdef void _init_buffers(self)
    cdef void _reset_buffers(self)
    cdef void _rewind_buffer(self, Buffer* buf, uint64_t max_ts)
    cdef void just_read(self)
    cdef void rewind(self, uint64_t max_ts, int winner)
