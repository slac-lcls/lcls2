## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from posix.unistd cimport read, sleep
from libc.errno cimport errno
from cpython cimport array
import array
from libc.stdint cimport uint32_t, uint64_t, int64_t

cdef struct Buffer:
    char*    chunk
    uint64_t got
    uint64_t ready_offset 
    uint64_t n_ready_events             
    uint64_t seen_offset 
    uint64_t n_seen_events             
    uint64_t timestamp                  # ts of the last dgram
    uint64_t* ts_arr                    # dgram timestamp
    unsigned* sv_arr                    # dgram service
    uint64_t* st_offset_arr             # start offset
    uint64_t* en_offset_arr             # end offset (start offset + size)
    int      found_endrun
    uint64_t endrun_ts

cdef class ParallelReader:
    cdef int[:]     file_descriptors
    cdef size_t     chunksize
    cdef Py_ssize_t nfiles
    cdef int        coarse_freq
    cdef Buffer     *bufs
    cdef Buffer     *step_bufs
    cdef unsigned   Configure
    cdef unsigned   BeginRun
    cdef unsigned   L1Accept
    cdef unsigned   EndRun
    cdef uint64_t   got                  # summing the size of new reads used by prometheus
    cdef uint64_t   chunk_overflown
    cdef int        num_threads
    cdef int        max_events

    cdef void _init_buffers(self, Buffer* bufs)
    cdef void _free_buffers(self, Buffer* bufs)
    cdef void just_read(self)
