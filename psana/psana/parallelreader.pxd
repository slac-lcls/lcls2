## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from posix.unistd cimport read, sleep
from cpython cimport array
import array
from libc.stdint cimport uint32_t, uint64_t

cdef struct Buffer:
    char*    chunk
    uint64_t got
    uint64_t ready_offset 
    uint64_t n_ready_events             
    uint64_t seen_offset 
    uint64_t n_seen_events             
    uint64_t timestamp                   # ts of the last dgram
    uint64_t ts_arr[0x100000]            # dgram timestamp
    unsigned sv_arr[0x100000]            # dgram service
    uint64_t next_offset_arr[0x100000]   # their offset + size of dgram and payload
    int      found_endrun

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
    cdef uint64_t   got                  # summing the size of new reads used by prometheus
    cdef uint64_t   chunk_overflown

    cdef void _init_buffers(self)
    cdef void _reset_buffers(self, Buffer* bufs)
    cdef void just_read(self)
