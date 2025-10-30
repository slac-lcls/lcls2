## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from posix.time cimport *
from posix.unistd cimport SEEK_CUR, lseek, read, sleep

from cpython cimport array
from libc.errno cimport errno
from libc.stdint cimport int64_t, uint32_t, uint64_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from libc.stddef cimport size_t

import array

from cpython.getargs cimport PyArg_ParseTupleAndKeywords
from cpython.object cimport PyObject


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
    uint64_t cp_offset                  # when force reread is set, cp_offset is the seen_offset
                                        # otherwise it's ready_offset (not using local var due to nogil)
    int err_code                        # for debugging

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
    cdef unsigned   L1Accept_EndOfBatch
    cdef unsigned   EndRun
    cdef uint64_t   got                  # summing the size of new reads used by prometheus
    cdef uint64_t   chunk_overflown
    cdef int        num_threads
    cdef uint64_t   max_events
    cdef array.array gots
    cdef int        max_retries
    cdef PyObject*  dsparms

    cdef void _init_buffers(self, Buffer* bufs)
    cdef void _free_buffers(self, Buffer* bufs)
    cdef void force_read(self)
