## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1
from parallelreader cimport Buffer

import os

from cython.parallel import prange

DEF DEBUG_MODULE = "ParallelReader"
include "cydebug.pxh"

cimport cython
from dgramlite cimport Dgram, Xtc

from psana.psexp import TransitionId


cdef class ParallelReader:

    def __cinit__(self, int[:] file_descriptors, size_t chunksize, *args, **kwargs):
        # Keyword args that need to be passed in once. To save some of
        # them as cpp class attributes, we need to read them in as PyObject*.
        cdef char* kwlist[2]
        kwlist[0] = "dsparms"
        kwlist[1] = NULL
        if PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist,
           &(self.dsparms)) is False:
            raise RuntimeError, "Invalid kwargs for SmdReader"

        self.file_descriptors   = file_descriptors
        self.chunksize          = chunksize
        self.nfiles             = self.file_descriptors.shape[0]
        self.Configure          = TransitionId.Configure
        self.BeginRun           = TransitionId.BeginRun
        self.L1Accept           = TransitionId.L1Accept
        self.L1Accept_EndOfBatch= TransitionId.L1Accept_EndOfBatch
        self.EndRun             = TransitionId.EndRun
        self.bufs               = <Buffer *>malloc(sizeof(Buffer) * self.nfiles)
        self.step_bufs          = <Buffer *>malloc(sizeof(Buffer) * self.nfiles)
        self.got                = 0
        self.chunk_overflown    = 0                         # set to dgram size if it's too big
        self.max_events         = int(self.chunksize / 70)  # guess no. of smd events in one chunk
        self.num_threads        = int(os.environ.get('PS_SMD0_NUM_THREADS', '16'))
        self.gots               = array.array('l', [0]*self.nfiles)
        self._init_buffers(self.bufs)
        self._init_buffers(self.step_bufs)

    def __dealloc__(self):
        self._free_buffers(self.bufs)
        self._free_buffers(self.step_bufs)

    cdef void _init_buffers(self, Buffer* bufs):
        cdef Py_ssize_t i
        cdef Buffer* buf
        for i in range(self.nfiles):
            buf                 = &(bufs[i])
            buf.got             = 0
            buf.ready_offset    = 0     # offset of the last event in the buffer
            buf.n_ready_events  = 0     # no. of total events in the buffer
            buf.seen_offset     = 0     # offset of the event seen (yielded) so far
            buf.n_seen_events   = 0     # no. of seen events
            buf.timestamp       = 0
            buf.found_endrun    = 0
            buf.endrun_ts       = 0
            buf.chunk      = <char *>malloc(self.chunksize)
            buf.ts_arr     = <uint64_t *>malloc(sizeof(uint64_t) * self.max_events)
            buf.sv_arr     = <unsigned *>malloc(sizeof(unsigned) * self.max_events)
            buf.st_offset_arr = <uint64_t *>malloc(sizeof(uint64_t) * self.max_events)
            buf.en_offset_arr = <uint64_t *>malloc(sizeof(uint64_t) * self.max_events)
            buf.err_code = 0

    cdef void _free_buffers(self, Buffer* bufs):
        cdef Py_ssize_t i
        cdef Buffer* buf
        if bufs:
            for i in range(self.nfiles):
                buf = &(bufs[i])
                free(buf.chunk)
                free(buf.ts_arr)
                free(buf.sv_arr)
                free(buf.st_offset_arr)
                free(buf.en_offset_arr)
            free(bufs)

    @cython.boundscheck(False)
    cdef void force_read(self):
        """
        Forcefully refill each buffer by copying any remaining data and reading from file descriptors.

        - Resets per-buffer tracking variables (ready_offset, seen_offset, n_ready_events, etc.)
        - Does NOT check if buffers still contain unseen events; it always overwrites.
        - Called by SmdReader.force_read() when a full reread is needed.
        """
        cdef int i           = 0
        cdef int64_t[:] gots = self.gots
        cdef Dgram* d
        cdef Buffer* buf
        cdef Buffer* step_buf
        cdef uint64_t payload = 0
        self.got = 0

        for i in prange(self.nfiles, nogil=True, num_threads=self.num_threads):
            gots[i] = 0
            buf = &(self.bufs[i])
            step_buf = &(self.step_bufs[i])

            # copy the unseen portion to the beginning of the buffer
            # [0            |seen_offset     |ready_offset |got         ]buf_size
            # copy this     { -----------------------------}
            buf.cp_offset = buf.seen_offset
            if buf.got - buf.cp_offset > 0:
                memcpy(buf.chunk,
                       buf.chunk + buf.cp_offset,
                       buf.got - buf.cp_offset)

            # read more data to fill up the buffer
            gots[i] = read(self.file_descriptors[i], buf.chunk + (buf.got - buf.cp_offset),
                           self.chunksize - (buf.got - buf.cp_offset))
            # summing the size of all the new reads
            self.got += gots[i]

            ## FOR DEBUGGING - Calling print with gil can slow down performance.
            #with gil:
            #    debug_print(f"Stream {i}: seen_offset:{buf.seen_offset} copied:{buf.got-buf.cp_offset} "
            #            f"read:{gots[i]} "
            #            f"total:{(buf.got - buf.cp_offset) + gots[i]}")

            buf.got = (buf.got - buf.cp_offset) + gots[i]

            # reset the offsets and no. of events
            buf.ready_offset        = 0
            buf.n_ready_events      = 0
            buf.seen_offset         = 0
            buf.n_seen_events       = 0
            step_buf.ready_offset   = 0
            step_buf.n_ready_events = 0
            step_buf.seen_offset    = 0
            step_buf.n_seen_events  = 0
            buf.err_code            = 0

            # Walk through and no. of events, offsets, timestamps, and services
            # for both L1 and transition buffers.
            while buf.ready_offset < buf.got and \
                  buf.n_ready_events < self.max_events and \
                  not buf.err_code:
                if buf.got - buf.ready_offset >= sizeof(Dgram):
                    d = <Dgram *>(buf.chunk + buf.ready_offset)

                    if d.xtc.extent == 0:
                        buf.err_code = 3
                        continue

                    payload = d.xtc.extent - sizeof(Xtc)

                    # check if this dgram is too big to fit in the chunk
                    if sizeof(Dgram) + payload > self.chunksize:
                        self.chunk_overflown = sizeof(Dgram) + payload

                    if (buf.got - buf.ready_offset) >= sizeof(Dgram) + payload:
                        buf.ts_arr[buf.n_ready_events] = <uint64_t>d.seq.high << 32 | d.seq.low
                        buf.sv_arr[buf.n_ready_events] = (d.env>>24)&0xf
                        buf.st_offset_arr[buf.n_ready_events] = buf.ready_offset
                        buf.en_offset_arr[buf.n_ready_events] = buf.ready_offset + sizeof(Dgram) + payload

                        # check if this a non L1
                        if buf.sv_arr[buf.n_ready_events] != self.L1Accept and \
                                buf.sv_arr[buf.n_ready_events] != self.L1Accept_EndOfBatch:
                            memcpy(step_buf.chunk + step_buf.ready_offset, d, sizeof(Dgram) + payload)
                            step_buf.ts_arr[step_buf.n_ready_events] = buf.ts_arr[buf.n_ready_events]
                            step_buf.st_offset_arr[step_buf.n_ready_events] = step_buf.ready_offset
                            step_buf.en_offset_arr[step_buf.n_ready_events] = step_buf.ready_offset + sizeof(Dgram) + payload
                            step_buf.sv_arr[step_buf.n_ready_events] = buf.sv_arr[buf.n_ready_events]
                            step_buf.n_ready_events += 1
                            step_buf.ready_offset += sizeof(Dgram) + payload
                            step_buf.timestamp = buf.ts_arr[buf.n_ready_events]

                            if buf.sv_arr[buf.n_ready_events] == self.EndRun:
                                buf.endrun_ts = buf.ts_arr[buf.n_ready_events]

                        buf.timestamp = buf.ts_arr[buf.n_ready_events]
                        buf.ready_offset += sizeof(Dgram) + payload
                        buf.n_ready_events += 1

                    else:  # if (buf.got - buf.ready_offset) >= sizeof(Dgram) + payload
                        buf.err_code = 2
                        continue
                else:  # if buf.got - buf.ready_offset >= sizeof(Dgram)
                    buf.err_code = 1
                    continue

            # end while buf.ready_offset < buf.got:

        # end for i

        # Check for fatal error
        cdef int fatal_error_detected = 0

        for i in range(self.nfiles):
            buf = &(self.bufs[i])

            if buf.err_code == 0:
                continue

            if buf.err_code == 3:
                fatal_error_detected = 1
                debug_print(f"Stream {i}: Found invalid dgram with xtc.extent==0 (fatal)")
            else:
                debug_print(f"Stream {i}: Buffer ended prematurely (err_code={buf.err_code})")

        if fatal_error_detected:
            raise RuntimeError("Data corruption detected during force_read(): xtc.extent == 0 in one or more streams.")
