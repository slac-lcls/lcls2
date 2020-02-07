## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

import numpy as np
from parallelreader cimport Buffer
from cython.parallel import prange
import os
from dgramlite cimport Xtc, Sequence, Dgram

cdef class ParallelReader:
    
    def __cinit__(self, int[:] file_descriptors, size_t chunksize, int max_events):
        self.file_descriptors = file_descriptors
        self.chunksize = chunksize
        self.max_events = max_events
        self.nfiles = self.file_descriptors.shape[0]
        self.L1Accept = 12
        self.bufs = <Buffer *>malloc(sizeof(Buffer) * self.nfiles)
        self.step_bufs = <Buffer *>malloc(sizeof(Buffer)*self.nfiles)
        self._init_buffers()


    def __dealloc__(self):
        if self.bufs:
            for i in range(self.nfiles):
                free(self.bufs[i].chunk)
            free(self.bufs)

        if self.step_bufs:
            for i in range(self.nfiles):
                free(self.step_bufs[i].chunk)
            free(self.step_bufs)


    cdef void _init_buffers(self):
        cdef Py_ssize_t i
        self._reset_buffers(self.bufs)
        self._reset_buffers(self.step_bufs)
        for i in prange(self.nfiles, nogil=True):
            self.bufs[i].chunk = <char *>malloc(self.chunksize)
            self.step_bufs[i].chunk = <char *>malloc(self.chunksize)
            self.bufs[i].got = read(self.file_descriptors[i], self.bufs[i].chunk, self.chunksize)

    
    cdef void _reset_buffers(self, Buffer* bufs):
        cdef Py_ssize_t i
        cdef Buffer* buf
        for i in range(self.nfiles):
            buf = &(bufs[i])
            buf.got = 0
            buf.offset = 0
            buf.nevents = 0
            buf.timestamp = 0
            buf.needs_reread = 0
            buf.lastget_offset = 0

    cdef void just_read(self):
        cdef Py_ssize_t i = 0
        cdef uint64_t remaining = 0
        cdef uint64_t got = 0
        cdef uint64_t offset = 0
        cdef Dgram* d
        cdef Buffer* buf
        cdef Buffer* step_buf
        cdef uint64_t payload = 0
        cdef unsigned service = 0
        
        self._reset_buffers(self.step_bufs) # step buffers always get reset when read

        #for i in prange(self.nfiles, nogil=True):
        for i in range(self.nfiles):
            buf = &(self.bufs[i])
            step_buf = &(self.step_bufs[i])
            
            buf.lastget_offset = buf.offset

            # Copy remaining to the beginning of the chunk (if needed)
            # then fill up the rest of the chunk
            if buf.needs_reread == 1:
                remaining = buf.got - buf.offset
                memcpy(buf.chunk, buf.chunk + buf.offset, remaining)
                
                # MONA: TODO there's a chance that below read will be wrong,
                # if the next part of the dgram cannot be read out in one retry (1s).
                # The next read will replace the remaining - possible segfault
                # when try to create Dgram.
                got = read(self.file_descriptors[i], buf.chunk + remaining, \
                        self.chunksize - remaining)
                
                buf.got = remaining + got
                buf.needs_reread = 0
                buf.offset = 0
                buf.lastget_offset = 0

            buf.nevents = 0
            
            while buf.offset <= buf.got and buf.got > 0:
                remaining = buf.got - buf.offset
                if remaining >= sizeof(Dgram):
                    d = <Dgram *>(buf.chunk + buf.offset)
                    payload = d.xtc.extent - sizeof(Xtc)

                    if remaining >= sizeof(Dgram) + payload:
                        buf.ts_arr[buf.nevents] = <uint64_t>d.seq.high << 32 | d.seq.low
                        buf.next_offset_arr[buf.nevents] = buf.offset + sizeof(Dgram) + payload
                        
                        # check if this a non L1
                        service = (d.env>>24)&0xf
                        if service != self.L1Accept:
                            if buf.ts_arr[buf.nevents] > step_buf.timestamp:
                                memcpy(step_buf.chunk + step_buf.offset, d, sizeof(Dgram) + payload)
                                step_buf.ts_arr[step_buf.nevents] = buf.ts_arr[buf.nevents]
                                step_buf.next_offset_arr[step_buf.nevents] = step_buf.offset + sizeof(Dgram) + payload
                                step_buf.nevents += 1
                                step_buf.offset += sizeof(Dgram) + payload
                                step_buf.timestamp = buf.ts_arr[buf.nevents]
                        
                        buf.offset += sizeof(Dgram) + payload
                        buf.nevents += 1

                        if buf.nevents == self.max_events:
                            buf.timestamp = <uint64_t>d.seq.high << 32 | d.seq.low
                            break
                    else:
                        buf.needs_reread = 1
                        break
                else:
                    buf.needs_reread = 1
                    break
            
            if buf.nevents < self.max_events:
                if buf.nevents > 0:
                    buf.timestamp = <uint64_t>d.seq.high << 32 | d.seq.low

    cdef void _rewind_buffer(self, Buffer* buf, uint64_t max_ts):
        cdef Py_ssize_t found_pos
        cdef uint64_t[:] ts_view
        ts_view = buf.ts_arr

        found_pos = np.searchsorted(ts_view[:buf.nevents], max_ts, side='right')
        if found_pos == 0:
            if ts_view[found_pos] == max_ts:
                buf.offset = buf.next_offset_arr[found_pos]
                buf.timestamp = buf.ts_arr[found_pos]
                buf.nevents = 1
            else:
                buf.offset = buf.lastget_offset
                buf.timestamp = 0
                buf.nevents = 0
        else:
            if found_pos < buf.nevents:
                found_pos -= 1
                buf.offset = buf.next_offset_arr[found_pos]
                buf.timestamp = buf.ts_arr[found_pos]
                buf.nevents = found_pos + 1
            

    cdef void rewind(self, uint64_t max_ts, int winner):
        cdef Py_ssize_t i
        for i in range(self.nfiles):
            if i == winner: continue
            self._rewind_buffer(&(self.bufs[i]), max_ts)
            self._rewind_buffer(&(self.step_bufs[i]), max_ts)

            

