## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from parallelreader cimport Buffer
from cython.parallel import prange
import os
from dgramlite cimport Xtc, Sequence, Dgram
cimport cython

cdef class ParallelReader:
    
    def __cinit__(self, int[:] file_descriptors, size_t chunksize):
        self.file_descriptors   = file_descriptors
        self.chunksize          = chunksize
        self.nfiles             = self.file_descriptors.shape[0]
        self.L1Accept           = 12
        self.BeginRun           = 4
        self.bufs               = <Buffer *>malloc(sizeof(Buffer) * self.nfiles)
        self.step_bufs          = <Buffer *>malloc(sizeof(Buffer)*self.nfiles)
        self.beginrun_buf       = <char *>malloc(self.chunksize)
        self.beginrun_offset    = 0
        self.n_beginruns        = 0
        self.got                = 0
        self.chunk_overflown    = 0     # set to dgram size if it's too big
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
        free(self.beginrun_buf)

    cdef void _init_buffers(self):
        cdef Py_ssize_t i
        self._reset_buffers(self.bufs)
        self._reset_buffers(self.step_bufs)
        for i in range(self.nfiles):
            self.bufs[i].chunk      = <char *>malloc(self.chunksize)
            self.step_bufs[i].chunk = <char *>malloc(self.chunksize)
    
    cdef void _reset_buffers(self, Buffer* bufs):
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
    
    @cython.boundscheck(False)
    cdef void just_read(self):
        """
        Reads only if the buffer has no more unseen events

        If there's some data left at the bottom of the buffer due to cutoff,
        copy this remaining data to the begining of the buffer then read to 
        fill the rest of the chunk. Sets the following variables when done:
        - got = remaining (from copying) + new got (from reading)
        - ready_offset = offset of the last event that fits in the buffer
        - n_ready_events = no. of total events that fit in the buffer

        """
        cdef Py_ssize_t i       = 0
        cdef uint64_t got       = 0
        cdef uint64_t offset    = 0
        cdef Dgram* d
        cdef Buffer* buf
        cdef Buffer* step_buf
        cdef uint64_t payload   = 0
        cdef unsigned service   = 0
        self.got                = 0
        self.beginrun_offset    = 0
        self.n_beginruns        = 0
        
        for i in prange(self.nfiles, nogil=True):
            buf = &(self.bufs[i])
            step_buf = &(self.step_bufs[i])

            # skip reading this buffer if there is/are still some event(s).
            if buf.n_ready_events - buf.n_seen_events > 0: continue 
            
            # copy remaining data if any 
            if buf.got - buf.ready_offset > 0 and buf.ready_offset > 0:
                memcpy(buf.chunk, buf.chunk + buf.ready_offset, buf.got - buf.ready_offset)
            
            # read more data to fill up the buffer
            got = read( self.file_descriptors[i], buf.chunk + (buf.got - buf.ready_offset), \
                    self.chunksize - (buf.got - buf.ready_offset) )
            
            # summing the size of all the new reads
            self.got += got
            
            buf.got = (buf.got - buf.ready_offset) + got
            
            # reset the offsets and no. of events
            buf.ready_offset        = 0
            buf.n_ready_events      = 0
            buf.seen_offset         = 0
            buf.n_seen_events       = 0
            step_buf.ready_offset   = 0
            step_buf.n_ready_events = 0
            step_buf.seen_offset    = 0
            step_buf.n_seen_events  = 0
            
            while buf.ready_offset < buf.got:
                if buf.got - buf.ready_offset >= sizeof(Dgram):
                    d = <Dgram *>(buf.chunk + buf.ready_offset)
                    payload = d.xtc.extent - sizeof(Xtc)

                    # check if this dgram is too big to fit in the chunk
                    if sizeof(Dgram) + payload > self.chunksize:
                        self.chunk_overflown = sizeof(Dgram) + payload

                    if (buf.got - buf.ready_offset) >= sizeof(Dgram) + payload:
                        buf.ts_arr[buf.n_ready_events] = <uint64_t>d.seq.high << 32 | d.seq.low
                        buf.next_offset_arr[buf.n_ready_events] = buf.ready_offset + sizeof(Dgram) + payload

                        # check if this a non L1
                        service = (d.env>>24)&0xf
                        if service != self.L1Accept:
                            memcpy(step_buf.chunk + step_buf.ready_offset, d, sizeof(Dgram) + payload)
                            step_buf.ts_arr[step_buf.n_ready_events] = buf.ts_arr[buf.n_ready_events]
                            step_buf.next_offset_arr[step_buf.n_ready_events] = step_buf.ready_offset + sizeof(Dgram) + payload
                            step_buf.n_ready_events += 1
                            step_buf.ready_offset += sizeof(Dgram) + payload
                            step_buf.timestamp = buf.ts_arr[buf.n_ready_events]

                            if service == self.BeginRun and i == 0:
                                memcpy(self.beginrun_buf + self.beginrun_offset, d, sizeof(Dgram) + payload)
                                self.beginrun_offset += sizeof(Dgram) + payload
                                self.n_beginruns += 1

                        
                        buf.timestamp = buf.ts_arr[buf.n_ready_events] 
                        buf.ready_offset += sizeof(Dgram) + payload
                        buf.n_ready_events += 1
                        
                    else:
                        break
                else:
                    break


