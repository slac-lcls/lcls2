## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from dgramlite cimport Xtc, Sequence, Dgram
from parallelreader cimport Buffer, ParallelReader
from libc.stdint cimport uint32_t, uint64_t
from cpython cimport array
import time, os
cimport cython


cdef class SmdReader:
    cdef ParallelReader prl_reader
    cdef int            winner, view_size
    cdef int            max_retries, sleep_secs
    cdef array.array    buf_offsets, stepbuf_offsets, buf_sizes, stepbuf_sizes
    cdef array.array    i_evts, founds

    def __init__(self, int[:] fds, int chunksize):
        assert fds.size > 0, "Empty file descriptor list (fds.size=0)."
        self.prl_reader         = ParallelReader(fds, chunksize)
        
        # max retries has no default value (set when creating datasource)
        self.max_retries        = int(os.environ['PS_SMD_MAX_RETRIES']) 
        self.sleep_secs         = int(os.environ.get('PS_SMD_SLEEP_SECS', '1'))
        self.buf_offsets        = array.array('Q', [0]*fds.size)
        self.stepbuf_offsets    = array.array('Q', [0]*fds.size)
        self.buf_sizes          = array.array('Q', [0]*fds.size)
        self.stepbuf_sizes      = array.array('Q', [0]*fds.size)
        self.i_evts             = array.array('Q', [0]*fds.size)
        self.founds             = array.array('Q', [0]*fds.size)

    def is_complete(self):
        """ Checks that all buffers have at least one event 
        """
        cdef int is_complete = 1
        cdef int i

        for i in range(self.prl_reader.nfiles):
            if self.prl_reader.bufs[i].n_ready_events - \
                    self.prl_reader.bufs[i].n_seen_events == 0:
                is_complete = 0

        return is_complete


    def get(self):
        self.prl_reader.just_read()
        
        if self.max_retries > 0:
            cn_retries = 0
            while not self.is_complete():
                time.sleep(self.sleep_secs)
                print('waiting for an event...')
                self.prl_reader.just_read()
                cn_retries += 1
                if cn_retries > self.max_retries:
                    break

    @cython.boundscheck(False)
    def view(self, int batch_size=1000):
        """ Returns memoryview of the data and step buffers.

        This function is called by SmdReaderManager only when is_complete is True (
        all buffers have at least one event). It returns events of batch_size if
        possible or as many as it has for the buffer.
        """

        # Find the winning buffer
        cdef int i, i_evt
        cdef uint64_t limit_ts=0
        
        for i in range(self.prl_reader.nfiles):
            if self.prl_reader.bufs[i].timestamp < limit_ts or limit_ts == 0:
                limit_ts = self.prl_reader.bufs[i].timestamp
                self.winner = i

        # Apply batch_size
        # Find the boundary or limit ts of the winning buffer
        # this is either the nth or the batch_size event.
        self.view_size = self.prl_reader.bufs[self.winner].n_ready_events - \
                self.prl_reader.bufs[self.winner].n_seen_events
        if self.view_size > batch_size:
            limit_ts = self.prl_reader.bufs[self.winner].ts_arr[\
                    self.prl_reader.bufs[self.winner].n_seen_events + batch_size - 1]
            self.view_size = batch_size

        # Locate the viewing window and update seen_offset for each buffer
        cdef uint64_t[:] ts_view
        cdef uint64_t prev_seen_offset  = 0
        cdef uint64_t block_size
        cdef Buffer* buf
        cdef uint64_t[:] buf_offsets    = self.buf_offsets
        cdef uint64_t[:] buf_sizes      = self.buf_sizes
        cdef uint64_t[:] stepbuf_offsets= self.stepbuf_offsets
        cdef uint64_t[:] stepbuf_sizes  = self.stepbuf_sizes
        cdef uint64_t[:] i_evts         = self.i_evts 
        cdef uint64_t[:] founds         = self.founds
        for i in range(self.prl_reader.nfiles):
            buf = &(self.prl_reader.bufs[i])
            buf_offsets[i] = buf.seen_offset
            
            # All the events before found_pos are within max_ts
            i_evts[i] = buf.n_seen_events
            founds[i] = 0
            while i_evts[i] < buf.n_ready_events and buf.ts_arr[i_evts[i]] < limit_ts:
                i_evts[i] += 1

            if i_evts[i] == buf.n_ready_events:     # all events should be yielded
                founds[i] = buf.n_ready_events - 1 
            if buf.ts_arr[i_evts[i]] == limit_ts:   # this event matches limit ts
                founds[i] = i_evts[i]              
            elif buf.ts_arr[i_evts[i]] > limit_ts:  # this event exceeds limt ts yield all the events prior to this one
                founds[i] = i_evts[i] - 1
            
            buf.seen_offset = buf.next_offset_arr[founds[i]]
            buf.n_seen_events = founds[i] + 1
            buf_sizes[i] = buf.seen_offset - buf_offsets[i]
            
            # Handle step buffers the same way
            buf = &(self.prl_reader.step_bufs[i])
            stepbuf_offsets[i] = buf.seen_offset
            
            # All the events before found_pos are within max_ts
            i_evts[i] = buf.n_seen_events
            founds[i] = 0
            while i_evts[i] < buf.n_ready_events and buf.ts_arr[i_evts[i]] < limit_ts:
                i_evts[i] += 1

            if i_evts[i] == buf.n_ready_events:     # all events should be yielded
                founds[i] = buf.n_ready_events - 1 
            if buf.ts_arr[i_evts[i]] == limit_ts:   # this event matches limit ts
                founds[i] = i_evts[i]              
            elif buf.ts_arr[i_evts[i]] > limit_ts:  # this event exceeds limt ts yield all the events prior to this one
                founds[i] = i_evts[i] - 1
            
            buf.seen_offset = buf.next_offset_arr[founds[i]]
            buf.n_seen_events = founds[i] + 1
            stepbuf_sizes[i] = buf.seen_offset - stepbuf_offsets[i]
            
        # output as a list of memoryviews for both L1 and step buffers
        mmrv_bufs = []
        mmrv_step_bufs = []
        cdef char[:] view
        for i in range(self.prl_reader.nfiles):
            buf = &(self.prl_reader.bufs[i])
            if self.buf_sizes[i] > 0:
                prev_seen_offset = self.buf_offsets[i]
                block_size = self.buf_sizes[i]
                view = <char [:block_size]> (buf.chunk + prev_seen_offset)
                mmrv_bufs.append(view)
            else:
                mmrv_bufs.append(0) # add 0 as a place=holder for empty buffer
            
            buf = &(self.prl_reader.step_bufs[i])
            if self.stepbuf_sizes[i] > 0:
                prev_seen_offset = self.stepbuf_offsets[i]
                block_size = self.stepbuf_sizes[i]
                view = <char [:block_size]> (buf.chunk + prev_seen_offset)
                mmrv_step_bufs.append(view)
            else:
                mmrv_step_bufs.append(0) # add 0 as a place=holder for empty buffer
        
        return mmrv_bufs, mmrv_step_bufs


    @property
    def view_size(self):
        return self.view_size


    @property
    def got(self):
        return self.prl_reader.got

    @property
    def chunk_overflown(self):
        return self.prl_reader.chunk_overflown

    @property
    def beginrun_offset(self):
        return self.prl_reader.beginrun_offset

    @property
    def n_beginruns(self):
        return self.prl_reader.n_beginruns

    def beginrun_view(self):
        if self.prl_reader.beginrun_offset == 0:
            return bytearray() 
        else:
            view = <char [:self.prl_reader.beginrun_offset]> (self.prl_reader.beginrun_buf)
            return view
    def reset_beginrun(self):
        self.prl_reader.beginrun_offset = 0
        self.prl_reader.n_beginruns = 0
