## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from dgramlite cimport Xtc, Sequence, Dgram
from parallelreader cimport Buffer, ParallelReader
from libc.stdint cimport uint32_t, uint64_t
import numpy as np
import time, os

cdef class SmdReader:
    cdef ParallelReader prl_reader
    cdef int winner, view_size
    cdef int max_retries, sleep_secs
    
    def __init__(self, int[:] fds, int chunksize):
        assert fds.size > 0, "Empty file descriptor list (fds.size=0)."
        self.prl_reader = ParallelReader(fds, chunksize)
        self.max_retries = int(os.environ['PS_SMD_MAX_RETRIES']) # no default (force set when creating datasource)
        self.sleep_secs = int(os.environ.get('PS_SMD_SLEEP_SECS', '1'))
        
    def is_complete(self):
        """ Checks that all buffers have at least one event 
        """
        cdef int is_complete = 1
        cdef int i
        for i in range(self.prl_reader.nfiles):
            if self.prl_reader.bufs[i].n_ready_events - self.prl_reader.bufs[i].n_seen_events == 0:
                is_complete = 0
                break
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

    def view(self, int batch_size=1000):
        """ Returns memoryview of the data and step buffers.

        This function is called by SmdReaderManager only when is_complete is True (
        all buffers have at least one event). It returns events of batch_size if
        possible or as many as it has for the buffer.
        """

        # Find the winning buffer
        cdef int i
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
        cdef uint64_t prev_seen_offset = 0
        cdef Py_ssize_t found_pos
        cdef uint64_t block_size
        cdef char[:] view
        
        mmrv_bufs = []
        mmrv_step_bufs = []
        cdef Buffer* buf, step_buf
        for i in range(self.prl_reader.nfiles):
            buf = &(self.prl_reader.bufs[i])
            ts_view = buf.ts_arr
            prev_seen_offset = buf.seen_offset
            
            # All the events before found_pos are within max_ts
            found_pos = np.searchsorted(ts_view[:buf.n_ready_events], limit_ts, side='right')
            buf.seen_offset = buf.next_offset_arr[found_pos-1]
            buf.n_seen_events = found_pos

            block_size = buf.seen_offset - prev_seen_offset
            if block_size > 0:
                view = <char [:block_size]> (buf.chunk + prev_seen_offset)
                mmrv_bufs.append(view)
            else:
                mmrv_bufs.append(0) # add 0 as a place=holder for empty buffer


            buf = &(self.prl_reader.step_bufs[i])
            ts_view = buf.ts_arr
            prev_seen_offset = buf.seen_offset
            
            # All the events before found_pos are within max_ts
            found_pos = np.searchsorted(ts_view[:buf.n_ready_events], limit_ts, side='right')
            buf.seen_offset = buf.next_offset_arr[found_pos-1]
            buf.n_seen_events = found_pos

            block_size = buf.seen_offset - prev_seen_offset
            if block_size > 0:
                view = <char [:block_size]> (buf.chunk + prev_seen_offset)
                mmrv_step_bufs.append(view)
            else:
                mmrv_step_bufs.append(0)
        
        return mmrv_bufs, mmrv_step_bufs

    @property
    def view_size(self):
        return self.view_size


