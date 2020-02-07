## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from dgramlite cimport Xtc, Sequence, Dgram
from parallelreader cimport Buffer, ParallelReader
from libc.stdint cimport uint32_t, uint64_t

cdef class SmdReader:
    cdef int got_events
    cdef uint64_t min_ts, max_ts
    cdef ParallelReader prl_reader
    
    def __init__(self, int[:] fds, int chunksize, int max_events):
        self.prl_reader = ParallelReader(fds, chunksize, max_events)
        self._reset()
        
    def _reset(self):
        self.got_events = 0
        self.min_ts = 0
        self.max_ts = 0

    def get(self):
        self._reset()
        self.prl_reader.just_read()
        
        cdef int i
        
        # Figure who's the fastest buffer (smalltest ts)
        cdef int winner=0
        cdef uint64_t limit_ts=0
        
        if self.prl_reader.nfiles > 0:
            limit_ts = self.prl_reader.bufs[0].timestamp
            for i in range(1, self.prl_reader.nfiles):
                if self.prl_reader.bufs[i].timestamp < limit_ts:
                    limit_ts = self.prl_reader.bufs[i].timestamp
                    winner = i
        
        # Make sure all buffers are time coherent
        self.prl_reader.rewind(limit_ts, winner)

        self.got_events = self.prl_reader.bufs[winner].nevents
        if self.got_events > 0:
            self.min_ts = self.prl_reader.bufs[winner].ts_arr[0]
            self.max_ts = self.prl_reader.bufs[winner].ts_arr[self.got_events-1]
        
    def view(self, int buf_id, int step=0):
        """ Returns memoryview of the buffer object.

        Set step to True to view step events.
        """
        assert buf_id < self.prl_reader.nfiles
        
        cdef char[:] view
        cdef Buffer* buf
        cdef uint64_t block_size

        if step == 0:
            buf = &(self.prl_reader.bufs[buf_id])
        else:
            buf = &(self.prl_reader.step_bufs[buf_id])

        if buf.nevents == 0:
            return 0
        
        if step == 0:
            block_size = buf.offset - buf.lastget_offset
            view = <char [:block_size]> (buf.chunk + buf.lastget_offset)
        else:
            view = <char [:buf.offset]> buf.chunk
        return view

    @property
    def got_events(self):
        return self.got_events

    @property
    def min_ts(self):
        return self.min_ts

    @property
    def max_ts(self):
        return self.max_ts
    
    def retry(self):
        self.prl_reader._reset_buffers(self.prl_reader.bufs)
        cdef int i
        for i in range(self.prl_reader.nfiles):
            self.prl_reader.bufs[i].needs_reread = 1
        self.get()
