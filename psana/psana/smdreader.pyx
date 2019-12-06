## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from dgramlite cimport Xtc, Sequence, Dgram
from parallelreader cimport Buffer, ParallelReader
from libc.stdint cimport uint32_t, uint64_t

cdef class SmdReader:
    cdef unsigned got_events
    cdef unsigned long limit_ts
    cdef size_t DGRAM_SIZE
    cdef size_t XTC_SIZE
    cdef unsigned long min_ts # minimum timestamp of the output chunks
    cdef ParallelReader prl_reader
    cdef unsigned L1_ACCEPT
    cdef Buffer* step_bufs
    
    def __init__(self, fds):
        self.got_events = 0
        self.limit_ts = 1
        self.DGRAM_SIZE = sizeof(Dgram)
        self.XTC_SIZE = sizeof(Xtc)
        self.L1_ACCEPT = 12
        self.min_ts = 0
        
        # smalldata buffers
        self.prl_reader = ParallelReader(fds)
        self.prl_reader.read() # fill up all buffers
        
        # step dgram buffers (epics, configs, etc.)
        self.step_bufs = <Buffer *>malloc(sizeof(Buffer)*self.prl_reader.nfiles)
        self._init_step_bufs()

    def _init_step_bufs(self):
        cdef int idx
        for idx in range(self.prl_reader.nfiles):
            self.step_bufs[idx].chunk = <char *>malloc(0x100000)
            self.step_bufs[idx].offset = 0
            self.step_bufs[idx].nevents = 0

    def _reset_step_bufs(self):
        cdef int idx
        for idx in range(self.prl_reader.nfiles):
            self.step_bufs[idx].offset = 0
            self.step_bufs[idx].nevents = 0

    def __dealloc__(self):
        cdef int idx
        for idx in range(self.prl_reader.nfiles):
            free(self.step_bufs[idx].chunk)
        free(self.step_bufs)

    def get(self, unsigned n_events = 1):
        """ Identifies the boundary of each smd chunk so that all exporting
        chunks have the same maximum timestamp. The maximum timestamp is 
        determined by n_events and is from the fastest detector.

        Note that this method only sets the boundary using offsets. Use view
        method to obtain chunks of memory as memoryview.
        """

        # Reset buffers and all bookkeeping variables
        self.got_events = 0
        self.min_ts = 0
        self.prl_reader.reset_buffers()
        self._reset_step_bufs()

        cdef Dgram* d
        cdef size_t dgram_offset = 0
        cdef int winner = 0 
        cdef int needs_reread = 0
        cdef int i_st = 0
        cdef unsigned long current_max_ts = 0
        cdef int current_winner = 0
        cdef unsigned current_got_events = 0
        cdef int idx = 0
        
        cdef size_t remaining = 0
        cdef size_t payload = 0
        cdef unsigned service = 0
        cdef Buffer* buf
        
        while self.got_events < n_events and self.prl_reader.bufs[winner].got > 0:
            for i in range(i_st, self.prl_reader.nfiles):
                while self.prl_reader.bufs[i].timestamp < self.limit_ts and self.prl_reader.bufs[i].got > 0:
                    buf = &(self.prl_reader.bufs[i])
                    # read until found an L1Accept
                    while True:
                        remaining = buf.got - buf.offset

                        if self.DGRAM_SIZE <= remaining:
                            # get payload and timestamp
                            d = <Dgram *>(buf.chunk + buf.offset)
                            payload = d.xtc.extent - self.XTC_SIZE
                            buf.offset += self.DGRAM_SIZE
                            buf.timestamp = <uint64_t>d.seq.high << 32 | d.seq.low
                            
                            # check if this a non L1
                            service = (d.env>>24)&0xf

                            remaining = buf.got - buf.offset
                            if payload <= remaining:
                                # got dgram
                                buf.offset += payload

                                if service == self.L1_ACCEPT:
                                    buf.nevents += 1
                                    break
                                elif payload > 0:
                                    memcpy(self.step_bufs[i].chunk + self.step_bufs[i].offset, d, self.DGRAM_SIZE + payload)
                                    self.step_bufs[i].offset += self.DGRAM_SIZE + payload
                                    self.step_bufs[i].nevents += 1
                            else:
                                needs_reread = 1 # not enough payload
                                break
                        else:
                            needs_reread = 1 # not enough dgram header
                            break
                    
                    
                    if needs_reread:
                        break
                    
                if needs_reread:
                    i_st = i # start with the current buffer
                    break

                # remember previous offsets in case reread is needed
                self.prl_reader.bufs[i].prev_offset = self.prl_reader.bufs[i].offset
                
                if self.prl_reader.bufs[i].timestamp > current_max_ts:
                    current_max_ts = self.prl_reader.bufs[i].timestamp
                    if self.min_ts == 0:
                        self.min_ts = current_max_ts # keep the first timestamp of this chunk
                    current_winner = i

                if self.prl_reader.bufs[i].nevents > current_got_events:
                    current_got_events = self.prl_reader.bufs[i].nevents

            # shift and reread in parallel
            if needs_reread:
                self.prl_reader.read_partial()
                needs_reread = 0
            else:
                i_st = 0 # make sure that unless reread, always start with buffer 0
                winner = current_winner
                self.limit_ts = current_max_ts + 1
                self.got_events = current_got_events
                current_got_events = 0

    def view(self, int buf_id, int step=0):
        """ Returns memoryview of the buffer object.

        Set step to True to view step events.
        """
        assert buf_id < self.prl_reader.nfiles
        cdef char[:] view
        cdef size_t block_size

        if step == 0:
            block_size = self.prl_reader.bufs[buf_id].offset - self.prl_reader.bufs[buf_id].block_offset
            if self.prl_reader.bufs[buf_id].nevents == 0:
                return 0
            view = <char [:block_size]> (self.prl_reader.bufs[buf_id].chunk + self.prl_reader.bufs[buf_id].block_offset)
        else:
            if self.step_bufs[buf_id].nevents == 0:
                return 0
            view = <char [:self.step_bufs[buf_id].offset]> self.step_bufs[buf_id].chunk

        return view

    @property
    def got_events(self):
        return self.got_events

    @property
    def min_ts(self):
        return self.min_ts

    @property
    def max_ts(self):
        return self.limit_ts - 1
