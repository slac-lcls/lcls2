from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from dgramlite cimport Xtc, Sequence, Dgram
from parallelreader cimport Buffer, ParallelReader

cdef class SmdReader:
    cdef unsigned got_events
    cdef unsigned long limit_ts
    cdef size_t DGRAM_SIZE
    cdef size_t XTC_SIZE
    cdef unsigned long min_ts # minimum timestamp of the output chunks
    cdef ParallelReader prl_reader
    cdef unsigned L1_ACCEPT
    cdef short v_cntrl
    cdef short v_service
    cdef unsigned long s_cntrl
    cdef unsigned long m_service
    cdef Buffer* update_bufs
    
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
        
        # update-dgram buffers (epics, configs, etc.)
        self.update_bufs = <Buffer *>malloc(sizeof(Buffer)*self.prl_reader.nfiles)
        self._init_update_bufs()
        
        # service calculation
        self.v_cntrl = 56
        self.v_service = 0
        k_cntrl = 8
        k_service = 4
        cdef unsigned long m_cntrl = ((1ULL << k_cntrl) - 1) 
        self.s_cntrl = (m_cntrl << self.v_cntrl)
        self.m_service = ((1 << k_service) - 1)
        
    def _init_update_bufs(self):
        cdef int idx
        for idx in range(self.prl_reader.nfiles):
            self.update_bufs[idx].chunk = <char *>malloc(0x100000)
            self.update_bufs[idx].offset = 0

    def _reset_update_bufs(self):
        cdef int idx
        for idx in range(self.prl_reader.nfiles):
            self.update_bufs[idx].offset = 0

    def __dealloc__(self):
        cdef int idx
        for idx in range(self.prl_reader.nfiles):
            free(self.update_bufs[idx].chunk)
        free(self.update_bufs)

    cdef inline int _check_dgram(self, Buffer *buf, unsigned smd_id):
        """ Checks status of the dgram.
        
        - If the entire dgram doesn't fit in the chunk, return needs_reread status.
        - If this dgram is a non L1Accept, save the dgram in update_buf
          and move on to the next dgram.
        """
        cdef size_t remaining = 0
        cdef size_t payload = 0
        cdef int needs_reread = 0
        cdef unsigned long pulse_id = 0
        cdef unsigned control = 0
        cdef unsigned service = 0
        cdef size_t prev_offset = 0

        while True:
            remaining = buf.got - buf.offset
            
            if self.DGRAM_SIZE <= remaining:
                # get payload
                d = <Dgram *>(buf.chunk + buf.offset)
                payload = d.xtc.extent - self.XTC_SIZE
                prev_offset = buf.offset # save the offset for updates if needed
                buf.offset += self.DGRAM_SIZE
                
                remaining = buf.got - buf.offset
                if payload <= remaining:
                    # got dgram
                    buf.offset += payload
                    buf.nevents += 1
                    buf.timestamp = <unsigned long>d.seq.high << 32 | d.seq.low

                    # check if this a non L1
                    pulse_id = d.seq.pulse_id
                    control = (pulse_id & self.s_cntrl) >> self.v_cntrl
                    service = (control >> self.v_service) & self.m_service
                    if service == self.L1_ACCEPT:
                        break
                    elif payload > 0: # not an empty non L1
                        memcpy(self.update_bufs[smd_id].chunk + self.update_bufs[smd_id].offset, buf.chunk + prev_offset, self.DGRAM_SIZE + payload)
                        self.update_bufs[smd_id].offset += self.DGRAM_SIZE + payload
                else:
                    needs_reread = 1
                    break
            else:
                needs_reread = 1
                break

        return needs_reread

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
        self._reset_update_bufs()

        cdef Dgram* d
        cdef size_t dgram_offset = 0
        cdef int winner = 0 
        cdef int needs_reread = 0
        cdef int i_st = 0
        cdef unsigned long current_max_ts = 0
        cdef int current_winner = 0
        cdef unsigned current_got_events = 0
        cdef int idx = 0
        
        while self.got_events < n_events and self.prl_reader.bufs[winner].got > 0:
            for i in range(i_st, self.prl_reader.nfiles):
                # read this file until hit limit timestamp
                while self.prl_reader.bufs[i].timestamp < self.limit_ts and self.prl_reader.bufs[i].got > 0:
                    # check that dgram fits in parallel reader buffer and it's non L1
                    needs_reread = self._check_dgram(&(self.prl_reader.bufs[i]), i)
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

    def view(self, int buf_id, int update=0):
        """ Returns memoryview of the buffer object.

        Set update to True to view update events.
        """
        assert buf_id < self.prl_reader.nfiles
        cdef char[:] view
        cdef size_t block_size

        if update == 0:
            block_size = self.prl_reader.bufs[buf_id].offset - self.prl_reader.bufs[buf_id].block_offset
            if self.prl_reader.bufs[buf_id].nevents == 0:
                return 0
            view = <char [:block_size]> (self.prl_reader.bufs[buf_id].chunk + self.prl_reader.bufs[buf_id].block_offset)
        else:
            view = <char [:self.update_bufs[buf_id].offset]> self.update_bufs[buf_id].chunk

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
