from libc.stdlib cimport abort, malloc, free
from libc.string cimport memcpy
from posix.unistd cimport read
from cython.parallel import parallel, prange
import numpy as np

cimport cython

cdef struct Xtc:
    int junks[4]
    unsigned extent

cdef struct Sequence:
    int junks[2]
    unsigned low
    unsigned high

cdef struct Dgram:
    Sequence seq
    int junks[4]
    Xtc xtc

cdef struct Buffer:
    char* chunk
    size_t got
    size_t offset
    size_t prev_offset
    unsigned nevents
    unsigned long timestamp
    size_t block_offset

cdef class SmdReader:
    cdef int* fds
    cdef size_t chunksize
    cdef int maxretries 
    cdef Buffer *bufs
    cdef int nfiles
    cdef unsigned got_events
    cdef unsigned long limit_ts
    cdef size_t dgram_size
    cdef size_t xtc_size
    
    def __init__(self, fds):
        self.chunksize = 0x100000
        self.maxretries = 5
        self.nfiles = len(fds)
        self.fds = <int *>malloc(sizeof(int) * self.nfiles)
        for i in range(self.nfiles):
            self.fds[i] = fds[i]
        self.bufs = NULL
        self.got_events = 0
        self.limit_ts = 1
        self.dgram_size = sizeof(Dgram)
        self.xtc_size = sizeof(Xtc)

    def __dealloc__(self):
        # FIXME with cppclass?
        if self.bufs:
            for i in range(self.nfiles):
                free(self.bufs[i].chunk)
            free(self.bufs)

    cdef inline void _init_buffers(self):
        cdef int i
        for i in prange(self.nfiles, nogil=True):
            self.bufs[i].chunk = <char *>malloc(self.chunksize)
            self.bufs[i].got = self._read_with_retries(i, 0, self.chunksize)
            self.bufs[i].offset = 0
            self.bufs[i].prev_offset = 0
            self.bufs[i].nevents = 0
            self.bufs[i].timestamp = 0
            self.bufs[i].block_offset = 0

    cdef inline void _reset_buffers(self):
        for i in range(self.nfiles):
            self.bufs[i].nevents = 0
            self.bufs[i].block_offset = self.bufs[i].offset
    
    cdef inline size_t _read_with_retries(self, int buf_id, size_t displacement, size_t count) nogil:
        cdef char* chunk = self.bufs[buf_id].chunk + displacement
        cdef size_t requested = count
        cdef size_t got = 0
        cdef int attempt
        for attempt in range(self.maxretries):
            got = read(self.fds[buf_id], chunk, count);
            if got == count:
                return requested
            else:
                chunk += got
                count -= got
        return requested - count
    
    cdef inline void _read_partial(self, int buf_id) nogil:
        """ Reads partial chunk
        First copy what remains in the chunk to the begining of 
        the chunk then re-read to fill in the chunk.
        """
        cdef char* chunk = self.bufs[buf_id].chunk
        cdef size_t remaining = self.chunksize - self.bufs[buf_id].block_offset
        if remaining > 0:
            memcpy(chunk, chunk + self.bufs[buf_id].block_offset, remaining)
        cdef size_t new_got = self._read_with_retries(buf_id, \
                remaining, self.chunksize - remaining)
        if new_got == 0:
            self.bufs[buf_id].got = 0 # nothing more to read
        else:
            self.bufs[buf_id].got = remaining + new_got
        self.bufs[buf_id].offset = self.bufs[buf_id].prev_offset - self.bufs[buf_id].block_offset
        self.bufs[buf_id].block_offset = 0

    cdef inline void _reread(self) nogil:
        cdef int idx = 0
        for idx in prange(self.nfiles, nogil=True):
            self._read_partial(idx)

    def get(self, unsigned n_events = 1):
        if not self.bufs:
            self.bufs = <Buffer *>malloc(sizeof(Buffer) * self.nfiles)
            self._init_buffers()
        
        self.got_events = 0
        self._reset_buffers()
        
        cdef Dgram* d
        cdef size_t payload = 0
        cdef size_t remaining = 0
        cdef size_t dgram_offset = 0
        cdef int winner = 0 
        cdef int needs_reread = 0
        cdef int i_st = 0
        cdef unsigned long current_max_ts = 0
        cdef int current_winner = 0
        cdef unsigned current_got_events = 0

        while self.got_events < n_events and self.bufs[winner].got > 0:
            for i in range(i_st, self.nfiles):
                # read this file until hit limit timestamp
                while self.bufs[i].timestamp < self.limit_ts and self.bufs[i].got > 0:
                    remaining = self.bufs[i].got - self.bufs[i].offset
                    if self.dgram_size <= remaining:
                        # get payload
                        d = <Dgram *>(self.bufs[i].chunk + self.bufs[i].offset)
                        payload = d.xtc.extent - self.xtc_size
                        self.bufs[i].offset += self.dgram_size
                        remaining = self.bufs[i].got - self.bufs[i].offset
                        if payload <= remaining:
                            # got dgram
                            self.bufs[i].offset += payload
                            self.bufs[i].nevents += 1
                            self.bufs[i].timestamp = <unsigned long>d.seq.high << 32 | d.seq.low
                        else:
                            needs_reread = 1 # not enough for the whole block, shift and reread all files
                            break
                    else:
                        needs_reread = 1
                        break
                
                if needs_reread:
                    i_st = i # start with the current buffer
                    break
                

                # remember previous offsets in case reread is needed
                self.bufs[i].prev_offset = self.bufs[i].offset
                
                if self.bufs[i].timestamp > current_max_ts:
                    current_max_ts = self.bufs[i].timestamp
                    current_winner = i

                if self.bufs[i].nevents > current_got_events:
                    current_got_events = self.bufs[i].nevents

                
            # shift and reread in parallel
            if needs_reread:
                self._reread()
                needs_reread = 0
            else:
                i_st = 0 # make sure that unless reread, always start with buffer 0
                winner = current_winner
                self.limit_ts = current_max_ts + 1
                self.got_events = current_got_events
                current_got_events = 0

    def view(self, int buf_id):
        cdef size_t block_size = self.bufs[buf_id].offset - self.bufs[buf_id].block_offset
        assert buf_id < self.nfiles
        if self.bufs[buf_id].nevents == 0:
            return 0
        cdef char [:] view = <char [:block_size]> (self.bufs[buf_id].chunk + self.bufs[buf_id].block_offset)
        return view

    @property
    def got_events(self):
        return self.got_events
