from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from posix.unistd cimport read

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
    unsigned nevents
    unsigned long timestamp
    size_t block_offset
    size_t block_size

cdef class SmdReader:
    cdef int* fds
    cdef size_t chunksize
    cdef int maxretries 
    cdef Buffer *bufs
    cdef int nfiles
    cdef unsigned got_events
    
    def __init__(self, fds):
        self.chunksize = 0x100000
        self.maxretries = 5
        self.nfiles = len(fds)
        self.fds = <int *>malloc(sizeof(int) * self.nfiles)
        for i in range(self.nfiles):
            self.fds[i] = fds[i]
        self.bufs = NULL
        self.got_events = 0

    def __dealloc__(self):
        # FIXME with cppclass?
        if self.bufs:
            for i in range(self.nfiles):
                free(self.bufs[i].chunk)

    def _init_buffers(self):
        for i in range(self.nfiles):
            self.bufs[i].chunk = <char *>malloc(self.chunksize)
            self.bufs[i].got = self._read_with_retries(i, 0, self.chunksize)
            self.bufs[i].offset = 0
            self.bufs[i].nevents = 0
            self.bufs[i].timestamp = 0
            self.bufs[i].block_offset = 0
            self.bufs[i].block_size = 0
    
    cdef inline size_t _read_with_retries(self, int buf_id, size_t displacement, size_t count):
        cdef char* chunk = self.bufs[buf_id].chunk + displacement
        cdef size_t requested = count
        cdef size_t got = 0
        for attempt in range(self.maxretries):
            got = read(self.fds[buf_id], chunk, count);
            if got == count:
                return requested
            else:
                chunk += got
                count -= got
        return requested - count
    
    cdef inline void _read_partial(self, int buf_id, size_t block_offset, size_t dgram_offset):
        """ Reads partial chunk
        First copy what remains in the chunk to the begining of 
        the chunk then re-read to fill in the chunk.
        """
        cdef char* chunk = self.bufs[buf_id].chunk
        cdef size_t remaining = self.chunksize - block_offset
        if remaining > 0:
            memcpy(chunk, chunk + block_offset, remaining)
        cdef size_t new_got = self._read_with_retries(buf_id, \
                remaining, self.chunksize - remaining)
        if new_got == 0:
            self.bufs[buf_id].got = 0 # nothing more to read
        else:
            self.bufs[buf_id].got = remaining + new_got
        self.bufs[buf_id].offset = dgram_offset - block_offset

    def get(self, unsigned n_events = 1, unsigned long limit_ts = 1):
        if not self.bufs:
            self.bufs = <Buffer *>malloc(sizeof(Buffer) * self.nfiles)
            self._init_buffers()
        
        self.got_events = 0
        for i in range(self.nfiles):
            self.bufs[i].nevents = 0
            self.bufs[i].block_size = 0
            self.bufs[i].block_offset = self.bufs[i].offset 

        cdef Dgram* d
        cdef size_t payload = 0
        cdef size_t remaining = 0
        cdef size_t dgram_offset = 0
        cdef int winner = 0 

        while self.got_events < n_events and self.bufs[winner].got > 0:
            for i in range(self.nfiles):
                # read this file until hit limit timestamp
                while self.bufs[i].timestamp < limit_ts and self.bufs[i].got > 0:
                    dgram_offset = self.bufs[i].offset
                    remaining = self.bufs[i].got - self.bufs[i].offset
                    if sizeof(Dgram) <= remaining:
                        # get payload
                        d = <Dgram *>(self.bufs[i].chunk + self.bufs[i].offset)
                        payload = d.xtc.extent - sizeof(Xtc)
                        self.bufs[i].offset += sizeof(Dgram)
                        remaining = self.bufs[i].got - self.bufs[i].offset
                        if payload <= remaining:
                            # got dgram
                            self.bufs[i].offset += payload
                            self.bufs[i].nevents += 1
                            self.bufs[i].timestamp = <unsigned long>d.seq.high << 32 | d.seq.low
                        else:
                            # not enough for the whole block, shift and reread
                            self._read_partial(i, self.bufs[i].block_offset, dgram_offset)
                            self.bufs[i].block_offset = 0
                    else:
                        # not enough for the whole block, shift and reread
                        self._read_partial(i, self.bufs[i].block_offset, dgram_offset)
                        self.bufs[i].block_offset = 0

                self.bufs[i].block_size = self.bufs[i].offset - self.bufs[i].block_offset
            
            #for i in range(self.nfiles):
                if self.bufs[i].timestamp > limit_ts:
                    limit_ts = self.bufs[i].timestamp + 1 
                    winner = i
                if self.bufs[i].nevents > self.got_events:
                    self.got_events = self.bufs[i].nevents

    def view(self, int buf_id):
        assert buf_id < self.nfiles
        if self.bufs[buf_id].nevents == 0:
            return 0
        cdef char [:] view = <char [:self.bufs[buf_id].block_size]> (self.bufs[buf_id].chunk + self.bufs[buf_id].block_offset)
        return view

    @property
    def got_events(self):
        return self.got_events
