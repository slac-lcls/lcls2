from libc.stdio cimport *
from posix.fcntl cimport *
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
from posix.unistd cimport read

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

cdef class BufferedReader:
    cdef int fd
    cdef size_t chunksize
    cdef int maxretries
    cdef char* chunk
    cdef size_t got
    cdef size_t offset

    def __init__(self, int fd):
        self.fd = fd
        self.chunksize = 0x1000000
        self.maxretries = 5
        self.chunk = NULL
        self.got = 0
        self.offset = 0

    def __dealloc__(self):
        free(self.chunk)
    
    def _read_with_retries(self, size_t displacement, size_t count):
        cdef char* buf = self.chunk + displacement
        cdef size_t requested = count
        cdef size_t got = 0
        for attempt in range(self.maxretries):
            got = read(self.fd, buf, count);
            if got == count:
                return requested
            else:
                buf += got
                count -= got
        return requested - count

    def _read_partial(self, size_t block_offset, size_t dgram_offset):
        """ Reads partial chunk
        First copy what remains in the chunk to the begining of 
        the chunk then re-read to fill in the chunk.
        """
        cdef size_t remaining = self.chunksize - block_offset
        memcpy(self.chunk, self.chunk + block_offset, remaining)
        cdef size_t new_got = self._read_with_retries(remaining, self.chunksize - remaining)
        if new_got == 0:
            self.got = 0 # nothing more to read
        else:
            self.got = remaining + new_got
        self.offset = dgram_offset - block_offset

    def get(self, unsigned n_events):
        if self.chunk == NULL:
            self.chunk = <char*> malloc(self.chunksize)
            self.got = self._read_with_retries(0, self.chunksize)
        
        cdef Dgram* d
        cdef size_t payload = 0
        cdef size_t remaining = 0
        cdef unsigned got_events = 0
        cdef size_t block_offset = self.offset
        cdef size_t dgram_offset = 0

        while (got_events < n_events and self.got > 0):
            dgram_offset = self.offset
            remaining = self.got - self.offset
            if sizeof(Dgram) <= remaining:
                # get payload
                d = <Dgram *>(self.chunk + self.offset)
                payload = d.xtc.extent - sizeof(Xtc)
                self.offset += sizeof(Dgram)
                remaining = self.got - self.offset
                if payload <= remaining:
                    # got dgram
                    self.offset += payload
                    got_events += 1
                else:
                    # not enough for the whole block, shift and reread
                    self._read_partial(block_offset, dgram_offset)
                    block_offset = 0
            else:
                # not enough for the whole block, shift and reread
                self._read_partial(block_offset, dgram_offset)
                block_offset = 0
            
        cdef size_t block_size = self.offset - block_offset
        if block_size == 0:
            return 0

        cdef char [:] view = <char [:self.offset - block_offset]> (self.chunk + block_offset)
        return view
