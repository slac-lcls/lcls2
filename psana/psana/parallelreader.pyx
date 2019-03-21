from libc.stdlib cimport malloc, free
from posix.unistd cimport read
from cython.parallel import parallel, prange
from cpython cimport array
import array

cdef struct Buffer:
    char* chunk
    size_t got
    size_t offset
    size_t prev_offset
    unsigned nevents
    unsigned long timestamp
    size_t block_offset
    
cdef class ParallelReader:
    cdef int[:] fds
    cdef size_t chunksize
    cdef int maxretries
    cdef int nfiles
    cdef Buffer *bufs

    def __init__(self, fds):
        self.fds = array.array('i', fds)
        self.chunksize = 0x100000
        self.maxretries = 5
        self.nfiles = len(self.fds)
        self.bufs = <Buffer *>malloc(sizeof(Buffer) * self.nfiles)
        self._init_buffers()

    def __dealloc__(self):
        if self.bufs:
            for i in range(self.nfiles):
                free(self.bufs[i].chunk)
            free(self.bufs)

    cdef inline void _init_buffers(self):
        cdef int i
        for i in prange(self.nfiles, nogil=True):
            self.bufs[i].chunk = <char *>malloc(self.chunksize)
            self.bufs[i].got = 0
            self.bufs[i].offset = 0
            self.bufs[i].prev_offset = 0
            self.bufs[i].nevents = 0
            self.bufs[i].timestamp = 0
            self.bufs[i].block_offset = 0

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

    cdef inline void _read(self, size_t displacement):
        """ Reads data in parallel (self.chunksize bytes) to the beginning
        of each buffer."""
        cdef int i
        for i in prange(self.nfiles, nogil=True):
            self.bufs[i].got = self._read_with_retries(i, displacement, self.chunksize)

    def get_block(self):
        """ Packs data in all buffers with footer."""
        self._read(0)

        block = bytearray()
        cdef array.array int_array_template = array.array('I', [])
        cdef array.array footer = array.clone(int_array_template, self.nfiles + 1, zero=True)
        cdef unsigned[:] footer_view = footer
        footer_view[-1] = self.nfiles
        
        cdef char [:] view
        for i in range(self.nfiles):
            if self.bufs[i].got > 0:
                view = <char [:self.bufs[i].got]> (self.bufs[i].chunk + self.bufs[i].block_offset)
                block.extend(bytearray(view))
                footer_view[i] = view.shape[0]

        if block:
            block.extend(footer_view)
        return block
        
        



