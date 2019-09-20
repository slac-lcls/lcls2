## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from parallelreader cimport Buffer
from cython.parallel import prange
import os

cdef class ParallelReader:
    
    def __init__(self, fds):
        self.fds = array.array('i', fds)
        self.chunksize = 0x100000
        self.maxretries = int(os.environ.get('PS_R_MAX_RETRIES', '5'))
        self.sleep_secs = int(os.environ.get('PS_R_SLEEP_SECS', '1'))
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
    
    cdef inline void reset_buffers(self):
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
                if attempt > 0:
                    sleep(self.sleep_secs) # sleep only when try more than once

        
        return requested - count

    cdef inline void read(self):
        """ Reads data in parallel (self.chunksize bytes) to the beginning
        of each buffer."""
        cdef int i
        for i in prange(self.nfiles, nogil=True):
            self.bufs[i].got = self._read_with_retries(i, 0, self.chunksize)

    cdef inline void read_partial(self):
        """ Reads partial chunk
        First copy what remains in the chunk to the begining of 
        the chunk then re-read to fill in the chunk.
        """
        cdef int i
        cdef char* chunk
        cdef size_t remaining = 0
        cdef size_t new_got = 0
        for i in prange(self.nfiles, nogil=True):
            chunk = self.bufs[i].chunk
            remaining = self.bufs[i].got - self.bufs[i].block_offset
            if remaining > 0:
                memcpy(chunk, chunk + self.bufs[i].block_offset, remaining)
            
            new_got = self._read_with_retries(i, \
                    remaining, self.chunksize - remaining)
            if new_got == 0:
                self.bufs[i].got = 0 # nothing more to read
            else:
                self.bufs[i].got = remaining + new_got
            
            self.bufs[i].offset = self.bufs[i].prev_offset - self.bufs[i].block_offset
            self.bufs[i].block_offset = 0

    def get_block(self):
        """ Packs data in all buffers with footer."""
        self.read()

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
        
        



