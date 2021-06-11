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
from psana.psexp import TransitionId
import numpy as np
from cython.parallel import prange


cdef class SmdReader:
    cdef ParallelReader prl_reader
    cdef int        winner, n_view_events
    cdef int        max_retries, sleep_secs
    cdef uint64_t   i_starts[100]           # these 5 are aux. local variables 
    cdef uint64_t   i_ends[100]             #    
    cdef uint64_t   i_stepbuf_starts[100]   #
    cdef uint64_t   i_stepbuf_ends[100]     #
    cdef uint64_t   block_sizes[100]        # 
    cdef uint64_t   i_st_bufs[100]          # these 4 are global - can be used for
    cdef uint64_t   block_size_bufs[100]    # sharing viewing windows.
    cdef uint64_t   i_st_stepbufs[100]      #
    cdef uint64_t   block_size_stepbufs[100]#
    cdef float      total_time
    cdef int        num_threads

    def __init__(self, int[:] fds, int chunksize, int max_retries):
        assert fds.size > 0, "Empty file descriptor list (fds.size=0)."
        self.prl_reader         = ParallelReader(fds, chunksize)
        
        # max retries has no default value (set when creating datasource)
        self.max_retries        = max_retries
        self.sleep_secs         = 1
        self.total_time         = 0
        self.num_threads        = int(os.environ.get('PS_SMD0_NUM_THREADS', '16'))

    def is_complete(self):
        """ Checks that all buffers have at least one event 
        """
        cdef int is_complete = 1
        cdef int i

        for i in range(self.prl_reader.nfiles):
            if self.prl_reader.bufs[i].n_ready_events - \
                    self.prl_reader.bufs[i].n_seen_events == 0:
                is_complete = 0
                break

        return is_complete

    def get(self, found_xtc2):
        # SmdReaderManager only calls this function when there's no more event
        # in one or more buffers. Reset the indices for buffers that need re-read.
        cdef int i=0
        for i in range(self.prl_reader.nfiles):
            buf = &(self.prl_reader.bufs[i])
            if buf.n_ready_events - buf.n_seen_events > 0: continue 
            self.i_starts[i] = 0
            self.i_ends[i] = 0
            self.i_stepbuf_starts[i] = 0
            self.i_stepbuf_ends[i] = 0

        self.prl_reader.just_read()
        
        if self.max_retries > 0:

            cn_retries = 0
            while not self.is_complete():
                flag_founds = found_xtc2('smd') 

                # Only when .inprogress file is used and ALL xtc2 files are found 
                # that this will return a list of all(True). If we have a mixed
                # of True and False, we let ParallelReader decides which file
                # to read but we'll still need to do sleep.
                if all(flag_founds): break

                time.sleep(self.sleep_secs)
                print(f'smdreader waiting for an event...retry#{cn_retries+1} (max_retries={self.max_retries}, use PS_R_MAX_RETRIES for different value)')
                self.prl_reader.just_read()
                cn_retries += 1
                if cn_retries >= self.max_retries:
                    break
        

    @cython.boundscheck(False)
    def view(self, int batch_size=1000):
        """ Returns memoryview of the data and step buffers.

        This function is called by SmdReaderManager only when is_complete is True (
        all buffers have at least one event). It returns events of batch_size if
        possible or as many as it has for the buffer.
        """
        st_all = time.monotonic()

        # Find the winning buffer
        cdef int i=0
        cdef uint64_t limit_ts=0
        
        for i in range(self.prl_reader.nfiles):
            if self.prl_reader.bufs[i].timestamp < limit_ts or limit_ts == 0:
                limit_ts = self.prl_reader.bufs[i].timestamp
                self.winner = i

        # Apply batch_size
        # Find the boundary or limit ts of the winning buffer
        # this is either the nth or the batch_size event.
        self.n_view_events = self.prl_reader.bufs[self.winner].n_ready_events - \
                self.prl_reader.bufs[self.winner].n_seen_events
        if self.n_view_events > batch_size:
            limit_ts = self.prl_reader.bufs[self.winner].ts_arr[\
                    self.prl_reader.bufs[self.winner].n_seen_events - 1 + batch_size]
            self.n_view_events = batch_size

        # Locate the viewing window and update seen_offset for each buffer
        cdef Buffer* buf
        #cdef uint64_t[:] buf_ts_arr     
        cdef uint64_t[:] i_starts           = self.i_starts
        cdef uint64_t[:] i_ends             = self.i_ends 
        cdef uint64_t[:] i_stepbuf_starts   = self.i_stepbuf_starts
        cdef uint64_t[:] i_stepbuf_ends     = self.i_stepbuf_ends 
        cdef uint64_t[:] block_sizes        = self.block_sizes 
        cdef uint64_t[:] i_st_bufs          = self.i_st_bufs
        cdef uint64_t[:] block_size_bufs    = self.block_size_bufs
        cdef uint64_t[:] i_st_stepbufs      = self.i_st_stepbufs
        cdef uint64_t[:] block_size_stepbufs= self.block_size_stepbufs
        cdef unsigned endrun_id = TransitionId.EndRun
        
        st_search = time.monotonic()
        
        for i in prange(self.prl_reader.nfiles, nogil=True, num_threads=self.num_threads):
            buf = &(self.prl_reader.bufs[i])
            i_st_bufs[i] = 0
            block_size_bufs[i] = 0
            
            # Find boundary using limit_ts
            i_ends[i] = i_starts[i] 
            if i_ends[i] < buf.n_ready_events:
                if buf.ts_arr[i_ends[i]] != limit_ts:
                    ## Reduce the size of the search by searching from the next unseen event
                    #i_ends[i] = np.searchsorted(buf_ts_arr[buf.n_seen_events:buf.n_ready_events], limit_ts) + buf.n_seen_events

                    ## Note that limit_ts should be at this found index or its left index
                    ## If the found index is 0, then this buffer has nothing to share.
                    #if buf.ts_arr[i_ends[i]] != limit_ts:
                    #    if i_ends[i] == 0:
                    #        i_ends[i] = i_starts[i] - 1
                    #    else:
                    #        i_ends[i] -= 1
                    
                    while buf.ts_arr[i_ends[i] + 1] <= limit_ts \
                            and i_ends[i] < buf.n_ready_events - 1:
                        i_ends[i] += 1
                
                block_sizes[i] = buf.en_offset_arr[i_ends[i]] - buf.st_offset_arr[i_starts[i]]
               
                i_st_bufs[i] = i_starts[i]
                block_size_bufs[i] = block_sizes[i]
                
                buf.seen_offset = buf.en_offset_arr[i_ends[i]]
                buf.n_seen_events =  i_ends[i] + 1
                if buf.sv_arr[i_ends[i]] == endrun_id:
                    buf.found_endrun = 1
                i_starts[i] = i_ends[i] + 1

            
            # Handle step buffers the same way
            buf = &(self.prl_reader.step_bufs[i])
            i_st_stepbufs[i] = 0
            block_size_stepbufs[i] = 0
            
            # Find boundary using limit_ts (omit check for exact match here because it's unlikely
            # for transition buffers.
            i_stepbuf_ends[i] = i_stepbuf_starts[i] 
            if i_stepbuf_ends[i] <  buf.n_ready_events \
                    and buf.ts_arr[i_stepbuf_ends[i]] <= limit_ts: 
                while buf.ts_arr[i_stepbuf_ends[i] + 1] <= limit_ts \
                        and i_stepbuf_ends[i] < buf.n_ready_events - 1:
                    i_stepbuf_ends[i] += 1
                
                block_sizes[i] = buf.en_offset_arr[i_stepbuf_ends[i]] - buf.st_offset_arr[i_stepbuf_starts[i]]
                
                i_st_stepbufs[i] = i_stepbuf_starts[i]
                block_size_stepbufs[i] = block_sizes[i]
                
                buf.seen_offset = buf.en_offset_arr[i_stepbuf_ends[i]]
                buf.n_seen_events = i_stepbuf_ends[i] + 1
                i_stepbuf_starts[i]  = i_stepbuf_ends[i] + 1
            
        # end for i in ...
        en_all = time.monotonic()

        self.total_time += en_all - st_all

    def show(self, int i_buf, step_buf=False):
        """ Returns memoryview of buffer i_buf at the current viewing
        i_st and block_size"""
        cdef Buffer* buf
        cdef uint64_t[:] block_size_bufs
        cdef uint64_t[:] i_st_bufs      
        if step_buf:
            buf = &(self.prl_reader.step_bufs[i_buf])
            block_size_bufs = self.block_size_stepbufs
            i_st_bufs = self.i_st_stepbufs
        else:
            buf = &(self.prl_reader.bufs[i_buf])
            block_size_bufs = self.block_size_bufs
            i_st_bufs = self.i_st_bufs
        
        cdef char[:] view
        if block_size_bufs[i_buf] > 0:
            view = <char [:block_size_bufs[i_buf]]> (buf.chunk + buf.st_offset_arr[i_st_bufs[i_buf]])
            return view
        else:
            return memoryview(bytearray()) 


    @property
    def view_size(self):
        return self.n_view_events

    @property
    def total_time(self):
        return self.total_time


    @property
    def got(self):
        return self.prl_reader.got

    @property
    def chunk_overflown(self):
        return self.prl_reader.chunk_overflown

    def found_endrun(self):
        cdef int i
        found = False
        cn_endruns = 0
        for i in range(self.prl_reader.nfiles):
            if self.prl_reader.bufs[i].found_endrun == 1:
                cn_endruns += 1
        if cn_endruns == self.prl_reader.nfiles:
            found = True
        return found


