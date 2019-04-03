from libc.stdlib cimport abort, malloc, free
from libc.string cimport memcpy
from posix.unistd cimport read, sleep
from cython.parallel import parallel, prange
import numpy as np
import os

from cpython cimport array
import array

cimport cython
from dgramlite cimport Xtc, Sequence, Dgram
from parallelreader cimport Buffer, ParallelReader

cdef class SmdReader:
    cdef unsigned got_events
    cdef unsigned long limit_ts
    cdef size_t dgram_size
    cdef size_t xtc_size
    cdef unsigned long min_ts # minimum timestamp of the output chunks
    cdef ParallelReader prl_reader
    
    def __init__(self, fds):
        self.got_events = 0
        self.limit_ts = 1
        self.dgram_size = sizeof(Dgram)
        self.xtc_size = sizeof(Xtc)
        self.min_ts = 0
        self.prl_reader = ParallelReader(fds)
        self.prl_reader.read() # fill up all buffers

    def get(self, unsigned n_events = 1):
        """ Uses parallelreader to manage data and offsets
        for each buffer that can be accessed when view is called."""

        # Reset buffers and all bookkeeping variables
        self.got_events = 0
        self.min_ts = 0
        self.prl_reader.reset_buffers()

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
        cdef int idx = 0
        
        while self.got_events < n_events and self.prl_reader.bufs[winner].got > 0:
            for i in range(i_st, self.prl_reader.nfiles):
                # read this file until hit limit timestamp
                while self.prl_reader.bufs[i].timestamp < self.limit_ts and self.prl_reader.bufs[i].got > 0:
                    remaining = self.prl_reader.bufs[i].got - self.prl_reader.bufs[i].offset
                    if self.dgram_size <= remaining:
                        # get payload
                        d = <Dgram *>(self.prl_reader.bufs[i].chunk + self.prl_reader.bufs[i].offset)
                        payload = d.xtc.extent - self.xtc_size
                        self.prl_reader.bufs[i].offset += self.dgram_size
                        remaining = self.prl_reader.bufs[i].got - self.prl_reader.bufs[i].offset
                        if payload <= remaining:
                            # got dgram
                            self.prl_reader.bufs[i].offset += payload
                            self.prl_reader.bufs[i].nevents += 1
                            self.prl_reader.bufs[i].timestamp = <unsigned long>d.seq.high << 32 | d.seq.low
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

    def view(self, int buf_id):
        cdef size_t block_size = self.prl_reader.bufs[buf_id].offset - self.prl_reader.bufs[buf_id].block_offset
        assert buf_id < self.prl_reader.nfiles
        if self.prl_reader.bufs[buf_id].nevents == 0:
            return 0
        cdef char [:] view = <char [:block_size]> (self.prl_reader.bufs[buf_id].chunk + self.prl_reader.bufs[buf_id].block_offset)
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
