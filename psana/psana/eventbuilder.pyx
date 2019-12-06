## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from cpython cimport array
import array
import numpy as np
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE

from dgramlite cimport Xtc, Sequence, Dgram

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from parallelreader cimport Buffer
from libc.stdint cimport uint32_t, uint64_t

cdef class EventBuilder:
    """Builds a batch of events
    Takes memoryslice 'views' and identifies matching timestamp
    dgrams as an event. Returns list of events (size=batch_size)
    as another memoryslice 'batch'.
    
    Input: views
    EventBuilder receives a views from SmdCore. Each views consists
    of 1 or more chunks of small data.
    
    Output: list of batches
    Without destination call back the build fn returns a batch of events (size = batch_size) at index 0. With destination call back, this fn returns list of batches. Each batch has the same destination rank.
    
    Note that reading chunks inside a views or events inside a batch can be done
    using PacketFooter class."""
    cdef short nsmds
    cdef array.array offsets 
    cdef array.array sizes
    cdef array.array timestamps
    cdef array.array dgram_sizes
    cdef array.array dgram_timestamps
    cdef array.array event_timestamps
    cdef list views
    cdef unsigned nevents
    cdef size_t DGRAM_SIZE
    cdef size_t XTC_SIZE
    cdef unsigned long min_ts
    cdef unsigned long max_ts
    cdef unsigned L1_ACCEPT
    cdef Buffer* step_bufs

    def __init__(self, views):
        self.nsmds = len(views)
        self.offsets = array.array('I', [0]*self.nsmds)
        self.sizes = array.array('I', [memoryview(view).shape[0] for view in views])
        self.timestamps = array.array('L', [0]*self.nsmds)
        self.dgram_sizes = array.array('I', [0]*self.nsmds)
        self.dgram_timestamps = array.array('L', [0]*self.nsmds)
        self.event_timestamps = array.array('L', [0]*self.nsmds)
        self.views = views
        self.nevents = 0
        self.DGRAM_SIZE = sizeof(Dgram)
        self.XTC_SIZE = sizeof(Xtc)
        self.L1_ACCEPT = 12
        
        # step dgram buffers (epics, configs, etc.)
        self.step_bufs = <Buffer *>malloc(sizeof(Buffer)*self.nsmds)
        self._init_step_bufs()

    def _has_more(self):
        for i in range(self.nsmds):
            if self.offsets[i] < self.sizes[i]:
                return True
        return False

    def _init_step_bufs(self):
        cdef int idx
        for idx in range(self.nsmds):
            self.step_bufs[idx].chunk = <char *>malloc(0x100000)
            self.step_bufs[idx].offset = 0
            self.step_bufs[idx].nevents = 0

    def _reset_step_bufs(self):
        cdef int idx
        for idx in range(self.nsmds):
            self.step_bufs[idx].offset = 0
            self.step_bufs[idx].nevents = 0

    def __dealloc__(self):
        cdef int idx
        for idx in range(self.nsmds):
            free(self.step_bufs[idx].chunk)
        free(self.step_bufs)
    
    def build(self, batch_size=1, filter_fn=0, destination=0, limit_ts=-1):
        """
        Builds a list of batches.

        Each batch is bytearray with this content:
        [ [[d0][d1][d2][evt_footer_view]] [[d0][d1][d2][evt_footer_view]] ][batch_footer_view]
        | ---------- evt 0 -------------| |------------evt 1 -----------| 
        evt_footer_view:    [sizeof(d0) | sizeof(d1) | sizeof(d2) | 3] (for 3 dgrams in 1 evt)
        batch_footer_view:  [sizeof(evt0) | sizeof(evt1) | 2] (for 2 evts in 1 batch)

        batch_size: no. of events in a batch
        filter_fn: takes an event and return True/False
        destination: takes a timestamp and return rank no.
        """
        cdef unsigned got = 0
        batch_dict = {} # keeps list of batches (w/o destination callback, only one batch is returned at index 0)
        self.min_ts = 0
        self.max_ts = 0

        # Storing python list of bytearray as c pointers
        cdef Dgram* d
        cdef size_t payload = 0
        cdef char* view_ptr
        cdef Py_buffer buf
        cdef list raw_dgrams = [0] * self.nsmds
        cdef list event_dgrams = [0] * self.nsmds

        # Setup event footer and batch footer - see above comments for the content of batch_dict
        cdef array.array int_array_template = array.array('I', [])
        cdef array.array evt_footer = array.clone(int_array_template, self.nsmds + 1, zero=False)
        cdef unsigned[:] evt_footer_view = evt_footer
        cdef unsigned evt_footer_size = sizeof(unsigned) * (self.nsmds + 1)
        evt_footer_view[-1] = self.nsmds
        cdef array.array batch_footer= array.clone(int_array_template, batch_size + 1, zero=True)
        cdef unsigned[:] batch_footer_view = batch_footer
        
        # Use typed variables for performance
        cdef unsigned evt_size = 0
        cdef short dgram_idx = 0
        cdef short view_idx = 0
        cdef unsigned evt_idx = 0

        cdef unsigned reach_limit_ts = 0
        
        # For checking step dgrams
        cdef unsigned service = 0
        self._reset_step_bufs()

        while got < batch_size and self._has_more() and not reach_limit_ts:
            array.zero(self.timestamps)
            array.zero(self.dgram_sizes)
            
            # Get dgrams for all smd files then
            # collect their timestamps and sizes for sorting.
            for view_idx in range(self.nsmds):
                view = self.views[view_idx]
                if self.offsets[view_idx] < self.sizes[view_idx]:
                    # Fill buf with data from memoryview 'view'.
                    PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
                    view_ptr = <char *>buf.buf
                    view_ptr += self.offsets[view_idx]
                    d = <Dgram *>(view_ptr)
                    payload = d.xtc.extent - self.XTC_SIZE
                    self.timestamps[view_idx] = <uint64_t>d.seq.high << 32 | d.seq.low
                    self.dgram_sizes[view_idx] = self.DGRAM_SIZE + payload
                    raw_dgrams[view_idx] = <char[:self.dgram_sizes[view_idx]]>view_ptr
                    # Copy to step buffers if non L1
                    service = (d.env>>24)&0xf
                    if service != self.L1_ACCEPT and payload > 0: 
                        memcpy(self.step_bufs[view_idx].chunk + self.step_bufs[view_idx].offset, d, self.DGRAM_SIZE + payload)
                        self.step_bufs[view_idx].offset += self.DGRAM_SIZE + payload
                        self.step_bufs[view_idx].nevents += 1
                    PyBuffer_Release(&buf)

            sorted_smd_id = np.argsort(self.timestamps)

            # Pick the oldest timestamp dgram as the first event
            # then look for other dgrams (in the rest of smd views)
            # for matching timestamps. An event is build (as bytearray)
            # with packet_footer. All dgrams in an event have the same timestamp.
            for smd_id in sorted_smd_id:
                if self.timestamps[smd_id] == 0:
                    continue

                array.zero(self.event_timestamps)
                self.event_timestamps[smd_id] = self.timestamps[smd_id]
                self.offsets[smd_id] += self.dgram_sizes[smd_id]
                event_dgrams[smd_id] = raw_dgrams[smd_id] # this is the selected dgram
                
                if self.min_ts == 0:
                    self.min_ts = self.event_timestamps[smd_id] # records first timestamp

                # In other smd views, find matching timestamp dgrams
                for view_idx in range(self.nsmds):
                    view = self.views[view_idx]
                    if view_idx == smd_id or self.offsets[view_idx] >= self.sizes[view_idx]:
                        continue
                    
                    event_dgrams[view_idx] = 0
                    PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
                    view_ptr = <char *>buf.buf
                    view_ptr += self.offsets[view_idx]
                    d = <Dgram *>(view_ptr)
                    self.max_ts = <unsigned long>d.seq.high << 32 | d.seq.low
                    payload = d.xtc.extent - self.XTC_SIZE
                    while self.max_ts <= self.event_timestamps[smd_id]:
                        if self.max_ts == self.event_timestamps[smd_id]:
                            self.event_timestamps[view_idx] = self.max_ts
                            self.timestamps[view_idx] = 0
                            event_dgrams[view_idx] = <char[:self.DGRAM_SIZE+payload]>view_ptr

                        self.offsets[view_idx] += (self.DGRAM_SIZE + payload)

                        if self.offsets[view_idx] == self.sizes[view_idx]:
                            break

                        view_ptr += (self.DGRAM_SIZE + payload)
                        d = <Dgram *>(view_ptr)
                        self.max_ts = <unsigned long>d.seq.high << 32 | d.seq.low
                        payload = d.xtc.extent - self.XTC_SIZE
                    
                    PyBuffer_Release(&buf)
                
                # Put this event in the correct batch (determined by destionation callback). 
                # If destination() is not specifed, use batch 0.
                dest_rank = 0
                if destination:
                    dest_rank = destination(self.event_timestamps[smd_id])
                
                if batch_dict:
                    if dest_rank not in batch_dict:
                        batch_dict[dest_rank] = (bytearray(), []) # (events as bytes, event sizes)
                else:
                    batch_dict[dest_rank] = (bytearray(), [])

                batch, evt_sizes = batch_dict[dest_rank]

                # Extend this batch bytearray to include this event and collect
                # the size of this event for batch footer.
                evt_size = 0
                for dgram_idx in range(self.nsmds):
                    dgram = event_dgrams[dgram_idx]
                    if dgram: 
                        batch.extend(bytearray(dgram))
                        evt_footer_view[dgram_idx] = dgram.nbytes
                    else:
                        evt_footer_view[dgram_idx] = 0

                    evt_size += evt_footer_view[dgram_idx]
                
                batch.extend(evt_footer_view)
                evt_sizes.append(evt_size + evt_footer_size)
               
                # mona removed evt._complete() - I think smd events do not
                # need det interface. The evt._complete() is called in def _from_bytes()
                # and this is how bigdata events are created.
                
                # TODO: 
                # Find a place for filter(evt)

                if limit_ts > -1:
                    if self.max_ts >= limit_ts:
                        reach_limit_ts = 1
                        break
                
                got += 1
                if got == batch_size:
                    break

        
        self.nevents = got
        
        # Add packet_footer for all events in each batch
        for _, val in batch_dict.items():
            batch, evt_sizes = val
            for evt_idx in range(len(evt_sizes)):
                batch_footer_view[evt_idx] = evt_sizes[evt_idx]
            batch_footer_view[-1] = evt_idx + 1
            batch.extend(batch_footer_view[:evt_idx+1])
            batch.extend(batch_footer_view[batch_size:]) # when convert to bytearray negative index doesn't work
        
        return batch_dict

    @property
    def nevents(self):
        return self.nevents

    @property
    def min_ts(self):
        return self.min_ts

    @property
    def max_ts(self):
        return self.max_ts
    
    @property
    def offsets(self):
        return self.offsets

    def step_view(self, int buf_id):
        """ Returns memoryview of the step buffer object."""
        assert buf_id < self.nsmds
        cdef char[:] view
        cdef size_t block_size

        if self.step_bufs[buf_id].nevents == 0:
            return 0
        view = <char [:self.step_bufs[buf_id].offset]> self.step_bufs[buf_id].chunk

        return view
