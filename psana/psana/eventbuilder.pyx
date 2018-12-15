## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from cpython cimport array
import array
import numpy as np
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE

from dgramlite cimport Xtc, Sequence, Dgram

cdef class EventBuilder:
    """Builds a batch of events
    Takes memoryslice 'views' and identifies matching timestamp
    dgrams as an event. Returns list of events (size=batch_size)
    as another memoryslice 'batch'.
    
    views: each smd file is seperated by b'endofstream'
    batch: each dgram and event is seperated by b'eod' and b'eoe'."""
    cdef short nsmds
    cdef array.array offsets 
    cdef array.array sizes
    cdef array.array timestamps
    cdef array.array dgram_sizes
    cdef array.array dgram_timestamps
    cdef array.array event_timestamps
    cdef list views
    cdef unsigned nevents
    cdef size_t dgram_size
    cdef size_t xtc_size

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
        self.dgram_size = sizeof(Dgram)
        self.xtc_size = sizeof(Xtc)

    def _has_more(self):
        for i in range(self.nsmds):
            if self.offsets[i] < self.sizes[i]:
                return True
        return False

    def build(self, unsigned batch_size=1, filter_fn=0):
        cdef unsigned got = 0
        batch = bytearray()

        cdef Dgram* d
        cdef size_t payload = 0
        cdef unsigned long ts
        cdef char* cview
        cdef Py_buffer buf
        cdef list raw_dgrams = [0] * self.nsmds
        cdef list event_dgrams = [0] * self.nsmds
        cdef char[:] to_view
        cdef array.array int_array_template = array.array('I', [])
        cdef array.array evt_footer = array.clone(int_array_template, self.nsmds + 1, zero=False)
        cdef unsigned[:] evt_footer_view = evt_footer
        cdef unsigned evt_footer_size = sizeof(unsigned) * (self.nsmds + 1)
        evt_footer_view[-1] = self.nsmds
        cdef array.array evt_sizes = array.clone(int_array_template, batch_size + 1, zero=True)
        cdef unsigned[:] evt_sizes_view = evt_sizes
        cdef unsigned evt_size = 0

        while got < batch_size and self._has_more():
            array.zero(self.timestamps)
            array.zero(self.dgram_sizes)
            
            # Get dgrams for all smd files then
            # collect their timestamps and sizes for sorting.
            for i, view in enumerate(self.views):
                if self.offsets[i] < self.sizes[i]:
                    PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
                    cview = <char *>buf.buf
                    cview += self.offsets[i]
                    d = <Dgram *>(cview)
                    payload = d.xtc.extent - self.xtc_size
                    self.timestamps[i] = <unsigned long>d.seq.high << 32 | d.seq.low
                    self.dgram_sizes[i] = self.dgram_size + payload
                    to_view = <char[:self.dgram_sizes[i]]>cview
                    raw_dgrams[i] = to_view
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

                # In other smd views, find matching timestamp dgrams
                for i, view in enumerate(self.views):
                    if i == smd_id or self.offsets[i] >= self.sizes[i]:
                        continue
                    
                    event_dgrams[i] = 0
                    PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
                    cview = <char *>buf.buf
                    cview += self.offsets[i]
                    d = <Dgram *>(cview)
                    ts = <unsigned long>d.seq.high << 32 | d.seq.low
                    payload = d.xtc.extent - self.xtc_size
                    while ts <= self.event_timestamps[smd_id]:
                        if ts == self.event_timestamps[smd_id]:
                            self.event_timestamps[i] = ts
                            self.timestamps[i] = 0
                            to_view = <char[:self.dgram_size+payload]>cview
                            event_dgrams[i] = to_view

                        self.offsets[i] += (self.dgram_size + payload)

                        if self.offsets[i] == self.sizes[i]:
                            break

                        cview += (self.dgram_size + payload)
                        d = <Dgram *>(cview)
                        ts = <unsigned long>d.seq.high << 32 | d.seq.low
                        payload = d.xtc.extent - self.xtc_size
                
                # Extend batch bytearray to include this event and collect
                # the size of this event for batch footer.
                evt_size = 0
                for i, dgram in enumerate(event_dgrams):
                    if dgram: 
                        batch.extend(bytearray(dgram))
                        evt_footer_view[i] = dgram.nbytes
                    else:
                        evt_footer_view[i] = 0

                    evt_size += evt_footer_view[i]
                
                batch.extend(evt_footer_view)
                evt_sizes_view[got] = evt_size + evt_footer_size
               
                # mona removed evt._complete() - I think smd events do not
                # need det interface. The evt._complete() is called in def _from_bytes()
                # and this is how bigdata events are created.
                
                # TODO: 
                # Find a place for filter(evt)

                got += 1
                if got == batch_size:
                    break
                
        
        self.nevents = got
        
        # Add packet_footer for all events
        evt_sizes_view[-1] = got
        batch.extend(evt_sizes_view[:got])
        batch.extend(evt_sizes_view[batch_size:]) # when convert to bytearray negative index doesn't work
        
        return batch

    @property
    def nevents(self):
        return self.nevents
