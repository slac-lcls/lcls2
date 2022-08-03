## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from cpython cimport array
import array
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from cpython.object cimport PyObject
from cpython.getargs cimport PyArg_ParseTupleAndKeywords

from dgramlite cimport Xtc, Sequence, Dgram

from libc.stdint cimport uint32_t, uint64_t

from psana.event import Event
import time

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
    cdef array.array services
    cdef list views
    cdef list configs
    cdef uint64_t[:] filter_timestamps
    cdef int batch_size
    cdef int intg_stream_id
    cdef PyObject* filter_fn
    cdef PyObject* destination
    cdef PyObject* run
    cdef PyObject* prometheus_counter
    cdef unsigned nevents
    cdef unsigned nsteps
    cdef size_t DGRAM_SIZE
    cdef size_t XTC_SIZE
    cdef unsigned long min_ts
    cdef unsigned long max_ts
    cdef unsigned L1Accept 

    def __init__(self, views, configs, uint64_t[:] filter_timestamps,
            *args, **kwargs):
        self.nsmds              = len(views)
        self.offsets            = array.array('I', [0]*self.nsmds)
        self.sizes              = array.array('I', [memoryview(view).shape[0] for view in views])
        self.timestamps         = array.array('L', [0]*self.nsmds)
        self.dgram_sizes        = array.array('I', [0]*self.nsmds)
        self.dgram_timestamps   = array.array('L', [0]*self.nsmds)
        self.event_timestamps   = array.array('L', [0]*self.nsmds)
        self.services           = array.array('i', [0]*self.nsmds)
        self.views              = views
        self.configs            = configs
        self.filter_timestamps  = filter_timestamps
        self.nevents            = 0
        self.nsteps             = 0
        self.DGRAM_SIZE         = sizeof(Dgram)
        self.XTC_SIZE           = sizeof(Xtc)
        self.L1Accept           = 12
    
        # Keyword args that need to be passed in once. To save some of
        # them as cpp class attributes, we need to read them in as PyObject*.
        cdef char* kwlist[7]
        kwlist[0] = "batch_size"
        kwlist[1] = "intg_stream_id"                    # Overrides counting every event (got) with counting only events in this stream
        kwlist[2] = "filter_fn"                         # A callback that takes an event and return True/False
        kwlist[3] = "destination"                       # A callback that takes an event and return rank no.
        kwlist[4] = "run"   
        kwlist[5] = "prometheus_counter"
        kwlist[6] = NULL

        if PyArg_ParseTupleAndKeywords(args, kwargs, "|iiOOOO", kwlist, 
                &(self.batch_size),
                &(self.intg_stream_id),
                &(self.filter_fn),
                &(self.destination),
                &(self.run),
                &(self.prometheus_counter)) == False:
            raise RuntimeError, "Invalid kwargs for EventBuilder"
        
    def _has_more(self):
        for i in range(self.nsmds):
            if self.offsets[i] < self.sizes[i]:
                return True
        return False

    def build(self):
        """
        Builds a list of batches.

        Each batch is bytearray with this content:
        [ [[d0][d1][d2][evt_footer]] [[d0][d1][d2][evt_footer]] ][batch_footer]
        | ---------- evt 0 -------------| |------------evt 1 -----------| 
        evt_footer:    [sizeof(d0) | sizeof(d1) | sizeof(d2) | 3] (for 3 dgrams in 1 evt)
        batch_footer:  [sizeof(evt0) | sizeof(evt1) | 2] (for 2 evts in 1 batch)

        """
        # Cast PyObject* to Python object
        filter_fn = <object> self.filter_fn
        destination = <object> self.destination
        run = <object> self.run
        prometheus_counter = <object> self.prometheus_counter

        cdef unsigned got       = 0
        cdef unsigned got_step  = 0
        batch_dict  = {} # keeps list of batches (w/o destination callback, only one batch is returned at index 0)
        step_dict   = {}
        self.min_ts = 0
        self.max_ts = 0

        # Storing python list of bytearray as c pointers
        cdef Dgram* d
        cdef size_t payload = 0
        cdef char* view_ptr
        cdef Py_buffer buf
        cdef list raw_dgrams    = [0] * self.nsmds
        cdef list event_dgrams  = [0] * self.nsmds

        # Setup event footer and batch footer - see above comments for the content of batch_dict
        cdef array.array int_array_template = array.array('I', [])
        cdef array.array evt_footer         = array.clone(int_array_template, self.nsmds + 1, zero=False)
        cdef unsigned evt_footer_size       = sizeof(unsigned) * (self.nsmds + 1)
        cdef unsigned MAX_BATCH_SIZE        = 10000 
        cdef array.array batch_footer       = array.clone(int_array_template, MAX_BATCH_SIZE+1, zero=True)
        cdef array.array step_batch_footer  = array.clone(int_array_template, MAX_BATCH_SIZE+1, zero=True)
        
        # Use typed variables for performance
        cdef unsigned evt_size  = 0
        cdef short dgram_idx    = 0
        cdef short view_idx     = 0
        cdef unsigned evt_idx   = 0
        cdef unsigned aux_idx   = 0

        # For checking step dgrams
        cdef unsigned service           = 0
        cdef unsigned aux_service       = 0
        cdef unsigned long aux_ts       = 0
        cdef unsigned long ts           = 0
        cdef unsigned long aux_min_ts   = 0
        cdef int smd_id                 = 0

        # For filter callback
        cdef int accept = 1 
        
        # For counting if all transtion dgrams show up
        cdef cn_dgrams = 0

        # For counting no. of events in integrating stream. If `intg_stream_id`
        # is not given, this counts no. of events (equivalent to `got`).
        cdef cn_intg_events = 0

        # Set no. of dgrams in an event 
        evt_footer[-1] = self.nsmds

        while cn_intg_events < self.batch_size and self._has_more():
            array.zero(self.timestamps)
            array.zero(self.dgram_sizes)
            array.zero(self.services)
            array.zero(self.event_timestamps)
            service = 0
            cn_dgrams = 0
            
            # Get dgrams and collect their timestamps for all smds, then locate
            # smd_id with the smallest timestamp.
            aux_min_ts = 0
            smd_id = -1 
            for view_idx in range(self.nsmds):
                event_dgrams[view_idx] = 0
                view = self.views[view_idx]
                if self.offsets[view_idx] < self.sizes[view_idx]:
                    # Fill buf with data from memoryview 'view'.
                    PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
                    view_ptr = <char *>buf.buf
                    view_ptr += self.offsets[view_idx]
                    d = <Dgram *>(view_ptr)
                    payload = d.xtc.extent - self.XTC_SIZE
                    aux_ts = <uint64_t>d.seq.high << 32 | d.seq.low
                    aux_service = (d.env>>24)&0xf
                    
                    # check if user selected this timestamp (only applies to L1)
                    if self.filter_timestamps.shape[0] > 0 and aux_service == self.L1Accept:
                        while aux_ts not in self.filter_timestamps and aux_service == self.L1Accept:
                            self.offsets[view_idx] += self.DGRAM_SIZE + payload
                            if self.offsets[view_idx] >= self.sizes[view_idx]:
                                aux_ts = 0
                                break
                            view_ptr += self.DGRAM_SIZE + payload
                            d = <Dgram *>(view_ptr)
                            payload = d.xtc.extent - self.XTC_SIZE
                            aux_ts = <uint64_t>d.seq.high << 32 | d.seq.low
                            aux_service = (d.env>>24)&0xf

                    self.timestamps[view_idx] = aux_ts
                    if aux_ts == 0: # there's nothing for this stream
                        continue

                    self.dgram_sizes[view_idx] = self.DGRAM_SIZE + payload
                    self.services[view_idx] = aux_service
                    raw_dgrams[view_idx] = <char[:self.dgram_sizes[view_idx]]>view_ptr

                    # Check for the smallest timestamp
                    if aux_min_ts == 0:
                        aux_min_ts = self.timestamps[view_idx]
                        smd_id = view_idx
                    else:
                        if self.timestamps[view_idx] < aux_min_ts:
                            aux_min_ts = self.timestamps[view_idx]
                            smd_id = view_idx

                    PyBuffer_Release(&buf)

                # end if self.offsets[view_idx] < ...
            
            # end for view_id in ...

            if smd_id == -1: # nothing matches user's selected timestamps 
                continue

            self.event_timestamps[smd_id] = self.timestamps[smd_id]
            self.timestamps[smd_id] = 0 # prepare for next reload
            self.offsets[smd_id] += self.dgram_sizes[smd_id]
            event_dgrams[smd_id] = raw_dgrams[smd_id] # this is the selected dgram
            service = self.services[smd_id]
            ts = self.event_timestamps[smd_id]
            cn_dgrams += 1
            
            if self.min_ts == 0:
                self.min_ts = self.event_timestamps[smd_id] # records first timestamp
            self.max_ts = self.event_timestamps[smd_id]

            # In other smd views, find matching timestamp dgrams
            for view_idx in range(self.nsmds):
                if view_idx == smd_id or self.offsets[view_idx] >= self.sizes[view_idx]:
                    continue
                
                view = self.views[view_idx]
                event_dgrams[view_idx] = 0
                PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
                view_ptr = <char *>buf.buf
                view_ptr += self.offsets[view_idx]
                d = <Dgram *>(view_ptr)
                aux_ts = <unsigned long>d.seq.high << 32 | d.seq.low
                payload = d.xtc.extent - self.XTC_SIZE
                if aux_ts == self.event_timestamps[smd_id]:
                    # found matching timestamp dgram
                    self.event_timestamps[view_idx] = aux_ts
                    self.timestamps[view_idx] = 0 # prepare for next reload
                    self.offsets[view_idx] += (self.DGRAM_SIZE + payload)
                    event_dgrams[view_idx] = <char[:self.DGRAM_SIZE+payload]>view_ptr
                    cn_dgrams += 1
                PyBuffer_Release(&buf)
            
            # Generate event as bytes from the dgrams
            evt_size = 0
            evt_bytes = bytearray()
            for dgram_idx in range(self.nsmds):
                dgram = event_dgrams[dgram_idx]
                evt_footer[dgram_idx] = 0
                if dgram: 
                    evt_footer[dgram_idx] = dgram.nbytes
                    evt_bytes.extend(bytearray(dgram))
                evt_size += evt_footer[dgram_idx]
            evt_bytes.extend(evt_footer)

            # If destination() is not specifed, use batch 0.
            dest_rank = 0
            accept = 1
            if (filter_fn or destination) and service == self.L1Accept:
                py_evt = Event._from_bytes(self.configs, evt_bytes, run=run) 
                py_evt._complete() 

                if filter_fn:
                    st_filter = time.time()
                    accept = filter_fn(py_evt)
                    en_filter = time.time()
                    if prometheus_counter is not None:
                        prometheus_counter.labels('seconds', 'None').inc(en_filter - st_filter)
                        prometheus_counter.labels('batches', 'None').inc()
                
                if destination:
                    dest_rank = destination(py_evt)
            
            if batch_dict:
                if dest_rank not in batch_dict:
                    batch_dict[dest_rank] = (bytearray(), []) # (events as bytes, event sizes)
            else:
                batch_dict[dest_rank] = (bytearray(), [])
            batch, evt_sizes = batch_dict[dest_rank]

            if step_dict:
                if dest_rank not in step_dict:
                    step_dict[dest_rank] = (bytearray(), [])
            else:
                step_dict[dest_rank] = (bytearray(), [])
            step_batch, step_sizes = step_dict[dest_rank] 

            # Extend this batch bytearray to include this event and collect
            # the size of this event for batch footer.
            if accept == 1:
                batch.extend(evt_bytes)
                evt_sizes.append(evt_size + evt_footer_size)

                # Either counting no. of events normally or counting only
                # events in integrating stream as set by `intg_stream_id`.
                if self.intg_stream_id > -1:
                    if self.event_timestamps[self.intg_stream_id] > 0:
                        cn_intg_events += 1
                else:
                    cn_intg_events += 1
                got += 1
                
                # Add step
                if service != self.L1Accept:
                    step_batch.extend(evt_bytes)
                    step_sizes.append(evt_size + evt_footer_size)
                    got_step += 1

                # Check that all transition dgrams show up in all streams
                if service != self.L1Accept:
                    if cn_dgrams != self.nsmds:
                        print(f'Error: Transtion: {(ts >> 32) & 0xffffffff}.{ts & 0xffffffff} (service:{service}) is missing one or more dgrams (found: {cn_dgrams}/ expected: {self.nsmds})')
                        raise

        # end while got < self.batch_size...
        
        assert got <= MAX_BATCH_SIZE, f"No. of events exceeds maximum allowed (max:{MAX_BATCH_SIZE} got:{got})"
        assert got_step <= MAX_BATCH_SIZE, f"No. of transition events exceeds maximum allowed (max:{MAX_BATCH_SIZE} got:{got_step})"
        self.nevents = got
        self.nsteps = got_step
        
        # Add packet_footer for all events in each batch
        for _, val in batch_dict.items():
            batch, evt_sizes = val
           
            if memoryview(batch).nbytes == 0: continue

            for evt_idx in range(len(evt_sizes)):
                batch_footer[evt_idx] = evt_sizes[evt_idx]
            batch_footer[evt_idx+1] = evt_idx + 1
            batch.extend(batch_footer[:evt_idx+2])

        for _, val in step_dict.items():
            step_batch, step_sizes = val

            if memoryview(step_batch).nbytes ==0: continue

            for evt_idx in range(len(step_sizes)):
                step_batch_footer[evt_idx] = step_sizes[evt_idx]
            step_batch_footer[evt_idx+1] = evt_idx + 1
            step_batch.extend(step_batch_footer[:evt_idx+2])
        
        return batch_dict, step_dict

    @property
    def nevents(self):
        return self.nevents

    @property
    def nsteps(self):
        return self.nsteps

    @property
    def min_ts(self):
        return self.min_ts

    @property
    def max_ts(self):
        return self.max_ts
    
    @property
    def offsets(self):
        return self.offsets

