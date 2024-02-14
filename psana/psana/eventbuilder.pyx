## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from cpython cimport array
import array
from cpython.object cimport PyObject
from cpython.getargs cimport PyArg_ParseTupleAndKeywords

from dgramlite cimport Xtc, Sequence, Dgram

from libc.stdint cimport uint32_t, uint64_t

from psana.event import Event
from psana.psexp import PacketFooter, TransitionId
from psana import dgram
import time
import numpy as np
from psana.dgramedit import PyDgram
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE

MAX_BATCH_SIZE = 1000000

cdef class ProxyEvent:
    """ EventBuilder uses this class to store event-related info
    while walking through each view buffer. Values such as timestamps, 
    dgram sizes, pointer to the dgrams, are kept here so that they
    can be reused when we build a byte-representation of event batch.
    """
    cdef short nsmds
    cdef list pydgrams
    cdef int destination
    cdef int service
    cdef uint64_t timestamp

    def __init__(self, nsmds):
        self.nsmds = nsmds
        self.pydgrams = [0] * self.nsmds
        self.destination = 0

    @property
    def pydgrams(self):
        return self.pydgrams
    
    def set_service(self, int service):
        self.service = service

    @property
    def service(self):
        return self.service

    def set_timestamp(self, uint64_t ts):
        self.timestamp = ts

    @property
    def timestamp(self):
        return self.timestamp

    def set_destination(self, dest_rank):
        self.destination = dest_rank
    
    @property
    def destination(self):
        return self.destination

    def as_bytearray(self):
        """Generate a bytearray representation of the event.
       
        An event as bytearray has this structure:
        [ [[d0][d1][d2][evt_footer]] [[d0][d1][d2][evt_footer]] ]
        evt_footer:    [sizeof(d0) | sizeof(d1) | sizeof(d2) | 3] (for 3 dgrams in 1 evt)
        """
        # Keep size of all dgrams in an event with the last item storing no. of dgrams
        cdef array.array evt_footer = array.array('I', [0]*(self.nsmds+1))
        evt_footer[-1] = self.nsmds
        evt_bytearray = bytearray()
        cdef int i
        for i, pydg in enumerate(self.pydgrams):
            if pydg == 0: continue
            evt_footer[i] = pydg.size()
            evt_bytearray.extend(bytearray(pydg.as_memoryview()))
        evt_bytearray.extend(evt_footer)
        return evt_bytearray
    
    def as_dgrams(self, configs):
        """ Returns a list of converted PyDgram objects (as dgram.Dgram)."""
        cdef int i
        dgrams = [None]*self.nsmds
        for i, pydg in enumerate(self.pydgrams):
            if pydg == 0: continue
            # FIXME: potentially makes dgram.cc view read-only with PyBUF_CONTIG_RO (currently PyBUF_SIMPLE)
            dgrams[i] = dgram.Dgram(config=configs[i], view=bytearray(pydg.as_memoryview()))
        return dgrams


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
    cdef list configs
    cdef PyObject* dsparms
    cdef PyObject* run
    cdef PyObject* prometheus_counter
    cdef unsigned nevents
    cdef unsigned nsteps
    cdef array.array offsets
    cdef array.array sizes
    cdef array.array timestamps
    cdef array.array dgram_sizes
    cdef array.array services
    cdef list views

    def __init__(self, views, configs,
            *args, **kwargs):
        self.nsmds  = len(views)
        self.configs= configs
        self.nevents= 0
        self.nsteps = 0
        self.views = views

        # Keep offsets and sizes for the input views
        self.offsets = array.array('I', [0]*self.nsmds)
        self.sizes = array.array('I', [memoryview(view).nbytes for view in views])
        
        # Keep dgram data while working along each stream during building step
        self.timestamps = array.array('L', [0]*self.nsmds)
        self.dgram_sizes= array.array('I', [0]*self.nsmds)
        self.services   = array.array('i', [0]*self.nsmds)
    
        # Keyword args that need to be passed in once. To save some of
        # them as cpp class attributes, we need to read them in as PyObject*.
        cdef char* kwlist[4]
        kwlist[0] = "dsparms"
        kwlist[1] = "run"   
        kwlist[2] = "prometheus_counter"
        kwlist[3] = NULL

        if PyArg_ParseTupleAndKeywords(args, kwargs, "|OOO", kwlist, 
                &(self.dsparms),
                &(self.run),
                &(self.prometheus_counter)) == False:
            raise RuntimeError, "Invalid kwargs for EventBuilder"
        
    def events(self):
        """A generator that yields an smd event.
        Note: Use either this generator or build(). They both call build()
        , which advances the offset of MyPyBuffers' view.
        """
        run = <object> self.run
        # Builds proxy events according to batch_size
        proxy_events = self.build(as_proxy_events=True)
        for proxy_evt in proxy_events:
            py_evt = Event(proxy_evt.as_dgrams(self.configs), run=run)
            # Smd event created this way will have proxy event set as its attribute.
            # This is so that SmdReaderManager can grab them and build batches/
            # step batches.
            py_evt._proxy_evt = proxy_evt
            yield py_evt

    def has_more(self):
        cdef int i
        for i in range(self.nsmds):
            if self.offsets[i] < self.sizes[i]:
                return True
        return False

    def gen_bytearray_batch(self, proxy_events, run_serial=False):
        """ Creates and returs batch_dict and step_batch_dict.
        The key of this batch is the destination (rank no.), which is default to 0
        (any rank). Each contain byte representation of event batch understood by
        PacketFooter.
        
        A batch is bytearray with this content:
        | ---------- evt 0 --------| |------------evt 1 --------| batch_footer |
        batch_footer:  [sizeof(evt0) | sizeof(evt1) | 2] (for 2 evts in 1 batch)
        """
        # Setup event footer and batch footer - see above comments for the content of batch_dict
        cdef array.array int_array_template = array.array('I', [])
        cdef array.array batch_footer       = array.clone(int_array_template, MAX_BATCH_SIZE+1, zero=True)
        cdef array.array step_batch_footer  = array.clone(int_array_template, MAX_BATCH_SIZE+1, zero=True)
    
        # Bytearray batch and step batch generation depends on the
        # run types and whether the user has the destination set.
        #
        # RunSerial:
        # Returns batch_dict of all events converted to a bytearray
        # with key 0. No step batch. Destination is irrelevant.
        #
        # RunParallel:
        # By default, bytearray (all events) is generated with key 0
        # for both batch (L1 + transitions) and step batch (transitions only).
        # If destination is set, these events are divided
        # into differnt bytearrays with key = destination number. Step batch
        # for each destination is the same.

        # Get list of all the destinations. If not all 0, then remove 0 key
        # because the 0 key represents the transitions (destination can not be
        # set by users).
        destinations = np.unique([proxy_evt.destination for proxy_evt in proxy_events])
        flag_dest = any(destinations)
        if flag_dest:
            destinations = destinations[destinations != 0]

        if run_serial:
            batch_dict = {0: (bytearray(), [])}
            step_dict = {}
        else:
            batch_dict  = {dest: (bytearray(), []) for dest in destinations} 
            step_dict  = {dest: (bytearray(), []) for dest in destinations} 
        
        for proxy_evt in proxy_events:
            evt_bytearray = proxy_evt.as_bytearray()
            if run_serial:
                batch, evt_sizes = batch_dict[0]
                batch.extend(evt_bytearray)
                evt_sizes.append(memoryview(evt_bytearray).nbytes)
            else:
                if proxy_evt.service != TransitionId.L1Accept:
                    for dest, (step_batch, step_sizes) in step_dict.items():
                        step_batch.extend(evt_bytearray)
                        step_sizes.append(memoryview(evt_bytearray).nbytes)
                    for dest, (batch, evt_sizes) in batch_dict.items():
                        batch.extend(evt_bytearray)
                        evt_sizes.append(memoryview(evt_bytearray).nbytes)
                else:
                    batch, evt_sizes = batch_dict[proxy_evt.destination]
                    batch.extend(evt_bytearray)
                    evt_sizes.append(memoryview(evt_bytearray).nbytes)

        # Add packet_footer for all events in each batch
        cdef int evt_idx = 0
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

    def build_proxy_event(self):
        """ Builds and returns a proxy event (None if filterred)"""
        #t0 = time.perf_counter()
        proxy_evt = ProxyEvent(self.nsmds)
        
        # Use typed variables for performance
        cdef short view_idx     = 0

        # For checking which smd stream has the smallest timestamp
        cdef uint64_t min_ts    = 0                          
        cdef int smd_id         = -1

        # For counting if all transtion dgrams show up
        cdef cn_dgrams = 0

        # Reset dgram data for all streams
        array.zero(self.timestamps)
        array.zero(self.dgram_sizes)
        array.zero(self.services)
        
        cdef list pydgrams = [0] * self.nsmds

        #t1 = time.perf_counter()

        # For retrieving pointers to the input memoryviews
        cdef Py_buffer buf
        cdef char* view_ptr
        cdef Dgram* dg
        cdef uint64_t payload
        
        # Get dgrams and collect their timestamps for all smds, then locate
        # smd_id with the smallest timestamp.
        for view_idx, view in enumerate(self.views):
            if self.offsets[view_idx] < self.sizes[view_idx]:
                # Read a dgram from this view and create PyDgram object
                PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
                view_ptr = <char *>buf.buf
                view_ptr += self.offsets[view_idx]
                dg = <Dgram *>(view_ptr)
                dgram_size = sizeof(Dgram) + (dg.xtc.extent - sizeof(Xtc))
                self.offsets[view_idx] += dgram_size
                pycap_dg = PyCapsule_New(<void *>dg, "dgram", NULL)
                pydg = PyDgram(pycap_dg, dgram_size)
                
                # Save dgram meta data for each stream
                self.timestamps[view_idx] = <uint64_t>dg.seq.high << 32 | dg.seq.low 
                self.dgram_sizes[view_idx] = dgram_size 
                self.services[view_idx] = (dg.env>>24)&0xf 
                pydgrams[view_idx] = pydg

                # Check for the smallest timestamp
                if min_ts == 0:
                    min_ts = self.timestamps[view_idx]
                    smd_id = view_idx
                else:
                    if self.timestamps[view_idx] < min_ts:
                        min_ts = self.timestamps[view_idx]
                        smd_id = view_idx
                PyBuffer_Release(&buf)

            # end if self.offsets[view_idx] ...
        # end for view_id in ...
        #t2 = time.perf_counter()
        
        # Nothing matches user's selected timestamps 
        if smd_id == -1:                                        
            return None 

        # Setup proxy event with its main dgram (smallest ts)
        proxy_evt.set_timestamp(self.timestamps[smd_id])
        proxy_evt.set_service(self.services[smd_id])
        proxy_evt.pydgrams[smd_id] = pydgrams[smd_id]
        cn_dgrams += 1
        #t3 = time.perf_counter()
        
        # In other smd views, find matching timestamp dgrams
        for view_idx in range(self.nsmds):
            # We use previous offset to check if this view has been used up. Only
            # the selected smd_id gets advanced, therefore its previous offset is
            # its current buffer offset.
            if view_idx == smd_id or self.offsets[view_idx] - self.dgram_sizes[view_idx] >= self.sizes[view_idx]:
                continue
            
            # Find matching timestamp dgram. If fails, we need to rewind the offset
            # of stream view back one dgram so that the new next call can get the
            # same dgram again.
            if self.timestamps[view_idx] == proxy_evt.timestamp:
                proxy_evt.pydgrams[view_idx] = pydgrams[view_idx]
                cn_dgrams += 1
            else:
                self.offsets[view_idx] -= self.dgram_sizes[view_idx]
        
        #t4 = time.perf_counter()

        # For Non L1, check that all dgrams show up
        if  proxy_evt.service != TransitionId.L1Accept and cn_dgrams != self.nsmds:
            msg = f'TransitionId {proxy_evt.service} incomplete (ts:{proxy_evt.timestamp}) expected:{self.nsmds} received:{cn_dgrams}'
            raise RuntimeError(msg)
        
        #t5 = time.perf_counter()
        #print(f'svr:{proxy_evt.service} total (micro-sec): {(t4-t0)*1e6:.2f} init:{(t1-t0)*1e6:.2f} loop1:{(t2-t1)*1e6:.2f} set:{(t3-t2)*1e6:.2f} loop2:{(t4-t3)*1e6:.2f} fin:{(t5-t4)*1e6:.2f}')
        return proxy_evt

    def build(self, as_proxy_events=False):
        """ Build proxy events according to batch size.
        
        Input: 
        as_proxy_events: set this to skip creating event and step batches

        Output:
        proxy_events: a list of proxy events (as_proxy_events=True)
        batch_dict, step_dict: batches of events with destination 
                               rank id as key
        """
        # Grab dsparms (cast PyObject* to Python object)
        dsparms = <object> self.dsparms
        
        # For counting no. of events. If `intg_stream_id`
        # is not given, this counts no. of events (equivalent to `got`).
        cdef unsigned got       = 0
        cdef unsigned got_step  = 0
        cdef unsigned cn_intg_events = 0

        # Keeping all built proxy event
        proxy_events = []

        while cn_intg_events < dsparms.batch_size and self.has_more():
            proxy_evt = self.build_proxy_event()
            if proxy_evt is not None:
                # Either counting no. of events normally or counting only
                # events in integrating stream as set by `intg_stream_id`.
                if dsparms.intg_stream_id > -1:
                    if proxy_evt.pydgrams[dsparms.intg_stream_id] != 0 and proxy_evt.service != TransitionId.SlowUpdate:
                        cn_intg_events += 1
                else:
                    cn_intg_events += 1
                
                # Counts no. of steps 
                if proxy_evt.service != TransitionId.L1Accept:
                    got_step += 1
                
                proxy_events.append(proxy_evt)
                got += 1

        assert got <= MAX_BATCH_SIZE, f"No. of events exceeds maximum allowed (max:{MAX_BATCH_SIZE} got:{got}). For integrating detector, lower the value of PS_SMD_N_EVENTS."
        assert got_step <= MAX_BATCH_SIZE, f"No. of transition events exceeds maximum allowed (max:{MAX_BATCH_SIZE} got:{got_step})"
        self.nevents = got
        self.nsteps = got_step

        # Eiter return the proxy_events (smd_callback) or bytearrays (grouped by destination)
        if as_proxy_events:
            return proxy_events
        else:
            batch_dict, step_dict = self.gen_bytearray_batch(proxy_events)
            return batch_dict, step_dict

    @property
    def nevents(self):
        return self.nevents

    @property
    def nsteps(self):
        return self.nsteps

