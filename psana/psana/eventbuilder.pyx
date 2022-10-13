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
from psana.mypybuffer import MyPyBuffer
from psana.psexp import TransitionId
import time
import numpy as np

MAX_BATCH_SIZE = 10000

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
    cdef list mypybufs

    def __init__(self, views, configs,
            *args, **kwargs):
        self.nsmds  = len(views)
        self.configs= configs
        self.nevents= 0
        self.nsteps = 0
    
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

        # Use MyPyBuffer to obtain pointers to all views
        self.mypybufs = []
        for view in views:
            self.mypybufs.append(MyPyBuffer(view))

    def events(self):
        """A generator that yields an smd event.
        Note: Use either this generator or build(). They both call build()
        , which advances the offset of MyPyBuffers' view.
        """
        run = <object> self.run
        # Builds proxy events according to batch_size
        proxy_events = self.build(as_proxy_events=True)
        for proxy_evt in proxy_events:
            py_evt = Event._from_bytes(self.configs, proxy_evt.as_bytearray(), run=run) 
            py_evt._complete() 
            # Smd event created this way will have proxy event set as its attribute.
            # This is so that SmdReaderManager can grab them and build batches/
            # step batches.
            py_evt._proxy_evt = proxy_evt
            yield py_evt

    def has_more(self):
        for mypybuf in self.mypybufs:
            if mypybuf.offset < mypybuf.size:
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
        proxy_evt = ProxyEvent(self.nsmds)
        
        # Cast PyObject* to Python object
        dsparms = <object> self.dsparms
        run = <object> self.run
        prometheus_counter = <object> self.prometheus_counter

        # Get parameters from dsparms
        filter_timestamps = dsparms.timestamps
        filter_fn = dsparms.filter
        destination = dsparms.destination
        
        # Use typed variables for performance
        cdef short view_idx     = 0

        # For checking which smd stream has the smallest timestamp
        cdef uint64_t min_ts    = 0                          
        cdef int smd_id         = -1

        # For filter and filter timestamp callbacks
        cdef int accept = 1, accept_filter_ts = 1
        
        # For counting if all transtion dgrams show up
        cdef cn_dgrams = 0

        # For storing temporary values when calculate either previous or next offsets
        cdef uint64_t next_offset, prev_offset = 0
        
        # For storing timestamps and services as we ask for the next dgram in each view
        cdef array.array timestamps = array.array('L', [0]*self.nsmds)
        cdef array.array dgram_sizes= array.array('I', [0]*self.nsmds)
        cdef array.array services   = array.array('i', [0]*self.nsmds)
        cdef list pydgrams = [0] * self.nsmds

        # Get dgrams and collect their timestamps for all smds, then locate
        # smd_id with the smallest timestamp.
        for view_idx, mypybuf in enumerate(self.mypybufs):
            if mypybuf.offset < mypybuf.size:
                pydg = next(mypybuf.dgrams())
                
                # check if user selected this timestamp (only applies to L1)
                accept_filter_ts = 1
                if filter_timestamps.shape[0] > 0:
                    while pydg.timestamp() not in filter_timestamps and pydg.service() == TransitionId.L1Accept:
                        if mypybuf.offset == mypybuf.size:         # Nothing left, we skipped everything in this view.
                            accept_filter_ts = 0
                            break
                        pydg = next(mypybuf.dgrams())

                # There's nothing for this stream
                if accept_filter_ts == 0:     
                    continue

                timestamps[view_idx] = pydg.timestamp()
                dgram_sizes[view_idx] = pydg.size()
                services[view_idx] = pydg.service()
                pydgrams[view_idx] = pydg

                # Check for the smallest timestamp
                if min_ts == 0:
                    min_ts = timestamps[view_idx]
                    smd_id = view_idx
                else:
                    if timestamps[view_idx] < min_ts:
                        min_ts = timestamps[view_idx]
                        smd_id = view_idx

            # end if mypybuf.offset < ...
        # end for view_id in ...
        
        # Nothing matches user's selected timestamps 
        if smd_id == -1:                                        
            return None 

        # Setup proxy event with its main dgram (smallest ts)
        proxy_evt.set_timestamp(timestamps[smd_id])
        proxy_evt.set_service(services[smd_id])
        proxy_evt.pydgrams[smd_id] = pydgrams[smd_id]
        cn_dgrams += 1
        
        # In other smd views, find matching timestamp dgrams
        for view_idx, mypybuf in enumerate(self.mypybufs): 
            # We use previous offset to check if this view has been used up. Only
            # the selected smd_id gets advanced, therefore its previous offset is
            # its current buffer offset.
            if view_idx == smd_id:
                prev_offset = mypybuf.offset
            else:
                prev_offset = mypybuf.offset - dgram_sizes[view_idx]

            if view_idx == smd_id or prev_offset >= mypybuf.size:
                continue
            
            pydg = mypybuf.dgram
            # Find matching timestamp dgram. If fails, we need to rewind the offset
            # of stream view back one dgram so that the new next call can get the
            # same dgram again.
            if pydg.timestamp() == proxy_evt.timestamp:
                proxy_evt.pydgrams[view_idx] = pydg
                cn_dgrams += 1
            else:
                mypybuf.rewind()

        
        # If destination() is not specifed, use batch 0.
        cdef dest_rank = 0
        if (filter_fn or destination) and proxy_evt.service == TransitionId.L1Accept:
            py_evt = Event._from_bytes(self.configs, proxy_evt.as_bytearray(), run=run) 
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
        
        # Add filterred proxy event to the list 
        if accept == 1:
            proxy_evt.set_destination(dest_rank)
            # For Non L1, check that all dgrams show up
            if  proxy_evt.service != TransitionId.L1Accept:
                if cn_dgrams != self.nsmds:
                    raise
        else: 
            return None

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
                    if proxy_evt.pydgrams[dsparms.intg_stream_id] != 0:
                        cn_intg_events += 1
                else:
                    cn_intg_events += 1
                got += 1
                
                # Counts no. of steps and check that all their dgrams show up
                if proxy_evt.service != TransitionId.L1Accept:
                    got_step += 1
                proxy_events.append(proxy_evt)

        assert got <= MAX_BATCH_SIZE, f"No. of events exceeds maximum allowed (max:{MAX_BATCH_SIZE} got:{got})"
        assert got_step <= MAX_BATCH_SIZE, f"No. of transition events exceeds maximum allowed (max:{MAX_BATCH_SIZE} got:{got_step})"
        self.nevents = got
        self.nsteps = got_step

        if as_proxy_events:
            return proxy_events
        else:
            # Generates bytearray representation in batches (grouped by destination rank id)
            batch_dict, step_dict = self.gen_bytearray_batch(proxy_events)
            return batch_dict, step_dict

    @property
    def nevents(self):
        return self.nevents

    @property
    def nsteps(self):
        return self.nsteps

