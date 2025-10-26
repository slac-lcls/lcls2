## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1
from cpython cimport array
import array

from cpython.getargs cimport PyArg_ParseTupleAndKeywords
from cpython.object cimport PyObject
from dgramlite cimport Dgram, Xtc
from libc.stdint cimport uint64_t

import numpy as np

from psana import dgram
from psana.dgramedit import PyDgram
from psana.psexp import TransitionId

from cpython.buffer cimport (PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE,
                             PyBuffer_Release, PyObject_GetBuffer)
from cpython.pycapsule cimport PyCapsule_New

MAX_BATCH_SIZE = 1000000

cdef class ProxyEvent:
    """
    Lightweight container for one SMD 'event' during batching.
    Holds per-stream PyDgram pointers and minimal metadata.
    """
    cdef short      nsmds
    cdef list       _pydgrams      # list[PyDgram or 0], length=nsmds
    cdef int        _destination   # routing hint (rank id), default 0
    cdef int        _service       # TransitionId code
    cdef uint64_t   _timestamp     # combined (high<<32 | low)

    def __init__(self, short nsmds):
        self.nsmds       = nsmds
        self._pydgrams   = [0] * nsmds
        self._destination = 0
        self._service     = 0
        self._timestamp   = 0

    # -------------------------
    # Properties / setters
    # -------------------------
    @property
    def pydgrams(self):
        """List of PyDgram handles (or 0 where missing), length == nsmds."""
        return self._pydgrams

    def set_service(self, int service):
        self._service = service

    @property
    def service(self):
        return self._service

    def set_timestamp(self, uint64_t ts):
        self._timestamp = ts

    @property
    def timestamp(self):
        return self._timestamp

    def set_destination(self, int dest_rank):
        self._destination = dest_rank

    @property
    def destination(self):
        return self._destination

    # -------------------------
    # Serialization helpers
    # -------------------------
    def as_bytearray(self):
        """
        Return a bytearray representation of the event:
        [ d0 | d1 | ... | footer ]
        where footer is an array('I') of per-dgram sizes plus count at the end.
        """
        cdef array.array evt_footer = array.array('I', [0] * (self.nsmds + 1))
        evt_footer[-1] = self.nsmds

        cdef bytearray out = bytearray()
        cdef int i
        cdef object pydg
        for i, pydg in enumerate(self._pydgrams):
            if pydg == 0:
                continue
            evt_footer[i] = pydg.size()
            out.extend(bytearray(pydg.as_memoryview()))
        out.extend(evt_footer)
        return out

    def as_dgrams(self, configs):
        """
        Convert stored PyDgram handles to Python dgram.Dgram objects using
        the provided per-stream configs.
        """
        cdef int i
        cdef object pydg
        dgrams = [None] * self.nsmds
        for i, pydg in enumerate(self._pydgrams):
            if pydg == 0:
                continue
            # Make a copy view for safety (matches existing behavior)
            dgrams[i] = dgram.Dgram(config=configs[i],
                                    view=bytearray(pydg.as_memoryview()))
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
    Without destination call back the build fn returns a batch of events
    (size = batch_size) at index 0. With destination call back, this fn
    returns list of batches. Each batch has the same destination rank.

    Note that reading chunks inside a views or events inside a batch can be done
    using PacketFooter class."""
    cdef short nsmds
    cdef list configs
    cdef unsigned nevents
    cdef unsigned nsteps
    cdef array.array offsets
    cdef array.array sizes
    cdef array.array timestamps
    cdef array.array dgram_sizes
    cdef array.array services
    cdef list views

    cdef PyObject* filter_timestamps_obj   # holds Python object (array-like) for filter timestamps
    cdef int intg_stream_id
    cdef int batch_size

    def __init__(self, views, configs, *args, **kwargs):
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

        # Parse kwargs: timestamps (O), intg_stream_id (i), batch_size (i)
        cdef char* kwlist[4]
        kwlist[0] = "filter_timestamps"
        kwlist[1] = "intg_stream_id"
        kwlist[2] = "batch_size"
        kwlist[3] = NULL

        # NOTE: args must be empty here; we accept kwargs only for these three
        if PyArg_ParseTupleAndKeywords(args, kwargs, "|Oii",
                                    kwlist,
                                    &(self.filter_timestamps_obj),
                                    &(self.intg_stream_id),
                                    &(self.batch_size)) is False:
            raise RuntimeError("Invalid kwargs for EventBuilder (expected: filter_timestamps, intg_stream_id, batch_size)")

    def events(self):
        """A generator that yields a list of dgrams for each event.
        Note: Use either this generator or build(); both advance the view offsets.
        """
        # Builds proxy events according to batch_size
        proxy_events = self.build(as_proxy_events=True)
        for proxy_evt in proxy_events:
            # Smd event created this way will have proxy event set as its attribute.
            # This is so that SmdReaderManager can grab them and build batches/
            # step batches.
            dgrams = proxy_evt.as_dgrams(self.configs)
            # Hand the lightweight proxy out so Python can batch/build as needed.
            # (Callers may attach it on their Event wrapper if desired.)
            yield (dgrams, proxy_evt)

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
                if not TransitionId.isEvent(proxy_evt.service):
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

            if memoryview(batch).nbytes == 0:
                continue

            for evt_idx in range(len(evt_sizes)):
                batch_footer[evt_idx] = evt_sizes[evt_idx]
            batch_footer[evt_idx+1] = evt_idx + 1
            batch.extend(batch_footer[:evt_idx+2])

        for _, val in step_dict.items():
            step_batch, step_sizes = val

            if memoryview(step_batch).nbytes ==0:
                continue

            for evt_idx in range(len(step_sizes)):
                step_batch_footer[evt_idx] = step_sizes[evt_idx]
            step_batch_footer[evt_idx+1] = evt_idx + 1
            step_batch.extend(step_batch_footer[:evt_idx+2])

        return batch_dict, step_dict

    def build_proxy_event(self):
        """ Builds and returns a proxy event (None if filterred)"""
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

        # For retrieving pointers to the input memoryviews
        cdef Py_buffer buf
        cdef char* view_ptr
        cdef Dgram* dg

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

        # Nothing matches user's selected timestamps
        if smd_id == -1:
            return None

        # Setup proxy event with its main dgram (smallest ts)
        proxy_evt.set_timestamp(self.timestamps[smd_id])
        proxy_evt.set_service(self.services[smd_id])
        proxy_evt.pydgrams[smd_id] = pydgrams[smd_id]
        cn_dgrams += 1

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

        # For Non L1, check that all dgrams show up
        if not TransitionId.isEvent(proxy_evt.service) and cn_dgrams != self.nsmds:
            msg = f'TransitionId {TransitionId.name(proxy_evt.service)} incomplete (ts:{proxy_evt.timestamp}) expected:{self.nsmds} received:{cn_dgrams}'
            raise RuntimeError(msg)
        return proxy_evt

    def build(self, as_proxy_events=False):
        """
        Build proxy events according to batch size or integrating mode.

        Input:
        as_proxy_events: if True, return list[ProxyEvent]; otherwise return
                        (batch_dict, step_dict)

        Uses:
        - self.intg_stream_id: integrating detector stream id (-1 means disabled)
        - self.batch_size: target batch size when not integrating
        - self.filter_timestamps_obj: array-like of uint64 timestamps to keep
        """
        # Prepare filter timestamps (if provided)
        filter_timestamps = <object> self.filter_timestamps_obj

        cdef int intg_stream_id = self.intg_stream_id
        cdef unsigned target_batch = MAX_BATCH_SIZE if self.batch_size <= 0 else self.batch_size

        # Counters
        cdef unsigned got      = 0
        cdef unsigned got_step = 0

        # Accumulators
        proxy_events    = []
        non_L1_indices  = []

        # Build loop
        if intg_stream_id > -1:
            # Integrating mode: consume as much as available
            while self.has_more():
                proxy_evt = self.build_proxy_event()
                if proxy_evt is not None:
                    if not TransitionId.isEvent(proxy_evt.service):
                        got_step += 1
                        if filter_timestamps.shape[0] > 0:
                            non_L1_indices.append(got)
                    proxy_events.append(proxy_evt)
                    got += 1
        else:
            # Batch mode
            while got < target_batch and self.has_more():
                proxy_evt = self.build_proxy_event()
                if proxy_evt is not None:
                    if not TransitionId.isEvent(proxy_evt.service):
                        got_step += 1
                        if filter_timestamps.shape[0] > 0:
                            non_L1_indices.append(got)
                    proxy_events.append(proxy_evt)
                    got += 1

        assert got      <= MAX_BATCH_SIZE, "No. of events exceeds maximum allowed"
        assert got_step <= MAX_BATCH_SIZE, "No. of transition events exceeds maximum allowed"
        self.nevents = got
        self.nsteps  = got_step

        # Timestamp filtering (keep non-L1 unfiltered)
        cdef int i, ia, ib
        if filter_timestamps.shape[0] and len(proxy_events) > 0:
            timestamps = np.asarray([proxy_evt.timestamp for proxy_evt in proxy_events], dtype=np.uint64)
            insert_indices = np.searchsorted(filter_timestamps, timestamps)

            found_indices = []
            found_insert_index = -1
            for ia in range(insert_indices.shape[0]-1, -1, -1):
                ib = insert_indices[ia]
                if ib == found_insert_index or ib == filter_timestamps.shape[0]:
                    continue
                if timestamps[ia] == filter_timestamps[ib]:
                    found_indices.append(ia)
                    found_insert_index = ib

            _proxy_events = [proxy_events[i] for i in sorted(list(found_indices) + non_L1_indices)]
            proxy_events = _proxy_events

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
