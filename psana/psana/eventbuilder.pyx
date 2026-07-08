## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1
from cpython cimport array
import array

from cpython.getargs cimport PyArg_ParseTupleAndKeywords
from cpython.object cimport PyObject
from psana.dgramlite cimport Dgram, Xtc
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t

import os
import sys
from time import perf_counter

import numpy as np

from psana import dgram
from psana.dgramedit import PyDgram
from psana.psexp import TransitionId

PROFILE_ENV = "PSANA_EB_PROFILE"
PROFILE_INTERVAL_ENV = "PSANA_EB_PROFILE_INTERVAL"
DEFAULT_PROFILE_INTERVAL = 10

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
    cdef list _pydgram_pool
    cdef bint _use_proxy_events
    cdef list _scratch_pydgrams
    cdef array.array _event_footer

    # Profiling helpers (enabled via PSANA_EB_PROFILE=1)
    cdef bint _profile_enabled
    cdef unsigned long _profile_emit_every
    cdef unsigned long _profile_batch_calls
    cdef unsigned long _profile_proxy_events
    cdef unsigned long _profile_batches_until_emit
    cdef double _profile_time_total
    cdef double _profile_time_gather
    cdef double _profile_time_gen_batch
    cdef double _profile_time_filter
    cdef double _profile_time_destinations
    cdef double _profile_time_serialize

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
        self._pydgram_pool = []

        # Parse kwargs: timestamps (O), intg_stream_id (i), batch_size (i), use_proxy_events (i)
        cdef char* kwlist[5]
        cdef int use_proxy_flag = 1
        kwlist[0] = "filter_timestamps"
        kwlist[1] = "intg_stream_id"
        kwlist[2] = "batch_size"
        kwlist[3] = "use_proxy_events"
        kwlist[4] = NULL

        # NOTE: args must be empty here; we accept kwargs only for these three
        if PyArg_ParseTupleAndKeywords(args, kwargs, "|Oiii",
                                    kwlist,
                                    &(self.filter_timestamps_obj),
                                    &(self.intg_stream_id),
                                    &(self.batch_size),
                                    &use_proxy_flag) is False:
            raise RuntimeError("Invalid kwargs for EventBuilder (expected: filter_timestamps, intg_stream_id, batch_size)")

        self._use_proxy_events = bool(use_proxy_flag)
        self._init_profile()
        self._scratch_pydgrams = [0] * self.nsmds
        self._event_footer = array.array('I', [0] * (self.nsmds + 1))

    cdef void _init_profile(self):
        cdef object env_val = os.environ.get(PROFILE_ENV)
        self._profile_enabled = bool(env_val and env_val != "0")
        if not self._profile_enabled:
            self._profile_emit_every = 0
            self._profile_batches_until_emit = 0
            self._reset_profile_counters()
            return

        cdef int interval
        try:
            interval = int(os.environ.get(PROFILE_INTERVAL_ENV, str(DEFAULT_PROFILE_INTERVAL)))
        except ValueError:
            interval = DEFAULT_PROFILE_INTERVAL

        if interval <= 0:
            interval = 1

        self._profile_emit_every = interval
        self._profile_batches_until_emit = interval
        self._reset_profile_counters()

    cdef void _reset_profile_counters(self):
        self._profile_batch_calls = 0
        self._profile_proxy_events = 0
        self._profile_time_total = 0.0
        self._profile_time_gather = 0.0
        self._profile_time_gen_batch = 0.0
        self._profile_time_filter = 0.0
        self._profile_time_destinations = 0.0
        self._profile_time_serialize = 0.0

    cdef void _profile_after_build(self, double total_time, bint force_flush):
        if not self._profile_enabled:
            return

        self._profile_time_total += total_time
        self._profile_batch_calls += 1
        self._profile_batches_until_emit -= 1

        if force_flush or self._profile_batches_until_emit <= 0:
            self._emit_profile()
            self._profile_batches_until_emit = self._profile_emit_every
            self._reset_profile_counters()

    cdef void _emit_profile(self):
        if self._profile_batch_calls == 0:
            return

        cdef double events = self._profile_proxy_events if self._profile_proxy_events > 0 else 1.0
        msg = (
            f"[EventBuilder pid={os.getpid()} nsmds={self.nsmds}] "
            f"batches={self._profile_batch_calls} events={self._profile_proxy_events} "
            f"gather={self._profile_time_gather:.6f}s "
            f"serialize={self._profile_time_serialize:.6f}s "
            f"destinations={self._profile_time_destinations:.6f}s "
            f"filter={self._profile_time_filter:.6f}s "
            f"gen_batch={self._profile_time_gen_batch:.6f}s "
            f"total_build={self._profile_time_total:.6f}s "
            f"avg_gather={self._profile_time_gather/events:.6e}s"
        )
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()

    cdef tuple _collect_destinations(self, list proxy_events):
        if not proxy_events:
            return (tuple(), False)
        cdef set dests = set()
        cdef bint has_explicit = False
        cdef object proxy_evt
        cdef int dest

        for proxy_evt in proxy_events:
            dest = proxy_evt.destination
            if dest != 0:
                has_explicit = True
                dests.add(dest)

        if has_explicit:
            return (tuple(sorted(dests)), True)
        else:
            return ((0,), False)

    cdef object _acquire_pydgram(self, Dgram* dg_ptr, uint64_t dgram_size):
        cdef object pydg
        if self._pydgram_pool:
            pydg = self._pydgram_pool.pop()
            pydg.reset_from_ptr(<size_t><void*>dg_ptr, dgram_size)
            return pydg
        pycap_dg = PyCapsule_New(<void *>dg_ptr, "dgram", NULL)
        return PyDgram(pycap_dg, dgram_size)

    cdef void _release_pydgram(self, object pydg):
        if pydg != 0:
            self._pydgram_pool.append(pydg)

    cdef void _release_pydgram_list(self, list pydgrams):
        cdef Py_ssize_t idx
        cdef object obj
        for idx in range(len(pydgrams)):
            obj = pydgrams[idx]
            if obj != 0:
                self._pydgram_pool.append(obj)
                pydgrams[idx] = 0

    cdef void _recycle_proxy_event(self, ProxyEvent proxy_evt):
        cdef int idx
        cdef object pydg
        for idx in range(proxy_evt.nsmds):
            pydg = proxy_evt.pydgrams[idx]
            if pydg != 0:
                self._pydgram_pool.append(pydg)
                proxy_evt.pydgrams[idx] = 0

    cdef int _gather_event(self, list pydgrams, short* out_service, uint64_t* out_timestamp):
        """Populate pydgrams with the next aligned event; return number of matching dgrams."""
        cdef short view_idx
        cdef uint64_t min_ts = 0
        cdef int smd_id = -1
        cdef int cn_dgrams = 0
        cdef Py_buffer buf
        cdef char* view_ptr
        cdef Dgram* dg
        cdef size_t dgram_size

        array.zero(self.timestamps)
        array.zero(self.dgram_sizes)
        array.zero(self.services)

        for view_idx, view in enumerate(self.views):
            if self.offsets[view_idx] < self.sizes[view_idx]:
                PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
                view_ptr = <char *>buf.buf
                view_ptr += self.offsets[view_idx]
                dg = <Dgram *>view_ptr
                dgram_size = sizeof(Dgram) + (dg.xtc.extent - sizeof(Xtc))
                self.offsets[view_idx] += dgram_size
                pydgrams[view_idx] = self._acquire_pydgram(dg, dgram_size)
                self.timestamps[view_idx] = <uint64_t>dg.seq.high << 32 | dg.seq.low
                self.dgram_sizes[view_idx] = dgram_size
                self.services[view_idx] = (dg.env >> 24) & 0xf
                if min_ts == 0 or self.timestamps[view_idx] < min_ts:
                    min_ts = self.timestamps[view_idx]
                    smd_id = view_idx
                PyBuffer_Release(&buf)

        if smd_id == -1:
            self._release_pydgram_list(pydgrams)
            return 0

        out_timestamp[0] = self.timestamps[smd_id]
        out_service[0] = self.services[smd_id]
        cn_dgrams = 1

        for view_idx in range(self.nsmds):
            if (
                view_idx == smd_id
                or self.offsets[view_idx] - self.dgram_sizes[view_idx] >= self.sizes[view_idx]
            ):
                continue

            if self.timestamps[view_idx] == out_timestamp[0]:
                cn_dgrams += 1
            else:
                self.offsets[view_idx] -= self.dgram_sizes[view_idx]
                self._release_pydgram(pydgrams[view_idx])
                pydgrams[view_idx] = 0

        if not TransitionId.isEvent(out_service[0]) and cn_dgrams != self.nsmds:
            self._release_pydgram_list(pydgrams)
            msg = (
                f'TransitionId {TransitionId.name(out_service[0])} incomplete '
                f'(ts:{out_timestamp[0]}) expected:{self.nsmds} received:{cn_dgrams}'
            )
            raise RuntimeError(msg)

        return cn_dgrams

    cdef tuple _event_to_bytearray(self, list pydgrams):
        """Return (bytearray, size) for the provided PyDgram list, releasing them afterward."""
        cdef bytearray evt = bytearray()
        cdef array.array footer = self._event_footer
        cdef Py_ssize_t i
        cdef object pydg
        for i in range(self.nsmds):
            pydg = pydgrams[i]
            if pydg == 0:
                footer[i] = 0
                continue
            footer[i] = pydg.size()
            evt.extend(bytearray(pydg.as_memoryview()))
            self._release_pydgram(pydg)
            pydgrams[i] = 0
        footer[self.nsmds] = self.nsmds
        evt.extend(footer)
        return evt, len(evt)

    cdef void _append_packet_footer(self, bytearray batch, list evt_sizes):
        if len(batch) == 0:
            return
        cdef array.array int_array_template = array.array('I', [])
        cdef array.array footer = array.clone(int_array_template, MAX_BATCH_SIZE + 1, zero=True)
        cdef Py_ssize_t evt_idx
        for evt_idx in range(len(evt_sizes)):
            footer[evt_idx] = evt_sizes[evt_idx]
        footer[len(evt_sizes)] = len(evt_sizes)
        batch.extend(footer[:len(evt_sizes) + 1])

    cdef tuple _build_fast_batch(self, unsigned target_batch):
        cdef bytearray batch = bytearray()
        cdef bytearray step_batch = bytearray()
        cdef list evt_sizes = []
        cdef list step_sizes = []
        cdef short service = 0
        cdef uint64_t timestamp = 0
        cdef int matched = 0
        cdef object evt_bytearray
        cdef Py_ssize_t evt_size
        cdef unsigned got = 0
        cdef unsigned got_step = 0
        cdef double t0

        while got < target_batch and self.has_more():
            if self._profile_enabled:
                t0 = perf_counter()
            matched = self._gather_event(self._scratch_pydgrams, &service, &timestamp)
            if self._profile_enabled:
                self._profile_time_gather += perf_counter() - t0
            if matched == 0:
                break
            if self._profile_enabled:
                self._profile_proxy_events += 1
                t0 = perf_counter()
            evt_bytearray, evt_size = self._event_to_bytearray(self._scratch_pydgrams)
            if self._profile_enabled:
                self._profile_time_serialize += perf_counter() - t0
            if not TransitionId.isEvent(service):
                got_step += 1
                step_batch.extend(evt_bytearray)
                step_sizes.append(evt_size)
            batch.extend(evt_bytearray)
            evt_sizes.append(evt_size)
            got += 1

        self.nevents = got
        self.nsteps = got_step

        cdef double t_footer
        if self._profile_enabled:
            t_footer = perf_counter()
        self._append_packet_footer(batch, evt_sizes)
        self._append_packet_footer(step_batch, step_sizes)
        if self._profile_enabled:
            self._profile_time_gen_batch += perf_counter() - t_footer

        return {0: (batch, evt_sizes)}, {0: (step_batch, step_sizes)}

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

        # Collect destination ids without relying on numpy allocations.
        if self._profile_enabled:
            t0 = perf_counter()
        destinations, _ = self._collect_destinations(proxy_events)
        if self._profile_enabled:
            self._profile_time_destinations += perf_counter() - t0

        if run_serial:
            batch_dict = {0: (bytearray(), [])}
            step_dict = {}
        else:
            batch_dict  = {dest: (bytearray(), []) for dest in destinations}
            step_dict  = {dest: (bytearray(), []) for dest in destinations}

        for proxy_evt in proxy_events:
            if self._profile_enabled:
                t0 = perf_counter()
            evt_bytearray = proxy_evt.as_bytearray()
            if self._profile_enabled:
                self._profile_time_serialize += perf_counter() - t0

            if run_serial:
                batch, evt_sizes = batch_dict[0]
                batch.extend(evt_bytearray)
                evt_sizes.append(len(evt_bytearray))
            else:
                if not TransitionId.isEvent(proxy_evt.service):
                    for dest, (step_batch, step_sizes) in step_dict.items():
                        step_batch.extend(evt_bytearray)
                        step_sizes.append(len(evt_bytearray))
                    for dest, (batch, evt_sizes) in batch_dict.items():
                        batch.extend(evt_bytearray)
                        evt_sizes.append(len(evt_bytearray))
                else:
                    batch, evt_sizes = batch_dict[proxy_evt.destination]
                    batch.extend(evt_bytearray)
                    evt_sizes.append(len(evt_bytearray))
            self._recycle_proxy_event(proxy_evt)

        # Add packet_footer for all events in each batch
        cdef int evt_idx = 0
        for _, val in batch_dict.items():
            batch, evt_sizes = val

            if len(batch) == 0:
                continue

            for evt_idx in range(len(evt_sizes)):
                batch_footer[evt_idx] = evt_sizes[evt_idx]
            batch_footer[evt_idx+1] = evt_idx + 1
            batch.extend(batch_footer[:evt_idx+2])

        for _, val in step_dict.items():
            step_batch, step_sizes = val

            if len(step_batch) == 0:
                continue

            for evt_idx in range(len(step_sizes)):
                step_batch_footer[evt_idx] = step_sizes[evt_idx]
            step_batch_footer[evt_idx+1] = evt_idx + 1
            step_batch.extend(step_batch_footer[:evt_idx+2])

        return batch_dict, step_dict

    def build_proxy_event(self):
        """ Builds and returns a proxy event (None if filterred)"""
        proxy_evt = ProxyEvent(self.nsmds)
        cdef short service = 0
        cdef uint64_t timestamp = 0
        if self._gather_event(proxy_evt.pydgrams, &service, &timestamp) == 0:
            return None
        proxy_evt.set_service(service)
        proxy_evt.set_timestamp(timestamp)
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
        if intg_stream_id > -1:
            target_batch = MAX_BATCH_SIZE  # integrating mode ignores user batch_size cap

        # Counters
        cdef unsigned got      = 0
        cdef unsigned got_step = 0

        # Accumulators
        proxy_events    = []
        non_L1_indices  = []
        cdef double t_total = 0.0
        cdef double t0 = 0.0

        if self._profile_enabled:
            t_total = perf_counter()

        if (
            not self._use_proxy_events
            and not as_proxy_events
            and filter_timestamps.shape[0] == 0
        ):
            batch_dict, step_dict = self._build_fast_batch(target_batch)
            if self._profile_enabled:
                self._profile_after_build(perf_counter() - t_total, self.nevents == 0 and self.nsteps == 0)
            return batch_dict, step_dict

        # Build loop
        while got < target_batch and self.has_more():
            if self._profile_enabled:
                t0 = perf_counter()
            proxy_evt = self.build_proxy_event()
            if self._profile_enabled:
                self._profile_time_gather += perf_counter() - t0

            if proxy_evt is not None:
                if not TransitionId.isEvent(proxy_evt.service):
                    got_step += 1
                    if filter_timestamps.shape[0] > 0:
                        non_L1_indices.append(got)
                proxy_events.append(proxy_evt)
                got += 1
                if self._profile_enabled:
                    self._profile_proxy_events += 1

        assert got      <= MAX_BATCH_SIZE, "No. of events exceeds maximum allowed"
        assert got_step <= MAX_BATCH_SIZE, "No. of transition events exceeds maximum allowed"
        self.nevents = got
        self.nsteps  = got_step

        # Timestamp filtering (keep non-L1 unfiltered)
        cdef int i, ia, ib
        if filter_timestamps.shape[0] and len(proxy_events) > 0:
            if self._profile_enabled:
                t0 = perf_counter()
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

            keep_indices = sorted(list(found_indices) + non_L1_indices)
            if len(keep_indices) < len(proxy_events):
                keep_mask = [False] * len(proxy_events)
                for idx in keep_indices:
                    if 0 <= idx < len(keep_mask):
                        keep_mask[idx] = True
                for idx, evt in enumerate(proxy_events):
                    if idx >= len(keep_mask) or not keep_mask[idx]:
                        self._recycle_proxy_event(evt)
            proxy_events = [proxy_events[i] for i in keep_indices]
            if self._profile_enabled:
                self._profile_time_filter += perf_counter() - t0

        if as_proxy_events:
            if self._profile_enabled:
                self._profile_after_build(perf_counter() - t_total, self.nevents == 0 and self.nsteps == 0)
            return proxy_events
        else:
            if self._profile_enabled:
                t0 = perf_counter()
            batch_dict, step_dict = self.gen_bytearray_batch(proxy_events)
            if self._profile_enabled:
                self._profile_time_gen_batch += perf_counter() - t0
                self._profile_after_build(perf_counter() - t_total, self.nevents == 0 and self.nsteps == 0)
            return batch_dict, step_dict

    @property
    def nevents(self):
        return self.nevents

    @property
    def nsteps(self):
        return self.nsteps
