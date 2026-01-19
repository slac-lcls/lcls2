# cython: language_level=3, boundscheck=False
from cpython cimport array
import array

from cpython.buffer cimport (
    PyBUF_ANY_CONTIGUOUS,
    PyBUF_SIMPLE,
    PyBuffer_Release,
    PyObject_GetBuffer,
    Py_buffer,
)
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t

from psana.dgramlite cimport Dgram, Xtc

import numpy as np
import time

from dataclasses import dataclass
from typing import Sequence

from psana import dgram, utils
from psana.psexp import TransitionId
from psana.psexp.packet_footer import PacketFooter
from psana.psexp.mpi_shmem import MPISharedMemory
from psana.psexp.tools import mode

if mode == "mpi":
    from mpi4py import MPI
else:
    MPI = None


@dataclass
class ParsedChunk:
    """Container for arrays derived from one SMD chunk."""

    n_events: int
    bd_offsets: np.ndarray
    bd_sizes: np.ndarray
    smd_offsets: np.ndarray
    smd_sizes: np.ndarray
    services: np.ndarray
    cutoff_flags: np.ndarray
    new_chunk_ids: np.ndarray


cdef class MarchingEventBuilder:
    """Prepare marching-read work slots from SMD chunks."""

    STATE_EMPTY = 0
    STATE_FILLING = 1
    STATE_READY = 2

    cdef:
        object configs
        object dsparms
        object shared_mem
        Py_ssize_t n_slots
        Py_ssize_t max_events
        Py_ssize_t max_chunk_bytes
        Py_ssize_t n_streams
        object logger
        object prefix
        object chunkinfo
        Py_ssize_t _next_slot_hint
        object use_smds
        object latest_chunk_ids
        object _views
        object _offsets
        object _sizes
        object _dgram_sizes
        object _services_arr
        object _timestamps
        object _stream_bases
        object _event_offsets
        object _event_sizes
        object _event_services
        object _event_cutoffs
        uint64_t _event_timestamp
        object bd_offsets
        object bd_sizes
        object smd_offsets
        object smd_sizes
        object services
        object cutoff_flags
        object new_chunk_ids
        object slot_events
        object slot_chunk_ids
        object slot_states
        object slot_next_event
        object slot_consumers_done
        object smd_chunk_sizes
        object smd_chunk_buffer
        object shutdown_flag
        object _dbg_l1_counts
        object _dbg_trans_counts
        int _rank

    def __init__(
        self,
        configs: Sequence,
        dsparms,
        shared_mem: MPISharedMemory,
        *,
        n_slots: int,
        max_events_per_chunk: int,
        max_chunk_bytes: int,
        name_prefix: str = "march",
    ):
        self.configs = list(configs)
        self.dsparms = dsparms
        self.shared_mem = shared_mem
        self.n_slots = n_slots
        self.max_events = max_events_per_chunk
        self.max_chunk_bytes = max_chunk_bytes
        self.n_streams = len(self.configs)
        self.logger = utils.get_logger(name=utils.get_class_name(self))
        if MPI is not None:
            try:
                self._rank = MPI.COMM_WORLD.Get_rank()
            except Exception:
                self._rank = 0
        else:
            self._rank = 0
        self.prefix = name_prefix
        self.chunkinfo = {}
        self._next_slot_hint = 0
        use_smds = getattr(dsparms, "use_smds", None)
        if not use_smds or len(use_smds) != self.n_streams:
            self.use_smds = np.zeros(self.n_streams, dtype=bool)
        else:
            self.use_smds = np.asarray(use_smds, dtype=bool)
        self.latest_chunk_ids = np.zeros(self.n_streams, dtype=np.int64)
        self._init_stream_state()
        self._allocate_shared_buffers()

    def _init_stream_state(self) -> None:
        self._views = [None] * self.n_streams
        self._offsets = array.array("Q", [0] * self.n_streams)
        self._sizes = array.array("Q", [0] * self.n_streams)
        self._dgram_sizes = array.array("Q", [0] * self.n_streams)
        self._services_arr = array.array("i", [0] * self.n_streams)
        self._timestamps = array.array("Q", [0] * self.n_streams)
        self._stream_bases = array.array("Q", [0] * self.n_streams)
        self._event_offsets = array.array("q", [-1] * self.n_streams)
        self._event_sizes = array.array("Q", [0] * self.n_streams)
        self._event_services = array.array("i", [0] * self.n_streams)
        self._event_cutoffs = array.array("b", [0] * self.n_streams)
        self._event_timestamp = 0

    def _allocate_shared_buffers(self) -> None:
        slot_shape = (self.n_slots, self.max_events, self.n_streams)
        self.bd_offsets = self._get_or_alloc(
            f"{self.prefix}_bd_offsets", slot_shape, np.int64
        )
        self.bd_sizes = self._get_or_alloc(
            f"{self.prefix}_bd_sizes", slot_shape, np.int64
        )
        self.smd_offsets = self._get_or_alloc(
            f"{self.prefix}_smd_offsets", slot_shape, np.int64
        )
        self.smd_sizes = self._get_or_alloc(
            f"{self.prefix}_smd_sizes", slot_shape, np.int64
        )
        self.services = self._get_or_alloc(
            f"{self.prefix}_services", slot_shape, np.int32
        )
        self.cutoff_flags = self._get_or_alloc(
            f"{self.prefix}_cutoff_flags", slot_shape, np.int8
        )
        self.new_chunk_ids = self._get_or_alloc(
            f"{self.prefix}_new_chunk_ids", slot_shape, np.int64
        )

        self.slot_events = self._get_or_alloc(
            f"{self.prefix}_slot_events", (self.n_slots,), np.int32
        )
        self.slot_chunk_ids = self._get_or_alloc(
            f"{self.prefix}_slot_chunk_ids", (self.n_slots,), np.int64
        )
        self.slot_states = self._get_or_alloc(
            f"{self.prefix}_slot_states", (self.n_slots,), np.int32
        )
        self.slot_next_event = self._get_or_alloc(
            f"{self.prefix}_slot_next_evt", (self.n_slots,), np.int64
        )
        self.slot_consumers_done = self._get_or_alloc(
            f"{self.prefix}_slot_consumers_done", (self.n_slots,), np.int32
        )

        self.smd_chunk_sizes = self._get_or_alloc(
            f"{self.prefix}_chunk_sizes", (self.n_slots,), np.int32
        )
        self.smd_chunk_buffer = self._get_or_alloc(
            f"{self.prefix}_chunk_bytes",
            (self.n_slots, self.max_chunk_bytes),
            np.uint8,
        )
        self.shutdown_flag = self._get_or_alloc(
            f"{self.prefix}_shutdown", (1,), np.int32
        )
        if self.shared_mem.is_leader:
            self.shutdown_flag[0] = 0

    def _get_or_alloc(self, name: str, shape, dtype):
        if self.shared_mem.has_array(name):
            return self.shared_mem.get_array(name)
        return self.shared_mem.allocate_array(name, shape, dtype)

    cpdef int ingest_chunk(self, object view, long chunk_id):
        """
        Parse `view` and publish its metadata into a shared-memory slot.

        Returns the slot index that transitioned to STATE_READY.
        """
        slot = self._acquire_slot()
        chunk_view = memoryview(view)
        chunk_bytes = chunk_view.nbytes
        parsed = self._parse_chunk(chunk_view)
        if parsed.n_events > self.max_events:
            self.slot_states[slot] = self.STATE_EMPTY
            raise ValueError(
                f"Chunk has {parsed.n_events} events but buffers support at most {self.max_events}"
            )

        self.bd_offsets[slot, : parsed.n_events, :] = parsed.bd_offsets[: parsed.n_events]
        self.bd_sizes[slot, : parsed.n_events, :] = parsed.bd_sizes[: parsed.n_events]
        self.smd_offsets[slot, : parsed.n_events, :] = parsed.smd_offsets[: parsed.n_events]
        self.smd_sizes[slot, : parsed.n_events, :] = parsed.smd_sizes[: parsed.n_events]
        self.services[slot, : parsed.n_events, :] = parsed.services[: parsed.n_events]
        self.cutoff_flags[slot, : parsed.n_events, :] = parsed.cutoff_flags[: parsed.n_events]
        self.new_chunk_ids[slot, : parsed.n_events, :] = parsed.new_chunk_ids[: parsed.n_events]

        if chunk_bytes > self.max_chunk_bytes:
            self.slot_states[slot] = self.STATE_EMPTY
            raise ValueError(
                f"SMD chunk is {chunk_bytes} bytes but buffers support {self.max_chunk_bytes}"
            )
        arr_view = np.frombuffer(chunk_view, dtype=np.uint8, count=chunk_bytes)
        self.smd_chunk_buffer[slot, :chunk_bytes] = arr_view
        self.smd_chunk_sizes[slot] = chunk_bytes

        self.slot_events[slot] = parsed.n_events
        self.slot_chunk_ids[slot] = chunk_id
        self.slot_next_event[slot] = 0
        self.slot_consumers_done[slot] = 0
        self.slot_states[slot] = self.STATE_READY
        return slot

    def finalize(self) -> None:
        """Signal consumers that ingestion is complete."""
        self.shutdown_flag[0] = 1

    def mark_slot_empty(self, slot: int) -> None:
        """Release a slot so it can be reused."""
        self.slot_states[slot] = self.STATE_EMPTY
        self.slot_events[slot] = 0
        self.slot_chunk_ids[slot] = -1
        self.slot_next_event[slot] = 0
        self.slot_consumers_done[slot] = 0
        self.smd_chunk_sizes[slot] = 0

    cpdef void publish_shutdown_slot(self, long chunk_id):
        """
        Publish a zero-event slot so BD consumers observe end-of-stream.
        This mirrors the legacy empty-bytearray handshake.
        """
        slot = self._acquire_slot()
        self.smd_chunk_sizes[slot] = 0
        self.slot_events[slot] = 0
        self.slot_chunk_ids[slot] = chunk_id
        self.slot_next_event[slot] = 0
        self.slot_consumers_done[slot] = 0
        self.bd_offsets[slot, :, :] = 0
        self.bd_sizes[slot, :, :] = 0
        self.smd_offsets[slot, :, :] = 0
        self.smd_sizes[slot, :, :] = 0
        self.services[slot, :, :] = 0
        self.cutoff_flags[slot, :, :] = 0
        self.new_chunk_ids[slot, :, :] = 0
        self.slot_states[slot] = self.STATE_READY

    cdef int _acquire_slot(self):
        cdef Py_ssize_t idx
        cdef int wait_iters = 0
        while True:
            for attempt in range(self.n_slots):
                idx = (self._next_slot_hint + attempt) % self.n_slots
                if self.slot_states[idx] == self.STATE_EMPTY:
                    self.slot_states[idx] = self.STATE_FILLING
                    self._next_slot_hint = (idx + 1) % self.n_slots
                    return idx
            wait_iters += 1
            if wait_iters % 10000 == 0:
                self.logger.debug(
                    "MarchingEventBuilder waiting for free slot; BD ranks may be stalled (n_slots=%d)",
                    self.n_slots,
                )
            time.sleep(0.001)

    cdef object _parse_chunk(self, object chunk_view):
        chunk_pf = PacketFooter(view=chunk_view)
        if chunk_pf.n_packets != self.n_streams:
            raise ValueError(
                f"SMD chunk reported {chunk_pf.n_packets} streams but builder expects {self.n_streams}"
            )
        stream_views = chunk_pf.split_packets()
        self._prepare_views(stream_views)

        slot_capacity = self.max_events
        n_streams = self.n_streams
        bd_offsets = np.zeros((slot_capacity, n_streams), dtype=np.int64)
        bd_sizes = np.zeros((slot_capacity, n_streams), dtype=np.int64)
        smd_offsets = np.zeros((slot_capacity, n_streams), dtype=np.int64)
        smd_sizes = np.zeros((slot_capacity, n_streams), dtype=np.int64)
        services = np.zeros((slot_capacity, n_streams), dtype=np.int32)
        cutoff_flags = np.zeros((slot_capacity, n_streams), dtype=np.int8)
        new_chunk_ids = np.zeros((slot_capacity, n_streams), dtype=np.int64)

        cdef Py_ssize_t event_idx = 0
        while True:
            matched = self._gather_event()
            if matched == 0:
                break
            if event_idx >= slot_capacity:
                raise ValueError(
                    f"Chunk exceeds configured event_idx={event_idx} max_events/slot_capacity={slot_capacity}; "
                    f"increase PS_MARCH_EVENTS_SCALE (default 1.2)"
                )

            for stream_idx in range(n_streams):
                rel_offset = self._event_offsets[stream_idx]
                if rel_offset < 0:
                    cutoff_flags[event_idx, stream_idx] = 0
                    continue
                global_offset = self._stream_bases[stream_idx] + rel_offset
                smd_offsets[event_idx, stream_idx] = global_offset
                smd_sizes[event_idx, stream_idx] = self._event_sizes[stream_idx]
                services[event_idx, stream_idx] = self._event_services[stream_idx]
                cutoff_flags[event_idx, stream_idx] = self._event_cutoffs[stream_idx]

                if self._event_sizes[stream_idx] <= 0:
                    continue

                dg = dgram.Dgram(
                    config=self.configs[stream_idx],
                    view=chunk_view,
                    offset=int(global_offset),
                )
                if (
                    self._is_event(self._event_services[stream_idx])
                    and not bool(self.use_smds[stream_idx])
                    and hasattr(dg, "smdinfo")
                    and dg.smdinfo
                ):
                    self._update_bd_arrays(
                        dg,
                        bd_offsets,
                        bd_sizes,
                        event_idx,
                        stream_idx,
                    )
                elif (
                    self._event_services[stream_idx] == TransitionId.Enable
                    and hasattr(dg, "chunkinfo")
                    and dg.chunkinfo
                ):
                    chunk_ids = [
                        getattr(seg.chunkinfo, "chunkid")
                        for seg in dg.chunkinfo.values()
                    ]
                    chunk_files = [
                        getattr(seg.chunkinfo, "filename")
                        for seg in dg.chunkinfo.values()
                    ]
                    if chunk_ids:
                        new_chunk_id = chunk_ids[0]
                        new_filename = chunk_files[0]
                        if new_chunk_id > self.latest_chunk_ids[stream_idx]:
                            new_chunk_ids[event_idx, stream_idx] = new_chunk_id
                            self.chunkinfo[(stream_idx, new_chunk_id)] = new_filename
                            self.latest_chunk_ids[stream_idx] = new_chunk_id

            event_idx += 1

        valid_events = int(event_idx)
        return ParsedChunk(
            n_events=valid_events,
            bd_offsets=bd_offsets[:valid_events],
            bd_sizes=bd_sizes[:valid_events],
            smd_offsets=smd_offsets[:valid_events],
            smd_sizes=smd_sizes[:valid_events],
            services=services[:valid_events],
            cutoff_flags=cutoff_flags[:valid_events],
            new_chunk_ids=new_chunk_ids[:valid_events],
        )

    cdef void _prepare_views(self, list stream_views):
        base_offset = 0
        for idx in range(self.n_streams):
            mv = memoryview(stream_views[idx])
            self._views[idx] = mv
            self._offsets[idx] = 0
            self._sizes[idx] = mv.nbytes
            self._stream_bases[idx] = base_offset
            base_offset += mv.nbytes
            self._dgram_sizes[idx] = 0
            self._services_arr[idx] = 0
            self._timestamps[idx] = 0
            self._event_offsets[idx] = -1
            self._event_sizes[idx] = 0
            self._event_services[idx] = 0
            self._event_cutoffs[idx] = 0

    cdef int _gather_event(self):
        cdef Py_buffer buf
        cdef char* view_ptr
        cdef Dgram* dg
        cdef size_t dgram_size
        cdef uint64_t ts
        cdef uint64_t min_ts = 0
        cdef int primary = -1
        cdef int matched = 0
        cdef int idx

        for idx in range(self.n_streams):
            if self._offsets[idx] < self._sizes[idx]:
                PyObject_GetBuffer(
                    self._views[idx], &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS
                )
                view_ptr = <char*>buf.buf
                view_ptr += self._offsets[idx]
                dg = <Dgram*>view_ptr
                dgram_size = sizeof(Dgram) + (dg.xtc.extent - sizeof(Xtc))
                self._event_offsets[idx] = self._offsets[idx]
                self._offsets[idx] += dgram_size
                self._dgram_sizes[idx] = dgram_size
                ts = (<uint64_t>dg.seq.high << 32) | <uint64_t>dg.seq.low
                self._timestamps[idx] = ts
                self._services_arr[idx] = (dg.env >> 24) & 0xF
                if min_ts == 0 or ts < min_ts:
                    min_ts = ts
                    primary = idx
                PyBuffer_Release(&buf)
            else:
                self._event_offsets[idx] = -1
                self._dgram_sizes[idx] = 0
                self._timestamps[idx] = 0
                self._services_arr[idx] = 0

        if primary == -1:
            return 0

        self._event_timestamp = self._timestamps[primary]

        for idx in range(self.n_streams):
            if self._dgram_sizes[idx] == 0:
                self._event_sizes[idx] = 0
                self._event_services[idx] = 0
                self._event_cutoffs[idx] = 0
                continue

            if self._timestamps[idx] == self._event_timestamp:
                matched += 1
                self._event_sizes[idx] = self._dgram_sizes[idx]
                self._event_services[idx] = self._services_arr[idx]
                self._event_cutoffs[idx] = 1
            else:
                self._offsets[idx] -= self._dgram_sizes[idx]
                self._event_offsets[idx] = -1
                self._event_sizes[idx] = 0
                self._event_services[idx] = 0
                self._event_cutoffs[idx] = 0
                self._dgram_sizes[idx] = 0
                self._services_arr[idx] = 0

        if (
            not TransitionId.isEvent(int(self._services_arr[primary]))
            and matched != self.n_streams
        ):
            raise RuntimeError(
                f"TransitionId {TransitionId.name(int(self._services_arr[primary]))} "
                f"incomplete (ts:{self._event_timestamp}) expected:{self.n_streams} received:{matched}"
            )

        return matched

    cdef void _update_bd_arrays(
        self,
        d,
        bd_offsets,
        bd_sizes,
        Py_ssize_t i_evt,
        Py_ssize_t i_smd,
    ):
        bd_offsets[i_evt, i_smd] = d.smdinfo[0].offsetAlg.intOffset
        bd_sizes[i_evt, i_smd] = d.smdinfo[0].offsetAlg.intDgramSize

    cdef bint _is_event(self, int service):
        return TransitionId.isEvent(service) or service == 0
