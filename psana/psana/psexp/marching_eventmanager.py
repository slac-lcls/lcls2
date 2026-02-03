"""
MarchingEventManager consumes marching slots populated by MarchingEventBuilder and
produces dgram lists compatible with the existing Run/Events pipeline.

Each Bd rank repeatedly:
  1. Waits for a slot to reach STATE_READY.
  2. Replays transition dgrams from the shared chunk buffer (updating EnvStore).
  3. Fetches L1 event indices via an atomic counter and issues concurrent pread()
     calls using ParallelPreader for the associated big-data streams.
  4. Marks the slot complete once every Bd consumer has drained it.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from psana import dgram, utils
from psana.psexp import TransitionId
from psana.psexp.tools import mode
from psana.marchingeventbuilder import MarchingEventBuilder
from psana.psexp.mpi_shmem import MPISharedMemory
from psana.psexp.parallel_pread import ParallelPreader

DEBUG_PRINT = False
DEBUG_ATOMIC_FETCH = False

if mode == "mpi":
    from mpi4py import MPI
else:
    MPI = None  # type: ignore[assignment]


class MarchingEventManager:
    """
    Iterator that yields events sourced from marching shared-memory slots.

    Parameters
    ----------
    configs : Sequence
        Configuration dgrams for per-stream reconstruction.
    dm : DgramManager
        Provides open big-data file descriptors and tracking of current chunk ids.
    shared_mem : MPISharedMemory
        Shared-memory allocator used by the matching MarchingEventBuilder instance.
    n_consumers : int
        Number of Bd ranks attached to this shared-memory domain.
    shared_state : SimpleNamespace, optional
        Carries terminate_flag, matching the contract used by psana.psexp.events.Events.
    name_prefix : str
        Prefix used when allocating shared-memory arrays (defaults to "march").
    poll_interval : float
        Seconds to sleep while waiting for a new slot.
    use_smds : Sequence[bool], optional
        Per-stream flag indicating whether SMD data should be used instead of big-data files.
    """

    def __init__(
        self,
        configs,
        dm,
        shared_mem: MPISharedMemory,
        *,
        n_consumers: int,
        shared_state=None,
        name_prefix: str = "march",
        poll_interval: float = 5e-4,
        use_smds=None,
        events_per_grant: int = 1,
        consumer_index: int = 0,
    ):
        if mode != "mpi":
            raise RuntimeError("MarchingEventManager requires MPI mode")

        self.configs = list(configs)
        self.dm = dm
        self.shared_mem = shared_mem
        self.prefix = name_prefix
        self.n_consumers = n_consumers
        self.shared_state = shared_state
        self.poll_interval = poll_interval

        self.n_streams = len(self.configs)
        self._attach_shared_arrays()

        if use_smds is None:
            use_smds = getattr(self.dm, "use_smds", None)
        if use_smds is None:
            use_smds = [False] * self.n_streams
        self._use_smds = np.asarray(use_smds, dtype=bool)
        self._bd_indices = [i for i in range(self.n_streams) if not self._use_smds[i]]
        if self._bd_indices:
            self._bd_fds = np.zeros(len(self._bd_indices), dtype=np.intc)
            self._refresh_bd_fds()
        else:
            self._bd_fds = np.empty(0, dtype=np.intc)

        self._preader = None
        self._preader_caps = None
        if self._bd_indices:
            self._pread_offsets = np.zeros(len(self._bd_indices), dtype=np.int64)
            self._pread_sizes = np.zeros(len(self._bd_indices), dtype=np.intp)

        self._one64 = np.array([1], dtype=np.int64)
        self._fetch_buf64 = np.zeros(1, dtype=np.int64)
        self._one32 = np.array([1], dtype=np.int32)
        self._fetch_buf32 = np.zeros(1, dtype=np.int32)

        self._chunk_view = None
        self._pending_transitions: List[Tuple[int, list]] = []
        self._active_slot: Optional[int] = None
        self._slot_event_count = 0
        self._last_chunk_id = -1
        self._chunk_filenames = {}
        self._l1_progress = 0
        self._stream_last_offsets = np.zeros(self.n_streams, dtype=np.int64)
        self._pending_chunk_switches: Dict[int, Tuple[int, str]] = {}

        self._logger = utils.get_logger(name=utils.get_class_name(self))
        self._rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        self._grant_size = max(1, int(events_per_grant))
        self._grant_remaining = 0
        self._grant_cursor = 0
        self._grant_request = np.array([self._grant_size], dtype=np.int64)
        self._grant_start = 0
        self._grant_span = 0
        self._grant_views = None
        self._grant_rel_offsets = None
        self._grant_sizes_arr = None
        self._grant_prints = 0
        self._slot_fetch_timings: Dict[int, List[float]] = {}
        self.consumer_index = int(consumer_index)
        if self.consumer_index < 0 or self.consumer_index >= self.n_consumers:
            self.consumer_index = max(0, min(self.n_consumers - 1, self.consumer_index))
        self._slot_local_events = 0

    def _debug(self, msg: str) -> None:
        if DEBUG_PRINT:
            print(msg, flush=True)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            # Respect termination flag used by Events iterators
            if self.shared_state and getattr(self.shared_state.terminate_flag, "value", 0):
                raise StopIteration

            if self._active_slot is None:
                slot = self._await_ready_slot()
                if slot is None:
                    raise StopIteration
                self._prepare_slot(slot)
                continue

            transition_evt = self._maybe_emit_transition()
            if transition_evt is not None:
                return transition_evt

            evt_idx = self._next_event_index()
            if evt_idx is None:
                self._l1_progress = self._slot_event_count
                transition_evt = self._maybe_emit_transition()
                if transition_evt is not None:
                    return transition_evt
                self._finish_slot_if_needed(self._active_slot)
                self._active_slot = None
                self._chunk_view = None
                continue

            # Updating grant start may unlock transitions; emit them before L1
            transition_evt = self._maybe_emit_transition()
            if transition_evt is not None:
                # Put the event index back so it can be reissued later
                self._grant_cursor -= 1
                self._grant_remaining += 1
                return transition_evt

            if not self._is_l1_event(self._active_slot, evt_idx):
                # Skip transitions and let the transition replay path handle them.
                continue

            dgrams = self._build_l1_event(self._active_slot, evt_idx)
            if not any(dgrams):
                # Missing event; continue marching.
                continue
            self._l1_progress = max(self._l1_progress, evt_idx + 1)
            self._slot_local_events += 1
            return dgrams

    # ------------------------------------------------------------------
    # Shared-memory wiring
    # ------------------------------------------------------------------
    def _attach_shared_arrays(self):
        arr = self.shared_mem
        pfx = self.prefix
        self.bd_offsets = arr.get_array(f"{pfx}_bd_offsets")
        self.bd_sizes = arr.get_array(f"{pfx}_bd_sizes")
        self.smd_offsets = arr.get_array(f"{pfx}_smd_offsets")
        self.smd_sizes = arr.get_array(f"{pfx}_smd_sizes")
        self.services = arr.get_array(f"{pfx}_services")
        self.new_chunk_ids = arr.get_array(f"{pfx}_new_chunk_ids")
        self.slot_events = arr.get_array(f"{pfx}_slot_events")
        self.slot_chunk_ids = arr.get_array(f"{pfx}_slot_chunk_ids")
        self.slot_states = arr.get_array(f"{pfx}_slot_states")
        self.slot_next_event = arr.get_array(f"{pfx}_slot_next_evt")
        self.slot_consumers_done = arr.get_array(f"{pfx}_slot_consumers_done")
        self.smd_chunk_sizes = arr.get_array(f"{pfx}_chunk_sizes")
        self.smd_chunk_buffer = arr.get_array(f"{pfx}_chunk_bytes")
        self.shutdown_flag = arr.get_array(f"{pfx}_shutdown")
        try:
            self.slot_reader_counts = arr.get_array(f"{pfx}_slot_reader_counts")
        except KeyError:
            self.slot_reader_counts = None
        try:
            self.slot_start_times = arr.get_array(f"{pfx}_slot_start_times")
        except KeyError:
            self.slot_start_times = None

        self._next_evt_handle = arr.get_handle(f"{pfx}_slot_next_evt")
        self._consumers_handle = arr.get_handle(f"{pfx}_slot_consumers_done")

    # ------------------------------------------------------------------
    # Slot lifecycle
    # ------------------------------------------------------------------
    def _await_ready_slot(self) -> Optional[int]:
        """Find the next slot marked READY, blocking until one is available."""
        while True:
            ready_mask = self.slot_states == MarchingEventBuilder.STATE_READY
            ready_indices = np.where(ready_mask)[0]
            candidate = None
            candidate_chunk = None
            for idx in ready_indices:
                chunk_id = int(self.slot_chunk_ids[idx])
                if chunk_id <= self._last_chunk_id:
                    continue
                if candidate is None or chunk_id < candidate_chunk:
                    candidate = int(idx)
                    candidate_chunk = chunk_id
            if candidate is not None:
                self._last_chunk_id = candidate_chunk
                return candidate

            if self.shared_state and getattr(self.shared_state.terminate_flag, "value", 0):
                return None
            if int(self.shutdown_flag[0]) == 1 and not np.any(
                self.slot_states == MarchingEventBuilder.STATE_READY
            ):
                return None
            time.sleep(self.poll_interval)

    def _prepare_slot(self, slot: int):
        self._active_slot = slot
        self._slot_event_count = int(self.slot_events[slot])
        chunk_size = int(self.smd_chunk_sizes[slot])
        if chunk_size == 0:
            self._chunk_view = None
        else:
            buffer = self.smd_chunk_buffer[slot, :chunk_size]
            self._chunk_view = memoryview(buffer)

        self._pending_transitions = self._collect_transition_events(slot)
        self._l1_progress = 0
        self._ensure_preader_capacity(slot)
        self._grant_remaining = 0
        self._grant_cursor = 0
        self._grant_start = 0
        self._grant_span = 0
        self._grant_views = None
        self._slot_local_events = 0
        if self.slot_reader_counts is not None and self.consumer_index < self.slot_reader_counts.shape[1]:
            self.slot_reader_counts[slot, self.consumer_index] = 0
        if self.slot_start_times is not None and self.slot_start_times[slot] == 0:
            self.slot_start_times[slot] = time.monotonic()
        if DEBUG_ATOMIC_FETCH:
            self._slot_fetch_timings[slot] = []
        else:
            self._slot_fetch_timings.pop(slot, None)

    def _finish_slot_if_needed(self, slot: int):
        """Increment the 'consumers done' counter and free slot if last consumer."""
        if (
            self.slot_reader_counts is not None
            and self.consumer_index < self.slot_reader_counts.shape[1]
        ):
            self.slot_reader_counts[slot, self.consumer_index] = self._slot_local_events
        self._slot_local_events = 0
        fetch_timings = None
        if DEBUG_ATOMIC_FETCH:
            fetch_timings = self._slot_fetch_timings.pop(slot, None)
        win = self._consumers_handle.window
        win.Lock(0)
        win.Fetch_and_op(
            self._one32,
            self._fetch_buf32,
            target_rank=0,
            target_disp=slot,
            op=MPI.SUM,
        )
        win.Unlock(0)
        previous = int(self._fetch_buf32[0])
        if previous >= self.n_consumers - 1:
            # Last consumer releases the slot.
            start_time = 0.0
            if self.slot_start_times is not None:
                start_time = float(self.slot_start_times[slot])
                self.slot_start_times[slot] = 0.0
            counts_summary = None
            if self.slot_reader_counts is not None:
                reader_cols = min(
                    self.slot_reader_counts.shape[1], max(self.n_consumers, 1)
                )
                counts_summary = np.array(
                    self.slot_reader_counts[slot, :reader_cols], copy=True
                )
                self.slot_reader_counts[slot, :reader_cols] = 0
            elapsed = time.monotonic() - start_time if start_time > 0 else 0.0
            total_events = int(self._slot_event_count)
            if counts_summary is not None and counts_summary.size:
                total_events = int(counts_summary.sum())
                readers = counts_summary.size
                min_ev = int(counts_summary.min())
                max_ev = int(counts_summary.max())
                avg_ev = total_events / readers if readers else 0.0
            else:
                readers = self.n_consumers
                min_ev = total_events // readers if readers else total_events
                max_ev = min_ev
                avg_ev = float(total_events) / readers if readers else float(total_events)
            fetch_stats = ""
            if DEBUG_ATOMIC_FETCH and fetch_timings:
                arr = np.asarray(fetch_timings, dtype=np.float64)
                arr_us = arr * 1e6
                count = arr_us.size
                avg_us = float(arr_us.mean())
                min_us = float(arr_us.min())
                max_us = float(arr_us.max())
                std_us = float(arr_us.std(ddof=0))
                fetch_stats = (
                    " fetch_us(count=%d avg=%.1f min=%.1f max=%.1f std=%.1f)"
                    % (count, avg_us, min_us, max_us, std_us)
                )
            self._logger.debug(
                "[marchbd] chunk_id=%d slot=%d readers=%d events=%d "
                "(avg=%.1f min=%d max=%d) duration=%.2fs%s",
                int(self.slot_chunk_ids[slot]),
                slot,
                readers,
                total_events,
                avg_ev,
                min_ev,
                max_ev,
                elapsed,
                fetch_stats,
            )
            self.slot_states[slot] = MarchingEventBuilder.STATE_EMPTY
            self.slot_events[slot] = 0
            self.slot_chunk_ids[slot] = -1
            self.slot_next_event[slot] = 0
            self.slot_consumers_done[slot] = 0
            self.smd_chunk_sizes[slot] = 0
            self._pending_transitions = []
            self._l1_progress = 0
            self._slot_event_count = 0

    def _next_event_index(self) -> Optional[int]:
        if self._active_slot is None:
            return None
        if self._grant_remaining <= 0:
            if not self._reserve_grant(self._active_slot):
                return None
        evt_idx = self._grant_cursor
        self._grant_cursor += 1
        self._grant_remaining -= 1
        return evt_idx

    def _reserve_grant(self, slot: int) -> bool:
        if self._slot_event_count <= 0:
            return False
        request = min(self._grant_size, self._slot_event_count)
        start = self._fetch_next_event_index(slot, request)
        self._l1_progress = max(self._l1_progress, start)
        if start >= self._slot_event_count:
            self._grant_remaining = 0
            return False
        end = min(start + request, self._slot_event_count)
        self._grant_cursor = start
        self._grant_remaining = end - start
        self._grant_start = start
        self._grant_span = self._grant_remaining
        self._prepare_grant_views(slot, start, end)
        return self._grant_remaining > 0

    def _prepare_grant_views(self, slot: int, start: int, end: int):
        if not self._bd_indices or self._preader is None:
            self._grant_views = None
            self._grant_rel_offsets = None
            self._grant_sizes_arr = None
            return
        span = end - start
        if span <= 0:
            self._grant_views = None
            self._grant_rel_offsets = None
            self._grant_sizes_arr = None
            return
        rel_offsets = np.full((span, len(self._bd_indices)), -1, dtype=np.int64)
        rel_sizes = np.zeros((span, len(self._bd_indices)), dtype=np.int64)
        if self._preader is None or self._preader_caps is None:
            self._ensure_preader_capacity(slot)
        chunk_offsets = np.zeros(len(self._bd_indices), dtype=np.int64)
        chunk_sizes = np.zeros(len(self._bd_indices), dtype=np.intp)
        chunk_bytes = np.zeros(len(self._bd_indices), dtype=np.float64)
        for local_idx, stream_idx in enumerate(self._bd_indices):
            stream_offsets = self.bd_offsets[slot, start:end, stream_idx].astype(np.int64, copy=False)
            stream_sizes = self.bd_sizes[slot, start:end, stream_idx].astype(np.int64, copy=False)
            valid = stream_sizes > 0
            if not np.any(valid):
                chunk_offsets[local_idx] = 0
                chunk_sizes[local_idx] = 0
                chunk_bytes[local_idx] = 0.0
                continue
            first_idx = int(np.argmax(valid))
            last_idx = int(len(valid) - 1 - np.argmax(valid[::-1]))
            first_offset = int(stream_offsets[first_idx])
            self._ensure_chunk_ready(stream_idx, first_offset)
            last_offset = int(stream_offsets[last_idx])
            last_size = int(stream_sizes[last_idx])
            chunk_offsets[local_idx] = first_offset
            chunk_sizes[local_idx] = max(1, (last_offset + last_size) - first_offset)
            chunk_bytes[local_idx] = chunk_sizes[local_idx] / (1024 * 1024)
            rel = stream_offsets - first_offset
            rel_offsets[:, local_idx] = np.where(valid, rel, -1)
            rel_sizes[:, local_idx] = stream_sizes
        if not np.any(chunk_sizes):
            self._grant_views = None
            self._grant_rel_offsets = rel_offsets
            self._grant_sizes_arr = rel_sizes
            return
        chunk_list = [int(sz) for sz in chunk_sizes]
        if self._preader_caps is None or len(self._preader_caps) != len(chunk_list):
            self._preader_caps = [max(1, sz) for sz in chunk_list]
            self._preader = ParallelPreader(len(self._bd_indices), self._preader_caps)
        else:
            new_caps = []
            need_resize = False
            for cap, required in zip(self._preader_caps, chunk_list):
                if required > cap:
                    need_resize = True
                    cap = required
                new_caps.append(max(1, cap))
            if need_resize:
                self._preader_caps = new_caps
                self._preader = ParallelPreader(len(self._bd_indices), self._preader_caps)
        try:
            views = self._preader.read(self._bd_fds, chunk_offsets, chunk_sizes)
        except Exception as e:
            print(
                "Marching grant pread failed: slot=%d start=%d end=%d offsets=%s sizes=%s error=%s",
                slot,
                start,
                end,
                chunk_offsets,
                chunk_sizes,
                str(e),
            )
            raise
        self._grant_views = views
        self._grant_rel_offsets = rel_offsets
        self._grant_sizes_arr = rel_sizes
        if DEBUG_PRINT and start < end and self._grant_prints < 3:
            grant_id = self._grant_prints + 1
            chunk_id = int(self.slot_chunk_ids[slot])
            stream_msgs = []
            for local_idx, stream_idx in enumerate(self._bd_indices):
                fname = os.path.basename(self.dm.xtc_files[stream_idx])
                stream_msgs.append(f"{fname}:{chunk_bytes[local_idx]:.2f} MiB")
            msg = (
                f"[marching grant] rank={self._rank} chunk={chunk_id} grant={grant_id} "
                f"events[{start},{end}) bytes per stream: {', '.join(stream_msgs)}"
            )
            self._debug(msg)
            self._grant_prints += 1

    # ------------------------------------------------------------------
    # Transition replay
    # ------------------------------------------------------------------
    def _maybe_emit_transition(self) -> Optional[list]:
        if not self._pending_transitions:
            return None
        evt_idx, dgrams = self._pending_transitions[0]
        if evt_idx > self._l1_progress:
            return None
        self._pending_transitions.pop(0)
        return dgrams

    def _collect_transition_events(self, slot: int) -> List[Tuple[int, list]]:
        """Extract transition dgrams from the shared chunk."""
        if self._chunk_view is None:
            return []

        transitions: List[Tuple[int, list]] = []
        for evt_idx in range(self._slot_event_count):
            if self._is_l1_event(slot, evt_idx):
                continue
            dgrams = self._build_smd_event(slot, evt_idx)
            if any(dgrams):
                self._apply_chunk_updates(dgrams)
                transitions.append((evt_idx, dgrams))
        if DEBUG_PRINT:
            self._debug(
                f"[march transition] rank={self._rank} slot={slot} transitions={len(transitions)}"
            )
        return transitions

    def _ensure_chunk_ready(self, stream_idx: int, next_offset: int) -> None:
        """Apply pending chunk switch when offsets wrap around to a new file."""
        pending = self._pending_chunk_switches.get(stream_idx)
        if pending is None:
            return
        if next_offset >= self._stream_last_offsets[stream_idx]:
            return
        new_chunk_id, filename = pending
        self._open_new_bd_file(stream_idx, new_chunk_id, filename)
        self._pending_chunk_switches.pop(stream_idx, None)

    def _build_smd_event(self, slot: int, evt_idx: int) -> List[Optional[dgram.Dgram]]:
        dgrams = [None] * self.n_streams
        offsets = self.smd_offsets[slot, evt_idx]
        sizes = self.smd_sizes[slot, evt_idx]
        for i, size in enumerate(sizes):
            if size <= 0:
                continue
            offset = int(offsets[i])
            dgrams[i] = dgram.Dgram(
                config=self.configs[i],
                view=self._chunk_view,
                offset=offset,
            )
        return dgrams

    def _apply_chunk_updates(self, dgrams: List[Optional[dgram.Dgram]]):
        """Detect Enable transitions carrying chunkinfo and reopen files as needed."""
        for i, dg in enumerate(dgrams):
            if dg is None or dg.service() != TransitionId.Enable or not hasattr(dg, "chunkinfo"):
                continue
            if not dg.chunkinfo:
                continue
            # Assume a single chunkinfo entry per Enable.
            meta = next(iter(dg.chunkinfo.values())).chunkinfo
            new_chunk_id = getattr(meta, "chunkid", None)
            filename = getattr(meta, "filename", None)
            if new_chunk_id is None or filename is None:
                continue
            current_id = self.dm.get_chunk_id(i)
            if current_id is not None and new_chunk_id <= current_id:
                continue
            self._chunk_filenames[(i, new_chunk_id)] = filename
            if DEBUG_PRINT:
                self._debug(
                    f"[marching chunkinfo] rank={self._rank} stream={i} chunk_id={new_chunk_id} "
                    f"filename={filename}"
                )
            self._pending_chunk_switches[i] = (new_chunk_id, filename)

    def _open_new_bd_file(self, stream_idx: int, new_chunk_id: int, filename: Optional[str] = None):
        """Switch the fd for a stream when chunkinfo indicates a new file."""
        if not filename:
            filename = self._chunk_filenames.get((stream_idx, new_chunk_id))
            if not filename:
                return
        os.close(int(self.dm.fds[stream_idx]))
        xtc_dir = os.path.dirname(self.dm.xtc_files[stream_idx])
        full_path = os.path.join(xtc_dir, filename)
        fd = os.open(full_path, os.O_RDONLY)
        self.dm.fds[stream_idx] = fd
        self.dm.xtc_files[stream_idx] = full_path
        self.dm.set_chunk_id(stream_idx, new_chunk_id)
        self._refresh_bd_fds()
        self._stream_last_offsets[stream_idx] = 0

    def _refresh_bd_fds(self):
        """Update cached big-data FD array to reflect current chunk assignments."""
        if not self._bd_indices:
            return
        for local_idx, stream_idx in enumerate(self._bd_indices):
            self._bd_fds[local_idx] = int(self.dm.fds[stream_idx])

    # ------------------------------------------------------------------
    # L1 processing
    # ------------------------------------------------------------------
    def _build_l1_event(self, slot: int, evt_idx: int) -> List[Optional[dgram.Dgram]]:
        dgrams = [None] * self.n_streams
        # Reuse SMD data for streams flagged via use_smds
        for i in range(self.n_streams):
            if self._use_smds[i]:
                size = self.smd_sizes[slot, evt_idx, i]
                if size > 0:
                    offset = int(self.smd_offsets[slot, evt_idx, i])
                    dgrams[i] = dgram.Dgram(
                        config=self.configs[i],
                        view=self._chunk_view,
                        offset=offset,
                    )

        if self._bd_indices:
            rel_idx = evt_idx - self._grant_start
            if (
                self._grant_views is not None
                and 0 <= rel_idx < self._grant_span
                and self._grant_rel_offsets is not None
            ):
                for local_idx, stream_idx in enumerate(self._bd_indices):
                    rel_off = self._grant_rel_offsets[rel_idx, local_idx]
                    size = self._grant_sizes_arr[rel_idx, local_idx]
                    if rel_off < 0 or size <= 0:
                        continue
                    view = self._grant_views[local_idx]
                    if size > 0 and view is not None:
                        sub_view = view[rel_off : rel_off + size]
                        dgrams[stream_idx] = dgram.Dgram(
                            config=self.dm.configs[stream_idx],
                            view=sub_view,
                        )
                        base_off = int(self.bd_offsets[slot, evt_idx, stream_idx])
                        self._stream_last_offsets[stream_idx] = max(
                            self._stream_last_offsets[stream_idx], base_off + size
                        )
                return dgrams

            offsets = self.bd_offsets[slot, evt_idx, self._bd_indices]
            sizes = self.bd_sizes[slot, evt_idx, self._bd_indices]
            self._pread_offsets[:] = offsets
            self._pread_sizes[:] = sizes
            for local_idx, stream_idx in enumerate(self._bd_indices):
                off = int(offsets[local_idx])
                if self._pread_sizes[local_idx] > 0:
                    self._ensure_chunk_ready(stream_idx, off)
            if self._preader is None:
                self._ensure_preader_capacity(slot)
            views = self._preader.read(self._bd_fds, self._pread_offsets, self._pread_sizes)
            for local_idx, stream_idx in enumerate(self._bd_indices):
                if self._pread_sizes[local_idx] <= 0:
                    continue
                dgrams[stream_idx] = dgram.Dgram(
                    config=self.dm.configs[stream_idx],
                    view=views[local_idx],
                )
                base_off = int(self.bd_offsets[slot, evt_idx, stream_idx])
                size = int(self._pread_sizes[local_idx])
                self._stream_last_offsets[stream_idx] = max(
                    self._stream_last_offsets[stream_idx], base_off + size
                )

        return dgrams

    def _ensure_preader_capacity(self, slot: int):
        if not self._bd_indices:
            return
        slot_sizes = self.bd_sizes[slot, : self._slot_event_count, :]
        if slot_sizes.ndim == 1:
            slot_sizes = slot_sizes.reshape(1, -1)
        sizes = slot_sizes[:, self._bd_indices]
        if sizes.size == 0:
            max_sizes = np.zeros(len(self._bd_indices), dtype=np.int64)
        else:
            max_sizes = sizes.max(axis=0).astype(np.int64)
        max_sizes = np.maximum(max_sizes, 1)
        caps_arr = np.atleast_1d(max_sizes).astype(np.int64).ravel()
        if caps_arr.size != len(self._bd_indices):
            self._logger.warning(
                "Marching pread capacity mismatch: len(bd_indices)=%d but computed sizes=%d; slot=%d events=%d",
                len(self._bd_indices),
                caps_arr.size,
                slot,
                self._slot_event_count,
            )
            adjusted = np.ones(len(self._bd_indices), dtype=np.int64)
            limit = min(len(self._bd_indices), caps_arr.size)
            if limit > 0:
                adjusted[:limit] = caps_arr[:limit]
            caps_arr = adjusted
        caps_list = caps_arr.tolist()
        if self._preader is None:
            self._preader_caps = caps_list
            self._preader = ParallelPreader(len(self._bd_indices), self._preader_caps)
            return
        if any(req > cap for req, cap in zip(caps_list, self._preader_caps)):
            self._preader_caps = caps_list
            self._preader = ParallelPreader(len(self._bd_indices), self._preader_caps)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_l1_event(self, slot: int, evt_idx: int) -> bool:
        services = self.services[slot, evt_idx]
        sizes = self.smd_sizes[slot, evt_idx]
        for svc, size in zip(services, sizes):
            if size <= 0:
                continue
            if TransitionId.isEvent(int(svc)):
                return True
        return False

    def _fetch_next_event_index(self, slot: int, count: int = 1) -> int:
        win = self._next_evt_handle.window
        request = max(1, int(count))
        self._grant_request[0] = request
        start = time.perf_counter() if DEBUG_ATOMIC_FETCH else None
        win.Lock(0)
        win.Fetch_and_op(
            self._grant_request,
            self._fetch_buf64,
            target_rank=0,
            target_disp=slot,
            op=MPI.SUM,
        )
        win.Unlock(0)
        if DEBUG_ATOMIC_FETCH and start is not None:
            elapsed = time.perf_counter() - start
            timings = self._slot_fetch_timings.get(slot)
            if timings is not None:
                timings.append(elapsed)
        return int(self._fetch_buf64[0])
