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
from typing import List, Optional

import numpy as np

from psana import dgram, utils
from psana.psexp import TransitionId
from psana.psexp.tools import mode
from psana.marchingeventbuilder import MarchingEventBuilder
from psana.psexp.marching_shmem import MarchingSharedMemory
from psana.psexp.parallel_pread import ParallelPreader

DEBUG_PRINT = False

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
    shared_mem : MarchingSharedMemory
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
        shared_mem: MarchingSharedMemory,
        *,
        n_consumers: int,
        shared_state=None,
        name_prefix: str = "march",
        poll_interval: float = 5e-4,
        use_smds=None,
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
        self._bd_fds = np.asarray(self.dm.fds, dtype=np.intc)
        self._bd_fds = self._bd_fds[self._bd_indices] if self._bd_indices else np.empty(0, dtype=np.intc)

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
        self._pending_transitions: List[list] = []
        self._active_slot: Optional[int] = None
        self._slot_event_count = 0
        self._last_chunk_id = -1
        self._chunk_filenames = {}

        self._logger = utils.get_logger(name=utils.get_class_name(self))
        self._rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

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

            if self._pending_transitions:
                return self._pending_transitions.pop(0)

            if self._active_slot is None:
                slot = self._await_ready_slot()
                if slot is None:
                    raise StopIteration
                self._prepare_slot(slot)
                continue

            evt_idx = self._fetch_next_event_index(self._active_slot)
            self._debug(f"[DEBUG-marchbd] _next__ evt_idx={evt_idx}")
            if evt_idx >= self._slot_event_count:
                self._finish_slot_if_needed(self._active_slot)
                self._active_slot = None
                self._chunk_view = None
                continue

            if not self._is_l1_event(self._active_slot, evt_idx):
                # Skip transitions and let the transition replay path handle them.
                self._debug("[DEBUG-marchbd] _next__ skipping non-L1 event")
                continue

            dgrams = self._build_l1_event(self._active_slot, evt_idx)
            if DEBUG_PRINT:
                ts_values = [dg.timestamp() for dg in dgrams if dg is not None]
                unique_ts = sorted(set(ts_values))
                if len(unique_ts) > 1:
                    self._debug(
                        f"[DEBUG-marchbd] _next__ multiple timestamps idx={evt_idx} ts={unique_ts}"
                    )
            if not any(dgrams):
                # Missing event; continue marching.
                self._debug("[DEBUG-marchbd] _next__ skipping missing event")
                continue
            if DEBUG_PRINT:
                slot = self._active_slot
                chunk_id = int(self.slot_chunk_ids[slot])
                ts = next((dg.timestamp() for dg in dgrams if dg is not None), None)
                self._debug(
                    f"[DEBUG-marchbd] rank={self._rank} slot={slot} chunk={chunk_id} evt_idx={evt_idx} ts={ts}"
                )
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
            if int(self.shutdown_flag[0]) == 1 and not np.any(self.slot_states == MarchingEventBuilder.STATE_READY):
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
        self._ensure_preader_capacity(slot)

    def _finish_slot_if_needed(self, slot: int):
        """Increment the 'consumers done' counter and free slot if last consumer."""
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
            self.slot_states[slot] = MarchingEventBuilder.STATE_EMPTY
            self.slot_events[slot] = 0
            self.slot_chunk_ids[slot] = -1
            self.slot_next_event[slot] = 0
            self.slot_consumers_done[slot] = 0
            self.smd_chunk_sizes[slot] = 0

    # ------------------------------------------------------------------
    # Transition replay
    # ------------------------------------------------------------------
    def _collect_transition_events(self, slot: int) -> List[list]:
        """Extract transition dgrams from the shared chunk."""
        if self._chunk_view is None:
            return []

        transitions = []
        for evt_idx in range(self._slot_event_count):
            if self._is_l1_event(slot, evt_idx):
                continue
            dgrams = self._build_smd_event(slot, evt_idx)
            if any(dgrams):
                self._apply_chunk_updates(dgrams)
                transitions.append(dgrams)
        if DEBUG_PRINT:
            self._debug(
                f"[DEBUG-marchbd] _collect_transition_events slot={slot} transitions={len(transitions)}"
            )
        return transitions

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
            self._open_new_bd_file(i, new_chunk_id, filename)

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
            offsets = self.bd_offsets[slot, evt_idx, self._bd_indices]
            sizes = self.bd_sizes[slot, evt_idx, self._bd_indices]
            self._pread_offsets[:] = offsets
            self._pread_sizes[:] = sizes
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

    def _fetch_next_event_index(self, slot: int) -> int:
        win = self._next_evt_handle.window
        win.Lock(0)
        win.Fetch_and_op(
            self._one64,
            self._fetch_buf64,
            target_rank=0,
            target_disp=slot,
            op=MPI.SUM,
        )
        win.Unlock(0)
        return int(self._fetch_buf64[0])
