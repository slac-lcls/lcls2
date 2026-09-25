import os
from dataclasses import dataclass, field, replace
import time
from typing import List, Tuple

import numpy as np

from .gpu_allocation import owned_empty, allocation_requirement, backing_capacity

from .gpu_read_plan import (
    LogicalDgram, ReadPlan, ReadRange, ResolvedDgram, ResolvedFile, build_read_plan,
)


# Descriptor-table columns shared by the reader and GPUDetector.  The table
# stays in CPU memory; only the raw XTC byte buffer is transferred to the GPU.
DESC_EVENT_INDEX = 0
DESC_STREAM_ID = 1
DESC_TIMESTAMP = 2
DESC_FILE_OFFSET = 3
DESC_READ_SIZE = 4
DESC_DEVICE_OFFSET = 5
DESC_NCOLS = 6


@dataclass
class KvikioBatchRead:
    desc_table: np.ndarray
    data_gpu: object = None
    _retain: object = None

    def retain_input(self):
        """Pin this read's raw storage; return an explicit release callback."""
        return self._retain() if self._retain is not None else lambda: None


@dataclass
class PendingBatch:
    """Holds in-flight KvikIO pread futures and the GPU-side buffers they write into.

    Returned by ``KvikioGpuReader.issue_batch()``.  Pass to ``wait_batch()``
    once the caller has done other work (e.g. CPU EventManager path) to collect
    the completed reads.
    """
    desc_table: np.ndarray
    data_gpu: object            # cp.ndarray uint8  (reads landing here)
    futures: List[Tuple]        # [(ReadRange, read_size, kvikio_future)]
    handles: list = field(default_factory=list)
    plan: object = None
    slot_id: object = None
    issued_ns: int = 0
    generation: int = 0
    completed: bool = False
    error: object = None


class KvikioGpuReader:
    def __init__(self, task_size=None, n_slots=2, budget=None, *, bulk_read=True):
        """Create a GPU reader with optional pre-allocated per-slot buffers.

        Parameters
        ----------
        task_size : int or None
            KvikIO task size for GDS reads.
        n_slots : int
            Number of raw input buffers. The current scheduler uses its
            execution depth; retained InputWindows independently guard reuse.
            One ``data_gpu`` buffer is pre-allocated per slot and grown
            lazily on the first batch that exceeds the current size.
            Reusing the same buffer per slot eliminates the per-batch
            ``cp.empty()`` allocation that causes CuPy pool fragmentation
            over long runs with large batch sizes.
        """
        import cupy as cp
        import kvikio

        self.cp = cp
        self.kvikio = kvikio
        self.task_size = task_size
        if type(bulk_read) is not bool:
            raise TypeError("bulk_read must be a bool")
        self.bulk_read = bulk_read
        self._files = {}
        self._latest_files = {}
        self._pending = []
        self._closed = False
        self._failure = None
        self._input_holds = {}
        self._generations = {}

        # Detect which I/O path kvikio will use for this run.
        # GDS (is_gds_available=True)  → NVMe → GPU VRAM direct via DMA
        # CPU fallback (False)         → NVMe → CPU DRAM → GPU VRAM via cudaMemcpy
        # GDS requires: local NVMe (not Lustre/GPFS), cuFile driver loaded,
        # KVIKIO_COMPAT_MODE not set.
        # On S3DF: /sdf/data/lcls/... is Lustre → GDS unavailable → CPU fallback.
        try:
            _dp = kvikio.DriverProperties()
            self._compat_mode: bool = not _dp.is_gds_available
        except Exception:
            self._compat_mode = True   # assume CPU fallback if detection fails

        self.io_path: str = 'CPU-fallback' if self._compat_mode else 'GDS'

        if self._compat_mode:
            import warnings
            warnings.warn(
                'KvikioGpuReader: kvikio is using the CPU-fallback path '
                '(NVMe → CPU DRAM → GPU VRAM).  True GDS '
                '(NVMe → GPU VRAM direct) is not available — possible causes: '
                'Lustre/GPFS filesystem, cuFile driver not loaded, or '
                'KVIKIO_COMPAT_MODE=1.  Performance will be limited by '
                'both NVMe read speed AND PCIe CPU→GPU transfer bandwidth.',
                stacklevel=2,
            )

        # Bandwidth tracking: accumulated across all issue+wait calls.
        # Reset via reset_io_stats(); read via io_stats().
        self._total_bytes_read: int = 0
        self._total_io_ns:      int = 0
        self._total_requests = 0
        self._total_requested_bytes = 0
        self._total_useful_bytes = 0
        self._total_issue_to_complete_ns = 0

        # Pre-allocated per-slot data buffers (Option D).
        # _slot_bufs[i] holds the current buffer for slot i.
        # Grown lazily: only re-allocated when total_nbytes exceeds the
        # existing buffer size (which happens at most a few times at the
        # start of a run as batch sizes stabilise).
        self._slot_bufs: list = [None] * n_slots
        self._n_slots: int = n_slots
        self._budget = budget  # _GpuBudget | None
        self._slot_idx: int = 0     # incremented on every issue_batch() call

    def io_stats(self) -> dict:
        """Return cumulative I/O statistics since last reset_io_stats().

        Returns
        -------
        dict with keys:
          io_path       : 'GDS' or 'CPU-fallback'
          compat_mode   : bool (True = CPU fallback)
          total_bytes   : int   total bytes read
          total_ns      : int   total wall-ns spent in wait_batch()
          bandwidth_gbs : float effective bandwidth in GB/s
          total_requests: int   successfully submitted pread calls
          requested_bytes: int  bytes requested by those calls
          useful_bytes  : int   logical bytes in fully successful batches
          issue_to_complete_ns: int summed submission-to-completion wall time

        total_bytes counts fully validated physical reads, including reads
        drained after another read failed. total_ns remains wait-only for
        compatibility. Concurrent batch durations can overlap, so summing
        issue_to_complete_ns does not measure end-to-end throughput.
        """
        bw = (self._total_bytes_read / self._total_io_ns
              if self._total_io_ns > 0 else 0.0)
        return {
            'io_path':       self.io_path,
            'compat_mode':   self._compat_mode,
            'total_bytes':   self._total_bytes_read,
            'total_ns':      self._total_io_ns,
            'bandwidth_gbs': bw,
            'bulk_read': self.bulk_read,
            'total_requests': self._total_requests,
            'requested_bytes': self._total_requested_bytes,
            'useful_bytes': self._total_useful_bytes,
            'issue_to_complete_ns': self._total_issue_to_complete_ns,
        }

    def reset_io_stats(self) -> None:
        """Reset cumulative I/O statistics."""
        if self._pending:
            raise RuntimeError("cannot reset I/O statistics with pending reads")
        self._total_bytes_read = 0
        self._total_io_ns      = 0
        self._total_requests = 0
        self._total_requested_bytes = 0
        self._total_useful_bytes = 0
        self._total_issue_to_complete_ns = 0

    def close(self):
        if self._closed:
            return
        self._closed = True
        error = None
        for pending in tuple(self._pending):
            try:
                self.wait_batch(pending)
            except BaseException as exc:
                if error is None:
                    error = exc
        for fh in tuple(self._files.values()):
            try:
                fh.close()
            except BaseException as exc:
                if error is None:
                    error = exc
        self._files.clear()
        if error is not None:
            raise error

    def memory_bytes(self) -> dict:
        """Return current VRAM usage for the raw input slot buffers.

        Used by GpuEventManager.log_memory() for Phase-0 accounting.
        """
        slot_sizes = [backing_capacity(b) if b is not None else 0
                      for b in self._slot_bufs]
        return {
            'raw_input_slots': sum(slot_sizes),
            'per_slot':        slot_sizes,
        }

    def _ensure_slot_buffer(self, slot: int, total_nbytes: int):
        """Grow slot ``slot``'s input buffer to hold at least total_nbytes.

        Charge both old and new buffers during replacement. Old capacity stays
        charged until its last alias is released; reusable buffers stay charged.
        """
        existing = self._slot_bufs[slot]
        if existing is not None and existing.nbytes >= total_nbytes:
            return

        new_buf = owned_empty(self.cp, total_nbytes, self.cp.uint8,
                              self._budget, 'reader')
        self._slot_bufs[slot] = new_buf
        del existing

    def allocation_requirements(self, nbytes, slot):
        old = self._slot_bufs[slot]
        return [allocation_requirement(self.cp, nbytes, old)]

    def trim_free_buffers(self):
        """Relinquish cached capacity only when neither I/O nor input owns it."""
        pending_slots = {p.slot_id for p in self._pending}
        for slot, buf in enumerate(self._slot_bufs):
            if buf is None or slot in pending_slots or self._input_holds.get(slot, 0):
                continue
            self._slot_bufs[slot] = None
            del buf

    def issue_group(self, group, *, slot_id):
        """Read one Stage 1 group into an independently leased reader slot.

        Opt-in adapter for the group pool. Keep the existing issue/wait path's
        generation, budget, file ownership and failure-draining protections.
        This does not change the production scheduler's issue_batch calls.
        """
        from types import SimpleNamespace
        from .gpu_batch import GpuReadDesc
        from .gpu_file_epochs import FileEpoch
        from .gpu_budget import allocation_growth_bytes, GpuMemoryPressureError

        if self._closed or self._failure is not None:
            raise RuntimeError('GPU reader is closed or failed') from self._failure
        if not self.bulk_read:
            raise ValueError('group reads require the adjacent-range reader')
        if not 0 <= slot_id < self._n_slots:
            raise IndexError(slot_id)
        cursor = group.file_offset
        for d in group.dgrams:
            if (d.file != group.file or d.stream_id != group.stream_id
                    or d.file_offset != cursor or d.size < 0
                    or (d.size == 0 and len(group.dgrams) != 1)):
                raise ValueError('input group must contain contiguous dgrams from one stream/file')
            cursor += d.size
        if not group.dgrams or cursor - group.file_offset != group.size:
            raise ValueError('input group byte count mismatch')
        growth = allocation_growth_bytes(self.allocation_requirements(group.size, slot_id))
        if self._budget is not None and growth > self._budget.allocation_available():
            raise GpuMemoryPressureError(f'input group needs {growth} allocation bytes')
        descs = tuple(GpuReadDesc(d.batch_event_index, d.timestamp, d.stream_id,
                                 d.file_offset, d.size, d.smd_size, 1)
                      for d in group.dgrams)
        view = SimpleNamespace(iter_read_descs=lambda _: iter(descs))
        epochs = {(d.batch_event_index, d.stream_id): FileEpoch(group.file, group.fence_id)
                  for d in group.dgrams}
        return self.issue_batch(view, None, slot_id=slot_id, file_epochs=epochs)

    def issue_batch(self, gpu_view, bd_dm, slot_id=None, *, file_epochs=None) -> "PendingBatch":
        """Issue GDS reads for a GPU batch non-blocking.

        All KvikIO pread() calls are issued immediately and return futures.
        The caller can do other work (e.g. CPU EventManager path) before
        calling wait_batch() to collect the completed reads.

        Parameters
        ----------
        gpu_view : GpuBatchView describing the batch
        bd_dm    : DgramManager holding bigdata file descriptors
        slot_id  : int or None
            Explicit reusable-buffer slot coordinated with EventPool.  When
            None, use this reader's internal round-robin order.
        file_epochs : mapping or None
            Required in bulk mode: (original event index, stream) to FileEpoch
            mapping resolved from the complete EB/SMD envelope before I/O.

        Returns
        -------
        PendingBatch with in-flight futures.  Pass to wait_batch().
        """
        if self._closed or self._failure is not None:
            raise RuntimeError("GPU reader is closed or failed") from self._failure
        read_descs = tuple(gpu_view.iter_read_descs(bd_dm))
        # Use the pre-allocated per-slot buffer when available.
        # Only re-allocate when the current buffer is too small (grows lazily).
        if slot_id is not None:
            slot = int(slot_id) % self._n_slots
        else:
            slot = self._slot_idx % self._n_slots
            self._slot_idx += 1
        if any(p.slot_id == slot for p in self._pending):
            raise RuntimeError(f"GPU input slot {slot} still has pending I/O")
        if self._input_holds.get(slot, 0):
            raise RuntimeError(f"GPU raw buffer {slot} is owned by an input window")
        existing = self._slot_bufs[slot]
        old_size = int(existing.nbytes) if existing is not None else 0
        capacity = (max(old_size, self._budget.allocation_available()) if self._budget is not None
                    else sum(d.size for d in read_descs))
        desc_table = self._build_desc_table(read_descs)
        plan = None
        if self.bulk_read:
            if file_epochs is None:
                raise ValueError("bulk reads require resolved file_epochs before submission")
            plan = self._coalesced_plan(read_descs, file_epochs, capacity)
            for row, logical in zip(desc_table, plan.logical_dgrams):
                row[DESC_DEVICE_OFFSET] = logical.device_offset
            ranges = plan.physical_ranges
            total_nbytes = plan.capacity_bytes
            for desc in read_descs:
                self._latest_files[desc.stream_id] = file_epochs[
                    (desc.batch_event_index, desc.stream_id)
                ].file
        else:
            ranges = []
            identities = {}
            for desc, row in zip(read_descs, desc_table):
                if desc.stream_id not in identities:
                    identities[desc.stream_id] = ResolvedFile(
                        os.path.realpath(str(bd_dm.xtc_files[desc.stream_id])),
                        int(bd_dm.get_chunk_id(desc.stream_id) or 0),
                    )
                identity = identities[desc.stream_id]
                self._latest_files[desc.stream_id] = identity
                if desc.size:
                    ranges.append(ReadRange(identity, desc.offset, desc.size,
                                            int(row[DESC_DEVICE_OFFSET])))
            total_nbytes = sum(d.size for d in read_descs)
        self._ensure_slot_buffer(slot, total_nbytes)
        data_gpu = self._slot_bufs[slot][:total_nbytes]
        if os.environ.get('PSANA_GPU_MEM_DEBUG'):
            try:
                from psana.gpu.gpu_mpi import log_gpu_mem
                grew = existing is None or existing.nbytes < total_nbytes
                log_gpu_mem(
                    f'issue_batch slot={slot} {total_nbytes/1e6:.0f} MB '
                    f'{"(grew)" if grew else "(reused)"}'
                )
            except Exception:
                pass

        generation = self._generations.get(slot, 0) + 1
        self._generations[slot] = generation
        pending = PendingBatch(desc_table, data_gpu, [], plan=plan, slot_id=slot,
                               generation=generation, issued_ns=time.perf_counter_ns())
        self._pending.append(pending)
        try:
            for r in ranges:
                cu_file = self._file_for_identity(r.file)
                pending.handles.append((r.file, cu_file))
                dst = data_gpu[r.device_offset:r.device_offset + r.size]
                future = cu_file.pread(dst, size=r.size, file_offset=r.file_offset,
                                       task_size=self.task_size)
                pending.futures.append((r, r.size, future))
                self._total_requests += 1
                self._total_requested_bytes += r.size
        except BaseException as exc:
            pending.error = self._read_error("submission", r, pending, exc)
            self.wait_batch(pending)  # drains partial submission, then raises
        return pending

    def wait_batch(self, pending: "PendingBatch") -> KvikioBatchRead:
        """Wait for in-flight reads from issue_batch() and return a KvikioBatchRead.

        Parameters
        ----------
        pending : PendingBatch returned by issue_batch()

        Returns
        -------
        KvikioBatchRead with the CPU descriptor table and GPU data populated.
        """
        if not pending.completed:
            if not any(p is pending for p in self._pending):
                raise ValueError("pending batch is not owned by this reader")
            start = time.perf_counter_ns()
            for r, read_size, future in pending.futures:
                try:
                    nread = int(future.get())
                    if nread != read_size:
                        raise RuntimeError(f"short read: asked={read_size} got={nread}")
                    self._total_bytes_read += nread
                except BaseException as exc:
                    if pending.error is None:
                        pending.error = self._read_error("completion", r, pending, exc)
            end = time.perf_counter_ns()
            self._total_io_ns += end - start
            self._total_issue_to_complete_ns += end - pending.issued_ns
            pending.completed = True
            self._pending = [p for p in self._pending if p is not pending]
            if pending.error is None:
                self._total_useful_bytes += sum(int(row[DESC_READ_SIZE]) for row in pending.desc_table)
            else:
                self._failure = pending.error  # never reuse failed input bytes
            try:
                self._prune_files()
            except BaseException as exc:
                if pending.error is None:
                    pending.error = exc
                    self._failure = exc
        if pending.error is not None:
            raise pending.error
        return KvikioBatchRead(pending.desc_table, pending.data_gpu,
                              lambda: self._retain_input(pending))

    def _retain_input(self, pending):
        slot = pending.slot_id
        if self._closed or self._generations.get(slot) != pending.generation:
            raise RuntimeError("cannot retain an obsolete GPU read")
        self._input_holds[slot] = self._input_holds.get(slot, 0) + 1
        released = False

        def release():
            nonlocal released
            if not released:
                self._input_holds[slot] -= 1
                released = True
        return release

    @staticmethod
    def _read_error(operation, r, pending, exc):
        if not isinstance(exc, Exception):
            return exc
        affected = [(int(row[DESC_EVENT_INDEX]), int(row[DESC_STREAM_ID]))
                    for row in pending.desc_table
                    if r.device_offset <= int(row[DESC_DEVICE_OFFSET]) < r.device_offset + r.size]
        error = RuntimeError(
            f"KvikIO {operation} failed: file={r.file.path} chunk={r.file.chunk_id} "
            f"offset={r.file_offset} size={r.size} events/streams={affected[:8]}: {exc}"
        )
        error.__cause__ = exc
        return error

    def _file_for_identity(self, identity):
        cu_file = self._files.get(identity)
        if cu_file is None:
            cu_file = self.kvikio.CuFile(identity.path, "r")
            self._files[identity] = cu_file
        return cu_file

    def _prune_files(self):
        retained = set(self._latest_files.values())
        retained.update(identity for p in self._pending for identity, _ in p.handles)
        for identity in tuple(self._files):
            if identity not in retained:
                self._files[identity].close()
                del self._files[identity]

    @staticmethod
    def _coalesced_plan(read_descs, file_epochs, capacity):
        # Each transition is a read fence. Plan independently on either side,
        # then rebase into one existing slot while retaining logical row order.
        groups = {}
        for i, d in enumerate(read_descs):
            epoch = file_epochs[(d.batch_event_index, d.stream_id)]
            groups.setdefault(epoch.fence, []).append((i, ResolvedDgram(
                d.batch_event_index, d.timestamp, d.stream_id, epoch.file,
                d.offset, d.size, d.smd_size,
            )))
        ranges, logical = [], [None] * len(read_descs)
        cursor = 0
        for group in groups.values():
            part = build_read_plan((d for _, d in group), capacity_bytes=capacity - cursor)
            first_range = len(ranges)
            ranges.extend(replace(r, device_offset=r.device_offset + cursor)
                          for r in part.physical_ranges)
            for (i, _), row in zip(group, part.logical_dgrams):
                logical[i] = LogicalDgram(
                    row.source,
                    None if row.range_index is None else row.range_index + first_range,
                    0 if row.range_index is None else row.device_offset + cursor,
                )
            cursor += part.capacity_bytes
        return ReadPlan(0, 0, tuple(ranges), tuple(logical), cursor, cursor, cursor)

    @staticmethod
    def _build_desc_table(read_descs):
        desc_table = np.empty((len(read_descs), DESC_NCOLS), dtype=np.uint64)

        device_offset = 0
        for row, desc in zip(desc_table, read_descs):
            row[DESC_EVENT_INDEX] = desc.batch_event_index
            row[DESC_STREAM_ID] = desc.stream_id
            row[DESC_TIMESTAMP] = desc.timestamp
            row[DESC_FILE_OFFSET] = desc.offset
            row[DESC_READ_SIZE] = desc.size
            row[DESC_DEVICE_OFFSET] = device_offset
            device_offset += desc.size

        return desc_table
