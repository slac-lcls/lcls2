import logging
import math
import os
import sys
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue

import numpy as np

_log = logging.getLogger(__name__)

from psana import dgram, utils
from psana.event import EventEnvelope
from psana.gpu.context import GpuEventState
from psana.gpu.gpu_batch import GPU_DESC_FLAG_VALID, GpuBatchView, GpuSubbatchView
from psana.gpu.gpu_calib import (
    GPUDetector,
    _compute_calib_constants_cpu,
    build_stream_seg_map,
    optimal_kernel_batch_size,
    prep_calib_constants,
)
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_stream import EventPool
from psana.psexp import TransitionId
from psana.psexp.event_manager import EventManager
from psana.psexp.packet_footer import PacketFooter


class _GpuOnlyDgram:
    """Minimal L1Accept metadata for an event whose streams all went to GPU.

    GPU splitting intentionally removes GPU-stream SMD dgrams from the CPU
    batch.  When every selected stream is a GPU stream, EventManager therefore
    returns an all-None dgram list. EventEnvelope still needs timestamp/service
    metadata so Run.events() can preserve the normal API without causing a
    redundant CPU BigData read.
    """

    def __init__(self, timestamp):
        self._timestamp = int(timestamp)
        self._env = int(TransitionId.L1Accept) << 24

    def timestamp(self):
        return self._timestamp

    def env(self):
        return self._env


def _iter_step_events(batch_bytes, configs):
    if not batch_bytes or len(batch_bytes) < 12:
        return

    batch_pf = PacketFooter(view=batch_bytes)
    event_offset = 0
    for event_index in range(batch_pf.n_packets):
        event_size = batch_pf.get_size(event_index)
        event_view = memoryview(batch_bytes)[event_offset : event_offset + event_size]
        event_offset += event_size

        event_pf = PacketFooter(view=event_view)
        event_footer_nbytes = memoryview(event_pf.footer).nbytes
        dgram_offset = 0
        dgrams = [None] * len(configs)
        for i_stream in range(event_pf.n_packets):
            dgram_size = event_pf.get_size(i_stream)
            if dgram_size:
                dgrams[i_stream] = dgram.Dgram(
                    config=configs[i_stream],
                    view=event_view,
                    offset=dgram_offset,
                )
            dgram_offset += dgram_size

        if dgram_offset + event_footer_nbytes != event_size:
            raise RuntimeError(f"Malformed step event {event_index}: dgrams={dgram_offset} footer={event_footer_nbytes} event_size={event_size}")

        service = 0
        for dg in dgrams:
            if dg is not None:
                service = dg.service()
                break
        yield service, dgrams


class _PendingD2H:
    """Token held by GPUResult while its async D→H is in-flight.

    Created by _D2hPipeline._schedule_chunk() immediately after issuing
    cudaMemcpyAsync.  GPUResult.on_cpu calls .get() to wait for the
    transfer and retrieve the host copy.

    Reference-counts the parent _PinnedSlot so the slot is not reused
    until every event in the chunk has called on_cpu (or been GC'd).
    """

    __slots__ = ("_pslot", "_row", "_n_segs")

    def __init__(self, pslot, row: int, n_segs: int):
        self._pslot = pslot
        self._row = row
        self._n_segs = n_segs

    def get(self) -> np.ndarray:
        """Block until D→H complete; return numpy copy; release slot ref."""
        self._pslot.done_event.synchronize()
        data = self._pslot.arr[self._row, : self._n_segs].copy()
        self._pslot.dec_ref()
        self._pslot = None
        return data

    def __del__(self):
        # Safety: if the user never calls on_cpu, release the ref anyway.
        if self._pslot is not None:
            self._pslot.dec_ref()
            self._pslot = None


class _PinnedSlot:
    """One pre-allocated page-locked host buffer for one D→H chunk.

    Pre-allocated during _D2hPipeline.__init__ so that cudaMallocHost
    page-lock latency does not appear in the event loop timing.

    Reference-counted: claim(n) marks n events in-flight; dec_ref()
    releases one reference, and when the count reaches 0 the slot puts
    itself back into the pipeline's _available SimpleQueue so
    _get_free_slot() can retrieve it on the next call.
    """

    def __init__(
        self,
        max_segs: int,
        nrows: int,
        ncols: int,
        chunk_size: int,
        available,
    ):
        import cupy as cp
        import threading

        nbytes = chunk_size * max_segs * nrows * ncols * 4  # float32
        self._mem = cp.cuda.alloc_pinned_memory(nbytes)
        self.arr = np.frombuffer(
            self._mem,
            dtype=np.float32,
            count=chunk_size * max_segs * nrows * ncols,
        ).reshape(chunk_size, max_segs, nrows, ncols)
        self.done_event = cp.cuda.Event(disable_timing=True)
        self._refs = 0
        self._available = available   # SimpleQueue[_PinnedSlot] from _D2hPipeline
        # Guards the decrement-and-check in dec_ref() so that concurrent
        # calls (e.g. multiple threads calling on_cpu on events from the
        # same chunk) cannot produce a lost-update on _refs and silently
        # prevent the slot from being returned to the free pool.
        self._refs_lock = threading.Lock()

    def claim(self, n: int):
        """Mark n events as in-flight on this slot."""
        self._refs = n

    def dec_ref(self):
        """Release one event reference; return slot to free pool when all done.

        The decrement-and-check is protected by _refs_lock so that
        concurrent dec_ref() calls from different threads (e.g. when
        multiple events from the same chunk have on_cpu called in parallel)
        cannot race and produce a lost update on _refs.  Releases after the
        reference count reaches zero are ignored so a slot is never queued
        in the free pool more than once.
        """
        with self._refs_lock:
            if self._refs <= 0:
                return
            self._refs -= 1
            freed = self._refs == 0
        if freed:
            self._available.put(self)  # return slot to the free pool


class _D2hPipeline:
    """Internal GpuEventManager D→H pipeline (not user-facing).

    Issues async D→H from an EventPool slot as soon as its final GPU work is
    submitted.  Result delivery remains separate: GPUResult.on_cpu later waits
    on the attached token and copies out of pinned memory.

    Activated by DataSource(gpu_d2h_chunk_size=N).  N=0 (default)
    bypasses the pipeline; on_cpu then triggers a blocking D→H on first
    access (existing behaviour).
    """

    def __init__(self, det_key: str, chunk_size: int, n_pinned_slots: int = 2):
        self._key = det_key
        self._chunk_size = chunk_size
        self._n_pinned_slots = max(2, int(n_pinned_slots))

        # _available is a SimpleQueue of free _PinnedSlot objects.
        # _get_free_slot() calls get_nowait() — O(1), thread-safe, no scan.
        # dec_ref() calls put(self) when a slot's ref count reaches 0.
        # Using a queue instead of a Semaphore + in_use flag + scan loop
        # reduces three separate mechanisms to one and also eliminates the
        # TOCTOU gap between "slot found free" and "slot claimed".
        self._available: SimpleQueue = SimpleQueue()

        # Lazy: shape not known until first event.
        self._pinned_pool: list = []
        self._d2h_stream = None
        self._n_segs: int | None = None
        self._nrows: int | None = None
        self._ncols: int | None = None

    def schedule(self, slot_record):
        """Arm D→H for every matching result in one execution slot.

        This runs immediately after EventPool.submit().  It never waits for
        normal asynchronous copies: the D→H stream waits on the slot's
        result-ready event and records its own completion event.  If all pinned
        buffers are retained, the result is copied synchronously now and cached
        so a later on_cpu() can never read a reused GPU slot.
        """
        items = []
        key = self._key
        for envelope in slot_record.event_envelopes:
            ts = utils.first_timestamp(envelope.dgrams)
            arr = slot_record.gpu_results_by_ts.get(ts, {}).get(key)
            if arr is None:
                continue
            lease = slot_record.leases_by_ts.get(ts, {}).get(key)
            items.append((ts, lease, arr))

        if not items:
            return
        if self._n_segs is None:
            self._init(items[0][2])

        for start in range(0, len(items), self._chunk_size):
            self._schedule_chunk(slot_record, items[start:start + self._chunk_size])

    def pinned_bytes(self) -> int:
        """Return bytes of pinned (page-locked) host memory currently
        allocated by this pipeline's _PinnedSlot pool.
        Used by GpuEventManager.log_memory() for Phase-0 accounting.
        """
        return sum(s.arr.nbytes for s in self._pinned_pool)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _init(self, arr):
        import cupy as cp

        self._n_segs = int(arr.shape[0])
        self._nrows = int(arr.shape[1])
        self._ncols = int(arr.shape[2])
        for _ in range(self._n_pinned_slots):
            slot = _PinnedSlot(self._n_segs, self._nrows, self._ncols, self._chunk_size, self._available)
            self._pinned_pool.append(slot)
            self._available.put(slot)   # all slots start free
        self._d2h_stream = cp.cuda.Stream(non_blocking=True)

    def _get_free_slot(self):
        """Return a free _PinnedSlot, or None if all slots are occupied.

        SimpleQueue.get_nowait() is O(1), thread-safe, and atomically
        removes the slot from the free pool — eliminating the Semaphore +
        in_use scan that three separate mechanisms previously handled.

        Returns None when the queue is empty.  schedule() then materializes an
        independent CPU result before the execution slot can be reused.
        """
        try:
            return self._available.get_nowait()
        except Empty:
            return None   # queue empty — schedule() materializes CPU data now

    def _schedule_chunk(self, slot_record, chunk):
        """Issue one pinned D→H chunk, or materialize a safe CPU fallback."""
        import cupy as cp
        n_evts = len(chunk)

        pslot = self._get_free_slot()

        # No free pinned slot: materialize while the device lease is valid.
        # Deferring arr.get() until on_cpu() would allow this execution slot to
        # be overwritten first when callers retain event contexts.
        if pslot is None:
            for ts, lease, arr in chunk:
                if lease is not None and lease.result_ready is not None:
                    lease.result_ready.synchronize()
                slot_record.cached_cpu_results_by_ts.setdefault(ts, {})[
                    self._key
                ] = arr.get()
            return

        # ── Issue async D→H ───────────────────────────────────────────────
        pslot.claim(n_evts)
        stream = self._d2h_stream
        row_nbytes = self._n_segs * self._nrows * self._ncols * 4
        dst_base = pslot.arr.ctypes.data
        leases_out = []

        for i, (_, lease, arr) in enumerate(chunk):
            if lease is not None and lease.result_ready is not None:
                stream.wait_event(lease.result_ready)
            cp.cuda.runtime.memcpyAsync(
                dst_base + i * row_nbytes,
                arr.data.ptr,
                arr.nbytes,
                cp.cuda.runtime.memcpyDeviceToHost,
                stream.ptr,
            )
            if lease is not None:
                leases_out.append(lease)

        # Record done_event and register on leases.
        pslot.done_event.record(stream)
        for lease in leases_out:
            lease.register_consumer_done(pslot.done_event)

        # Store host-result tokens on the execution record.  Context delivery
        # later attaches them to GPUResult without scheduling new CUDA work.
        for i, (ts, _, _) in enumerate(chunk):
            slot_record.pending_d2h_by_ts.setdefault(ts, {})[self._key] = (
                _PendingD2H(pslot, i, self._n_segs)
            )


def _fmt_mib(n: int) -> str:
    """Format byte count as MiB string for logging."""
    return f"{n / 1024**2:.1f} MiB"


@dataclass
class _GpuMemStats:
    """Snapshot of GPU and pinned-host memory broken down by owner.

    All values are bytes.  Recorded by GpuEventManager.log_memory() and used
    to update per-category high-water marks.

    GPU categories (device VRAM):
        constants    calibration constants per detector (peds + gmask)
        geometry     scatter-index arrays for image assembly
        calib_slots  per-slot calibrated-output buffers (grow lazily)
        raw_slots    per-slot raw-gather buffers (grow lazily)
        raw_input    KvikioGpuReader per-slot input buffers
        cupy_pool    CuPy memory-pool total committed bytes
        device_used  bytes in use according to CUDA (total - free)
        device_total total device memory

    Pinned-host category:
        pinned       _D2hPipeline _PinnedSlot allocations
    """

    # per-detector breakdowns
    det_constants: dict = field(default_factory=dict)  # {det_name: bytes}
    det_geometry: dict = field(default_factory=dict)
    det_calib_slots: dict = field(default_factory=dict)
    det_raw_slots: dict = field(default_factory=dict)
    # aggregate GPU
    raw_input: int = 0
    cupy_pool: int = 0
    device_used: int = 0
    device_total: int = 0
    # pinned host
    pinned: int = 0
    # label for logging
    label: str = ""

    # _mb is now the module-level _fmt_mib; kept as alias for log() callers
    _mb = staticmethod(lambda n: _fmt_mib(n))

    def log(self):
        """Emit a structured INFO log summarising the snapshot."""
        det_names = sorted(self.det_constants)
        for name in det_names:
            _log.info(
                "GPU mem [%s] det=%s  constants=%s  geometry=%s  calib_slots=%s  raw_slots=%s",
                self.label,
                name,
                self._mb(self.det_constants.get(name, 0)),
                self._mb(self.det_geometry.get(name, 0)),
                self._mb(self.det_calib_slots.get(name, 0)),
                self._mb(self.det_raw_slots.get(name, 0)),
            )
        _log.info(
            "GPU mem [%s] raw_input=%s  cupy_pool=%s  device_used=%s / %s  pinned=%s",
            self.label,
            self._mb(self.raw_input),
            self._mb(self.cupy_pool),
            self._mb(self.device_used),
            self._mb(self.device_total),
            self._mb(self.pinned),
        )


class GpuEventManager:
    """Run-scoped GPU event processor.

    The serial path still drives this object through its iterator compatibility
    interface. The MPI path supplies coherent SMD/GPU batches directly through
    ``process_batch``; the manager never owns MPI communication.
    """

    def __init__(
        self,
        configs,
        dm,
        max_retries,
        use_smds,
        shared_state,
        dsparms,
        run,
        smdr_man=None,
        setup_geometry=True,
        prebuilt_geometry=None,
        calib_leader=True,
    ):
        self.configs = configs
        self.dm = dm
        self.max_retries = max_retries
        self.use_smds = use_smds
        self.shared_state = shared_state
        self.dsparms = dsparms
        self.run = run
        self._d2h_pipelines: dict = {}  # populated at end of __init__
        self.smdr_man = smdr_man
        self._setup_geometry = setup_geometry
        self._prebuilt_geometry = prebuilt_geometry  # {det_name: (ix_all, iy_all)}

        self._batch_iter = iter([])
        self._iter = None
        self._has_gpu_batch_iter = False   # cached in beginrun; avoids per-batch hasattr
        self._n_events = 0
        self._done = False
        self._closed = False

        self.gpu_det_names = self._normalize_gpu_det(dsparms.gpu_det)
        self.gpu_detectors = {}
        self.event_pool = None
        self.gpu_reader = None
        # At most one KvikIO read is pre-issued ahead of the CPU event loop.
        # Keep explicit ownership so generator close/early termination can
        # drain it before gpu_reader.close() releases its buffers.
        self._pending_gpu_read = None

        self._setup_detectors(calib_leader=calib_leader)

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = self._events()
        return next(self._iter)

    def _snapshot_memory(self, label: str) -> _GpuMemStats:
        """Collect a _GpuMemStats snapshot from all pipeline components."""
        s = _GpuMemStats(label=label)
        for name, (_, det) in self.gpu_detectors.items():
            m = det.memory_bytes()
            s.det_constants[name] = m["constants"]
            s.det_geometry[name] = m["geometry"]
            s.det_calib_slots[name] = m["calib_slots"]
            s.det_raw_slots[name] = m["raw_slots"]
        if self.gpu_reader is not None and hasattr(self.gpu_reader, "memory_bytes"):
            s.raw_input = self.gpu_reader.memory_bytes()["raw_input_slots"]
        s.pinned = sum(p.pinned_bytes() for p in self._d2h_pipelines.values())
        # Query CuPy pool and CUDA device info only when a GPU is active.
        # These calls fail on CPU-only nodes and are skipped silently.
        cupy_mod = sys.modules.get("cupy")
        if cupy_mod is not None:
            try:
                # Probe the runtime before constructing/accessing CuPy's
                # device-local memory pool.  On CPU-only hosts, touching the
                # pool first can leave a partially initialized object whose
                # destructor raises an unraisable CUDA driver exception.
                if cupy_mod.cuda.runtime.getDeviceCount() <= 0:
                    return s
                s.cupy_pool = cupy_mod.get_default_memory_pool().total_bytes()
                free, total = cupy_mod.cuda.Device().mem_info
                s.device_used = total - free
                s.device_total = total
            except Exception:
                pass
        return s

    def log_memory(self, label: str = ""):
        """Snapshot memory usage, update high-water marks, and log.

        Emits one INFO log line per detector plus a summary line.
        Call after GPU setup, after the first batch, and at EndRun.

        High-water marks track the peak value seen for each category
        across all calls within this run.
        """
        s = self._snapshot_memory(label)
        s.log()
        hw = self._high_water
        for name in s.det_constants:
            hw["constants"] = max(hw.get("constants", 0), s.det_constants.get(name, 0))
            hw["geometry"] = max(hw.get("geometry", 0), s.det_geometry.get(name, 0))
            hw["calib_slots"] = max(hw.get("calib_slots", 0), s.det_calib_slots.get(name, 0))
            hw["raw_slots"] = max(hw.get("raw_slots", 0), s.det_raw_slots.get(name, 0))
        hw["raw_input"] = max(hw.get("raw_input", 0), s.raw_input)
        hw["cupy_pool"] = max(hw.get("cupy_pool", 0), s.cupy_pool)
        hw["device_used"] = max(hw.get("device_used", 0), s.device_used)
        hw["pinned"] = max(hw.get("pinned", 0), s.pinned)

    def log_high_water(self):
        """Log the peak memory values seen since the last reset."""
        hw = self._high_water
        _log.info(
            "GPU mem high-water  constants=%s  geometry=%s  calib_slots=%s  raw_slots=%s  raw_input=%s  cupy_pool=%s  device_used=%s  pinned=%s",
            _fmt_mib(hw.get("constants", 0)),
            _fmt_mib(hw.get("geometry", 0)),
            _fmt_mib(hw.get("calib_slots", 0)),
            _fmt_mib(hw.get("raw_slots", 0)),
            _fmt_mib(hw.get("raw_input", 0)),
            _fmt_mib(hw.get("cupy_pool", 0)),
            _fmt_mib(hw.get("device_used", 0)),
            _fmt_mib(hw.get("pinned", 0)),
        )

    @staticmethod
    def _normalize_gpu_det(gpu_det):
        if gpu_det is None:
            return []
        if isinstance(gpu_det, str):
            return [gpu_det]
        return list(gpu_det)

    def _setup_detectors(self, calib_leader=True):
        # Budget must exist before constructing GPUDetector objects.
        from psana.gpu.gpu_budget import _GpuBudget

        budget_gb = float(getattr(self.dsparms, "gpu_memory_budget_gb", 0) or 0)
        if budget_gb > 0:
            self._gpu_budget = _GpuBudget(limit_bytes=int(budget_gb * 1024**3))
        else:
            n_bd = max(1, int(os.environ.get("PS_BD_NODES", 1)))
            self._gpu_budget = _GpuBudget.auto(n_bd_ranks=n_bd)

        opt_batch_sizes = []
        ids_table = getattr(self.dsparms, "det_stream_ids_table", {})
        segments_table = getattr(
            self.dsparms, "det_stream_segments_table", {}
        )
        streams_by_detector = {
            name: sorted(ids_table.get(name) or segments_table.get(name, {}).keys())
            for name in self.gpu_det_names
        }
        missing = [name for name, stream_ids in streams_by_detector.items()
                   if not stream_ids]
        if missing:
            raise RuntimeError(
                f"gpu_det did not resolve to any stream ids: {missing}"
            )

        all_gpu_stream_ids = {
            stream_id
            for stream_ids in streams_by_detector.values()
            for stream_id in stream_ids
        }
        requested_stream_ids = getattr(self.dsparms, "gpu_stream_ids", None)
        if (requested_stream_ids is not None
                and set(requested_stream_ids) != all_gpu_stream_ids):
            raise RuntimeError(
                "GPU stream routing must include every stream for each "
                f"gpu_det: expected {sorted(all_gpu_stream_ids)}, got "
                f"{sorted(requested_stream_ids)}"
            )

        selected_detectors = set(self.gpu_det_names)
        stream_owners = getattr(self.dsparms, "stream_id_to_detnames", {})
        for stream_id in sorted(all_gpu_stream_ids):
            cpu_only = set(stream_owners.get(stream_id, ())) - selected_detectors
            if cpu_only:
                raise RuntimeError(
                    f"GPU stream {stream_id} also contains detector(s) "
                    f"{sorted(cpu_only)}. EventBuilder routes whole streams, "
                    "so every detector on that stream must be selected by "
                    "gpu_det."
                )
        self.dsparms.gpu_stream_ids = sorted(all_gpu_stream_ids)

        from psana.gpu.gpu_mpi import log_gpu_mem

        try:
            from mpi4py import MPI

            _rank = MPI.COMM_WORLD.Get_rank()
        except Exception:
            _rank = None

        log_gpu_mem("_setup_detectors entry", rank=_rank)
        for det_name in self.gpu_det_names:
            det = self.run.Detector(det_name)
            det_type = getattr(det, "_dettype", None)

            # Determine whether bigdata carries raw uint16 ADC data ('raw'
            # drp_class) or DRP-calibrated float32 data ('fex' or similar).
            drp_classes = {k[1] for k in self.run.detinfo if k[0] == det_name}
            is_pre_calibrated = 'raw' not in drp_classes

            if not is_pre_calibrated and det_type != "jungfrau":
                raise NotImplementedError(
                    f"gpu_det={det_name!r} has detector type {det_type!r}; "
                    "the integrated GPU calibration path currently supports "
                    "only Jungfrau.  Pre-calibrated (fex) data can be used "
                    "via passthrough mode regardless of detector type."
                )

            if is_pre_calibrated:
                peds_gpu  = None
                gmask_gpu = None
                _log.info(
                    "gpu_det=%r: drp_classes=%s — using passthrough mode "
                    "(bigdata is pre-calibrated float32; fused_calib_gpu skipped)",
                    det_name, sorted(drp_classes),
                )
            elif not calib_leader:
                # Follower BD rank sharing a GPU with the leader.
                # is_calib_leader() returned False before _setup_detectors() was
                # called, so this rank must NOT allocate peds_gpu/gmask_gpu here.
                # share_calib_between_gpu_peers() will populate them later via
                # CUDA IPC handles from the leader — at zero allocation cost.
                peds_gpu  = None
                gmask_gpu = None
                _log.info(
                    "gpu_det=%r: follower BD rank — skipping prep_calib_constants; "
                    "calibration constants will be shared from leader via CUDA IPC",
                    det_name,
                )
            else:
                peds_gpu, gmask_gpu = prep_calib_constants(det)
                log_gpu_mem(f"after prep_calib_constants ({det_name})", rank=_rank)
            det_shape = det.calibconst["pedestals"][0].shape[1:]

            stream_segments = dict(segments_table.get(det_name, {}))
            gpu_stream_ids = streams_by_detector[det_name]

            # Configure identifies which physical segments belong to each
            # stream, but its dictionary order is not necessarily the order
            # of ShapesData children in L1Accept.  The fixed-stride GPU gather
            # preserves L1 child order, so discover that order from the first
            # detector event in each routed bigdata stream.
            xtc_files = getattr(self.dm, "xtc_files", None)
            if xtc_files is None:
                xtc_files = getattr(self.dsparms, "xtc_files", [])
            stream_files = {stream_id: xtc_files[stream_id] for stream_id in gpu_stream_ids if stream_id < len(xtc_files)}
            stream_seg_map = build_stream_seg_map(stream_files, det_name)

            for stream_id in gpu_stream_ids:
                segment_ids = stream_seg_map.get(stream_id)
                if not segment_ids:
                    raise RuntimeError(f"gpu_det={det_name!r} could not determine L1Accept segment order for stream {stream_id}")
                configured = set(stream_segments.get(stream_id, []))
                if configured and set(segment_ids) != configured:
                    raise RuntimeError(f"gpu_det={det_name!r} stream {stream_id} segment mismatch: Configure={sorted(configured)} L1Accept={segment_ids}")
            configured_segment_ids = sorted({
                segment_id
                for stream_id in gpu_stream_ids
                for segment_id in stream_segments.get(stream_id, ())
            })
            detector_api = next(
                (
                    getattr(det, drp_class, None)
                    for drp_class in sorted(drp_classes)
                    if hasattr(
                        getattr(det, drp_class, None),
                        "_sorted_segment_inds",
                    )
                ),
                None,
            )
            canonical_segment_ids = list(
                getattr(detector_api, "_sorted_segment_inds", configured_segment_ids)
            )
            routed_segment_ids = {
                segment_id
                for stream_id in gpu_stream_ids
                for segment_id in stream_seg_map.get(stream_id, ())
            }
            if routed_segment_ids != set(canonical_segment_ids):
                raise RuntimeError(
                    f"gpu_det={det_name!r} must route all detector segments: "
                    f"configured={canonical_segment_ids}, "
                    f"routed={sorted(routed_segment_ids)}"
                )

            gpu_detector = GPUDetector(
                det_shape=det_shape,
                peds_gpu=peds_gpu,
                gmask_gpu=gmask_gpu,
                stream_seg_map=stream_seg_map or None,
                canonical_segment_ids=canonical_segment_ids,
                n_slots=getattr(self.dsparms, "n_gpu_streams", 2),
                budget=self._gpu_budget,
                passthrough=is_pre_calibrated,
            )
            if self._prebuilt_geometry and det_name in self._prebuilt_geometry:
                ix_all, iy_all = self._prebuilt_geometry[det_name]
                gpu_detector.setup_geometry_from_arrays(ix_all, iy_all)
            elif self._setup_geometry:
                gpu_detector.setup_geometry(det)
            log_gpu_mem(f"after setup_geometry ({det_name})", rank=_rank)

            opt_batch_sizes.append(optimal_kernel_batch_size(det_shape))
            self.gpu_detectors[det_name] = (det, gpu_detector)

        if not self.dsparms.batch_size:
            self.dsparms.batch_size = min(opt_batch_sizes) if opt_batch_sizes else 1

        pool_depth = getattr(self.dsparms, "n_gpu_streams", 2)
        self.event_pool = EventPool(n=pool_depth)

        # KvikioGpuReader: pre-allocate one data_gpu buffer per slot.
        # _gpu_budget was already created in _setup_detectors() above and
        # is shared with every GPUDetector so all allocations are counted
        # against the same limit.
        self.gpu_reader = KvikioGpuReader(n_slots=pool_depth, budget=self._gpu_budget)

        # Internal D→H pipeline — activated when gpu_d2h_chunk_size > 0.
        # Transfers calibrated results to pinned host memory in chunks so that
        # evt.gpu.get('det.calib').on_cpu returns without triggering
        # an additional synchronous D→H at the user's call site.
        chunk_size = getattr(self.dsparms, "gpu_d2h_chunk_size", 0) or 0
        if chunk_size > 0 and self.gpu_det_names:
            # One pipeline per GPU detector key.
            self._d2h_pipelines = {
                f"{det_name}.calib": _D2hPipeline(
                    det_key=f"{det_name}.calib",
                    chunk_size=chunk_size,
                    n_pinned_slots=pool_depth,
                )
                for det_name in self.gpu_det_names
            }
        else:
            self._d2h_pipelines = {}

        # Report which I/O path kvikio will use for this run.
        # GDS (compat_mode=False) reads NVMe → GPU VRAM directly (fast).
        # CPU-fallback (compat_mode=True) reads NVMe → CPU DRAM → GPU VRAM
        # via cudaMemcpy (slower; common on Lustre/GPFS filesystems like S3DF).
        _path = self.gpu_reader.io_path
        if self.gpu_reader._compat_mode:
            _log.warning(
                "GpuEventManager: kvikio I/O path = %s "
                "(NVMe → CPU DRAM → GPU VRAM via cudaMemcpy). "
                "True GDS is not available — likely Lustre/GPFS filesystem "
                "or cuFile driver not loaded.  GDS would give NVMe → GPU VRAM "
                "directly, bypassing CPU DRAM entirely.",
                _path,
            )
        else:
            _log.info("GpuEventManager: kvikio I/O path = %s (NVMe → GPU VRAM direct)", _path)

        # Phase-0 accounting: high-water marks reset each run.
        self._high_water: dict = {}
        self._first_batch_logged = False

        # Log fixed allocations (constants + geometry already on GPU).
        try:
            self.log_memory("after_setup")
        except Exception:
            pass

        # Phase-3: per-subbatch byte budget for byte-bounded splitting.
        # Computed once after all GPU detectors are set up.
        self._subbatch_budget_bytes = self._compute_subbatch_budget()

    # ------------------------------------------------------------------
    # Phase 3: byte-bounded subbatch helpers
    # ------------------------------------------------------------------

    def _compute_subbatch_budget(self) -> int:
        """Compute per-subbatch VRAM byte budget for Phase 3 splitting.

        The budget is:
          (total_limit - fixed_bytes - 10% margin) / n_slots

        where fixed_bytes = calibration constants + geometry arrays already
        reserved in _gpu_budget._committed.  The 10% margin covers CuPy
        allocator overhead, input buffers (KvikioGpuReader), and rounding.

        Configurable override: set DsParms.gpu_subbatch_budget_bytes > 0.

        Returns
        -------
        int  — bytes per subbatch.  At least 256 MiB to prevent splitting
               every single event on low-budget or CPU-only nodes.
        """
        _min = 256 * 1024 * 1024   # 256 MiB floor

        override = int(getattr(self.dsparms, 'gpu_subbatch_budget_bytes', 0) or 0)
        if override > 0:
            return override

        if not self.gpu_detectors:
            return _min

        try:
            fixed_bytes = 0
            for _, (_, det) in self.gpu_detectors.items():
                mb = det.memory_bytes()
                fixed_bytes += mb['constants'] + mb['geometry']
        except Exception:
            fixed_bytes = 0

        limit   = self._gpu_budget.limit()
        margin  = int(limit * 0.10)   # 10% headroom for CuPy pool etc.
        n_slots = max(1, getattr(self.dsparms, 'n_gpu_streams', 2))
        variable = max(0, limit - fixed_bytes - margin)
        budget   = variable // n_slots

        return max(_min, budget)

    def _split_subbatches(self, gpu_view) -> list:
        """Partition gpu_view into byte-bounded GpuSubbatchViews.

        Each subbatch is sized so that:
          calib_detector_bytes + raw_input_bytes <= _subbatch_budget_bytes

        The first event is always included even if it alone exceeds the
        budget (a single oversized event cannot be split further).

        Events from the same EB batch that fit within the budget are grouped
        together to fill one EventPool slot efficiently.

        Parameters
        ----------
        gpu_view : GpuBatchView

        Returns
        -------
        list[GpuSubbatchView]  — at least one element; may be one entry
                                 equal to the whole batch if no split needed.
        """
        n_events = gpu_view.header.n_events
        if n_events == 0:
            return []

        budget = self._subbatch_budget_bytes

        # Estimate calibration bytes per event (sum across all GPU detectors).
        calib_bytes_per_event = sum(
            det_obj.estimate_subbatch_bytes(1)
            for _, (_, det_obj) in self.gpu_detectors.items()
        )

        # Per-event raw input bytes from the desc table (varies by event).
        per_event_raw = []
        for i in range(n_events):
            raw_bytes = 0
            for desc in gpu_view.desc_rows_for_event(i):
                if int(desc['flags']) & GPU_DESC_FLAG_VALID:
                    raw_bytes += int(desc['bd_size'])
            per_event_raw.append(raw_bytes)

        # Greedy bin-packing: accumulate events until budget exceeded.
        subbatches = []
        start = 0
        current_bytes = 0

        for i in range(n_events):
            event_bytes = calib_bytes_per_event + per_event_raw[i]

            if i == start:
                # Always include at least one event (even if over budget).
                current_bytes = event_bytes
            elif current_bytes + event_bytes > budget:
                # Current accumulation would exceed budget — flush subbatch.
                subbatches.append(GpuSubbatchView(gpu_view, start, i))
                start = i
                current_bytes = event_bytes
            else:
                current_bytes += event_bytes

        # Final subbatch (always present).
        subbatches.append(GpuSubbatchView(gpu_view, start, n_events))
        return subbatches

    def _next_batch(self):
        if self.smdr_man is None:
            raise StopIteration

        while True:
            if self.shared_state.terminate_flag.value:
                raise StopIteration

            try:
                if self._has_gpu_batch_iter:
                    return self._batch_iter.next_with_gpu()
                batch_dict, step_dict = next(self._batch_iter)
                return batch_dict, {}, step_dict
            except StopIteration:
                self._batch_iter = next(self.smdr_man)
                self._has_gpu_batch_iter = hasattr(self._batch_iter, "next_with_gpu")

    def _dispatch_transition(self, service, dgrams):
        if service == TransitionId.BeginStep:
            for det_info in self.gpu_detectors.values():
                det, gpu_detector = det_info[0], det_info[1]
                # Skip constant computation for passthrough detectors — they have
                # no calibration constants, and beginstep() is a no-op for them.
                if getattr(gpu_detector, '_passthrough', False):
                    continue
                peds, gmask = _compute_calib_constants_cpu(det)
                gpu_detector.beginstep(peds, gmask)

        self.run._handle_transition(dgrams)

    def _handle_steps(self, step_dict):
        end_run_seen = False
        if not step_dict:
            return end_run_seen

        pending_transitions = []
        for step_batch, _ in step_dict.values():
            for service, dgrams in _iter_step_events(step_batch, self.configs):
                if TransitionId.isEvent(service):
                    continue
                pending_transitions.append((service, dgrams))

        needs_drain = any(service in (TransitionId.BeginStep, TransitionId.EndRun) for service, _ in pending_transitions)
        if needs_drain:
            yield from self._flush_event_pool()

        for service, dgrams in pending_transitions:
            if service == TransitionId.EndRun:
                end_run_seen = True
                try:
                    self.log_memory("end_run")
                    self.log_high_water()
                except Exception:
                    pass
            self._dispatch_transition(service, dgrams)

        return end_run_seen

    def _attach_gpu(self, envelope, gpu_results, leases=None,
                    pending_d2h=None, cached_cpu_results=None,
                    device_released=False):
        state = GpuEventState(
            gpu_results=gpu_results,
            detector_names=self.gpu_det_names,
            leases=leases,
            pending_d2h=pending_d2h,
            cached_cpu_results=cached_cpu_results,
            device_released=device_released,
        )
        return EventEnvelope(dgrams=envelope.dgrams, gpu_state=state)

    def _submit_gpu(self, subbatch, gpu_read, event_envelopes):
        """Submit one device slot and arm its automatic D→H immediately."""
        record = self.event_pool.submit(
            subbatch,
            gpu_read,
            event_envelopes,
            self.gpu_detectors,
        )
        for pipe in self._d2h_pipelines.values():
            pipe.schedule(record)
        return record

    @staticmethod
    def _is_fully_host_backed(ready):
        """Return True when every slot-backed result has a host handoff.

        A pending pinned D2H token or an independent CPU fallback both make
        the corresponding result safe after its device slot is reused.
        """
        if ready is None:
            return True
        for ts, results in ready.gpu_results_by_ts.items():
            pending = ready.pending_d2h_by_ts.get(ts, {})
            cached = ready.cached_cpu_results_by_ts.get(ts, {})
            if any(key not in pending and key not in cached for key in results):
                return False
        return True

    def _yield_ready(self, ready, device_released=False):
        if ready is None:
            return
        # Log after the first batch: slot buffers have grown to their
        # initial sizes so this shows the steady-state allocation.
        if not self._first_batch_logged and ready.event_envelopes:
            self._first_batch_logged = True
            self.log_memory("first_batch")
        for envelope in ready.event_envelopes:
            ts = utils.first_timestamp(envelope.dgrams)
            gpu_results = ready.gpu_results_by_ts.get(ts, {})
            if device_released:
                # Preserve the result-key API but never retain a stale view
                # into a slot which the replacement H2D may now overwrite.
                gpu_results = {key: None for key in gpu_results}
            yield self._attach_gpu(
                envelope,
                gpu_results,
                leases={} if device_released else ready.leases_by_ts.get(ts, {}),
                pending_d2h=ready.pending_d2h_by_ts.pop(ts, {}),
                cached_cpu_results=ready.cached_cpu_results_by_ts.get(ts, {}),
                device_released=device_released,
            )

    def _issue_gpu_read(self, subbatch, slot_id):
        """Issue and own the single read allowed ahead of CPU processing."""
        if getattr(self, '_pending_gpu_read', None) is not None:
            raise RuntimeError("a pre-issued GPU read is already outstanding")
        pending = self.gpu_reader.issue_batch(subbatch, self.dm, slot_id=slot_id)
        self._pending_gpu_read = pending
        return pending

    def _wait_gpu_read(self, pending):
        """Complete a read and relinquish its controller-side ownership."""
        if getattr(self, '_pending_gpu_read', None) is not pending:
            raise RuntimeError("attempted to wait for an unowned GPU read")
        try:
            return self.gpu_reader.wait_batch(pending)
        finally:
            self._pending_gpu_read = None

    def _drain_pending_gpu_read(self):
        """Finish a pre-issued read before the reader and buffers are closed."""
        pending = getattr(self, '_pending_gpu_read', None)
        if pending is None:
            return
        try:
            self.gpu_reader.wait_batch(pending)
        finally:
            self._pending_gpu_read = None

    def _retire_issue_and_yield(self, subbatch):
        """Retire one slot, issue its replacement read, and yield its result.

        Automatic-D2H results are already host-backed.  Their terminal D2H
        consumer is joined first, then the freed slot receives the replacement
        H2D before CPU code sees the old event.  External-GPU mode retains the
        original yield-first registration window so a user kernel can attach a
        completion event before the slot is released.
        """
        ready = self.event_pool.begin_retire_next()
        release_before_yield = (
            bool(self._d2h_pipelines) and self._is_fully_host_backed(ready)
        )

        if release_before_yield:
            self.event_pool.finish_retire_next()
            slot = self.event_pool.next_slot_id
            pending = self._issue_gpu_read(subbatch, slot)
            yield from self._yield_ready(ready, device_released=True)
            return pending

        try:
            yield from self._yield_ready(ready)
        finally:
            self.event_pool.finish_retire_next()
        slot = self.event_pool.next_slot_id
        return self._issue_gpu_read(subbatch, slot)

    def _flush_event_pool(self):
        for slot_data in self.event_pool.flush():
            yield from self._yield_ready(slot_data)

    def _process_batch(self, batch_dict, gpu_batch_dict, step_dict):
        n_events = self._n_events
        try:
            while True:
                end_run_seen = yield from self._handle_steps(step_dict)

                # ── Phase 3: GPU path — split batch into subbatches ──────────
                # Parse every GPU batch from this EB communication and split
                # each into byte-bounded GpuSubbatchViews.  Issue the FIRST
                # subbatch's reads now (before the CPU EventManager loop) so
                # GDS/PCIe I/O overlaps with CPU SMD deserialization.
                all_subbatches = []
                first_pending  = None   # (subbatch_0, PendingBatch)

                for gpu_batch, _ in gpu_batch_dict.values():
                    gpu_view = GpuBatchView(gpu_batch, validate=True)
                    if not gpu_view.has_work:
                        continue
                    all_subbatches.extend(self._split_subbatches(gpu_view))

                if all_subbatches:
                    first_pending = (
                        all_subbatches[0],
                        (yield from self._retire_issue_and_yield(
                            all_subbatches[0]
                        )),
                    )

                # ── CPU path ─────────────────────────────────────────────────
                # EventManager loop runs while subbatch 0 reads are in-flight.
                stop_after = False
                event_envelopes = []
                for smd_batch, _ in batch_dict.values():
                    if not smd_batch:
                        continue
                    event_manager = EventManager(
                        smd_batch,
                        self.configs,
                        self.dm,
                        self.max_retries,
                        self.use_smds,
                    )
                    for envelope in event_manager:
                        dgrams = envelope.dgrams
                        # All GPU streams are represented in GPUBAT1, not the
                        # CPU batch.  A GPU-only dataset therefore has no CPU
                        # dgram from which the envelope could obtain service/time.
                        # Synthesize that metadata below from GPUBAT1 instead.
                        if not any(dgrams):
                            continue
                        if not TransitionId.isEvent(utils.first_service(dgrams)):
                            continue
                        event_envelopes.append(envelope)
                    if event_manager.exit_id:
                        raise RuntimeError(f"EventManager exit {event_manager.exit_id}")

                # ── Submit subbatches ─────────────────────────────────────────
                if all_subbatches:
                    # Build a timestamp → envelope lookup, filling GPU-only
                    # events from GPUBAT1 timestamps without reading BigData
                    # through the CPU path.
                    ts_to_envelope = {
                        utils.first_timestamp(envelope.dgrams): envelope
                        for envelope in event_envelopes
                    }
                    selected_envelopes = []
                    for subbatch in all_subbatches:
                        for timestamp in subbatch.timestamps:
                            if (
                                self.dsparms.max_events > 0
                                and n_events >= self.dsparms.max_events
                            ):
                                stop_after = True
                                break
                            timestamp = int(timestamp)
                            envelope = ts_to_envelope.get(timestamp)
                            if envelope is None:
                                dgrams = [None] * len(self.configs)
                                dgrams[0] = _GpuOnlyDgram(timestamp)
                                envelope = EventEnvelope(dgrams=dgrams)
                            selected_envelopes.append(envelope)
                            n_events += 1
                        if stop_after:
                            break
                    event_envelopes = selected_envelopes
                    ts_to_envelope = {
                        utils.first_timestamp(envelope.dgrams): envelope
                        for envelope in event_envelopes
                    }

                    for i, subbatch in enumerate(all_subbatches):
                        # Event envelopes whose timestamps appear in this subbatch.
                        sb_ts  = subbatch.timestamps
                        sb_envelopes = [
                            ts_to_envelope[ts]
                            for ts in sb_ts
                            if ts in ts_to_envelope
                        ]

                        if i == 0 and first_pending is not None:
                            # Subbatch 0: reads were already issued before the
                            # CPU loop.  Just wait for them to complete.
                            _, pending_0 = first_pending
                            gpu_read = self._wait_gpu_read(pending_0)
                            self._submit_gpu(subbatch, gpu_read, sb_envelopes)
                        else:
                            pending = yield from self._retire_issue_and_yield(
                                subbatch
                            )
                            gpu_read = self._wait_gpu_read(pending)
                            self._submit_gpu(subbatch, gpu_read, sb_envelopes)
                else:
                    # No GPU batch — yield CPU-only events directly.
                    for envelope in event_envelopes:
                        if (
                            self.dsparms.max_events > 0
                            and n_events >= self.dsparms.max_events
                        ):
                            stop_after = True
                            break
                        n_events += 1
                        yield self._attach_gpu(envelope, {})

                if stop_after or end_run_seen:
                    yield from self._flush_event_pool()
                    self._done = True
                return
        finally:
            self._n_events = n_events

    def process_batch(self, smd_batch, gpu_batch=None):
        """Process one coherent EB-to-BD batch and yield EventEnvelopes."""
        if self._done:
            return
        if self.gpu_reader is not None:
            self.gpu_reader.reset_io_stats()
        batch_dict = {0: (smd_batch, [])}
        gpu_batch_dict = {0: (gpu_batch, [])} if gpu_batch else {}
        # MPI transition history is embedded in the SMD packet itself.
        step_dict = {0: (smd_batch, [])}
        yield from self._process_batch(batch_dict, gpu_batch_dict, step_dict)

    def get_bd_read_stats(self):
        """Return bytes and seconds spent in GPU big-data reads this batch."""
        if self.gpu_reader is None:
            return 0, 0.0
        stats = self.gpu_reader.io_stats()
        return int(stats["total_bytes"]), stats["total_ns"] / 1e9

    def finish(self):
        """Drain in-flight work and close GPU reader resources once."""
        if self._closed:
            return
        try:
            yield from self._flush_event_pool()
            self._drain_pending_gpu_read()
        finally:
            if self.gpu_reader is not None:
                self.gpu_reader.close()
            self._closed = True

    def close(self):
        """Discard remaining deliveries while safely retiring their slots."""
        for _ in self.finish():
            pass

    def _events(self):
        try:
            while not self._done:
                try:
                    batch_dict, gpu_batch_dict, step_dict = self._next_batch()
                except StopIteration:
                    break
                yield from self._process_batch(
                    batch_dict, gpu_batch_dict, step_dict
                )
        except BaseException:
            self.close()
            raise
        else:
            yield from self.finish()
