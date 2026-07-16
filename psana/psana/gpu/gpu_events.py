import logging
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

_log = logging.getLogger(__name__)

from psana import dgram
from psana.event import Event
from psana.gpu.context import GpuEventContext
from psana.gpu.detector_router import DetectorRouter
from psana.gpu.gpu_batch import GpuBatchView
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


def _apply_full_routing(gpu_results, evt, gpu_detectors, router):
    if not router or not hasattr(router, "has_full_routing"):
        return gpu_results

    for det_name, det_info in gpu_detectors.items():
        if not router.has_full_routing(det_name):
            continue
        calib_key = f"{det_name}.calib"
        gpu_calib = gpu_results.get(calib_key)
        if gpu_calib is None:
            continue

        det = det_info[0]
        cpu_calib = router.compute_cpu_calib(det_name, det, evt)
        combined = router.assemble_full_calib(det_name, gpu_calib, cpu_calib)
        if combined is not None:
            gpu_results[calib_key] = combined

    return gpu_results


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

    Created by _D2hPipeline._flush_chunk() immediately after issuing
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
    releases one; when the count reaches 0 the slot is free for reuse.
    """

    def __init__(self, max_segs: int, nrows: int, ncols: int, chunk_size: int):
        import cupy as cp

        nbytes = chunk_size * max_segs * nrows * ncols * 4  # float32
        self._mem = cp.cuda.alloc_pinned_memory(nbytes)
        self.arr = np.frombuffer(
            self._mem,
            dtype=np.float32,
            count=chunk_size * max_segs * nrows * ncols,
        ).reshape(chunk_size, max_segs, nrows, ncols)
        self.done_event = cp.cuda.Event(disable_timing=True)
        self.in_use = False
        self._refs = 0

    def claim(self, n: int):
        """Mark n events as in-flight on this slot."""
        self.in_use = True
        self._refs = n

    def dec_ref(self):
        """Release one event reference; mark free when all events done."""
        self._refs -= 1
        if self._refs <= 0:
            self._refs = 0
            self.in_use = False


class _D2hPipeline:
    """Internal GpuEvents D→H pipeline (not user-facing).

    Issues async D→H from EventPool slot views to pinned host memory,
    then yields contexts IMMEDIATELY — before the transfer completes.
    GPUResult.on_cpu waits for the CUDA done-event lazily at the call
    site, so the generator stays thin and D→H overlaps with whatever
    work the user does between receiving a context and calling on_cpu.

    Activated by DataSource(gpu_d2h_chunk_size=N).  N=0 (default)
    bypasses the pipeline; on_cpu then triggers a blocking D→H on first
    access (existing behaviour).
    """

    def __init__(self, det_key: str, chunk_size: int):
        self._key = det_key
        self._chunk_size = chunk_size
        self._max_inflight = 2  # pinned slots; 2 lets one transfer while next fills

        # Lazy: shape not known until first event.
        self._pinned_pool: list = []
        self._d2h_stream = None
        self._n_segs: int | None = None
        self._nrows: int | None = None
        self._ncols: int | None = None

        # Accumulator for the current in-progress chunk.
        self._chunk_buf: list = []  # list of (ctx, lease, arr)

    def add(self, ctx) -> list:
        """Add one context.  Returns list of contexts ready to yield.

        Contexts are returned immediately once their D→H has been
        *issued* (not waited for).  The caller yields them; the user
        completes the sync lazily via GPUResult.on_cpu.
        """
        arr = ctx._gpu_results.get(self._key)
        lease = ctx._leases.get(self._key)

        if arr is None:
            return [ctx]  # no GPU data for this key — pass through

        if self._n_segs is None:
            self._init(arr)

        self._chunk_buf.append((ctx, lease, arr))

        if len(self._chunk_buf) >= self._chunk_size:
            return self._flush_chunk()
        return []

    def flush(self) -> list:
        """Issue D→H for any partial chunk; return all buffered contexts.

        Called at EndRun, BeginStep, and end of each batch so no event
        is stranded in _chunk_buf indefinitely.
        """
        if self._chunk_buf:
            return self._flush_chunk()
        return []

    def pinned_bytes(self) -> int:
        """Return bytes of pinned (page-locked) host memory currently
        allocated by this pipeline's _PinnedSlot pool.
        Used by GpuEvents.log_memory() for Phase-0 accounting.
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
        for _ in range(self._max_inflight):
            self._pinned_pool.append(_PinnedSlot(self._n_segs, self._nrows, self._ncols, self._chunk_size))
        self._d2h_stream = cp.cuda.Stream(non_blocking=True)

    def _get_free_slot(self):
        """Return a free pinned slot, or None if all are in use.

        A slot is free when its reference count is 0 — every event in its
        chunk has called on_cpu (releasing the slot via _PendingD2H →
        pslot.dec_ref()).

        Returning None signals _flush_chunk to skip async D→H for this
        chunk and yield events without _pending_d2h.  on_cpu then falls
        back to synchronous self._arr.get() — correct but slower.

        This prevents Race 2 (pinned-slot overwrite causing silent data
        corruption) without raising an error or deadlocking for any access
        pattern.

        Phase 3 will replace the None path with generator backpressure:
        block here instead of returning None so the EB rank is rate-limited
        when the user is slow to consume results.
        """
        for slot in self._pinned_pool:
            if not slot.in_use:
                return slot
        return None   # all slots held — caller falls back to sync on_cpu

    def _flush_chunk(self) -> list:
        """Issue async D→H for the current chunk; attach _pending_d2h;
        return all events immediately.

        If no pinned slot is available (all busy with pending _PendingD2H
        tokens), skip async D→H and yield events without _pending_d2h.
        on_cpu will fall back to synchronous self._arr.get() for those
        events — correct but without the async overlap benefit.
        """
        import cupy as cp
        from psana.gpu.context import GPUResult

        chunk = self._chunk_buf
        self._chunk_buf = []
        n_evts = len(chunk)

        pslot = self._get_free_slot()

        # ── Fallback: no free pinned slot — skip async D→H ────────────────
        if pslot is None:
            # Yield events without _pending_d2h. on_cpu will call
            # self._arr.get() synchronously (the pre-existing fallback path).
            # No data corruption, no error, just slower D→H at call site.
            return [ctx for ctx, _, _ in chunk]

        # ── Normal path: issue async D→H ──────────────────────────────────
        pslot.claim(n_evts)
        stream = self._d2h_stream
        row_nbytes = self._n_segs * self._nrows * self._ncols * 4
        dst_base = pslot.arr.ctypes.data
        leases_out = []

        for i, (ctx, lease, arr) in enumerate(chunk):
            if lease is not None and lease.calib_done is not None:
                stream.wait_event(lease.calib_done)
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
            lease.register_d2h_done(pslot.done_event)

        # Attach _pending_d2h to each GPUResult and yield immediately.
        key = self._key
        results = []
        for i, (ctx, lease, arr) in enumerate(chunk):
            # Ensure GPUResult is cached so we can attach pending_d2h.
            if key not in ctx._cache:
                ctx._cache[key] = GPUResult(
                    ctx._gpu_results.get(key),
                    stream=None,
                    lease=ctx._leases.get(key),
                )
            ctx._cache[key]._pending_d2h = _PendingD2H(pslot, i, self._n_segs)
            results.append(ctx)
        return results


@dataclass
class _GpuMemStats:
    """Snapshot of GPU and pinned-host memory broken down by owner.

    All values are bytes.  Recorded by GpuEvents.log_memory() and used
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

    def _mb(self, n: int) -> str:
        return f"{n / 1024**2:.1f} MiB"

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


class GpuEvents:
    """GPU-aware event iterator for the existing serial psana read path.

    This mirrors Events, but consumes the GPU split batch produced by the
    existing SmdReaderManager/BatchIterator/EventBuilder stack.  It does not
    create DsParms, SmdReaderManager, DgramManager, or EventBuilderManager.
    """

    is_gpu_events = True

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

        self.gpu_det_names = self._normalize_gpu_det(dsparms.gpu_det)
        self.gpu_detectors = {}
        self.cpu_dets = {}
        self.router = DetectorRouter()
        self.event_pool = None
        self.gpu_reader = None

        self._setup_detectors()

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = self._events()
        return next(self._iter)

    def _snapshot_memory(self, label: str) -> _GpuMemStats:
        """Collect a _GpuMemStats snapshot from all pipeline components."""
        s = _GpuMemStats(label=label)
        for name, (_, det, _) in self.gpu_detectors.items():
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
        _mb = lambda n: f"{n / 1024**2:.1f} MiB"
        _log.info(
            "GPU mem high-water  constants=%s  geometry=%s  calib_slots=%s  raw_slots=%s  raw_input=%s  cupy_pool=%s  device_used=%s  pinned=%s",
            _mb(hw.get("constants", 0)),
            _mb(hw.get("geometry", 0)),
            _mb(hw.get("calib_slots", 0)),
            _mb(hw.get("raw_slots", 0)),
            _mb(hw.get("raw_input", 0)),
            _mb(hw.get("cupy_pool", 0)),
            _mb(hw.get("device_used", 0)),
            _mb(hw.get("pinned", 0)),
        )

    @staticmethod
    def _normalize_gpu_det(gpu_det):
        if gpu_det is None:
            return []
        if isinstance(gpu_det, str):
            return [gpu_det]
        return list(gpu_det)

    def _setup_detectors(self):
        # Budget must exist before constructing GPUDetector objects.
        if not hasattr(self, "_gpu_budget"):
            from psana.gpu.gpu_budget import _GpuBudget

            budget_gb = float(getattr(self.dsparms, "gpu_memory_budget_gb", 0) or 0)
            if budget_gb > 0:
                self._gpu_budget = _GpuBudget(limit_bytes=int(budget_gb * 1024**3))
            else:
                n_bd = max(1, int(os.environ.get("PS_BD_NODES", 1)))
                self._gpu_budget = _GpuBudget.auto(n_bd_ranks=n_bd)

        all_gpu_stream_ids = set()
        opt_batch_sizes = []
        requested_stream_ids = getattr(self.dsparms, "gpu_stream_ids", None)
        requested_stream_ids = set(requested_stream_ids) if requested_stream_ids is not None else None

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
            if det_type != "jungfrau":
                raise NotImplementedError(f"gpu_det={det_name!r} has detector type {det_type!r}; the integrated GPU path currently supports only Jungfrau")
            peds_gpu, gmask_gpu = prep_calib_constants(det)
            log_gpu_mem(f"after prep_calib_constants ({det_name})", rank=_rank)
            det_shape = det.calibconst["pedestals"][0].shape[1:]

            stream_segments = dict(getattr(self.dsparms, "det_stream_segments_table", {}).get(det_name, {}))
            det_stream_ids = sorted(getattr(self.dsparms, "det_stream_ids_table", {}).get(det_name, stream_segments.keys()))
            if requested_stream_ids is None:
                gpu_stream_ids = det_stream_ids
            else:
                gpu_stream_ids = [stream_id for stream_id in det_stream_ids if stream_id in requested_stream_ids]
            if not gpu_stream_ids:
                raise RuntimeError(f"gpu_det={det_name!r} did not resolve to any stream ids")

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
            cpu_stream_seg_map = {stream_id: sorted(segment_ids) for stream_id, segment_ids in stream_segments.items() if stream_id not in gpu_stream_ids}
            all_gpu_stream_ids.update(gpu_stream_ids)

            gpu_detector = GPUDetector(
                det_shape=det_shape,
                peds_gpu=peds_gpu,
                gmask_gpu=gmask_gpu,
                stream_seg_map=stream_seg_map or None,
                n_slots=getattr(self.dsparms, "n_gpu_streams", 2),
                budget=self._gpu_budget,
            )
            if self._prebuilt_geometry and det_name in self._prebuilt_geometry:
                ix_all, iy_all = self._prebuilt_geometry[det_name]
                gpu_detector.setup_geometry_from_arrays(ix_all, iy_all)
            elif self._setup_geometry:
                gpu_detector.setup_geometry(det)
            log_gpu_mem(f"after setup_geometry ({det_name})", rank=_rank)

            opt_batch_sizes.append(optimal_kernel_batch_size(det_shape))
            self.gpu_detectors[det_name] = (det, gpu_detector, cpu_stream_seg_map)
            self.router.register_gpu(det_name)

            gpu_seg_ids = []
            for stream_id in sorted(stream_seg_map):
                gpu_seg_ids.extend(stream_seg_map[stream_id])
            cpu_seg_ids = []
            for stream_id in sorted(cpu_stream_seg_map):
                cpu_seg_ids.extend(cpu_stream_seg_map[stream_id])

            self.router.setup_full_routing(
                det_name=det_name,
                gpu_seg_ids=gpu_seg_ids,
                cpu_seg_ids=cpu_seg_ids,
                calibconst_n_segs=det_shape[0],
                nrows=det_shape[1],
                ncols=det_shape[2],
                gpu_det_obj=gpu_detector,
            )

        if all_gpu_stream_ids:
            self.dsparms.gpu_stream_ids = sorted(all_gpu_stream_ids)

        if not self.dsparms.batch_size:
            self.dsparms.batch_size = min(opt_batch_sizes) if opt_batch_sizes else 1

        gpu_det_set = set(self.gpu_det_names)
        for det_name in self.run.detnames:
            if det_name in gpu_det_set:
                continue
            self.cpu_dets[det_name] = self.run.Detector(det_name)
            self.router.register_cpu(det_name)

        n_streams = getattr(self.dsparms, "n_gpu_streams", 2)
        self.event_pool = EventPool(n=n_streams)

        # KvikioGpuReader: pre-allocate one data_gpu buffer per slot.
        # _gpu_budget was already created in _setup_detectors() above and
        # is shared with every GPUDetector so all allocations are counted
        # against the same limit.
        self.gpu_reader = KvikioGpuReader(n_slots=n_streams, budget=self._gpu_budget)

        # Internal D→H pipeline — activated when gpu_d2h_chunk_size > 0.
        # Transfers calibrated results to pinned host memory in chunks so that
        # ctx.get('det.calib').on_cpu returns immediately without triggering
        # an additional synchronous D→H at the user's call site.
        chunk_size = getattr(self.dsparms, "gpu_d2h_chunk_size", 0) or 0
        if chunk_size > 0 and self.gpu_det_names:
            # One pipeline per GPU detector key.
            self._d2h_pipelines = {
                f"{det_name}.calib": _D2hPipeline(
                    det_key=f"{det_name}.calib",
                    chunk_size=chunk_size,
                )
                for det_name in self.gpu_det_names
            }
        else:
            self._d2h_pipelines = {}

        # Report which I/O path kvikio will use for this run.
        # GDS (compat_mode=False) reads NVMe → GPU VRAM directly (fast).
        # CPU-fallback (compat_mode=True) reads NVMe → CPU DRAM → GPU VRAM
        # via cudaMemcpy (slower; common on Lustre/GPFS filesystems like S3DF).
        _logger = _log
        _path = self.gpu_reader.io_path
        if self.gpu_reader._compat_mode:
            _logger.warning(
                "GpuEvents: kvikio I/O path = %s "
                "(NVMe → CPU DRAM → GPU VRAM via cudaMemcpy). "
                "True GDS is not available — likely Lustre/GPFS filesystem "
                "or cuFile driver not loaded.  GDS would give NVMe → GPU VRAM "
                "directly, bypassing CPU DRAM entirely.",
                _path,
            )
        else:
            _logger.info("GpuEvents: kvikio I/O path = %s (NVMe → GPU VRAM direct)", _path)

        # Phase-0 accounting: high-water marks reset each run.
        self._high_water: dict = {}
        self._first_batch_logged = False

        # Log fixed allocations (constants + geometry already on GPU).
        try:
            self.log_memory("after_setup")
        except Exception:
            pass

    def _next_batch(self):
        if self.smdr_man is None:
            raise StopIteration

        while True:
            if self.shared_state.terminate_flag.value:
                raise StopIteration

            try:
                if hasattr(self._batch_iter, "next_with_gpu"):
                    return self._batch_iter.next_with_gpu()
                batch_dict, step_dict = next(self._batch_iter)
                return batch_dict, {}, step_dict
            except StopIteration:
                self._batch_iter = next(self.smdr_man)

    def free_calib_bufs(self):
        """Release pre-allocated calib_gpu slot buffers for all GPU detectors.

        Delegates to GPUDetector.free_calib_bufs() for each detector.
        See GPUDetector.free_calib_bufs() for usage guidance.
        """
        for det_info in self.gpu_detectors.values():
            det_info[1].free_calib_bufs()

    def _dispatch_transition(self, service, dgrams):
        if service == TransitionId.BeginStep:
            for det_info in self.gpu_detectors.values():
                det, gpu_detector = det_info[0], det_info[1]
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

    def _make_context(self, evt, gpu_results, leases=None):
        gpu_results = _apply_full_routing(
            gpu_results,
            evt,
            self.gpu_detectors,
            self.router,
        )
        return GpuEventContext(
            evt=evt,
            gpu_results=gpu_results,
            cpu_dets=self.cpu_dets,
            stream=None,
            router=self.router,
            leases=leases,
        )

    def _push_context(self, ctx):
        """Feed one context through D→H pipelines; yield when ready."""
        if not self._d2h_pipelines:
            yield ctx
            return
        # Feed through every active pipeline (one per det key).
        # A context is only yielded once ALL pipelines have finished with it.
        # For the common single-detector case this is a single iteration.
        ready = [ctx]
        for pipe in self._d2h_pipelines.values():
            next_ready = []
            for c in ready:
                next_ready.extend(pipe.add(c))
            ready = next_ready
        yield from ready

    def _yield_ready(self, ready):
        if ready is None:
            return
        gpu_results_by_ts, cpu_evts, leases_by_ts = ready
        # Log after the first batch: slot buffers have grown to their
        # initial sizes so this shows the steady-state allocation.
        if not self._first_batch_logged and cpu_evts:
            self._first_batch_logged = True
            self.log_memory("first_batch")
        for evt in cpu_evts:
            ctx = self._make_context(
                evt,
                gpu_results_by_ts.get(evt.timestamp, {}),
                leases=leases_by_ts.get(evt.timestamp, {}),
            )
            yield from self._push_context(ctx)
        # Flush any partial D→H chunk at the batch boundary so events are
        # never stranded in _chunk_buf when batch_size % chunk_size != 0.
        yield from self._flush_d2h_pipelines()

    def _flush_d2h_pipelines(self):
        """Flush partial D→H chunks at EndRun, BeginStep, or end of pool drain.

        With the lazy-sync design, all events have already been yielded;
        this only handles events still buffered in _chunk_buf (partial
        chunk not yet flushed because it didn't reach chunk_size).
        """
        if not self._d2h_pipelines:
            return
        for pipe in self._d2h_pipelines.values():
            yield from pipe.flush()

    def _flush_event_pool(self):
        for gpu_results_by_ts, cpu_evts, leases_by_ts in self.event_pool.flush():
            for evt in cpu_evts:
                ctx = self._make_context(
                    evt,
                    gpu_results_by_ts.get(evt.timestamp, {}),
                    leases=leases_by_ts.get(evt.timestamp, {}),
                )
                yield from self._push_context(ctx)
        yield from self._flush_d2h_pipelines()

    def _events(self):
        n_events = 0
        try:
            while True:
                try:
                    batch_dict, gpu_batch_dict, step_dict = self._next_batch()
                except StopIteration:
                    yield from self._flush_event_pool()
                    return

                end_run_seen = yield from self._handle_steps(step_dict)

                gpu_pending = None
                for gpu_batch, _ in gpu_batch_dict.values():
                    gpu_view = GpuBatchView(gpu_batch, validate=True)
                    if gpu_view.has_work:
                        # Retire the batch occupying the next slot before
                        # KvikIO overwrites that slot's raw input buffer.  Its
                        # calibrated output aliases the same logical slot, so
                        # yield it before submitting the replacement batch.
                        ready = self.event_pool.retire_next()
                        slot_id = self.event_pool.next_slot_id
                        gpu_pending = (
                            gpu_view,
                            self.gpu_reader.issue_batch(gpu_view, self.dm, slot_id=slot_id),
                        )
                        yield from self._yield_ready(ready)

                stop_after = False
                cpu_evts = []
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
                    for dgrams in event_manager:
                        evt = Event(dgrams=dgrams, run=self.run._run_ctx)
                        if not TransitionId.isEvent(evt.service()):
                            continue
                        cpu_evts.append(evt)
                        n_events += 1
                        if self.dsparms.max_events > 0 and n_events >= self.dsparms.max_events:
                            stop_after = True
                            break
                    if event_manager.exit_id:
                        raise RuntimeError(f"EventManager exit {event_manager.exit_id}")
                    if stop_after:
                        break

                if gpu_pending is not None:
                    gpu_view, pending = gpu_pending
                    gpu_read = self.gpu_reader.wait_batch(pending)
                    self.event_pool.submit(
                        gpu_view,
                        gpu_read,
                        cpu_evts,
                        self.gpu_detectors,
                    )
                else:
                    for evt in cpu_evts:
                        yield self._make_context(evt, {})

                if stop_after or end_run_seen:
                    yield from self._flush_event_pool()
                    break
        finally:
            if self.gpu_reader is not None:
                self.gpu_reader.close()
