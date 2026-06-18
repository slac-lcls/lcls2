"""
psana/gpu/gpu_events.py — Generic GPU event loop for psana2.

This module provides ``gpu_events()``, the prototype implementation of the
user-facing event loop described in the implementation guide §1a:

    for ctx in gpu_events(smd_glob, gpu_det='jungfrau'):
        calib  = ctx.get('jungfrau.calib').on_gpu  # CuPy, stays on GPU
        energy = ctx.raw('gmd').energy              # CPU, unchanged
        n_hit  = int(cp.sum(calib > threshold))
        if n_hit > min_bright:
            save(ctx.get('jungfrau.calib').on_cpu, energy)

This is detector-agnostic: ``gpu_det`` may name any large area detector
(Jungfrau, ePix, CSPAD, …) whose bigdata format follows the XTC2 convention.
The calibration kernel (``GPUDetector``) auto-detects the XTC header overhead
and per-segment stride from the first bigdata dgram, so no detector-specific
constants need to be hard-coded.

Integration path to full psana2
--------------------------------
When ``DataSource(gpu_det='jungfrau', ...)`` is implemented (guide §4):
  - ``_setup_run()`` calls ``DetectorRouter._init_gpu(det_name, ...)``
  - ``run.events()`` calls this generator internally
  - No change to user analysis code
Until then, this function is the drop-in replacement for ``run.events()``.

Key prototype features used
----------------------------
  Step 1/2  EventBuilder GPU split + GDS read via KvikIO
  Step 3    compute_digest=False (no per-descriptor D→H in production)
  Step 4    GPUDetector.process_batch + EventContext wiring
  Step 5a   fused_calib_gpu CUDA kernel
  Step 8    issue_batch / wait_batch double-buffering
  A1        stream_seg_map (correct per-stream calibconst indices)
  A2        beginstep() in-place calibconst refresh
  B1        StreamPool non-blocking CUDA streams
"""

from __future__ import annotations

import glob
import os
from typing import Iterator

import numpy as np

from psana.psexp.ds_base import DsParms
from psana.psexp.eventbuilder_manager import EventBuilderManager
from psana.psexp.packet_footer import PacketFooter
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.event_manager import EventManager
from psana.dgrammanager import DgramManager
from psana.event import Event
from psana.psexp import TransitionId
from psana.gpu.gpu_batch import GpuBatchView
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_calib import (GPUDetector, prep_calib_constants,
                                  build_stream_seg_map,
                                  _compute_calib_constants_cpu,
                                  optimal_kernel_batch_size)
from psana.gpu.gpu_stream import StreamPool, EventPool
from psana.gpu.gpu_kernel_registry import GPUKernelRegistry, default_registry
from psana.gpu.detector_router import DetectorRouter
from psana.gpu.context import GPUResult, GpuEventContext


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_smd_files(smd_glob: str) -> list:
    files = sorted(glob.glob(smd_glob))
    if not files:
        raise FileNotFoundError(f'No SMD files found: {smd_glob}')
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); unique.append(f)
    return unique


def _smd_to_xtc(smd_file: str) -> str:
    xtc_dir  = os.path.dirname(os.path.dirname(smd_file))
    xtc_name = os.path.basename(smd_file).split('.smd')[0] + '.xtc2'
    p = os.path.join(xtc_dir, xtc_name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'bigdata file not found: {p}  (derived from {smd_file})'
        )
    return p


def _make_dsparms(batch_size: int) -> DsParms:
    return DsParms(
        batch_size=batch_size,
        max_events=0,
        max_retries=0,
        live=False,
        timestamps=np.empty(0, dtype=np.uint64),
        intg_det='',
        intg_delta_t=0,
        use_calib_cache=False,
        cached_detectors=[],
        fetch_calib_cache_max_retries=60,
        skip_calib_load=[],
        dbsuffix='',
    )


def _build_eb_chunk(smd_manager) -> bytearray:
    empty_steps = [bytearray() for _ in range(smd_manager.n_files)]
    return bytearray(
        smd_manager.smdr.repack_parallel(
            empty_steps, 1,
            intg_stream_id=smd_manager.dsparms.intg_stream_id,
        )
    )


def _apply_full_routing(gpu_results: dict, evt, gpu_detectors: dict,
                        router) -> dict:
    """Combine GPU + CPU calibrated segments into complete detector arrays.

    For each GPU-split detector where setup_full_routing() was called,
    computes CPU-path segment calibration and scatters both sets into a
    (calibconst_n_segs, nrows, ncols) array.  Replaces the partial GPU-only
    calib in gpu_results with the combined result.

    Parameters
    ----------
    gpu_results   : {ts: {'det.calib': cp.ndarray, ...}} — modified in-place
    evt           : psana2 Event (provides raw CPU-path pixels)
    gpu_detectors : {det_name: (psana_det, gpu_det_obj, cpu_seg_map) | ...}
    router        : DetectorRouter with full routing configured

    Returns
    -------
    The modified gpu_results dict (same object).
    """
    if not router or not hasattr(router, 'has_full_routing'):
        return gpu_results

    for det_name, det_info in gpu_detectors.items():
        if not router.has_full_routing(det_name):
            continue
        calib_key = f'{det_name}.calib'

        # Nothing to combine for this event if GPU produced no result.
        gpu_calib = gpu_results.get(calib_key)
        if gpu_calib is None:
            continue

        psana_det = det_info[0]
        cpu_calib = router.compute_cpu_calib(det_name, psana_det, evt)

        combined = router.assemble_full_calib(det_name, gpu_calib, cpu_calib)
        if combined is not None:
            gpu_results[calib_key] = combined

    return gpu_results


def _iter_step_services(batch_bytes):
    """Yield (service, dgram_memoryview) for every transition dgram in a
    PacketFooter-formatted step batch.

    The old _service_from_batch() only read the first dgram.  A single step
    batch can contain multiple consecutive transitions (e.g. Enable immediately
    followed by BeginStep in the same batch), so all must be dispatched.
    """
    if not batch_bytes or len(batch_bytes) < 12:
        return
    try:
        pf     = PacketFooter(view=batch_bytes)
        n      = pf.n_packets
        offset = 0
        for _ in range(n):
            if offset + 12 > len(batch_bytes):
                break
            mv   = memoryview(batch_bytes)[offset:]
            env  = int.from_bytes(mv[8:12], 'little')
            svc  = (env >> 24) & 0xFF
            ext  = int.from_bytes(mv[20:24], 'little')   # XTC extent in Dgram header
            yield svc, mv
            dgram_bytes = 24 + ext                        # Dgram header + XTC payload
            offset += dgram_bytes
    except Exception:
        # Fallback: yield service from first dgram only
        env = int.from_bytes(memoryview(batch_bytes)[8:12], 'little')
        yield (env >> 24) & 0xFF, memoryview(batch_bytes)


# ---------------------------------------------------------------------------
# GPU detector setup (one per run)
# ---------------------------------------------------------------------------

def _setup_gpu_detector(
    smd_files: list,
    xtc_files: list,
    det_name: str,
    registry: GPUKernelRegistry = None,
) -> tuple:
    """Open calibconst, build stream_seg_map, return (det, GPUDetector, det_shape).

    Parameters
    ----------
    smd_files, xtc_files : lists of file paths
    det_name             : detector name (e.g. 'jungfrau')

    Returns
    -------
    (psana_det, gpu_detector, det_shape)
    """
    import psana

    ds_cal  = psana.DataSource(files=list(smd_files))
    run_cal = next(ds_cal.runs())
    det     = run_cal.Detector(det_name)

    peds_gpu, gmask_gpu = prep_calib_constants(det)
    det_shape = det.calibconst['pedestals'][0].shape[1:]  # (n_segs, nrows, ncols)

    # Discover GPU-routed stream IDs and per-stream segment mapping from the
    # SMD Configure dgrams.  No bigdata files are opened.
    #
    # The Configure dgram for each SMD stream contains the detector type
    # information, including which physical segment IDs are present.
    # Streams whose Configure dgram has the target detector attribute are
    # GPU-routed; streams where that detector is absent (e.g. stream 0 for
    # the CPU Jungfrau segments in the MFX run) keep the CPU path.
    #
    # PS_TEST_GPU_STREAM_IDS is still honoured as an explicit override so
    # that existing test commands continue to work unchanged.
    gpu_ids_str = os.environ.get('PS_TEST_GPU_STREAM_IDS', '')
    stream_seg_map    = {}
    gpu_ids_discovered = []
    cpu_stream_seg_map = {}   # populated after GPU stream discovery

    if gpu_ids_str:
        # Explicit override: use the env var as before.
        gpu_ids_discovered = [int(x) for x in gpu_ids_str.split(',')]
        bd_map = {i: xtc_files[i] for i in gpu_ids_discovered
                  if i < len(xtc_files)}
        stream_seg_map = build_stream_seg_map(bd_map, det_name)
    else:
        # Auto-discover: inspect each SMD stream's Configure dgram.
        # The Configure dgrams are the first set of dgrams produced by
        # SmdReaderManager — they are read here from the individual SMD files
        # without opening any bigdata.
        import numpy as _np
        for i, smd_f in enumerate(smd_files):
            try:
                _fd = os.open(smd_f, os.O_RDONLY)
                _dsp = _make_dsparms(1)
                _dsp.update_smd_state([smd_f], [False])
                from psana.psexp.smdreader_manager import SmdReaderManager as _SRM
                _smr = _SRM(_np.array([_fd], dtype=_np.int32), _dsp)
                _cfgs = _smr.get_next_dgrams()  # Configure dgrams
                os.close(_fd)
                if _cfgs and _cfgs[0] is not None:
                    cfg = _cfgs[0]
                    if hasattr(cfg, det_name):
                        seg_ids = sorted(getattr(cfg, det_name).keys())
                        gpu_ids_discovered.append(i)
                        stream_seg_map[i] = seg_ids
            except Exception:
                pass

    # Look up calibration kernel from registry.
    reg    = registry if registry is not None else default_registry()
    kernel = reg.get(det_name, 'calib')
    if kernel is None:
        import warnings
        warnings.warn(
            f'GPUKernelRegistry: no calibration kernel registered for '
            f"detector type '{det_name}'.  Falling back to JungfrauCalibKernel "
            f"(likely wrong for this detector).  Register a kernel with "
            f"GPUKernelRegistry.register() to fix this."
        )

    gpu_detector = GPUDetector(
        det_shape=det_shape,
        peds_gpu=peds_gpu,
        gmask_gpu=gmask_gpu,
        stream_seg_map=stream_seg_map or None,
        calib_kernel=kernel,
    )
    if kernel is not None:
        kernel.setup(det, gpu_detector)

    # Load pixel coordinate geometry for image assembly.
    gpu_detector.setup_geometry(det)

    # Compute the optimal batch size for this detector on the current GPU.
    opt_batch = optimal_kernel_batch_size(det_shape)

    return det, gpu_detector, det_shape, gpu_ids_discovered, cpu_stream_seg_map, opt_batch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gpu_events(
    smd_glob_or_files,
    gpu_det,
    *,
    cpu_det: list | None = None,
    _cpu_dets_dict: dict | None = None,
    _on_transition=None,
    batch_size: int = 0,   # 0 = auto-compute from GPU properties + detector shape
    max_events: int = 0,
    n_streams: int = 4,    # EventPool depth; 4 concurrent streams optimal for NVMe io_depth
    registry: GPUKernelRegistry = None,
) -> Iterator[GpuEventContext]:
    """Yield one GpuEventContext per L1Accept event.

    This is the prototype's implementation of ``run.events()`` when
    ``DataSource(gpu_det=...)`` is used (guide §1a, §4).  It hides the full
    GPU BD pipeline behind a simple iterator and exposes the standard
    ctx.get() / ctx.raw() API to user analysis code.

    Parameters
    ----------
    smd_glob   : str
        Glob pattern matching the SMD .smd.xtc2 files, e.g.
        ``'/path/to/smalldata/expXXXXXX-rNNNN*'``.
    gpu_det    : str or list of str
        Detector name(s) to route to GPU.  Their bigdata is read via GDS and
        calibrated by the GPU kernel.  Currently only one GPU detector per
        call is fully supported; a list will route each to a separate
        GPUDetector (future: DetectorRouter §5).
    cpu_det    : list of str or None
        Additional CPU detector names to pre-load for ``ctx.raw()`` access.
        Scalar diagnostics (GMD, IPM, EVR, timetool) should go here.
    batch_size : int
        L1Accept events per GPU batch.  Larger = better GPU utilisation at
        the cost of higher per-event latency (guide §12b).  Default 32.
    max_events : int
        Stop after this many L1Accepts (0 = all).
    n_streams  : int
        StreamPool size.  2 = compute + future prefetch ping-pong (B1).

    Yields
    ------
    GpuEventContext
        ctx.get('det.calib').on_gpu  → cp.ndarray float32
        ctx.get('det.calib').on_cpu  → np.ndarray float32 (D→H)
        ctx.raw('cpu_det')           → psana2 detector result (unchanged)
        ctx.timestamp                → uint64
        ctx.service()                → int (12 = L1Accept)

    Notes
    -----
    - ``PS_TEST_GPU_STREAM_IDS`` env var selects which bigdata streams are
      GPU-routed.  Set before calling gpu_events().
    - The calibration kernel (``fused_calib_gpu``) implements pedestal
      subtract + gain mode select + bad pixel mask for Jungfrau.  Future
      detectors need their own kernel registered via GPUKernelRegistry (§4b).
    - XTC header overhead and per-segment stride are auto-detected from the
      first bigdata dgram; no detector-specific constants are hard-coded.

    Example
    -------
    >>> import cupy as cp
    >>> from psana.gpu import gpu_events
    >>>
    >>> for ctx in gpu_events('/data/smalldata/mfx*', gpu_det='jungfrau',
    ...                       cpu_det=['gmd', 'ipm'], batch_size=32):
    ...     calib  = ctx.get('jungfrau.calib').on_gpu  # CuPy
    ...     energy = ctx.raw('gmd').energy              # float64
    ...     n_hit  = int(cp.sum(calib > 5.0))
    ...     if n_hit > 100:
    ...         save(ctx.get('jungfrau.calib').on_cpu, energy)
    """
    # Normalise gpu_det to a list.
    if isinstance(gpu_det, str):
        gpu_det_names = [gpu_det]
    else:
        gpu_det_names = list(gpu_det)

    # Resolve files — accept either a glob string or an explicit list.
    if isinstance(smd_glob_or_files, (list, tuple)):
        smd_files = list(dict.fromkeys(smd_glob_or_files))   # deduplicate
    else:
        smd_files = _resolve_smd_files(smd_glob_or_files)
    xtc_files = [_smd_to_xtc(f) for f in smd_files]

    # Set up one GPUDetector per GPU-routed detector name.
    gpu_detectors = {}   # {det_name: (psana_det, GPUDetector)}
    all_gpu_stream_ids = set()
    _gpu_setup_results = {}
    opt_batch_sizes    = []   # one per GPU detector
    for name in gpu_det_names:
        result = _setup_gpu_detector(smd_files, xtc_files, name, registry=registry)
        psana_det, gpu_det_obj, det_shape, stream_ids, cpu_stream_seg_map, opt_batch = result
        gpu_detectors[name] = (psana_det, gpu_det_obj)
        all_gpu_stream_ids.update(stream_ids)
        _gpu_setup_results[name] = result
        opt_batch_sizes.append(opt_batch)

    # Auto-compute batch_size when caller passes 0 or None: use the smallest
    # optimal size across all GPU-routed detectors (most conservative).
    if not batch_size:
        batch_size = min(opt_batch_sizes) if opt_batch_sizes else 1

    # Pre-load CPU detectors for ctx.raw().
    # Priority: _cpu_dets_dict (pre-loaded by Run) > cpu_det (name list).
    import psana as _psana
    cpu_dets: dict = {}
    if _cpu_dets_dict:
        cpu_dets = dict(_cpu_dets_dict)
    elif cpu_det:
        ds_cpu  = _psana.DataSource(files=list(smd_files))
        run_cpu = next(ds_cpu.runs())
        for name in cpu_det:
            cpu_dets[name] = run_cpu.Detector(name)

    # EventPool — keeps n_streams batches in flight simultaneously (B2).
    # Calibration kernels for batch N are non-blocking; results are
    # synchronised and yielded when that stream slot is recycled.
    event_pool = EventPool(n=n_streams)

    # DetectorRouter — resolves unqualified ctx.get('calib') keys and
    # assembles full (GPU + CPU segment) calibrated arrays (D1).
    router = DetectorRouter()
    for _name in gpu_det_names:
        router.register_gpu(_name)
    for _name in cpu_dets:
        router.register_cpu(_name)

    # Setup full routing for each GPU-split detector: precompute scatter
    # indices for combining GPU-calibrated + CPU-calibrated segments.
    for name, (psana_det, gpu_det_obj, det_shape, _gpu_ids, cpu_seg_map, _opt) in \
            _gpu_setup_results.items():
        seg_map = gpu_det_obj._stream_seg_map
        if not seg_map:
            continue
        # GPU seg_ids in process_batch output order.
        gpu_seg_ids_ordered = []
        for sid in sorted(seg_map.keys()):
            gpu_seg_ids_ordered.extend(seg_map[sid])
        # CPU seg_ids: all streams with the detector that are NOT GPU-routed,
        # ordered by stream_id ascending and seg_id ascending within each stream.
        cpu_seg_ids_ordered = []
        for sid in sorted(cpu_seg_map.keys()):
            cpu_seg_ids_ordered.extend(cpu_seg_map[sid])
        # Total calibconst segments and panel dimensions.
        calibconst_n_segs = det_shape[0]
        nrows, ncols = det_shape[1], det_shape[2]
        router.setup_full_routing(
            det_name=name,
            gpu_seg_ids=gpu_seg_ids_ordered,
            cpu_seg_ids=cpu_seg_ids_ordered,
            calibconst_n_segs=calibconst_n_segs,
            nrows=nrows,
            ncols=ncols,
            gpu_det_obj=gpu_det_obj,    # for pre-extracting CPU calibconst rows
        )
        # Store 3-tuple so yield sites can access psana_det for CPU calib.
        gpu_detectors[name] = (psana_det, gpu_det_obj, cpu_seg_map)

    # Pipeline setup.
    dsparms = _make_dsparms(batch_size)
    dsparms.update_smd_state(list(smd_files), [False] * len(smd_files))
    # Propagate discovered stream IDs so EventBuilder can split them without
    # reading PS_TEST_GPU_STREAM_IDS from the environment.
    dsparms.gpu_stream_ids = sorted(all_gpu_stream_ids) if all_gpu_stream_ids else None
    smd_fds = np.array(
        [os.open(f, os.O_RDONLY) for f in smd_files], dtype=np.int32
    )

    gpu_reader = KvikioGpuReader()
    n_events   = 0

    try:
        smd_manager = SmdReaderManager(smd_fds, dsparms)
        if smd_manager.get_next_dgrams() is None:
            raise RuntimeError('missing Configure transition')
        if smd_manager.get_next_dgrams() is None:
            raise RuntimeError('missing BeginRun transition')

        bd_dm = DgramManager(xtc_files, configs=smd_manager.configs)

        for _chunk_id in smd_manager.chunks():
            smd_chunk  = _build_eb_chunk(smd_manager)
            eb_manager = EventBuilderManager(
                smd_chunk, smd_manager.configs, dsparms
            )

            for batch_dict, gpu_batch_dict, step_dict in \
                    eb_manager.batches_with_gpu():

                # --------------------------------------------------------
                # Transition handling: iterate ALL dgrams in step_dict,
                # not just the first.  A single step batch may contain
                # multiple consecutive transitions (e.g. Enable then
                # BeginStep).
                #
                # Ordering constraints
                # --------------------
                # BeginStep MAY change calibration constants (gain mode,
                # thresholds, etc.).  beginstep() calls peds_gpu.set()
                # which overwrites the shared GPU buffer in-place.  Any
                # calibration kernel from a previous batch that is still
                # running on a non-blocking CUDA stream would see a mix
                # of old and new constants — incorrect calibration.
                #
                # FIX: flush the EventPool (sync all in-flight streams)
                # BEFORE calling beginstep() so the buffer write is safe.
                #
                # EndRun: flush so all accumulated results are yielded
                # before the iterator terminates — no events are lost.
                #
                # Enable / Disable / EndStep: no calibration dependency;
                # process immediately (just update envstore metadata).
                # --------------------------------------------------------
                end_run_seen = False
                if step_dict:
                    # Collect all transition services first so we know
                    # whether a drain is needed before starting GPU work.
                    pending_transitions = []
                    for step_batch, _ in step_dict.values():
                        for svc, dgram_mv in _iter_step_services(step_batch):
                            pending_transitions.append((svc, bytes(dgram_mv)))

                    # If BeginStep or EndRun are present, drain the EventPool
                    # so no in-flight kernels are still reading calibration
                    # buffers when beginstep() (or termination) arrives.
                    needs_drain = any(
                        svc in (TransitionId.BeginStep, TransitionId.EndRun)
                        for svc, _ in pending_transitions
                    )
                    if needs_drain:
                        for old_results, old_evts in event_pool.flush():
                            for evt in old_evts:
                                yield GpuEventContext(
                                    evt=evt,
                                    gpu_results=old_results.get(
                                        evt.timestamp, {}),
                                    cpu_dets=cpu_dets,
                                    stream=None,
                                    router=router,
                                )

                    # Now dispatch each transition in order.
                    for svc, dgram_bytes in pending_transitions:
                        if svc == TransitionId.BeginStep:
                            # Safe to update constants: EventPool drained above.
                            for det_name, det_info in gpu_detectors.items():
                                psana_det, gpu_det_obj = det_info[0], det_info[1]
                                pn, gn = _compute_calib_constants_cpu(psana_det)
                                gpu_det_obj.beginstep(pn, gn)
                        elif svc == TransitionId.EndRun:
                            end_run_seen = True   # EventPool drained above
                        # Forward ALL transitions to the on_transition callback
                        # so the caller (Run._handle_transition) can update
                        # the envstore for scan variables, step counts, etc.
                        if _on_transition is not None:
                            try:
                                _on_transition(svc, dgram_bytes)
                            except Exception:
                                pass

                # --------------------------------------------------------
                # Step 8: issue GDS reads immediately (non-blocking).
                # --------------------------------------------------------
                gpu_pending = None
                for gpu_batch, _ in gpu_batch_dict.values():
                    gv = GpuBatchView(gpu_batch, validate=True)
                    if gv.has_work:
                        gpu_pending = (gv, gpu_reader.issue_batch(gv, bd_dm))

                # --------------------------------------------------------
                # CPU EventManager — runs while GDS reads are in-flight.
                # --------------------------------------------------------
                stop_after = False
                cpu_evts   = []
                for smd_batch, _ in batch_dict.values():
                    if not smd_batch:
                        continue
                    em = EventManager(
                        smd_batch,
                        smd_manager.configs,
                        bd_dm,
                        dsparms.max_retries,
                        [False] * len(smd_manager.configs),
                    )
                    for dgrams in em:
                        evt = Event(dgrams=dgrams)
                        if not TransitionId.isEvent(evt.service()):
                            continue
                        cpu_evts.append(evt)
                        n_events += 1
                        if max_events > 0 and n_events >= max_events:
                            stop_after = True
                            break
                    if em.exit_id:
                        raise RuntimeError(f'EventManager exit {em.exit_id}')
                    if stop_after:
                        break

                # --------------------------------------------------------
                # Wait for GDS reads; submit to EventPool (non-blocking
                # kernel launch).  EventPool returns synced results from
                # n_streams batches ago — those are ready to yield now.
                # --------------------------------------------------------
                if gpu_pending is not None:
                    gv, pending = gpu_pending
                    gpu_read    = gpu_reader.wait_batch(pending)
                    old = event_pool.submit(gv, gpu_read, cpu_evts,
                                            gpu_detectors)
                    if old is not None:
                        old_results, old_evts = old
                        for evt in old_evts:
                            gpu_res = _apply_full_routing(
                                old_results.get(evt.timestamp, {}),
                                evt, gpu_detectors, router)
                            yield GpuEventContext(
                                evt=evt, gpu_results=gpu_res,
                                cpu_dets=cpu_dets, stream=None, router=router,
                            )
                else:
                    # Transition-only batch: no GPU work, yield CPU events now.
                    for evt in cpu_evts:
                        yield GpuEventContext(
                            evt=evt, gpu_results={},
                            cpu_dets=cpu_dets, stream=None, router=router,
                        )

                if stop_after or end_run_seen:
                    # Flush pool; apply full routing before yielding.
                    for old_results, old_evts in event_pool.flush():
                        for evt in old_evts:
                            gpu_res = _apply_full_routing(
                                old_results.get(evt.timestamp, {}),
                                evt, gpu_detectors, router)
                            yield GpuEventContext(
                                evt=evt, gpu_results=gpu_res,
                                cpu_dets=cpu_dets, stream=None, router=router,
                            )
                    break

            if max_events > 0 and n_events >= max_events:
                break
            if end_run_seen:
                break

        # Flush any remaining in-flight batches at end of chunk.
        for old_results, old_evts in event_pool.flush():
            for evt in old_evts:
                gpu_res = _apply_full_routing(
                    old_results.get(evt.timestamp, {}),
                    evt, gpu_detectors, router)
                yield GpuEventContext(
                    evt=evt, gpu_results=gpu_res,
                    cpu_dets=cpu_dets, stream=None, router=router,
                )

    finally:
        for fd in smd_fds:
            os.close(int(fd))
        gpu_reader.close()
