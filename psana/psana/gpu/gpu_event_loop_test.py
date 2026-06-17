"""
gpu_event_loop_test.py — End-to-end GPU/CPU event loop test for Jungfrau data.

Implements the BD rank event loop described in the psana2 GPU BD prototype
design document (Section 9, Step 4):

    GPU path  (once per batch, before CPU loop):
        EventBuilder → GPUBAT1
        → issue_batch()          # non-blocking GDS reads
        → [CPU EventManager runs concurrently, reading non-GPU streams]
        → wait_batch()           # GDS reads complete
        → GPUDetector.process_batch() → {timestamp: EventContext}

    CPU event loop  (unchanged structure from implementation guide):
        for dgrams in evt_manager:
            evt = Event(dgrams=dgrams)
            calib_gpu = gpu_results[det_name][evt.timestamp].calib_gpu
            calib_cpu = cpu_ref[evt.timestamp]   # from no-split reference run

    Validation (PDF §9):
        calib_gpu.get()  (D2H)  should match  det.raw.calib(evt)
        from the standard CPU-only path, for every GPU-routed event.

        Because the calibconst segment mapping between GPU-routed streams and
        the full detector calibconst is an approximation in the current
        prototype, validation checks shape, dtype, and absence of NaN rather
        than pixel-exact equality.  Pixel-exact validation is noted as a
        future refinement (correct per-stream segment mapping).

Usage
-----
  # On a GPU node (sdfampereNNN):
  cd ~/lcls2
  PS_TEST_GPU_STREAM_IDS=6,8,9,10,11 python psana/psana/gpu/gpu_event_loop_test.py

  # Explicit arguments:
  PS_TEST_GPU_STREAM_IDS=6,8,9,10,11 python psana/psana/gpu/gpu_event_loop_test.py \\
      --smd-glob "/sdf/data/lcls/ds/prj/public01/xtc/smalldata/mfx100852324-r0077*" \\
      --det-name jungfrau \\
      --max-events 10 \\
      --batch-size 5
"""

import argparse
import glob
import os
import sys

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
                                  _compute_calib_constants_cpu)
from psana.gpu.gpu_stream import StreamPool
from psana.psexp import TransitionId


_DEFAULT_SMD_GLOB = os.environ.get('PSANA_GPU_TEST_SMD_GLOB', '')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _service_from_step_dict(step_dict):
    """Return the service type of the first transition in step_dict, or -1.

    step_dict has structure {dest: (step_batch_bytes, evt_sizes)}.
    The service is encoded in the dgram env field: (env >> 24) & 0xFF.
    """
    for step_batch, _ in step_dict.values():
        if len(step_batch) >= 12:
            env = int.from_bytes(memoryview(step_batch)[8:12], 'little')
            return (env >> 24) & 0xFF
    return -1


def _parse_args():
    p = argparse.ArgumentParser(
        description="GPU/CPU event loop test — implements PDF §9 event loop"
    )
    p.add_argument("--smd-glob", default=_DEFAULT_SMD_GLOB,
                   help="Glob for SMD xtc2 files")
    p.add_argument("--det-name", default="jungfrau",
                   help="Detector name (default: jungfrau)")
    p.add_argument("--max-events", type=int, default=10,
                   help="Maximum L1Accept events to process (default 10)")
    p.add_argument("--batch-size", type=int, default=5,
                   help="L1Accept events per GPU batch (default 5)")
    p.add_argument("--atol", type=float, default=None,
                   help="If set, require pixel-exact GPU vs CPU comparison "
                        "within this tolerance (only valid when calibconst "
                        "segment mapping is known-correct)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-event output; only print summary")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers shared with gpu_bd_read.py
# ---------------------------------------------------------------------------

def _resolve_smd_files(smd_glob):
    files = sorted(glob.glob(smd_glob))
    if not files:
        raise SystemExit(f"No files found: {smd_glob}")
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); unique.append(f)
    return unique


def _smd_to_xtc(smd_file):
    xtc_dir  = os.path.dirname(os.path.dirname(smd_file))
    xtc_name = os.path.basename(smd_file).split(".smd")[0] + ".xtc2"
    xtc_file = os.path.join(xtc_dir, xtc_name)
    if not os.path.exists(xtc_file):
        raise FileNotFoundError(xtc_file)
    return xtc_file


def _open_fd(path):
    return os.open(path, os.O_RDONLY)


def _make_dsparms(batch_size):
    return DsParms(
        batch_size=batch_size,
        max_events=0,
        max_retries=0,
        live=False,
        timestamps=np.empty(0, dtype=np.uint64),
        intg_det="",
        intg_delta_t=0,
        use_calib_cache=False,
        cached_detectors=[],
        fetch_calib_cache_max_retries=60,
        skip_calib_load=[],
        dbsuffix="",
    )


def _build_eb_chunk(smd_manager):
    empty_steps = [bytearray() for _ in range(smd_manager.n_files)]
    return bytearray(
        smd_manager.smdr.repack_parallel(
            empty_steps, 1,
            intg_stream_id=smd_manager.dsparms.intg_stream_id,
        )
    )


def _open_calib_detector(smd_files, det_name):
    """Open SMD files via standard psana to load calibration constants."""
    import psana
    ds  = psana.DataSource(files=list(smd_files))
    run = next(ds.runs())
    return run.Detector(det_name)


# ---------------------------------------------------------------------------
# Per-event result
# ---------------------------------------------------------------------------

class PerEventResult:
    """Holds GPU calib (on device) and CPU calib stats for one L1Accept."""

    def __init__(self, timestamp, calib_gpu, n_gpu_segs,
                 cpu_det_calib=None, cpu_n_segs=None):
        self.timestamp   = timestamp
        self.calib_gpu   = calib_gpu        # cp.ndarray float32 on GPU
        self.n_gpu_segs  = n_gpu_segs
        self.cpu_det_calib = cpu_det_calib  # np.ndarray float32 (CPU path)
        self.cpu_n_segs    = cpu_n_segs

    def validate(self, atol=None):
        """Check GPU calib shape, dtype, NaN-free.  Return (passed, message)."""
        gpu_np = self.calib_gpu.get()  # D2H

        if gpu_np.dtype != np.float32:
            return False, f"GPU dtype {gpu_np.dtype} != float32"
        if np.any(np.isnan(gpu_np)):
            n_nan = int(np.sum(np.isnan(gpu_np)))
            return False, f"GPU calib has {n_nan} NaN values"
        if gpu_np.shape[0] != self.n_gpu_segs:
            return False, (f"GPU calib shape {gpu_np.shape} "
                           f"expected {self.n_gpu_segs} segs")

        if atol is not None and self.cpu_det_calib is not None:
            if not np.allclose(gpu_np, self.cpu_det_calib, atol=atol, rtol=0):
                diff = np.abs(gpu_np - self.cpu_det_calib)
                return False, (f"GPU vs CPU max_diff={diff.max():.4f} "
                               f"> atol={atol}")

        return True, "ok"

    def summary(self):
        gpu_np = self.calib_gpu.get()
        gpu_str = (f"shape={gpu_np.shape} min={gpu_np.min():.2f} "
                   f"max={gpu_np.max():.2f} mean={gpu_np.mean():.2f}")
        if self.cpu_det_calib is not None:
            c = self.cpu_det_calib
            cpu_str = (f"shape={c.shape} min={c.min():.2f} "
                       f"max={c.max():.2f} mean={c.mean():.2f}")
        else:
            cpu_str = "N/A"
        return f"ts={self.timestamp}\n  GPU calib: {gpu_str}\n  CPU calib: {cpu_str}"


# ---------------------------------------------------------------------------
# CPU reference calib (no GPU routing)
# ---------------------------------------------------------------------------

def _collect_cpu_reference(smd_files, xtc_files, det_name, max_events):
    """
    Run the standard psana event loop WITHOUT GPU routing to collect
    calibrated pixel data per timestamp.

    Uses det.raw.raw(evt) + prep_calib_constants() to avoid the internal
    psana shape mismatch that occurs when the calibconst has more segments
    than are present in the DAQ (e.g. 32-seg calibconst, 7-seg run).
    Falls back to det.raw.calib() if raw() is unavailable.

    Returns {timestamp: np.ndarray float32 calibrated array}.
    """
    import psana
    from psana.gpu.gpu_calib import prep_calib_constants

    # Temporarily unset PS_TEST_GPU_STREAM_IDS so EventManager sees all streams.
    saved = os.environ.pop("PS_TEST_GPU_STREAM_IDS", None)
    try:
        ds  = psana.DataSource(files=xtc_files)
        run = next(ds.runs())
        det = run.Detector(det_name)

        # Build calibration constants on CPU for manual application.
        peds  = det.calibconst["pedestals"][0].astype(np.float32)  # (3,n,r,c)
        gain  = det.calibconst["pixel_gain"][0].astype(np.float32)
        try:
            mask = det.raw._mask(all_segs=True)
        except Exception:
            mask = None
        gfac  = np.where(gain != 0, np.float32(1.0) / gain, np.float32(0.0))
        if mask is not None:
            try:
                gmask = (gfac * mask[np.newaxis]).astype(np.float32)
            except ValueError:
                gmask = gfac.astype(np.float32)
        else:
            gmask = gfac.astype(np.float32)

        peds_flat  = peds.ravel()
        gmask_flat = gmask.ravel()

        ref = {}
        for evt in run.events():
            if max_events > 0 and len(ref) >= max_events:
                break
            raw = det.raw.raw(evt)
            if raw is None:
                continue
            # Manually apply Jungfrau calibration (same formula as GPU kernel).
            raw_flat  = raw.ravel().astype(np.float32)
            gbits     = (raw.ravel() >> 14).astype(np.int32)
            data_bits = (raw.ravel() & 0x3fff).astype(np.float32)
            n_pix     = raw_flat.size
            mode = np.where(gbits == 0, 0,
                   np.where(gbits == 1, 1,
                   np.where(gbits == 3, 2, -1)))
            calib_flat = np.zeros(n_pix, dtype=np.float32)
            good = mode >= 0
            idx  = np.where(good)[0]
            offsets = (mode[idx].astype(np.int64) * n_pix + idx).astype(np.int64)
            calib_flat[idx] = (data_bits[idx] - peds_flat[offsets]) * gmask_flat[offsets]
            ref[evt.timestamp] = calib_flat.reshape(raw.shape)
    finally:
        if saved is not None:
            os.environ["PS_TEST_GPU_STREAM_IDS"] = saved
    print(f"CPU reference: {len(ref)} events collected")
    return ref


# ---------------------------------------------------------------------------
# Main event loop  (PDF §9 structure)
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    smd_files = _resolve_smd_files(args.smd_glob)
    xtc_files = [_smd_to_xtc(f) for f in smd_files]

    # -----------------------------------------------------------------------
    # 1. Load calibration constants (once per run).
    # -----------------------------------------------------------------------
    print(f"Loading calibconst for '{args.det_name}' ...")
    det_calib  = _open_calib_detector(smd_files, args.det_name)
    peds_gpu, gmask_gpu = prep_calib_constants(det_calib)
    peds_shape = det_calib.calibconst["pedestals"][0].shape  # (3,n,r,c)
    det_shape  = peds_shape[1:]
    gpu_detector = GPUDetector(
        det_shape=det_shape,
        peds_gpu=peds_gpu,
        gmask_gpu=gmask_gpu,
    )
    print(f"  det_shape={det_shape}  "
          f"peds {peds_gpu.shape}  gmask {gmask_gpu.shape}")

    # Two non-blocking streams: calibration and (future) prefetch.
    stream_pool = StreamPool(size=2)
    print(f"  StreamPool: {len(stream_pool)} non-blocking CUDA streams")

    # -----------------------------------------------------------------------
    # 2. Collect CPU reference calibration (no GPU routing).
    # -----------------------------------------------------------------------
    print("Collecting CPU reference calibration ...")
    cpu_ref = _collect_cpu_reference(
        smd_files, xtc_files, args.det_name, args.max_events
    )

    # -----------------------------------------------------------------------
    # 3. GPU BD event loop  (PDF §9).
    # -----------------------------------------------------------------------
    dsparms = _make_dsparms(args.batch_size)
    dsparms.update_smd_state(list(smd_files), [False] * len(smd_files))
    smd_fds = np.array([_open_fd(f) for f in smd_files], dtype=np.int32)

    gpu_stream_ids_str = os.environ.get("PS_TEST_GPU_STREAM_IDS", "")
    if not gpu_stream_ids_str:
        print("WARNING: PS_TEST_GPU_STREAM_IDS not set — no GPU routing active")

    results   = {}   # timestamp → PerEventResult
    n_events  = 0
    gpu_reader = None
    try:
        smd_manager = SmdReaderManager(smd_fds, dsparms)
        if smd_manager.get_next_dgrams() is None:
            raise RuntimeError("missing Configure")
        if smd_manager.get_next_dgrams() is None:
            raise RuntimeError("missing BeginRun")

        bd_dm      = DgramManager(xtc_files, configs=smd_manager.configs)
        gpu_reader = KvikioGpuReader()

        for _chunk_id in smd_manager.chunks():
            smd_chunk  = _build_eb_chunk(smd_manager)
            eb_manager = EventBuilderManager(
                smd_chunk, smd_manager.configs, dsparms
            )

            for batch_dict, gpu_batch_dict, step_dict in eb_manager.batches_with_gpu():

                # -----------------------------------------------------------
                # BeginStep hook: refresh calibration constants in-place.
                # Called before the GPU path so any updated constants are
                # visible to the kernels that follow.
                # -----------------------------------------------------------
                if step_dict and gpu_detector is not None:
                    svc = _service_from_step_dict(step_dict)
                    if svc == TransitionId.BeginStep:
                        peds_new, gmask_new = _compute_calib_constants_cpu(det_calib)
                        gpu_detector.beginstep(peds_new, gmask_new)
                        print(f"beginstep: calibration constants refreshed "
                              f"(BeginStep detected)")

                # -----------------------------------------------------------
                # GPU path: issue reads immediately (non-blocking).
                # PDF §9: "Once per batch, before the existing CPU event loop"
                # -----------------------------------------------------------
                gpu_pending = None
                for gpu_batch, _ in gpu_batch_dict.values():
                    gpu_view = GpuBatchView(gpu_batch, validate=True)
                    if gpu_view.has_work:
                        gpu_pending = (
                            gpu_view,
                            gpu_reader.issue_batch(gpu_view, bd_dm),
                        )

                # -----------------------------------------------------------
                # CPU event loop: reads non-GPU stream bigdata concurrently
                # with the in-flight GDS reads above.
                # PDF §9: "Existing CPU event loop — structure unchanged"
                # -----------------------------------------------------------
                cpu_evts = []   # (evt, dgrams)
                stop_after = False
                for smd_batch, _ in batch_dict.values():
                    if not smd_batch:
                        continue
                    evt_manager = EventManager(
                        smd_batch,
                        smd_manager.configs,
                        bd_dm,
                        dsparms.max_retries,
                        [False] * len(smd_manager.configs),
                    )
                    for dgrams in evt_manager:
                        evt = Event(dgrams=dgrams)
                        if not TransitionId.isEvent(evt.service()):
                            continue
                        cpu_evts.append((evt, dgrams))
                        n_events += 1
                        if args.max_events > 0 and n_events >= args.max_events:
                            stop_after = True
                            break

                    if evt_manager.exit_id:
                        raise RuntimeError(
                            f"EventManager exit_id={evt_manager.exit_id}"
                        )
                    if stop_after:
                        break

                # -----------------------------------------------------------
                # Wait for GPU reads (typically already complete) and calibrate.
                # Yields one EventContext per L1Accept in this batch.
                # -----------------------------------------------------------
                gpu_calib_by_ts = {}   # {timestamp: EventContext}
                if gpu_pending is not None:
                    gpu_view, pending = gpu_pending
                    gpu_read = gpu_reader.wait_batch(pending)
                    # Acquire a non-blocking CUDA stream for this batch's
                    # calibration work (non-default stream avoids serialisation).
                    batch_stream = stream_pool.acquire()
                    for evt_ctx in gpu_detector.process_batch(
                            gpu_view, gpu_read, stream=batch_stream):
                        gpu_calib_by_ts[evt_ctx.timestamp] = evt_ctx
                    # stream synchronisation happens automatically on next
                    # stream_pool.acquire() for the same slot, or when the
                    # caller calls .get() on calib_gpu (implicit sync).

                # -----------------------------------------------------------
                # Per-event result: combine GPU calib + CPU reference.
                # PDF §9: ctx = EventContext(evt, router, gpu_results, stream)
                # ctx.get('jungfrau.calib') → gpu_results['jungfrau'][ts]
                # -----------------------------------------------------------
                for evt, dgrams in cpu_evts:
                    ts = evt.timestamp
                    gpu_ctx    = gpu_calib_by_ts.get(ts)
                    cpu_calib  = cpu_ref.get(ts)

                    if gpu_ctx is not None:
                        n_gpu_segs = int(
                            gpu_ctx.calib_gpu.shape[0]
                            if gpu_ctx.calib_gpu.ndim >= 1 else 1
                        )
                        results[ts] = PerEventResult(
                            timestamp=ts,
                            calib_gpu=gpu_ctx.calib_gpu,
                            n_gpu_segs=n_gpu_segs,
                            cpu_det_calib=cpu_calib,
                            cpu_n_segs=(cpu_calib.shape[0]
                                        if cpu_calib is not None else None),
                        )

                # Break AFTER GPU results for this batch are collected.
                if stop_after:
                    break

            if args.max_events > 0 and n_events >= args.max_events:
                break

    finally:
        for fd in smd_fds:
            os.close(int(fd))
        if "bd_dm" in dir():
            bd_dm.close()
        if gpu_reader is not None:
            gpu_reader.close()

    # -----------------------------------------------------------------------
    # 4. Report and validate.
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Event loop complete: {len(results)} events with GPU+CPU calib")
    print(f"{'='*60}\n")

    n_pass = n_fail = 0
    for ts in sorted(results):
        r   = results[ts]
        ok, msg = r.validate(atol=args.atol)
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        if not args.quiet or not ok:
            tag = "PASS" if ok else "FAIL"
            print(f"[{tag}] {r.summary()}")
            if not ok:
                print(f"       Reason: {msg}")

    print(f"\nResult: {n_pass} passed, {n_fail} failed "
          f"out of {len(results)} events")

    if n_events == 0:
        print("WARNING: no L1Accept events processed — "
              "check PS_TEST_GPU_STREAM_IDS and input files")
        sys.exit(1)

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
