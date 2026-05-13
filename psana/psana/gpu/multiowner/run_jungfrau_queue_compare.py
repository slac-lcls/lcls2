from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

from psana import DataSource
from psana.psexp.tools import mode as psana_mode

if psana_mode == "mpi":
    from mpi4py import MPI
else:  # pragma: no cover
    MPI = None


QUEUE_BASELINE = "baseline"
QUEUE_ONE_SIDE = "one-side"
QUEUE_BOTH_SIDES_LEASE = "both-sides-lease"
QUEUE_MODES = (QUEUE_BASELINE, QUEUE_ONE_SIDE, QUEUE_BOTH_SIDES_LEASE)


def add_datasource_args(parser):
    source_group = parser.add_argument_group("data source")
    source_group.add_argument("--file", default=None, help="Input xtc2 file.")
    source_group.add_argument(
        "-e",
        "--exp",
        default=None,
        help="Experiment name for DataSource(exp=..., run=...).",
    )
    source_group.add_argument(
        "-r",
        "--run",
        type=int,
        default=None,
        help="Run number for DataSource(exp=..., run=...).",
    )
    source_group.add_argument(
        "--xtc-dir",
        default=None,
        help="Pass dir=... to DataSource for experiment/run mode.",
    )
    return parser


def datasource_kwargs_from_args(args):
    has_file = args.file is not None
    has_exp_run = args.exp is not None or args.run is not None

    if has_file and has_exp_run:
        raise SystemExit("Use either --file or --exp/--run, not both.")
    if has_file:
        if args.xtc_dir is not None:
            raise SystemExit("--xtc-dir is only valid with --exp/--run.")
        return {"files": args.file}
    if args.exp is not None and args.run is not None:
        ds_kwargs = {"exp": args.exp, "run": args.run}
        if args.xtc_dir is not None:
            ds_kwargs["dir"] = args.xtc_dir
        return ds_kwargs
    if args.exp is None and args.run is None:
        raise SystemExit("Provide either --file or both --exp and --run.")
    raise SystemExit("When using experiment mode, both --exp and --run are required.")


def apply_max_events(ds_kwargs, max_events):
    if max_events is None:
        return dict(ds_kwargs)
    limited = dict(ds_kwargs)
    limited["max_events"] = max_events
    return limited


def apply_skip_calib_load(ds_kwargs, skip_calib_load):
    if skip_calib_load is None:
        return dict(ds_kwargs)
    updated = dict(ds_kwargs)
    updated["skip_calib_load"] = skip_calib_load
    return updated


def _should_print_event(index, interval, total_events=None):
    if interval <= 0:
        return True
    if (index + 1) % interval == 0:
        return True
    if total_events is not None and (index + 1) >= total_events:
        return True
    return False


def _world_rank():
    if MPI is None:
        return 0
    return MPI.COMM_WORLD.Get_rank()


def _local_rank():
    for env_name in ("OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "MV2_COMM_WORLD_LOCAL_RANK"):
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return 0


def _probe_visible_gpu_count():
    if cp is None:
        return None, "CuPy is unavailable"
    try:
        return int(cp.cuda.runtime.getDeviceCount()), None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, "%s: %s" % (type(exc).__name__, exc)


def _gpu_visibility_summary():
    visible_gpu_count, visible_gpu_error = _probe_visible_gpu_count()
    fields = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "SLURM_STEP_GPUS": os.environ.get("SLURM_STEP_GPUS", "<unset>"),
        "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS", "<unset>"),
        "visible_gpu_count": visible_gpu_count,
    }
    if visible_gpu_error is not None:
        fields["visible_gpu_error"] = visible_gpu_error
    return " ".join("%s=%s" % (key, value) for key, value in fields.items())


@contextmanager
def _nvtx_range(name):
    if cp is None or not hasattr(cp.cuda, "nvtx"):
        yield
        return
    try:
        cp.cuda.nvtx.RangePush(name)
        yield
    finally:
        try:
            cp.cuda.nvtx.RangePop()
        except Exception:
            pass


def _compare_cpu(det, evt, gpu_calib):
    cpu_calib = det.raw.calib(evt, cversion=3)
    cpu_host = np.asarray(cpu_calib, dtype=np.float32)
    gpu_host = cp.asnumpy(gpu_calib)
    if gpu_host.shape != cpu_host.shape:
        squeezed = np.squeeze(gpu_host)
        if squeezed.shape != cpu_host.shape:
            raise SystemExit(
                "Shape mismatch during compare: cpu=%s gpu=%s"
                % (cpu_host.shape, gpu_host.shape)
            )
        gpu_host = squeezed
    max_abs_diff = float(np.max(np.abs(cpu_host - gpu_host)))
    allclose = bool(np.allclose(cpu_host, gpu_host, rtol=1e-4, atol=1e-5))
    return max_abs_diff, allclose


def _print_global_summary(rank, is_bd, processed, loop_s):
    if MPI is None:
        total_events = int(processed)
        elapsed_s = float(loop_s)
        total_rate_evt_s = (total_events / elapsed_s) if elapsed_s > 0 else 0.0
        print(
            "[Rank %d] all_ranks_summary total_events=%d elapsed_s=%.3f "
            "total_rate_evt_s=%.3f bd_ranks=%d world_size=1"
            % (rank, total_events, elapsed_s, total_rate_evt_s, 1 if is_bd else 0),
            flush=True,
        )
        return

    gathered = MPI.COMM_WORLD.gather(
        {"is_bd": bool(is_bd), "processed": int(processed), "loop_s": float(loop_s)},
        root=0,
    )
    if rank != 0:
        return

    total_events = sum(item["processed"] for item in gathered)
    elapsed_s = max((item["loop_s"] for item in gathered), default=0.0)
    bd_ranks = sum(1 for item in gathered if item["is_bd"])
    total_rate_evt_s = (total_events / elapsed_s) if elapsed_s > 0 else 0.0
    print(
        "[Rank 0] all_ranks_summary total_events=%d elapsed_s=%.3f "
        "total_rate_evt_s=%.3f bd_ranks=%d world_size=%d"
        % (total_events, elapsed_s, total_rate_evt_s, bd_ranks, len(gathered)),
        flush=True,
    )


class JungfrauMultiOwnerGpu:
    def __init__(self, det_raw, device_id, extra_gpu_work=0):
        if cp is None:
            raise SystemExit("CuPy is required for this script.")
        self.det_raw = det_raw
        self.device_id = int(device_id)
        self.extra_gpu_work = int(extra_gpu_work)
        visible_gpu_count, visible_gpu_error = _probe_visible_gpu_count()
        if visible_gpu_count is not None and visible_gpu_count <= self.device_id:
            raise SystemExit(
                "Requested CUDA device %d, but only %d visible device(s) are available "
                "on this rank. %s"
                % (self.device_id, visible_gpu_count, _gpu_visibility_summary())
            )
        if visible_gpu_error is not None:
            raise SystemExit(
                "Unable to query visible CUDA devices before activating the GPU. %s"
                % _gpu_visibility_summary()
            )
        try:
            self.device = cp.cuda.Device(self.device_id)
            self.device.use()
        except Exception as exc:
            raise SystemExit(
                "Failed to activate CUDA device %d: %s: %s. %s"
                % (
                    self.device_id,
                    type(exc).__name__,
                    exc,
                    _gpu_visibility_summary(),
                )
            )
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.ccons_host = None
        self.ccons_dev = None
        self._pixel_index_cache = {}
        self._ccons_upload_s = 0.0

    def _ensure_ccons(self, evt):
        if self.ccons_dev is not None:
            return

        import psana.detector.UtilsJungfrau as uj

        shared = getattr(self.det_raw, "_jf_shared", None)
        if shared and shared.get("ccons") is not None:
            ccons_host = shared["ccons"]
        else:
            odc = getattr(self.det_raw, "_odc", None)
            if odc is None or getattr(odc, "cversion", None) != uj.CALIB_CPP_V3:
                self.det_raw._odc = odc = uj.DetCache(
                    self.det_raw,
                    evt,
                    cversion=uj.CALIB_CPP_V3,
                )
            ccons_host = odc.ccons

        self.ccons_host = np.ascontiguousarray(ccons_host, dtype=np.float32)
        start_evt = cp.cuda.Event()
        stop_evt = cp.cuda.Event()
        with self.stream:
            start_evt.record()
            self.ccons_dev = cp.asarray(self.ccons_host)
            stop_evt.record()
        stop_evt.synchronize()
        self._ccons_upload_s = cp.cuda.get_elapsed_time(start_evt, stop_evt) / 1e3

    def _pixel_index(self, size):
        idx = self._pixel_index_cache.get(size)
        if idx is None:
            idx = cp.arange(size, dtype=cp.int32)
            self._pixel_index_cache[size] = idx
        return idx


@dataclass
class AsyncGpuSlot:
    slot_id: int
    stream: Any
    busy: bool = False
    evt: Optional[Any] = None
    raw_host: Optional[Any] = None
    raw_dev: Optional[Any] = None
    calib_dev: Optional[Any] = None
    copy_start_evt: Optional[Any] = None
    copy_stop_evt: Optional[Any] = None
    kernel_start_evt: Optional[Any] = None
    kernel_stop_evt: Optional[Any] = None
    extra_start_evt: Optional[Any] = None
    extra_stop_evt: Optional[Any] = None
    done_evt: Optional[Any] = None
    raw_extract_s: float = 0.0
    lease_wait_s: float = 0.0
    lease_spins: int = 0
    submit_wall_t: float = 0.0

    def reset(self):
        self.busy = False
        self.evt = None
        self.raw_host = None
        self.raw_dev = None
        self.calib_dev = None
        self.copy_start_evt = None
        self.copy_stop_evt = None
        self.kernel_start_evt = None
        self.kernel_stop_evt = None
        self.extra_start_evt = None
        self.extra_stop_evt = None
        self.done_evt = None
        self.raw_extract_s = 0.0
        self.lease_wait_s = 0.0
        self.lease_spins = 0
        self.submit_wall_t = 0.0


class NodeGpuLeasePool:
    """Node-local GPU admission-control pool backed by MPI shared memory."""

    def __init__(self, max_leases, sleep_us=50):
        self.max_leases = max(1, int(max_leases))
        self.sleep_s = max(0, int(sleep_us)) / 1e6
        self.enabled = MPI is not None
        self.comm = None
        self.win = None
        self.state = None
        self.node_rank = 0

        if not self.enabled:
            return

        self.comm = MPI.COMM_WORLD.Split_type(
            MPI.COMM_TYPE_SHARED,
            MPI.COMM_WORLD.Get_rank(),
            MPI.INFO_NULL,
        )
        self.node_rank = self.comm.Get_rank()
        n_items = 3
        itemsize = np.dtype(np.int64).itemsize
        local_bytes = n_items * itemsize if self.node_rank == 0 else 0
        self.win = MPI.Win.Allocate_shared(local_bytes, itemsize, comm=self.comm)
        buf, _itemsize = self.win.Shared_query(0)
        self.state = np.ndarray(buffer=buf, dtype=np.int64, shape=(n_items,))
        if self.node_rank == 0:
            self.state.fill(0)
        self.comm.Barrier()

    def acquire(self):
        if not self.enabled:
            return 0.0, 0

        start = time.perf_counter()
        spins = 0
        while True:
            self.win.Lock(0, MPI.LOCK_EXCLUSIVE)
            self.win.Sync()
            active = int(self.state[0])
            if active < self.max_leases:
                self.state[0] = active + 1
                self.state[1] += 1
                self.win.Sync()
                self.win.Unlock(0)
                return time.perf_counter() - start, spins
            self.state[2] += 1
            self.win.Sync()
            self.win.Unlock(0)
            spins += 1
            if self.sleep_s:
                time.sleep(self.sleep_s)

    def release(self):
        if not self.enabled:
            return

        self.win.Lock(0, MPI.LOCK_EXCLUSIVE)
        self.win.Sync()
        self.state[0] = max(0, int(self.state[0]) - 1)
        self.win.Sync()
        self.win.Unlock(0)

    def snapshot(self):
        if not self.enabled:
            return {"active": 0, "acquire_count": 0, "contention_count": 0}

        self.win.Lock(0, MPI.LOCK_SHARED)
        self.win.Sync()
        active, acquire_count, contention_count = (int(value) for value in self.state)
        self.win.Unlock(0)
        return {
            "active": active,
            "acquire_count": acquire_count,
            "contention_count": contention_count,
        }

    def close(self):
        if self.win is None:
            return
        self.comm.Barrier()
        self.win.Free()
        self.win = None


class MeasuredJungfrauMultiOwnerGpu(JungfrauMultiOwnerGpu):
    """Synchronous baseline that preserves current behavior and returns CPU timing."""

    def process_event(self, evt):
        submit_start = time.perf_counter()
        raw_start = time.perf_counter()
        with _nvtx_range("psana2-gpu/raw_extract"):
            raw = self.det_raw.raw(evt, copy=False)
            if raw is None:
                return None
            raw_host = np.ascontiguousarray(raw, dtype=np.uint16)
        raw_extract_s = time.perf_counter() - raw_start

        self._ensure_ccons(evt)

        copy_start_evt = cp.cuda.Event()
        copy_stop_evt = cp.cuda.Event()
        kernel_start_evt = cp.cuda.Event()
        kernel_stop_evt = cp.cuda.Event()
        extra_start_evt = cp.cuda.Event()
        extra_stop_evt = cp.cuda.Event()

        with self.stream:
            with _nvtx_range("psana2-gpu/h2d"):
                copy_start_evt.record()
                raw_dev = cp.asarray(raw_host)
                copy_stop_evt.record()

            with _nvtx_range("psana2-gpu/jungfrau_v3"):
                kernel_start_evt.record()
                out_dev = cp.empty(raw_host.shape, dtype=np.float32)
                raw_flat = raw_dev.reshape(-1)
                out_flat = out_dev.reshape(-1)
                size = int(raw_flat.size)
                base_idx = self._pixel_index(size)
                gain_idx = (raw_flat >> 14).astype(cp.int32, copy=False)
                cc_idx = 2 * (base_idx + size * gain_idx)
                pedoff = cp.take(self.ccons_dev, cc_idx)
                gain = cp.take(self.ccons_dev, cc_idx + 1)
                out_flat[...] = (
                    cp.bitwise_and(raw_flat, 0x3FFF).astype(cp.float32) - pedoff
                ) * gain
                if self.extra_gpu_work > 0:
                    extra_start_evt.record()
                    scale = np.float32(1.000001)
                    offset = np.float32(0.000001)
                    for _ in range(self.extra_gpu_work):
                        cp.multiply(out_flat, scale, out=out_flat)
                        cp.add(out_flat, offset, out=out_flat)
                    extra_stop_evt.record()
                kernel_stop_evt.record()

        kernel_stop_evt.synchronize()
        copy_s = cp.cuda.get_elapsed_time(copy_start_evt, copy_stop_evt) / 1e3
        kernel_s = cp.cuda.get_elapsed_time(kernel_start_evt, kernel_stop_evt) / 1e3
        extra_s = (
            cp.cuda.get_elapsed_time(extra_start_evt, extra_stop_evt) / 1e3
            if self.extra_gpu_work > 0
            else 0.0
        )
        return {
            "raw_host": raw_host,
            "raw_dev": raw_dev,
            "calib_dev": out_dev,
            "copy_s": copy_s,
            "kernel_s": kernel_s,
            "extra_s": extra_s,
            "raw_extract_s": raw_extract_s,
            "lease_wait_s": 0.0,
            "lease_spins": 0,
            "completion_latency_s": time.perf_counter() - submit_start,
            "raw_bytes": int(raw_host.nbytes),
            "shape": tuple(raw_host.shape),
        }


class AsyncJungfrauMultiOwnerGpu(JungfrauMultiOwnerGpu):
    """Per-BD async queue using local streams/events and optional GPU leases."""

    def __init__(self, det_raw, device_id, extra_gpu_work=0, local_depth=2, lease_pool=None):
        super().__init__(det_raw, device_id, extra_gpu_work=extra_gpu_work)
        self.local_depth = max(1, int(local_depth))
        self.lease_pool = lease_pool
        self.slots = [
            AsyncGpuSlot(slot_id=slot_id, stream=cp.cuda.Stream(non_blocking=True))
            for slot_id in range(self.local_depth)
        ]

    def has_free_slot(self):
        return any(not slot.busy for slot in self.slots)

    def pending_count(self):
        return sum(1 for slot in self.slots if slot.busy)

    def _free_slot(self):
        for slot in self.slots:
            if not slot.busy:
                return slot
        return None

    def submit_event(self, evt):
        slot = self._free_slot()
        if slot is None:
            raise RuntimeError("No free async GPU slot available")

        raw_start = time.perf_counter()
        with _nvtx_range("psana2-gpu/raw_extract"):
            raw = self.det_raw.raw(evt, copy=False)
            if raw is None:
                return None
            raw_host = np.ascontiguousarray(raw, dtype=np.uint16)
        raw_extract_s = time.perf_counter() - raw_start

        self._ensure_ccons(evt)

        lease_wait_s = 0.0
        lease_spins = 0
        if self.lease_pool is not None:
            lease_wait_s, lease_spins = self.lease_pool.acquire()

        copy_start_evt = cp.cuda.Event()
        copy_stop_evt = cp.cuda.Event()
        kernel_start_evt = cp.cuda.Event()
        kernel_stop_evt = cp.cuda.Event()
        extra_start_evt = cp.cuda.Event()
        extra_stop_evt = cp.cuda.Event()
        done_evt = cp.cuda.Event()

        with slot.stream:
            with _nvtx_range("psana2-gpu/h2d"):
                copy_start_evt.record()
                raw_dev = cp.asarray(raw_host)
                copy_stop_evt.record()

            with _nvtx_range("psana2-gpu/jungfrau_v3"):
                kernel_start_evt.record()
                out_dev = cp.empty(raw_host.shape, dtype=np.float32)
                raw_flat = raw_dev.reshape(-1)
                out_flat = out_dev.reshape(-1)
                size = int(raw_flat.size)
                base_idx = self._pixel_index(size)
                gain_idx = (raw_flat >> 14).astype(cp.int32, copy=False)
                cc_idx = 2 * (base_idx + size * gain_idx)
                pedoff = cp.take(self.ccons_dev, cc_idx)
                gain = cp.take(self.ccons_dev, cc_idx + 1)
                out_flat[...] = (
                    cp.bitwise_and(raw_flat, 0x3FFF).astype(cp.float32) - pedoff
                ) * gain
                if self.extra_gpu_work > 0:
                    extra_start_evt.record()
                    scale = np.float32(1.000001)
                    offset = np.float32(0.000001)
                    for _ in range(self.extra_gpu_work):
                        cp.multiply(out_flat, scale, out=out_flat)
                        cp.add(out_flat, offset, out=out_flat)
                    extra_stop_evt.record()
                kernel_stop_evt.record()
                done_evt.record()

        slot.busy = True
        slot.evt = evt
        slot.raw_host = raw_host
        slot.raw_dev = raw_dev
        slot.calib_dev = out_dev
        slot.copy_start_evt = copy_start_evt
        slot.copy_stop_evt = copy_stop_evt
        slot.kernel_start_evt = kernel_start_evt
        slot.kernel_stop_evt = kernel_stop_evt
        slot.extra_start_evt = extra_start_evt
        slot.extra_stop_evt = extra_stop_evt
        slot.done_evt = done_evt
        slot.raw_extract_s = raw_extract_s
        slot.lease_wait_s = lease_wait_s
        slot.lease_spins = lease_spins
        slot.submit_wall_t = time.perf_counter()
        return slot

    def poll_completed(self, wait_one=False):
        completed = []
        forced_ready_slot = None
        if wait_one and self.pending_count() and not self._has_completed_slot():
            first_busy = next(slot for slot in self.slots if slot.busy)
            first_busy.done_evt.synchronize()
            forced_ready_slot = first_busy

        for slot in self.slots:
            if not slot.busy:
                continue
            ready = slot is forced_ready_slot or self._event_is_ready(slot.done_evt)
            if not ready:
                continue
            completed.append(self._complete_slot(slot))
        return completed

    def drain(self):
        completed = []
        while self.pending_count():
            completed.extend(self.poll_completed(wait_one=True))
        return completed

    def _has_completed_slot(self):
        for slot in self.slots:
            if not slot.busy:
                continue
            if self._event_is_ready(slot.done_evt):
                return True
        return False

    def _event_is_ready(self, event):
        try:
            query = getattr(event, "query", None)
            if query is not None:
                return bool(query())
            event_ptr = getattr(event, "ptr", None)
            if event_ptr is None:
                event_ptr = getattr(event, "_ptr", None)
            if event_ptr is None:
                return False
            cp.cuda.runtime.eventQuery(event_ptr)
            return True
        except Exception as exc:
            if "cudaErrorNotReady" in str(exc) or "not ready" in str(exc).lower():
                return False
            raise

    def _complete_slot(self, slot):
        slot.done_evt.synchronize()
        copy_s = cp.cuda.get_elapsed_time(slot.copy_start_evt, slot.copy_stop_evt) / 1e3
        kernel_s = cp.cuda.get_elapsed_time(slot.kernel_start_evt, slot.kernel_stop_evt) / 1e3
        extra_s = (
            cp.cuda.get_elapsed_time(slot.extra_start_evt, slot.extra_stop_evt) / 1e3
            if self.extra_gpu_work > 0
            else 0.0
        )
        result = {
            "evt": slot.evt,
            "raw_host": slot.raw_host,
            "raw_dev": slot.raw_dev,
            "calib_dev": slot.calib_dev,
            "copy_s": copy_s,
            "kernel_s": kernel_s,
            "extra_s": extra_s,
            "raw_extract_s": slot.raw_extract_s,
            "lease_wait_s": slot.lease_wait_s,
            "lease_spins": slot.lease_spins,
            "completion_latency_s": time.perf_counter() - slot.submit_wall_t,
            "raw_bytes": int(slot.raw_host.nbytes),
            "shape": tuple(slot.raw_host.shape),
        }
        if self.lease_pool is not None:
            self.lease_pool.release()
        slot.reset()
        return result


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Jungfrau GPU queue comparison for multi-BD, one-GPU "
            "experiments. Modes compare current synchronous multiowner behavior, "
            "per-BD one-side async queues, and shared GPU lease admission control."
        )
    )
    add_datasource_args(parser)
    parser.add_argument(
        "--queue-mode",
        choices=QUEUE_MODES,
        default=QUEUE_ONE_SIDE,
        help="Queue model to benchmark.",
    )
    parser.add_argument(
        "--detector",
        default="jungfrau",
        help="Detector name to process (default: jungfrau).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Pass max_events to DataSource to cap events read from the source.",
    )
    parser.add_argument(
        "--skip-calib-load",
        default=None,
        help="Pass skip_calib_load=... to DataSource, for example --skip-calib-load all.",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="CUDA device id used by every active BD rank (default: 0).",
    )
    parser.add_argument(
        "--bd-local-depth",
        type=int,
        default=2,
        help="Per-BD async GPU queue depth for one-side and both-sides-lease modes.",
    )
    parser.add_argument(
        "--gpu-leases",
        type=int,
        default=2,
        help="Max active BD CUDA submissions per node in both-sides-lease mode.",
    )
    parser.add_argument(
        "--lease-sleep-us",
        type=int,
        default=50,
        help="Sleep interval while waiting for a GPU lease (default: 50 us).",
    )
    parser.add_argument(
        "--extra-gpu-work",
        type=int,
        default=0,
        help="Run this many extra in-place GPU math iterations after calibration.",
    )
    parser.add_argument(
        "--compare-cpu-events",
        type=int,
        default=0,
        help="Compare the first N completed BD events against det.raw.calib(evt).",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=1000,
        help="Print every N completed BD events (default: 1000).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level forwarded to DataSource (default: INFO).",
    )
    parser.add_argument(
        "--dry-run-gpu",
        action="store_true",
        help="Print rank/GPU visibility and exit without constructing DataSource.",
    )
    return parser


def _process_completion(rank, det, result, counters, args):
    counters["processed"] += 1
    processed = counters["processed"]
    counters["copied_bytes"] += result["raw_bytes"]
    counters["total_copy_s"] += result["copy_s"]
    counters["total_kernel_s"] += result["kernel_s"]
    counters["total_extra_s"] += result["extra_s"]
    counters["total_raw_extract_s"] += result["raw_extract_s"]
    counters["total_lease_wait_s"] += result["lease_wait_s"]
    counters["total_lease_spins"] += result["lease_spins"]
    counters["total_completion_latency_s"] += result["completion_latency_s"]

    if processed <= args.compare_cpu_events:
        diff, ok = _compare_cpu(det, result["evt"], result["calib_dev"])
        counters["max_abs_diff"] = max(counters["max_abs_diff"], diff)
        if not ok:
            counters["compare_failures"] += 1

    if _should_print_event(processed - 1, args.print_interval, args.max_events):
        print(
            f"[Rank {rank}] event {processed - 1} "
            f"timestamp={int(result['evt'].timestamp)} "
            f"queue_mode={args.queue_mode} "
            f"gpu_type={type(result['calib_dev']).__module__}.{type(result['calib_dev']).__name__} "
            f"storage=device shape={result['shape']} "
            f"raw_extract_s={result['raw_extract_s']:.6f} "
            f"h2d_s={result['copy_s']:.6f} kernel_s={result['kernel_s']:.6f} "
            f"extra_s={result['extra_s']:.6f} lease_wait_s={result['lease_wait_s']:.6f}",
            flush=True,
        )


def _new_counters():
    return {
        "processed": 0,
        "copied_bytes": 0,
        "total_copy_s": 0.0,
        "total_kernel_s": 0.0,
        "total_extra_s": 0.0,
        "total_raw_extract_s": 0.0,
        "total_lease_wait_s": 0.0,
        "total_lease_spins": 0,
        "total_free_slot_wait_s": 0.0,
        "total_completion_latency_s": 0.0,
        "max_abs_diff": 0.0,
        "compare_failures": 0,
    }


def _build_rank_summary(rank, assigned_gpu_device, worker, counters, loop_s, args):
    processed = counters["processed"]
    copied_bytes = counters["copied_bytes"]
    mib = copied_bytes / float(1024 ** 2)
    rate_evt_s = (processed / loop_s) if loop_s > 0 else 0.0
    copy_rate_mib_s = (
        mib / counters["total_copy_s"] if counters["total_copy_s"] > 0 else 0.0
    )
    avg_raw_extract_s = (
        counters["total_raw_extract_s"] / processed if processed else 0.0
    )
    avg_lease_wait_s = counters["total_lease_wait_s"] / processed if processed else 0.0
    avg_completion_latency_s = (
        counters["total_completion_latency_s"] / processed if processed else 0.0
    )
    ccons_upload_s = getattr(worker, "_ccons_upload_s", 0.0) if worker is not None else 0.0
    return {
        "rank": int(rank),
        "queue_mode": args.queue_mode,
        "events": int(processed),
        "loop_s": float(loop_s),
        "rate_evt_s": float(rate_evt_s),
        "gpu_device": int(assigned_gpu_device),
        "bd_local_depth": int(args.bd_local_depth),
        "gpu_leases": int(args.gpu_leases) if args.queue_mode == QUEUE_BOTH_SIDES_LEASE else -1,
        "extra_gpu_work": int(args.extra_gpu_work),
        "ccons_upload_s": float(ccons_upload_s),
        "raw_extract_total_s": float(counters["total_raw_extract_s"]),
        "raw_extract_avg_s": float(avg_raw_extract_s),
        "free_slot_wait_total_s": float(counters["total_free_slot_wait_s"]),
        "lease_wait_total_s": float(counters["total_lease_wait_s"]),
        "lease_wait_avg_s": float(avg_lease_wait_s),
        "lease_spins": int(counters["total_lease_spins"]),
        "completion_latency_avg_s": float(avg_completion_latency_s),
        "copy_total_s": float(counters["total_copy_s"]),
        "kernel_total_s": float(counters["total_kernel_s"]),
        "extra_total_s": float(counters["total_extra_s"]),
        "copied_mib": float(mib),
        "copy_rate_mib_s": float(copy_rate_mib_s),
        "compare_events": int(min(processed, args.compare_cpu_events)),
        "compare_failures": int(counters["compare_failures"]),
        "max_abs_diff": float(counters["max_abs_diff"]),
    }


def _format_value(value):
    return "%.2f" % float(value)


def _format_summary_table(first, gathered, stat_fields):
    copied_mib_total = sum(item["copied_mib"] for item in gathered)
    copy_total_s = sum(item["copy_total_s"] for item in gathered)
    copy_rate_global = copied_mib_total / copy_total_s if copy_total_s > 0 else 0.0
    lines = [
        "[Rank 0] all_bd_summary",
        (
            "queue_mode=%s bd_ranks=%d events_total=%d "
            "copied_mib_total=%.3f copy_rate_mib_s_global=%.3f"
            % (
                first["queue_mode"],
                len(gathered),
                sum(item["events"] for item in gathered),
                copied_mib_total,
                copy_rate_global,
            )
        ),
        "%-32s %14s %14s %14s" % ("metric", "avg", "min", "max"),
        "%-32s %14s %14s %14s" % ("-" * 6, "-" * 3, "-" * 3, "-" * 3),
    ]
    for field in stat_fields:
        arr = np.asarray([item[field] for item in gathered], dtype=np.float64)
        lines.append(
            "%-32s %14s %14s %14s"
            % (
                field,
                _format_value(float(arr.mean())),
                _format_value(float(arr.min())),
                _format_value(float(arr.max())),
            )
        )
    return "\n".join(lines)


def _print_rank_summary(rank_summary, args):
    gpu_leases = (
        rank_summary["gpu_leases"]
        if args.queue_mode == QUEUE_BOTH_SIDES_LEASE
        else "none"
    )
    print(
        f"[Rank {rank_summary['rank']}] summary queue_mode={rank_summary['queue_mode']} "
        f"events={rank_summary['events']} "
        f"loop_s={rank_summary['loop_s']:.3f} rate_evt_s={rank_summary['rate_evt_s']:.3f} "
        f"gpu_device={rank_summary['gpu_device']} "
        f"bd_local_depth={rank_summary['bd_local_depth']} "
        f"gpu_leases={gpu_leases} "
        f"extra_gpu_work={rank_summary['extra_gpu_work']} "
        f"ccons_upload_s={rank_summary['ccons_upload_s']:.6f} "
        f"raw_extract_total_s={rank_summary['raw_extract_total_s']:.6f} "
        f"raw_extract_avg_s={rank_summary['raw_extract_avg_s']:.6f} "
        f"free_slot_wait_total_s={rank_summary['free_slot_wait_total_s']:.6f} "
        f"lease_wait_total_s={rank_summary['lease_wait_total_s']:.6f} "
        f"lease_wait_avg_s={rank_summary['lease_wait_avg_s']:.6f} "
        f"lease_spins={rank_summary['lease_spins']} "
        f"completion_latency_avg_s={rank_summary['completion_latency_avg_s']:.6f} "
        f"copy_total_s={rank_summary['copy_total_s']:.6f} "
        f"kernel_total_s={rank_summary['kernel_total_s']:.6f} "
        f"extra_total_s={rank_summary['extra_total_s']:.6f} "
        f"copied_mib={rank_summary['copied_mib']:.3f} "
        f"copy_rate_mib_s={rank_summary['copy_rate_mib_s']:.3f} "
        f"compare_events={rank_summary['compare_events']} "
        f"compare_failures={rank_summary['compare_failures']} "
        f"max_abs_diff={rank_summary['max_abs_diff']:.6g}",
        flush=True,
    )


def _print_all_bd_summary(rank_summary, is_bd):
    if MPI is None:
        gathered = [rank_summary] if is_bd else []
        rank = 0
    else:
        gathered = MPI.COMM_WORLD.gather(rank_summary if is_bd else None, root=0)
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            return
        gathered = [item for item in gathered if item is not None]

    if not gathered:
        return

    first = gathered[0]
    stat_fields = (
        "events",
        "loop_s",
        "rate_evt_s",
        "ccons_upload_s",
        "raw_extract_total_s",
        "raw_extract_avg_s",
        "free_slot_wait_total_s",
        "lease_wait_total_s",
        "lease_wait_avg_s",
        "lease_spins",
        "completion_latency_avg_s",
        "copy_total_s",
        "kernel_total_s",
        "extra_total_s",
        "copied_mib",
        "copy_rate_mib_s",
        "compare_events",
        "compare_failures",
        "max_abs_diff",
    )
    print(_format_summary_table(first, gathered, stat_fields), flush=True)


def main():
    args = _build_parser().parse_args()
    if cp is None:
        raise SystemExit("CuPy is unavailable in the current environment.")
    if args.extra_gpu_work < 0:
        raise SystemExit("--extra-gpu-work must be >= 0.")
    if args.bd_local_depth < 1:
        raise SystemExit("--bd-local-depth must be >= 1.")
    if args.gpu_leases < 1:
        raise SystemExit("--gpu-leases must be >= 1.")

    rank = _world_rank()
    local_rank = _local_rank()
    assigned_gpu_device = args.gpu_device

    lease_pool = None
    if args.queue_mode == QUEUE_BOTH_SIDES_LEASE:
        lease_pool = NodeGpuLeasePool(
            max_leases=args.gpu_leases,
            sleep_us=args.lease_sleep_us,
        )

    if args.dry_run_gpu:
        visible_gpus, _error = _probe_visible_gpu_count()
        print(
            f"[Rank {rank}] dry_run local_rank={local_rank} "
            f"visible_gpus={visible_gpus} assigned_gpu={assigned_gpu_device} "
            f"queue_mode={args.queue_mode} {_gpu_visibility_summary()}",
            flush=True,
        )
        if lease_pool is not None:
            lease_pool.close()
        return

    ds_kwargs = datasource_kwargs_from_args(args)
    ds_kwargs = apply_max_events(ds_kwargs, args.max_events)
    ds_kwargs = apply_skip_calib_load(ds_kwargs, args.skip_calib_load)
    ds = DataSource(**ds_kwargs, log_level=args.log_level)
    is_bd = bool(ds.is_bd())
    if rank == 0:
        print(_gpu_visibility_summary(), flush=True)

    run = next(ds.runs())
    det = run.Detector(args.detector)

    worker = None
    if is_bd:
        if args.queue_mode == QUEUE_BASELINE:
            worker = MeasuredJungfrauMultiOwnerGpu(
                det.raw,
                assigned_gpu_device,
                extra_gpu_work=args.extra_gpu_work,
            )
        else:
            worker = AsyncJungfrauMultiOwnerGpu(
                det.raw,
                assigned_gpu_device,
                extra_gpu_work=args.extra_gpu_work,
                local_depth=args.bd_local_depth,
                lease_pool=lease_pool,
            )
        print(
            f"[Rank {rank}] pid={os.getpid()} "
            f"local_rank={local_rank} using CUDA device {assigned_gpu_device} "
            f"queue_mode={args.queue_mode} bd_local_depth={args.bd_local_depth} "
            f"gpu_leases={args.gpu_leases if args.queue_mode == QUEUE_BOTH_SIDES_LEASE else 'none'}",
            flush=True,
        )

    counters = _new_counters()
    loop_start = time.perf_counter()

    for evt in run.events():
        if not is_bd:
            continue

        if args.queue_mode == QUEUE_BASELINE:
            result = worker.process_event(evt)
            if result is None:
                continue
            result["evt"] = evt
            _process_completion(rank, det, result, counters, args)
            continue

        free_wait_start = None
        while not worker.has_free_slot():
            if free_wait_start is None:
                free_wait_start = time.perf_counter()
            completions = worker.poll_completed(wait_one=True)
            for result in completions:
                _process_completion(rank, det, result, counters, args)
        if free_wait_start is not None:
            counters["total_free_slot_wait_s"] += time.perf_counter() - free_wait_start

        submitted = worker.submit_event(evt)
        if submitted is None:
            continue
        for result in worker.poll_completed(wait_one=False):
            _process_completion(rank, det, result, counters, args)

    if is_bd and args.queue_mode != QUEUE_BASELINE:
        for result in worker.drain():
            _process_completion(rank, det, result, counters, args)

    loop_s = time.perf_counter() - loop_start

    rank_summary = None
    if is_bd:
        rank_summary = _build_rank_summary(
            rank,
            assigned_gpu_device,
            worker,
            counters,
            loop_s,
            args,
        )

    _print_all_bd_summary(rank_summary, is_bd)

    _print_global_summary(
        rank,
        is_bd,
        counters["processed"],
        loop_s,
    )

    if lease_pool is not None:
        if lease_pool.node_rank == 0:
            snapshot = lease_pool.snapshot()
            print(
                f"[Rank {rank}] lease_summary max_leases={args.gpu_leases} "
                f"active={snapshot['active']} acquire_count={snapshot['acquire_count']} "
                f"contention_count={snapshot['contention_count']}",
                flush=True,
            )
        lease_pool.close()


if __name__ == "__main__":
    main()
