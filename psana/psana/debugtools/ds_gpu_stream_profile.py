#!/usr/bin/env python3
"""
ds_gpu_stream_profile.py

Simple MPI-friendly profiling script for psana event streaming with optional
Host-to-Device copies. It is designed to answer:
1) How fast can the current CPU-event-loop + H2D path run?
2) What changes when multiple ranks share one GPU?
"""

import argparse
import json
import os
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import psutil
from mpi4py import MPI
from psana import DataSource

try:
    from psana.psexp import parallel_pread as _parallel_pread
except ImportError:
    _parallel_pread = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile psana detector event loop with optional GPU staging copies."
    )
    parser.add_argument("-e", "--exp", help="Experiment name, e.g. mfx101344525")
    parser.add_argument("-r", "--run", type=int, help="Run number, e.g. 125")
    parser.add_argument("--xtc_files", nargs="+", help="Explicit list of XTC2 files")
    parser.add_argument("--dir", help="Path to directory containing XTC2 files")
    parser.add_argument("-d", "--detectors", nargs="*", default=[], help="DataSource detector list")
    parser.add_argument("-c", "--cached_detectors", nargs="*", default=[], help="DataSource cached_detectors list")
    parser.add_argument("--max_events", type=int, default=0, help="Max events per rank (0=all)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Events per batch (DataSource)")
    parser.add_argument("--log_level", default="INFO", help="DataSource log level")
    parser.add_argument("--log_file", default=None, help="Optional DataSource log file")
    parser.add_argument("--monitor", action="store_true", help="Enable DataSource monitor mode")
    parser.add_argument("--live", action="store_true", help="Enable DataSource live mode")
    parser.add_argument("--use_calib_cache", action="store_true", help="Use cached calibration constants")
    parser.add_argument(
        "--skip_calib_load",
        nargs="+",
        default=None,
        help="Detectors to skip calibration loading, or 'all'",
    )
    parser.add_argument("--debug_detector", required=True, help="Detector to profile")
    parser.add_argument(
        "--detector-method",
        choices=("raw", "calib", "image"),
        default="raw",
        help="Detector method to run per event",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cupy", "torch", "none"),
        default="auto",
        help="GPU copy backend; use 'none' for CPU-only baseline",
    )
    parser.add_argument(
        "--host-mem",
        choices=("pageable", "pinned"),
        default="pinned",
        help="Host staging mode for H2D copy",
    )
    parser.add_argument(
        "--gpu-map",
        choices=("local_rank", "global_rank", "fixed"),
        default="local_rank",
        help="How to map ranks to GPU id",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id when --gpu-map fixed")
    parser.add_argument("--print_interval", type=int, default=200, help="Progress print interval in events")
    parser.add_argument("--show_rank_stats", action="store_true", help="Print per-rank final stats on rank0")
    parser.add_argument("--json_out", default=None, help="Write summary JSON to this path (rank0 only)")
    args = parser.parse_args()
    if args.skip_calib_load is not None and any(v.lower() == "all" for v in args.skip_calib_load):
        args.skip_calib_load = "all"
    return args


def create_datasource(args, rank):
    common_kwargs = dict(
        max_events=args.max_events,
        batch_size=args.batch_size,
        log_level=args.log_level,
        detectors=args.detectors,
        cached_detectors=args.cached_detectors,
        use_calib_cache=args.use_calib_cache,
        monitor=args.monitor,
        log_file=args.log_file,
    )
    if args.skip_calib_load is not None:
        common_kwargs["skip_calib_load"] = args.skip_calib_load

    if args.xtc_files:
        if rank == 0:
            print(f"Using explicit XTC2 files ({len(args.xtc_files)}):")
            for path in args.xtc_files:
                print(f"  {path}")
        return DataSource(files=args.xtc_files, **common_kwargs)

    if args.exp is None or args.run is None:
        raise ValueError("Either --xtc_files or both --exp and --run must be provided.")

    if args.dir:
        dir_path = args.dir
    elif args.live:
        dir_path = f"/sdf/data/lcls/drpsrcf/ffb/{args.exp[:3]}/{args.exp}/xtc"
    else:
        dir_path = None

    if rank == 0:
        print(f"Using exp={args.exp} run={args.run} dir={dir_path}")

    return DataSource(exp=args.exp, run=args.run, live=args.live, dir=dir_path, **common_kwargs)


def _iter_numpy_arrays(obj) -> Iterable[np.ndarray]:
    if obj is None:
        return
    if isinstance(obj, np.ndarray):
        if obj.size > 0 and obj.dtype != object:
            yield np.ascontiguousarray(obj)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_numpy_arrays(item)
        return
    if isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_numpy_arrays(item)
        return
    try:
        arr = np.asarray(obj)
    except Exception:
        return
    if isinstance(arr, np.ndarray) and arr.size > 0 and arr.dtype != object:
        yield np.ascontiguousarray(arr)


def _get_detector_data(det, method, evt):
    if method == "raw":
        return det.raw.raw(evt)
    if method == "calib":
        return det.raw.calib(evt)
    if method == "image":
        return det.raw.image(evt)
    raise ValueError(f"Unknown detector method: {method}")


class NoCopyBackend:
    name = "none"

    def copy_array(self, arr: np.ndarray) -> Tuple[int, float]:
        return arr.nbytes, 0.0


class CupyBackend:
    name = "cupy"

    def __init__(self, gpu_id: int, host_mem: str):
        import cupy as cp

        self.cp = cp
        self.device = cp.cuda.Device(gpu_id)
        self.device.use()
        self.host_mem = host_mem
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.device_buf = None
        self.device_cap = 0
        self.pin_mem = None
        self.pin_view = None
        self.pin_cap = 0

    def _ensure_device(self, nbytes: int):
        if nbytes > self.device_cap:
            self.device_buf = self.cp.empty((nbytes,), dtype=self.cp.uint8)
            self.device_cap = nbytes

    def _ensure_pinned(self, nbytes: int):
        if nbytes > self.pin_cap:
            self.pin_mem = self.cp.cuda.alloc_pinned_memory(nbytes)
            self.pin_view = np.frombuffer(self.pin_mem, dtype=np.uint8, count=nbytes)
            self.pin_cap = nbytes

    def copy_array(self, arr: np.ndarray) -> Tuple[int, float]:
        src = arr.view(np.uint8).reshape(-1)
        nbytes = src.nbytes
        self._ensure_device(nbytes)
        t0 = time.perf_counter()
        if self.host_mem == "pinned":
            self._ensure_pinned(nbytes)
            self.pin_view[:nbytes] = src
            self.cp.cuda.runtime.memcpyAsync(
                self.device_buf.data.ptr,
                int(self.pin_view.ctypes.data),
                nbytes,
                self.cp.cuda.runtime.memcpyHostToDevice,
                self.stream.ptr,
            )
            self.stream.synchronize()
        else:
            self.cp.cuda.runtime.memcpy(
                self.device_buf.data.ptr,
                int(src.ctypes.data),
                nbytes,
                self.cp.cuda.runtime.memcpyHostToDevice,
            )
        return nbytes, time.perf_counter() - t0


class TorchBackend:
    name = "torch"

    def __init__(self, gpu_id: int, host_mem: str):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch CUDA is not available")
        self.torch = torch
        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(self.device)
        self.host_mem = host_mem
        self.device_buf = None
        self.device_cap = 0
        self.pin_tensor = None
        self.pin_view = None
        self.pin_cap = 0

    def _ensure_device(self, nbytes: int):
        if nbytes > self.device_cap:
            self.device_buf = self.torch.empty((nbytes,), dtype=self.torch.uint8, device=self.device)
            self.device_cap = nbytes

    def _ensure_pinned(self, nbytes: int):
        if nbytes > self.pin_cap:
            self.pin_tensor = self.torch.empty((nbytes,), dtype=self.torch.uint8, pin_memory=True)
            self.pin_view = self.pin_tensor.numpy()
            self.pin_cap = nbytes

    def copy_array(self, arr: np.ndarray) -> Tuple[int, float]:
        src = arr.view(np.uint8).reshape(-1)
        nbytes = src.nbytes
        self._ensure_device(nbytes)
        t0 = time.perf_counter()
        if self.host_mem == "pinned":
            self._ensure_pinned(nbytes)
            self.pin_view[:nbytes] = src
            self.device_buf[:nbytes].copy_(self.pin_tensor[:nbytes], non_blocking=True)
            self.torch.cuda.synchronize(self.device)
        else:
            src_t = self.torch.from_numpy(src)
            self.device_buf[:nbytes].copy_(src_t, non_blocking=False)
            self.torch.cuda.synchronize(self.device)
        return nbytes, time.perf_counter() - t0


def _gpu_count() -> int:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pass
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _resolve_gpu_id(args, rank: int, size: int) -> int:
    n_gpu = _gpu_count()
    if n_gpu <= 0:
        return -1
    if args.gpu_map == "fixed":
        return args.gpu_id % n_gpu
    if args.gpu_map == "global_rank":
        return rank % n_gpu
    local_rank = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("SLURM_LOCALID"))
    if local_rank is None:
        local_rank = str(rank % max(size, 1))
    return int(local_rank) % n_gpu


def _build_backend(args, gpu_id: int):
    if args.backend == "none":
        return NoCopyBackend()
    if gpu_id < 0:
        raise RuntimeError("No visible CUDA devices for requested backend.")
    errors = []
    if args.backend in ("auto", "cupy"):
        try:
            return CupyBackend(gpu_id, args.host_mem)
        except Exception as exc:
            errors.append(f"cupy: {exc}")
            if args.backend == "cupy":
                raise
    if args.backend in ("auto", "torch"):
        try:
            return TorchBackend(gpu_id, args.host_mem)
        except Exception as exc:
            errors.append(f"torch: {exc}")
            if args.backend == "torch":
                raise
    raise RuntimeError("Failed to initialize backend. " + "; ".join(errors))


def _rss_gb(process):
    return process.memory_info().rss / (1024 ** 3)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    process = psutil.Process(os.getpid())

    gpu_id = _resolve_gpu_id(args, rank, size)
    backend = _build_backend(args, gpu_id)

    comm.Barrier()
    t_start = MPI.Wtime()
    ds = create_datasource(args, rank)
    if _parallel_pread is not None:
        _parallel_pread.reset_parallel_pread_stats()
    run = next(ds.runs())
    det = run.Detector(args.debug_detector)

    n_evt = 0
    n_evt_with_data = 0
    n_copy_calls = 0
    det_s = 0.0
    h2d_s = 0.0
    data_bytes = 0
    rss_start = _rss_gb(process)
    rss_peak = rss_start
    ti = time.perf_counter()
    bytes_i = 0

    for evt in run.events():
        t0 = time.perf_counter()
        data = _get_detector_data(det, args.detector_method, evt)
        det_s += time.perf_counter() - t0
        evt_bytes = 0
        for arr in _iter_numpy_arrays(data):
            nbytes, dt = backend.copy_array(arr)
            evt_bytes += nbytes
            h2d_s += dt
            n_copy_calls += 1
        n_evt += 1
        data_bytes += evt_bytes
        bytes_i += evt_bytes
        if evt_bytes > 0:
            n_evt_with_data += 1
        if n_evt % max(1, args.print_interval) == 0:
            now = time.perf_counter()
            dt = max(now - ti, 1e-12)
            hz = args.print_interval / dt
            e2e = bytes_i / dt / (1024 ** 3)
            rss_peak = max(rss_peak, _rss_gb(process))
            print(
                f"[rank {rank}] evt={n_evt} hz={hz:.1f} e2e={e2e:.2f} GiB/s "
                f"bytes={bytes_i / (1024 ** 3):.2f} GiB rss={rss_peak:.2f} GiB"
            )
            ti = now
            bytes_i = 0

    t_end = MPI.Wtime()
    rss_end = _rss_gb(process)
    rss_peak = max(rss_peak, rss_end)
    wall_s = max(t_end - t_start, 1e-12)

    if _parallel_pread is not None:
        pread_s, pread_bytes, pread_calls = _parallel_pread.parallel_pread_stats()
    else:
        pread_s, pread_bytes, pread_calls = 0.0, 0.0, 0.0

    rank_stats = dict(
        rank=rank,
        gpu_id=gpu_id,
        backend=getattr(backend, "name", "unknown"),
        host_mem=args.host_mem,
        events=n_evt,
        events_with_data=n_evt_with_data,
        data_bytes=data_bytes,
        copy_calls=n_copy_calls,
        det_s=det_s,
        h2d_s=h2d_s,
        wall_s=wall_s,
        e2e_gibs=(data_bytes / wall_s / (1024 ** 3)),
        h2d_gibs=(data_bytes / h2d_s / (1024 ** 3)) if h2d_s > 0 else 0.0,
        rss_start_gb=rss_start,
        rss_peak_gb=rss_peak,
        rss_end_gb=rss_end,
        pread_s=float(pread_s),
        pread_bytes=float(pread_bytes),
        pread_calls=int(pread_calls),
    )
    gathered = comm.gather(rank_stats, root=0)

    if rank == 0:
        total_events = int(sum(v["events"] for v in gathered))
        total_data_bytes = float(sum(v["data_bytes"] for v in gathered))
        max_wall_s = float(max(v["wall_s"] for v in gathered))
        agg_e2e_gibs = total_data_bytes / max_wall_s / (1024 ** 3) if max_wall_s > 0 else 0.0
        print("ds_gpu_stream_profile")
        print(f"  ranks          : {size}")
        print(f"  detector       : {args.debug_detector}")
        print(f"  method         : {args.detector_method}")
        print(f"  backend        : {gathered[0]['backend']}")
        print(f"  host_mem       : {args.host_mem}")
        print(f"  total_events   : {total_events}")
        print(f"  total_data     : {total_data_bytes / (1024 ** 3):.2f} GiB")
        print(f"  agg_e2e        : {agg_e2e_gibs:.2f} GiB/s")
        print(
            f"  per_rank_e2e   : min={min(v['e2e_gibs'] for v in gathered):.2f} "
            f"max={max(v['e2e_gibs'] for v in gathered):.2f} GiB/s"
        )
        print(
            f"  per_rank_h2d   : min={min(v['h2d_gibs'] for v in gathered):.2f} "
            f"max={max(v['h2d_gibs'] for v in gathered):.2f} GiB/s"
        )
        if args.show_rank_stats:
            for v in sorted(gathered, key=lambda x: x["rank"]):
                print(
                    f"  rank={v['rank']:>3} gpu={v['gpu_id']:>2} evt={v['events']:>7} "
                    f"e2e={v['e2e_gibs']:.2f} h2d={v['h2d_gibs']:.2f} "
                    f"rss_peak={v['rss_peak_gb']:.2f} GiB"
                )
        if args.json_out:
            payload = dict(
                config=vars(args),
                aggregate=dict(
                    ranks=size,
                    total_events=total_events,
                    total_data_bytes=total_data_bytes,
                    agg_e2e_gibs=agg_e2e_gibs,
                ),
                ranks=gathered,
            )
            with open(args.json_out, "w", encoding="utf-8") as out:
                json.dump(payload, out, indent=2, sort_keys=True)
            print(f"  json_out       : {args.json_out}")


if __name__ == "__main__":
    main()
