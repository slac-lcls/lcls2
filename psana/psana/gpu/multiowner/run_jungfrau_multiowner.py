from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

from psana import DataSource


def add_datasource_args(parser):
    source_group = parser.add_argument_group("data source")
    source_group.add_argument(
        "--file",
        default=None,
        help="Input xtc2 file.",
    )
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


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone multi-owner Jungfrau GPU baseline. Each bigdata rank "
            "creates its own CuPy context and launches its own GPU work, with "
            "all ranks optionally pinned to the same GPU device."
        )
    )
    add_datasource_args(parser)
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
        help=(
            "Pass skip_calib_load=... to DataSource to bypass calibration DB access "
            "(for example: --skip-calib-load all)."
        ),
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="CUDA device id to use for every active BD rank (default: 0).",
    )
    parser.add_argument(
        "--gpu-auto",
        action="store_true",
        help=(
            "Assign one visible GPU per BD rank on each node. "
            "Fails if the number of local BD ranks exceeds the number of visible GPUs."
        ),
    )
    parser.add_argument(
        "--cpu-raw-only",
        action="store_true",
        help=(
            "Skip all GPU work and only measure CPU/I-O scaling for "
            "det.raw.raw(evt, copy=False)."
        ),
    )
    parser.add_argument(
        "--compare-cpu-events",
        type=int,
        default=0,
        help="Compare the first N BD events against det.raw.calib(evt).",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=1000,
        help="Print every N processed BD events (default: 1000).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level forwarded to DataSource (default: INFO).",
    )
    parser.add_argument(
        "--dry-run-gpu-map",
        action="store_true",
        help=(
            "Print the per-rank GPU assignment and exit without constructing a "
            "DataSource. Assumes rank 0 is smd0, rank 1 is EB, and ranks >=2 are BD."
        ),
    )
    return parser


def _should_print_event(index, interval, total_events=None):
    if interval <= 0:
        return True
    if (index + 1) % interval == 0:
        return True
    if total_events is not None and (index + 1) >= total_events:
        return True
    return False


def _world_rank():
    try:
        from mpi4py import MPI
    except Exception:
        return 0
    return MPI.COMM_WORLD.Get_rank()


def _mpi_comm():
    try:
        from mpi4py import MPI
    except Exception:
        return None
    return MPI.COMM_WORLD


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


def _visible_gpu_count():
    if cp is None:
        raise SystemExit("CuPy is required for GPU device discovery.")
    return int(cp.cuda.runtime.getDeviceCount())


def _auto_gpu_device(rank, local_rank, is_bd):
    visible_gpu_count = _visible_gpu_count()
    if visible_gpu_count <= 0:
        raise SystemExit("No visible CUDA devices found for --gpu-auto.")

    hostname = os.environ.get("HOSTNAME") or os.uname().nodename
    comm = _mpi_comm()
    if comm is None:
        if not is_bd:
            return None
        bd_ordinal = 0
        local_bd_count = 1
    else:
        gathered = comm.allgather(
            {
                "rank": rank,
                "host": hostname,
                "local_rank": local_rank,
                "is_bd": bool(is_bd),
            }
        )
        local_bd_ranks = sorted(
            item["local_rank"]
            for item in gathered
            if item["host"] == hostname and item["is_bd"]
        )
        local_bd_count = len(local_bd_ranks)
        if not is_bd:
            return None
        try:
            bd_ordinal = local_bd_ranks.index(local_rank)
        except ValueError as exc:
            raise SystemExit(
                f"Failed to determine local BD ordinal for rank {rank} on host {hostname}."
            ) from exc

    if local_bd_count > visible_gpu_count:
        raise SystemExit(
            "Requested --gpu-auto but there are more local BD ranks than visible GPUs "
            f"on host {hostname}: bd_ranks={local_bd_count} visible_gpus={visible_gpu_count}"
        )

    return bd_ordinal


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


class JungfrauMultiOwnerGpu:
    def __init__(self, det_raw, device_id):
        if cp is None:
            raise SystemExit("CuPy is required for this script.")
        self.det_raw = det_raw
        self.device_id = int(device_id)
        self.device = cp.cuda.Device(self.device_id)
        self.device.use()
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

    def process_event(self, evt):
        with _nvtx_range("psana2-gpu/raw_extract"):
            raw = self.det_raw.raw(evt, copy=False)
            if raw is None:
                return None
            raw_host = np.ascontiguousarray(raw, dtype=np.uint16)

        self._ensure_ccons(evt)

        copy_start_evt = cp.cuda.Event()
        copy_stop_evt = cp.cuda.Event()
        kernel_start_evt = cp.cuda.Event()
        kernel_stop_evt = cp.cuda.Event()

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
                kernel_stop_evt.record()

        kernel_stop_evt.synchronize()
        copy_s = cp.cuda.get_elapsed_time(copy_start_evt, copy_stop_evt) / 1e3
        kernel_s = cp.cuda.get_elapsed_time(kernel_start_evt, kernel_stop_evt) / 1e3
        return {
            "raw_host": raw_host,
            "raw_dev": raw_dev,
            "calib_dev": out_dev,
            "copy_s": copy_s,
            "kernel_s": kernel_s,
            "raw_bytes": int(raw_host.nbytes),
            "shape": tuple(raw_host.shape),
        }


def _process_cpu_raw_only(det, evt):
    with _nvtx_range("psana2-gpu/raw_extract"):
        raw = det.raw.raw(evt, copy=False)
        if raw is None:
            return None
        raw_host = np.ascontiguousarray(raw, dtype=np.uint16)
    return {
        "raw_host": raw_host,
        "raw_dev": None,
        "calib_dev": None,
        "copy_s": 0.0,
        "kernel_s": 0.0,
        "raw_bytes": int(raw_host.nbytes),
        "shape": tuple(raw_host.shape),
    }


def _compare_cpu(det, evt, gpu_calib):
    cpu_calib = det.raw.calib(evt, cversion=3)
    cpu_host = np.asarray(cpu_calib, dtype=np.float32)
    gpu_host = cp.asnumpy(gpu_calib)
    if gpu_host.shape != cpu_host.shape:
        squeezed = np.squeeze(gpu_host)
        if squeezed.shape != cpu_host.shape:
            raise SystemExit(
                f"Shape mismatch during compare: cpu={cpu_host.shape} gpu={gpu_host.shape}"
            )
        gpu_host = squeezed
    max_abs_diff = float(np.max(np.abs(cpu_host - gpu_host)))
    allclose = bool(np.allclose(cpu_host, gpu_host, rtol=1e-4, atol=1e-5))
    return max_abs_diff, allclose


def _print_global_summary(rank, is_bd, processed, loop_s, cpu_raw_only=False, cpu_raw_type=None):
    comm = _mpi_comm()
    if comm is None:
        total_events = int(processed)
        elapsed_s = float(loop_s)
        total_rate_evt_s = (total_events / elapsed_s) if elapsed_s > 0 else 0.0
        summary = (
            f"[Rank {rank}] all_ranks_summary total_events={total_events} "
            f"elapsed_s={elapsed_s:.3f} total_rate_evt_s={total_rate_evt_s:.3f} "
            f"bd_ranks={1 if is_bd else 0} world_size=1"
        )
        if cpu_raw_only:
            summary += (
                f" cpu_raw_only=true "
                f"cpu_raw_type={cpu_raw_type if cpu_raw_type is not None else 'unknown'}"
            )
        print(summary, flush=True)
        return

    gathered = comm.gather(
        {
            "is_bd": bool(is_bd),
            "processed": int(processed),
            "loop_s": float(loop_s),
            "cpu_raw_type": cpu_raw_type,
        },
        root=0,
    )
    if rank != 0:
        return

    total_events = sum(item["processed"] for item in gathered)
    elapsed_s = max((item["loop_s"] for item in gathered), default=0.0)
    bd_ranks = sum(1 for item in gathered if item["is_bd"])
    total_rate_evt_s = (total_events / elapsed_s) if elapsed_s > 0 else 0.0
    summary = (
        f"[Rank 0] all_ranks_summary total_events={total_events} "
        f"elapsed_s={elapsed_s:.3f} total_rate_evt_s={total_rate_evt_s:.3f} "
        f"bd_ranks={bd_ranks} world_size={len(gathered)}"
    )
    if cpu_raw_only:
        cpu_raw_type_global = next(
            (item["cpu_raw_type"] for item in gathered if item["cpu_raw_type"] is not None),
            "unknown",
        )
        summary += f" cpu_raw_only=true cpu_raw_type={cpu_raw_type_global}"
    print(summary, flush=True)


def main():
    args = _build_parser().parse_args()
    if cp is None and not args.cpu_raw_only:
        raise SystemExit("CuPy is unavailable in the current environment.")
    if args.cpu_raw_only and args.compare_cpu_events:
        raise SystemExit("--compare-cpu-events is incompatible with --cpu-raw-only.")
    if args.gpu_auto and "--gpu-device" in os.sys.argv:
        raise SystemExit("Use either --gpu-device or --gpu-auto, not both.")

    rank = _world_rank()
    local_rank = _local_rank()
    if args.dry_run_gpu_map:
        is_bd = rank >= 2
        assigned_gpu_device = None
        assigned_gpu_device = (
            _auto_gpu_device(rank, local_rank, is_bd)
            if args.gpu_auto
            else args.gpu_device
        )
        visible_gpus = 0 if args.cpu_raw_only else _visible_gpu_count()
        print(
            f"[Rank {rank}] dry_run local_rank={local_rank} is_bd={is_bd} "
            f"visible_gpus={visible_gpus} assigned_gpu={assigned_gpu_device}",
            flush=True,
        )
        return

    ds_kwargs = datasource_kwargs_from_args(args)
    ds_kwargs = apply_max_events(ds_kwargs, args.max_events)
    ds_kwargs = apply_skip_calib_load(ds_kwargs, args.skip_calib_load)
    ds = DataSource(**ds_kwargs, log_level=args.log_level)
    is_bd = bool(ds.is_bd())
    assigned_gpu_device = None
    if not args.cpu_raw_only:
        assigned_gpu_device = (
            _auto_gpu_device(rank, local_rank, is_bd)
            if args.gpu_auto
            else args.gpu_device
        )
    print(
        f"[Rank {rank}] pid={os.getpid()} local_rank={local_rank} "
        f"assigned CUDA device {assigned_gpu_device} before run setup",
        flush=True,
    )

    run = next(ds.runs())
    det = run.Detector(args.detector)

    worker = None
    if is_bd and not args.cpu_raw_only:
        worker = JungfrauMultiOwnerGpu(det.raw, assigned_gpu_device)
        print(
            f"[Rank {rank}] pid={os.getpid()} local_rank={local_rank} "
            f"using CUDA device {assigned_gpu_device} as an independent owner"
            ,
            flush=True,
        )

    processed = 0
    copied_bytes = 0
    total_copy_s = 0.0
    total_kernel_s = 0.0
    max_abs_diff = 0.0
    compare_failures = 0
    cpu_raw_type = None
    loop_start = time.perf_counter()

    for evt in run.events():
        if not is_bd:
            continue

        result = (
            _process_cpu_raw_only(det, evt)
            if args.cpu_raw_only
            else worker.process_event(evt)
        )
        if result is None:
            continue

        copied_bytes += result["raw_bytes"]
        total_copy_s += result["copy_s"]
        total_kernel_s += result["kernel_s"]
        if args.cpu_raw_only and cpu_raw_type is None:
            cpu_raw_type = (
                f"{type(result['raw_host']).__module__}.{type(result['raw_host']).__name__}"
            )

        if processed < args.compare_cpu_events:
            diff, ok = _compare_cpu(det, evt, result["calib_dev"])
            max_abs_diff = max(max_abs_diff, diff)
            if not ok:
                compare_failures += 1

        if _should_print_event(processed, args.print_interval, args.max_events):
            if args.cpu_raw_only:
                print(
                    f"[Rank {rank}] event {processed} timestamp={int(evt.timestamp)} "
                    f"cpu_raw_type={type(result['raw_host']).__module__}.{type(result['raw_host']).__name__} "
                    f"storage=host shape={result['shape']}",
                    flush=True,
                )
            else:
                print(
                    f"[Rank {rank}] event {processed} timestamp={int(evt.timestamp)} "
                    f"gpu_type={type(result['calib_dev']).__module__}.{type(result['calib_dev']).__name__} "
                    f"storage=device shape={result['shape']} "
                    f"h2d_s={result['copy_s']:.6f} kernel_s={result['kernel_s']:.6f}",
                    flush=True,
                )
        processed += 1

    loop_s = time.perf_counter() - loop_start

    if is_bd:
        mib = copied_bytes / float(1024 ** 2)
        rate_evt_s = (processed / loop_s) if loop_s > 0 else 0.0
        copy_rate_mib_s = (mib / total_copy_s) if total_copy_s > 0 else 0.0
        if args.cpu_raw_only:
            print(
                f"[Rank {rank}] summary events={processed} loop_s={loop_s:.3f} "
                f"rate_evt_s={rate_evt_s:.3f} gpu_device=None "
                f"cpu_raw_only=true copied_mib={mib:.3f}",
                flush=True,
            )
        else:
            print(
                f"[Rank {rank}] summary events={processed} loop_s={loop_s:.3f} "
                f"rate_evt_s={rate_evt_s:.3f} gpu_device={assigned_gpu_device} "
                f"ccons_upload_s={worker._ccons_upload_s:.6f} "
                f"copy_total_s={total_copy_s:.6f} kernel_total_s={total_kernel_s:.6f} "
                f"copied_mib={mib:.3f} copy_rate_mib_s={copy_rate_mib_s:.3f} "
                f"compare_events={min(processed, args.compare_cpu_events)} "
                f"compare_failures={compare_failures} max_abs_diff={max_abs_diff:.6g}",
                flush=True,
            )

    _print_global_summary(
        rank,
        is_bd,
        processed,
        loop_s,
        cpu_raw_only=args.cpu_raw_only,
        cpu_raw_type=cpu_raw_type,
    )


if __name__ == "__main__":
    main()
