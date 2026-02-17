#!/usr/bin/env python3
"""
Benchmark file->GPU transfer throughput for:
  - pattern 4: host-staged read + H2D copy (--mode host)
  - pattern 3: cuFile/GDS direct read to device buffer (--mode cufile)

Example:
  python -m psana.debugtools.gpu_host_stage_bench \
      --path /sdf/data/lcls/ds/mfx/mfx101344525/xtc/mfx101344525-r0125-s007-c000.xtc2 \
      --mode host --chunk-mb 64 --max-gb 8 --backend auto --gpu 0
"""

import argparse
import os
import sys
import time
from typing import Optional, Sequence

import numpy as np

GIB = 1024**3


def _positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return out


def _positive_float(value: str) -> float:
    out = float(value)
    if out <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive float")
    return out


def _bytes_to_gib_per_s(nbytes: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return (nbytes / GIB) / seconds


def _fmt_bytes(nbytes: int) -> str:
    return f"{nbytes / GIB:.2f} GiB"


class GpuBackendBase:
    name = "base"

    def copy_from_host(self, host_np: np.ndarray, host_t=None) -> None:
        raise NotImplementedError

    def synchronize(self) -> None:
        raise NotImplementedError


class CupyBackend(GpuBackendBase):
    name = "cupy"

    def __init__(self, gpu: int, max_chunk_bytes: int) -> None:
        import cupy as cp

        self.cp = cp
        cp.cuda.runtime.setDevice(gpu)
        self._d_buf = cp.empty(max_chunk_bytes, dtype=cp.uint8)
        self._dst_ptr = int(self._d_buf.data.ptr)
        self._h2d_kind = cp.cuda.runtime.memcpyHostToDevice

    def copy_from_host(self, host_np: np.ndarray, host_t=None) -> None:
        nbytes = int(host_np.nbytes)
        self.cp.cuda.runtime.memcpy(
            self._dst_ptr, int(host_np.ctypes.data), nbytes, self._h2d_kind
        )

    def synchronize(self) -> None:
        self.cp.cuda.runtime.deviceSynchronize()


class TorchBackend(GpuBackendBase):
    name = "torch"

    def __init__(self, gpu: int, max_chunk_bytes: int) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        self.torch = torch
        self.device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(self.device)
        self._d_buf = torch.empty(max_chunk_bytes, dtype=torch.uint8, device=self.device)

    def copy_from_host(self, host_np: np.ndarray, host_t=None) -> None:
        if host_t is None:
            host_t = self.torch.from_numpy(host_np)
        self._d_buf[: host_t.numel()].copy_(host_t, non_blocking=False)

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)


def _make_backend(name: str, gpu: int, chunk_bytes: int) -> GpuBackendBase:
    errors = []
    if name in ("auto", "cupy"):
        try:
            return CupyBackend(gpu=gpu, max_chunk_bytes=chunk_bytes)
        except Exception as exc:  # pragma: no cover - env dependent
            errors.append(f"cupy backend unavailable: {exc}")
            if name == "cupy":
                raise

    if name in ("auto", "torch"):
        try:
            return TorchBackend(gpu=gpu, max_chunk_bytes=chunk_bytes)
        except Exception as exc:  # pragma: no cover - env dependent
            errors.append(f"torch backend unavailable: {exc}")
            if name == "torch":
                raise

    msg = "No GPU backend available.\n" + "\n".join(f"- {e}" for e in errors)
    msg += "\nInstall cupy or torch in your environment, or pick --backend explicitly."
    raise RuntimeError(msg)


class Stats:
    def __init__(self) -> None:
        self.measured_chunks = 0
        self.measured_bytes = 0
        self.read_seconds = 0.0
        self.copy_seconds = 0.0
        self.t0 = None  # type: Optional[float]
        self.t1 = None  # type: Optional[float]


class HostBuffer:
    def __init__(self, np_arr: np.ndarray, read_view, torch_tensor=None, keepalive=None) -> None:
        self.np_arr = np_arr
        self.read_view = read_view
        self.torch_tensor = torch_tensor
        self.keepalive = keepalive


def _make_host_buffer(mode: str, chunk_bytes: int, backend: GpuBackendBase) -> HostBuffer:
    if mode == "pageable":
        raw = bytearray(chunk_bytes)
        np_arr = np.frombuffer(raw, dtype=np.uint8, count=chunk_bytes)
        return HostBuffer(np_arr=np_arr, read_view=memoryview(raw), keepalive=raw)

    if mode != "pinned":
        raise ValueError(f"Unsupported host memory mode: {mode}")

    if backend.name == "torch":
        t = backend.torch.empty(chunk_bytes, dtype=backend.torch.uint8, pin_memory=True)
        np_arr = t.numpy()
        return HostBuffer(np_arr=np_arr, read_view=memoryview(np_arr), torch_tensor=t, keepalive=t)

    if backend.name == "cupy":
        pbuf = backend.cp.cuda.alloc_pinned_memory(chunk_bytes)
        np_arr = np.frombuffer(pbuf, dtype=np.uint8, count=chunk_bytes)
        return HostBuffer(np_arr=np_arr, read_view=memoryview(np_arr), keepalive=pbuf)

    raise RuntimeError(
        f"Pinned host memory is not implemented for backend={backend.name}"
    )


def _kvikio_mode_value_to_str(value) -> str:
    sval = str(value).upper()
    if "OFF" in sval:
        return "OFF"
    if "ON" in sval:
        return "ON"
    if "AUTO" in sval:
        return "AUTO"
    return str(value)


def _run_cufile_mode(
    args: argparse.Namespace,
    file_size: int,
    chunk_bytes: int,
    max_bytes_per_pass: int,
    warmup_chunks: int,
) -> int:
    mode = args.cufile_compat_mode.upper()
    if mode in ("AUTO", "ON", "OFF"):
        os.environ["KVIKIO_COMPAT_MODE"] = mode

    try:
        import cupy as cp
    except Exception as exc:
        print(
            "cuFile mode requires cupy. Install cupy first "
            "(e.g. `python -m pip install --user cupy-cuda12x`).",
            file=sys.stderr,
        )
        print(f"cupy import failed: {exc}", file=sys.stderr)
        return 5

    try:
        import kvikio
    except Exception as exc:
        print(
            "cuFile mode requires kvikio. Install kvikio first "
            "(e.g. `python -m pip install --user kvikio-cu12 libkvikio-cu12`).",
            file=sys.stderr,
        )
        print(f"kvikio import failed: {exc}", file=sys.stderr)
        return 6

    cp.cuda.runtime.setDevice(args.gpu)

    compat_state = "unknown"
    try:
        compat_state = _kvikio_mode_value_to_str(kvikio.defaults.get("compat_mode"))
    except Exception:
        try:
            compat_state = _kvikio_mode_value_to_str(kvikio.defaults.compat_mode())
        except Exception:
            pass

    gds_available = "unknown"
    cufile_driver = getattr(kvikio, "cufile_driver", None)
    if cufile_driver is not None:
        try:
            gds_available = str(cufile_driver.get("is_gds_available"))
        except Exception:
            pass

    if args.cufile_require_gds and compat_state == "ON":
        print(
            "cuFile mode requested `--cufile-require-gds` but KvikIO is in compatibility mode (ON).",
            file=sys.stderr,
        )
        return 7

    print("gpu_host_stage_bench")
    print(f"  file            : {args.path}")
    print(f"  file_size       : {_fmt_bytes(file_size)}")
    print("  mode            : cufile")
    print("  backend         : kvikio+cupy")
    print(f"  gpu             : {args.gpu}")
    print(f"  chunk_size      : {args.chunk_mb:.2f} MiB")
    print(f"  bytes_per_pass  : {_fmt_bytes(max_bytes_per_pass)}")
    print(f"  passes          : {args.passes}")
    print(f"  warmup_chunks   : {warmup_chunks}")
    print(f"  cufile_compat   : {compat_state}")
    print(f"  gds_available   : {gds_available}")
    print()

    stats = Stats()
    warmup_left = warmup_chunks
    d_buf = cp.empty(chunk_bytes, dtype=cp.uint8)

    with kvikio.CuFile(args.path, "r") as cuf:
        for pass_idx in range(args.passes):
            remaining = max_bytes_per_pass
            file_offset = 0

            while remaining > 0:
                this_read = min(chunk_bytes, remaining)

                t_io0 = time.perf_counter()
                fut_or_n = cuf.pread(d_buf[:this_read], size=this_read, file_offset=file_offset)
                if hasattr(fut_or_n, "get"):
                    nread = fut_or_n.get()
                else:
                    nread = fut_or_n
                cp.cuda.runtime.deviceSynchronize()
                t_io1 = time.perf_counter()

                if nread is None:
                    nread = this_read
                nread = int(nread)
                if nread <= 0:
                    break

                file_offset += nread
                remaining -= nread

                if warmup_left > 0:
                    warmup_left -= 1
                    continue

                if stats.t0 is None:
                    stats.t0 = t_io0
                stats.t1 = t_io1
                stats.measured_chunks += 1
                stats.measured_bytes += nread
                stats.read_seconds += t_io1 - t_io0

                if (
                    args.report_every > 0
                    and stats.measured_chunks % args.report_every == 0
                    and stats.t0 is not None
                    and stats.t1 is not None
                ):
                    wall = stats.t1 - stats.t0
                    e2e = _bytes_to_gib_per_s(stats.measured_bytes, wall)
                    print(
                        f"[pass {pass_idx + 1}] chunks={stats.measured_chunks} "
                        f"data={_fmt_bytes(stats.measured_bytes)} e2e={e2e:.2f} GiB/s"
                    )

    if stats.measured_bytes == 0 or stats.t0 is None or stats.t1 is None:
        print("No measured data. Increase --passes or reduce --warmup-chunks.", file=sys.stderr)
        return 3

    wall_seconds = stats.t1 - stats.t0
    io_gibs = _bytes_to_gib_per_s(stats.measured_bytes, stats.read_seconds)
    e2e_gibs = _bytes_to_gib_per_s(stats.measured_bytes, wall_seconds)

    print("Results")
    print(f"  measured_data   : {_fmt_bytes(stats.measured_bytes)}")
    print(f"  measured_chunks : {stats.measured_chunks}")
    print(f"  cufile_io_time  : {stats.read_seconds:.3f} s")
    print(f"  wall_time       : {wall_seconds:.3f} s")
    print(f"  cufile_throughput: {io_gibs:.2f} GiB/s")
    print(f"  end_to_end      : {e2e_gibs:.2f} GiB/s")
    return 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "File->GPU benchmark for two paths:\n"
            "  --mode host   : pattern 4 (host read + H2D copy)\n"
            "  --mode cufile : pattern 3 (cuFile/GDS read to device)\n\n"
            "Notes:\n"
            "  - Filesystem cache can affect read throughput.\n"
            "  - Keep chunk/max/passes identical when comparing modes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("host", "cufile"),
        default="host",
        help=(
            "Transfer path: host-staged pattern 4 or cuFile pattern 3 "
            "(default: host)."
        ),
    )
    parser.add_argument(
        "--path",
        default="/sdf/data/lcls/ds/mfx/mfx101344525/xtc/mfx101344525-r0125-s007-c000.xtc2",
        help="Path to the input file.",
    )
    parser.add_argument(
        "--chunk-mb",
        type=_positive_float,
        default=64.0,
        help="Chunk size in MiB for each read/copy (default: 64).",
    )
    parser.add_argument(
        "--max-gb",
        type=_positive_float,
        default=8.0,
        help=(
            "GiB to process per pass (default: 8). "
            "If larger than file size, it is capped at file size."
        ),
    )
    parser.add_argument(
        "--passes",
        type=_positive_int,
        default=1,
        help="Number of full passes over the file slice (default: 1).",
    )
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=2,
        help="Warmup chunks to exclude from throughput stats (default: 2).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cupy", "torch"),
        default="auto",
        help=(
            "GPU backend for --mode host (default: auto, prefers cupy then torch). "
            "Ignored in --mode cufile."
        ),
    )
    parser.add_argument(
        "--host-mem",
        choices=("pageable", "pinned"),
        default="pageable",
        help=(
            "Host staging buffer type for pattern 4 (default: pageable). "
            "Use pinned for best H2D bandwidth."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (default: 0).",
    )
    parser.add_argument(
        "--cufile-compat-mode",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "For --mode cufile, sets KVIKIO_COMPAT_MODE. "
            "'off' asks kvikio to use cuFile/GDS path."
        ),
    )
    parser.add_argument(
        "--cufile-require-gds",
        action="store_true",
        help=(
            "For --mode cufile, fail if kvikio reports compatibility mode ON "
            "(POSIX fallback)."
        ),
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=0,
        help="Print intermediate throughput every N measured chunks (default: 0 = off).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if not os.path.exists(args.path):
        print(f"Input file not found: {args.path}", file=sys.stderr)
        return 1

    file_size = os.path.getsize(args.path)
    if file_size <= 0:
        print(f"Input file is empty: {args.path}", file=sys.stderr)
        return 1

    chunk_bytes = int(args.chunk_mb * 1024 * 1024)
    max_bytes_per_pass = min(int(args.max_gb * GIB), file_size)
    warmup_chunks = max(0, int(args.warmup_chunks))

    if args.mode == "cufile":
        return _run_cufile_mode(
            args=args,
            file_size=file_size,
            chunk_bytes=chunk_bytes,
            max_bytes_per_pass=max_bytes_per_pass,
            warmup_chunks=warmup_chunks,
        )

    try:
        backend = _make_backend(args.backend, args.gpu, chunk_bytes)
    except Exception as exc:
        print(f"Backend initialization failed: {exc}", file=sys.stderr)
        return 2

    print("gpu_host_stage_bench")
    print(f"  file            : {args.path}")
    print(f"  file_size       : {_fmt_bytes(file_size)}")
    print("  mode            : host")
    print(f"  backend         : {backend.name}")
    print(f"  host_mem        : {args.host_mem}")
    print(f"  gpu             : {args.gpu}")
    print(f"  chunk_size      : {args.chunk_mb:.2f} MiB")
    print(f"  bytes_per_pass  : {_fmt_bytes(max_bytes_per_pass)}")
    print(f"  passes          : {args.passes}")
    print(f"  warmup_chunks   : {warmup_chunks}")
    print()

    stats = Stats()
    try:
        host_buf = _make_host_buffer(args.host_mem, chunk_bytes, backend)
    except Exception as exc:
        print(f"Host buffer initialization failed: {exc}", file=sys.stderr)
        return 4
    warmup_left = warmup_chunks

    with open(args.path, "rb", buffering=0) as fobj:
        for pass_idx in range(args.passes):
            remaining = max_bytes_per_pass
            fobj.seek(0)

            while remaining > 0:
                this_read = min(chunk_bytes, remaining)

                t_read0 = time.perf_counter()
                nread = fobj.readinto(host_buf.read_view[:this_read])
                t_read1 = time.perf_counter()

                if nread <= 0:
                    break

                host_np = host_buf.np_arr[:nread]
                host_t = None
                if host_buf.torch_tensor is not None:
                    host_t = host_buf.torch_tensor[:nread]

                t_copy0 = time.perf_counter()
                backend.copy_from_host(host_np, host_t=host_t)
                backend.synchronize()
                t_copy1 = time.perf_counter()

                remaining -= nread

                if warmup_left > 0:
                    warmup_left -= 1
                    continue

                if stats.t0 is None:
                    stats.t0 = t_read0
                stats.t1 = t_copy1
                stats.measured_chunks += 1
                stats.measured_bytes += nread
                stats.read_seconds += t_read1 - t_read0
                stats.copy_seconds += t_copy1 - t_copy0

                if (
                    args.report_every > 0
                    and stats.measured_chunks % args.report_every == 0
                    and stats.t0 is not None
                    and stats.t1 is not None
                ):
                    wall = stats.t1 - stats.t0
                    e2e = _bytes_to_gib_per_s(stats.measured_bytes, wall)
                    print(
                        f"[pass {pass_idx + 1}] chunks={stats.measured_chunks} "
                        f"data={_fmt_bytes(stats.measured_bytes)} e2e={e2e:.2f} GiB/s"
                    )

    if stats.measured_bytes == 0 or stats.t0 is None or stats.t1 is None:
        print("No measured data. Increase --passes or reduce --warmup-chunks.", file=sys.stderr)
        return 3

    wall_seconds = stats.t1 - stats.t0
    read_gibs = _bytes_to_gib_per_s(stats.measured_bytes, stats.read_seconds)
    copy_gibs = _bytes_to_gib_per_s(stats.measured_bytes, stats.copy_seconds)
    e2e_gibs = _bytes_to_gib_per_s(stats.measured_bytes, wall_seconds)

    print("Results")
    print(f"  measured_data   : {_fmt_bytes(stats.measured_bytes)}")
    print(f"  measured_chunks : {stats.measured_chunks}")
    print(f"  read_time       : {stats.read_seconds:.3f} s")
    print(f"  h2d_copy_time   : {stats.copy_seconds:.3f} s")
    print(f"  wall_time       : {wall_seconds:.3f} s")
    print(f"  read_throughput : {read_gibs:.2f} GiB/s")
    print(f"  h2d_throughput  : {copy_gibs:.2f} GiB/s")
    print(f"  end_to_end      : {e2e_gibs:.2f} GiB/s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
