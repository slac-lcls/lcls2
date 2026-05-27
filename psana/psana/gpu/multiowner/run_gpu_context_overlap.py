#!/usr/bin/env python3

import argparse
import json
import os
import socket
import time
from contextlib import contextmanager

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - diagnostic path
    cp = None


SPIN_KERNEL = r"""
extern "C" __global__
void spin_kernel(const unsigned char* src,
                 float* dst,
                 long long n,
                 long long src_n,
                 int spin_iters,
                 float seed)
{
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float x = ((float)src[i % src_n]) * 0.001f + seed + ((float)(i & 1023)) * 0.000001f;
    for (int k = 0; k < spin_iters; ++k) {
        x = fmaf(x, 1.000001f, 0.000001f);
        if (x > 4096.0f) {
            x -= 4096.0f;
        }
    }
    dst[i] = x;
}
"""


def _mpi():
    try:
        from mpi4py import MPI
    except Exception:
        return None
    return MPI


def _rank_size():
    mpi = _mpi()
    if mpi is not None:
        comm = mpi.COMM_WORLD
        return comm.Get_rank(), comm.Get_size(), comm

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    size = int(os.environ.get("SLURM_NTASKS", "1"))
    return rank, size, None


def _local_rank():
    for name in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            pass
    return 0


@contextmanager
def _nvtx(name):
    if cp is None or not hasattr(cp.cuda, "nvtx"):
        yield
        return
    cp.cuda.nvtx.RangePush(name)
    try:
        yield
    finally:
        cp.cuda.nvtx.RangePop()


def _parse_size(text):
    suffixes = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
    }
    lowered = text.strip().lower()
    for suffix, scale in sorted(suffixes.items(), key=lambda item: len(item[0]), reverse=True):
        if lowered.endswith(suffix):
            return int(float(lowered[: -len(suffix)]) * scale)
    return int(float(lowered) * 1024**2)


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "MPI/CuPy microbenchmark for inspecting whether multiple MPI ranks "
            "sharing one GPU can overlap CPU-side I/O with CUDA kernel work."
        )
    )
    parser.add_argument("--iterations", type=int, default=50, help="Loop iterations per rank.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations before timing; still visible in Nsight.",
    )
    parser.add_argument(
        "--data-size",
        default="64M",
        help="Host bytes read/touched and copied H2D each iteration. Bare numbers mean MiB.",
    )
    parser.add_argument(
        "--compute-elements",
        type=int,
        default=0,
        help="Float32 output elements touched by the kernel. 0 means data_size / 4.",
    )
    parser.add_argument(
        "--compute-iters",
        type=int,
        default=2048,
        help="Arithmetic loop iterations inside the CUDA kernel.",
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=2,
        help="Number of per-rank host/device slots. 1 is sequential; 2+ allows CPU I/O while prior GPU work runs.",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=("slot", "staged"),
        default="slot",
        help=(
            "slot keeps H2D and kernel work on each slot's stream. "
            "staged uses one H2D stream and one kernel stream with CUDA event dependencies."
        ),
    )
    parser.add_argument(
        "--streams-per-rank",
        "--stream-per-rank",
        type=int,
        default=1,
        help=(
            "Independent CUDA streams launched by each rank per outer iteration. "
            "Total timed GPU workloads per rank is iterations * streams_per_rank."
        ),
    )
    parser.add_argument(
        "--io-mode",
        choices=("fill", "sleep", "pread"),
        default="fill",
        help="CPU-side I/O surrogate: fill host memory, sleep, or os.preadv from a file.",
    )
    parser.add_argument(
        "--io-file",
        default=None,
        help="File to read when --io-mode=pread. Reads wrap around the file size.",
    )
    parser.add_argument(
        "--io-sleep-us",
        type=float,
        default=0.0,
        help="Additional sleep inside the I/O NVTX range.",
    )
    parser.add_argument(
        "--host-memory",
        choices=("pinned", "pageable"),
        default="pinned",
        help="Host staging memory used for H2D copies.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA device id. Default intentionally maps every rank to GPU 0.",
    )
    parser.add_argument(
        "--gpu-map",
        choices=("fixed", "local-rank"),
        default="fixed",
        help="fixed makes all ranks use --gpu-id; local-rank spreads ranks across visible GPUs.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="CUDA threads per block for the spin kernel.",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=10,
        help="Rank 0 progress interval in timed iterations. 0 disables progress.",
    )
    parser.add_argument("--json-out", default=None, help="Write gathered summary JSON on rank 0.")
    return parser


class Slot:
    def __init__(self, cp_module, data_nbytes, compute_elements, host_memory):
        self.cp = cp_module
        self.host_memory = host_memory
        if host_memory == "pinned":
            self.pinned = cp_module.cuda.alloc_pinned_memory(data_nbytes)
            self.host = np.frombuffer(self.pinned, dtype=np.uint8, count=data_nbytes)
        else:
            self.pinned = None
            self.host = np.empty((data_nbytes,), dtype=np.uint8)
        self.dev_in = cp_module.empty((data_nbytes,), dtype=cp_module.uint8)
        self.dev_out = cp_module.empty((compute_elements,), dtype=cp_module.float32)
        self.stream = cp_module.cuda.Stream(non_blocking=True)

    @property
    def host_ptr(self):
        return int(self.host.ctypes.data)


def _read_preadv(fd, file_size, host, nbytes, rank, size, workload_iteration):
    if file_size <= 0:
        raise RuntimeError("Cannot pread from an empty file.")

    offset = ((workload_iteration * size + rank) * nbytes) % file_size
    view = memoryview(host)
    done = 0
    while done < nbytes:
        chunk = min(nbytes - done, file_size - offset)
        got = os.preadv(fd, [view[done : done + chunk]], offset)
        if got == 0:
            offset = 0
            continue
        done += got
        offset = (offset + got) % file_size


def _do_io(args, slot, fd, file_size, rank, size, workload_iteration, timed_iteration, stream_index):
    with _nvtx(f"rank{rank}/iter{timed_iteration}/stream{stream_index}/io_{args.io_mode}"):
        t0 = time.perf_counter()
        if args.io_mode == "pread":
            _read_preadv(fd, file_size, slot.host, slot.host.nbytes, rank, size, workload_iteration)
        elif args.io_mode == "fill":
            slot.host.fill((rank * 17 + workload_iteration) & 0xFF)
        if args.io_sleep_us > 0:
            time.sleep(args.io_sleep_us / 1e6)
        return time.perf_counter() - t0


def _select_gpu(args):
    if args.gpu_map == "fixed":
        return args.gpu_id
    visible = cp.cuda.runtime.getDeviceCount()
    if visible <= 0:
        raise SystemExit("No visible CUDA devices.")
    return _local_rank() % visible


def _event_elapsed_s(start, stop):
    return cp.cuda.get_elapsed_time(start, stop) / 1e3


def main():
    args = _build_parser().parse_args()
    if cp is None:
        raise SystemExit("CuPy is required. Activate the psana GPU environment first.")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be > 0.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0.")
    if args.compute_iters < 0:
        raise SystemExit("--compute-iters must be >= 0.")
    if args.pipeline_depth <= 0:
        raise SystemExit("--pipeline-depth must be > 0.")
    if args.streams_per_rank <= 0:
        raise SystemExit("--streams-per-rank must be > 0.")
    if args.pipeline_mode == "staged" and args.streams_per_rank != 1:
        raise SystemExit("--pipeline-mode=staged currently requires --streams-per-rank=1.")
    if args.io_mode == "pread" and not args.io_file:
        raise SystemExit("--io-file is required when --io-mode=pread.")

    rank, size, comm = _rank_size()
    local_rank = _local_rank()
    data_nbytes = _parse_size(args.data_size)
    compute_elements = args.compute_elements or max(1, data_nbytes // np.dtype(np.float32).itemsize)
    gpu_id = _select_gpu(args)
    cp.cuda.Device(gpu_id).use()
    kernel = cp.RawKernel(SPIN_KERNEL, "spin_kernel")

    fd = None
    file_size = 0
    if args.io_mode == "pread":
        fd = os.open(args.io_file, os.O_RDONLY)
        file_size = os.fstat(fd).st_size

    slot_sets = [
        [
            Slot(cp, data_nbytes, compute_elements, args.host_memory)
            for _ in range(args.streams_per_rank)
        ]
        for _ in range(args.pipeline_depth)
    ]
    for slot_set in slot_sets:
        for slot in slot_set:
            slot.host.fill(rank & 0xFF)

    total_iterations = args.warmup + args.iterations
    records = []
    io_s = 0.0
    slot_wait_s = 0.0
    start_wall = None
    block_size = args.block_size
    grid_size = (compute_elements + block_size - 1) // block_size
    host = socket.gethostname()

    print(
        f"[Rank {rank}] pid={os.getpid()} host={host} local_rank={local_rank} "
        f"world_size={size} gpu_id={gpu_id} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"data_nbytes={data_nbytes} compute_elements={compute_elements} "
        f"compute_iters={args.compute_iters} pipeline_depth={args.pipeline_depth} "
        f"pipeline_mode={args.pipeline_mode} streams_per_rank={args.streams_per_rank} "
        f"io_mode={args.io_mode} host_memory={args.host_memory}",
        flush=True,
    )

    if comm is not None:
        comm.Barrier()
    wall0 = time.perf_counter()

    try:
        if args.pipeline_mode == "slot":
            for iteration in range(total_iterations):
                timed_iteration = iteration - args.warmup
                slot_set = slot_sets[iteration % args.pipeline_depth]

                if iteration >= args.pipeline_depth:
                    with _nvtx(f"rank{rank}/iter{timed_iteration}/wait_for_slot"):
                        t0 = time.perf_counter()
                        for slot in slot_set:
                            slot.stream.synchronize()
                        if timed_iteration >= 0:
                            slot_wait_s += time.perf_counter() - t0

                if timed_iteration == 0:
                    start_wall = time.perf_counter()

                for stream_index, slot in enumerate(slot_set):
                    # Simulate CPU-side work
                    workload_iteration = iteration * args.streams_per_rank + stream_index
                    io_time = _do_io(
                        args,
                        slot,
                        fd,
                        file_size,
                        rank,
                        size,
                        workload_iteration,
                        timed_iteration,
                        stream_index,
                    )
                    if timed_iteration >= 0:
                        io_s += io_time

                    h2d_start = cp.cuda.Event()
                    h2d_stop = cp.cuda.Event()
                    kernel_start = cp.cuda.Event()
                    kernel_stop = cp.cuda.Event()
                    with slot.stream:
                        with _nvtx(f"rank{rank}/iter{timed_iteration}/stream{stream_index}/h2d_enqueue"):
                            h2d_start.record()
                            cp.cuda.runtime.memcpyAsync(
                                slot.dev_in.data.ptr,
                                slot.host_ptr,
                                data_nbytes,
                                cp.cuda.runtime.memcpyHostToDevice,
                                slot.stream.ptr,
                            )
                            h2d_stop.record()
                        with _nvtx(f"rank{rank}/iter{timed_iteration}/stream{stream_index}/kernel_enqueue"):
                            kernel_start.record()
                            kernel(
                                (grid_size,),
                                (block_size,),
                                (
                                    slot.dev_in,
                                    slot.dev_out,
                                    np.int64(compute_elements),
                                    np.int64(data_nbytes),
                                    np.int32(args.compute_iters),
                                    np.float32(rank + workload_iteration * 0.001),
                                ),
                            )
                            kernel_stop.record()

                    if timed_iteration >= 0:
                        records.append((h2d_start, h2d_stop, kernel_start, kernel_stop))

                if timed_iteration >= 0:
                    if args.print_interval > 0 and rank == 0 and (timed_iteration + 1) % args.print_interval == 0:
                        launched = (timed_iteration + 1) * args.streams_per_rank
                        total = args.iterations * args.streams_per_rank
                        print(
                            f"[Rank 0] launched {timed_iteration + 1}/{args.iterations} "
                            f"outer_iterations ({launched}/{total} stream_workloads)",
                            flush=True,
                        )

            with _nvtx(f"rank{rank}/drain"):
                t0 = time.perf_counter()
                for slot_set in slot_sets:
                    for slot in slot_set:
                        slot.stream.synchronize()
                slot_wait_s += time.perf_counter() - t0
        else:
            h2d_stream = cp.cuda.Stream(non_blocking=True)
            kernel_stream = cp.cuda.Stream(non_blocking=True)
            slots = [slot_set[0] for slot_set in slot_sets]
            # Host memory is reusable after H2D; device input is reusable after
            # the kernel. Keep the compute dependency on the GPU timeline so the
            # host can enqueue future H2Ds instead of blocking on kernel_done.
            host_reuse_events = [None for _ in slots]
            device_reuse_events = [None for _ in slots]

            for iteration in range(total_iterations):
                timed_iteration = iteration - args.warmup
                slot_index = iteration % args.pipeline_depth
                slot = slots[slot_index]

                if host_reuse_events[slot_index] is not None:
                    with _nvtx(f"rank{rank}/iter{timed_iteration}/wait_for_host_slot"):
                        t0 = time.perf_counter()
                        host_reuse_events[slot_index].synchronize()
                        if timed_iteration >= 0:
                            slot_wait_s += time.perf_counter() - t0

                if timed_iteration == 0:
                    start_wall = time.perf_counter()

                workload_iteration = iteration
                io_time = _do_io(
                    args,
                    slot,
                    fd,
                    file_size,
                    rank,
                    size,
                    workload_iteration,
                    timed_iteration,
                    0,
                )
                if timed_iteration >= 0:
                    io_s += io_time

                h2d_start = cp.cuda.Event()
                h2d_stop = cp.cuda.Event()
                kernel_start = cp.cuda.Event()
                kernel_stop = cp.cuda.Event()
                with h2d_stream:
                    with _nvtx(f"rank{rank}/iter{timed_iteration}/stage/h2d_enqueue"):
                        if device_reuse_events[slot_index] is not None:
                            cp.cuda.runtime.streamWaitEvent(
                                h2d_stream.ptr,
                                device_reuse_events[slot_index].ptr,
                                0,
                            )
                        h2d_start.record()
                        cp.cuda.runtime.memcpyAsync(
                            slot.dev_in.data.ptr,
                            slot.host_ptr,
                            data_nbytes,
                            cp.cuda.runtime.memcpyHostToDevice,
                            h2d_stream.ptr,
                        )
                        h2d_stop.record()
                with kernel_stream:
                    with _nvtx(f"rank{rank}/iter{timed_iteration}/stage/kernel_enqueue"):
                        cp.cuda.runtime.streamWaitEvent(kernel_stream.ptr, h2d_stop.ptr, 0)
                        kernel_start.record()
                        kernel(
                            (grid_size,),
                            (block_size,),
                            (
                                slot.dev_in,
                                slot.dev_out,
                                np.int64(compute_elements),
                                np.int64(data_nbytes),
                                np.int32(args.compute_iters),
                                np.float32(rank + workload_iteration * 0.001),
                            ),
                        )
                        kernel_stop.record()
                host_reuse_events[slot_index] = h2d_stop
                device_reuse_events[slot_index] = kernel_stop

                if timed_iteration >= 0:
                    records.append((h2d_start, h2d_stop, kernel_start, kernel_stop))
                    if args.print_interval > 0 and rank == 0 and (timed_iteration + 1) % args.print_interval == 0:
                        print(
                            f"[Rank 0] launched {timed_iteration + 1}/{args.iterations} "
                            f"outer_iterations ({timed_iteration + 1}/{args.iterations} stream_workloads)",
                            flush=True,
                        )

            with _nvtx(f"rank{rank}/drain"):
                t0 = time.perf_counter()
                for event in device_reuse_events:
                    if event is not None:
                        event.synchronize()
                h2d_stream.synchronize()
                kernel_stream.synchronize()
                slot_wait_s += time.perf_counter() - t0

        elapsed_s = time.perf_counter() - (start_wall or wall0)
        h2d_s = 0.0
        kernel_s = 0.0
        for h2d_start, h2d_stop, kernel_start, kernel_stop in records:
            h2d_s += _event_elapsed_s(h2d_start, h2d_stop)
            kernel_s += _event_elapsed_s(kernel_start, kernel_stop)

        total_workloads = args.iterations * args.streams_per_rank
        result = {
            "rank": rank,
            "host": host,
            "pid": os.getpid(),
            "gpu_id": gpu_id,
            "iterations": args.iterations,
            "streams_per_rank": args.streams_per_rank,
            "total_workloads": total_workloads,
            "data_nbytes": data_nbytes,
            "compute_elements": compute_elements,
            "compute_iters": args.compute_iters,
            "pipeline_depth": args.pipeline_depth,
            "pipeline_mode": args.pipeline_mode,
            "io_mode": args.io_mode,
            "host_memory": args.host_memory,
            "wall_s": elapsed_s,
            "io_s": io_s,
            "slot_wait_s": slot_wait_s,
            "h2d_cuda_s": h2d_s,
            "kernel_cuda_s": kernel_s,
            "h2d_gib_s": (data_nbytes * total_workloads / (1024**3) / h2d_s) if h2d_s > 0 else 0.0,
            "iter_s": elapsed_s / args.iterations,
            "workload_s": elapsed_s / total_workloads,
        }
        print(
            f"[Rank {rank}] summary wall_s={elapsed_s:.6f} iter_s={result['iter_s']:.6f} "
            f"workload_s={result['workload_s']:.6f} pipeline_mode={args.pipeline_mode} "
            f"streams_per_rank={args.streams_per_rank} "
            f"total_workloads={total_workloads} "
            f"io_s={io_s:.6f} slot_wait_s={slot_wait_s:.6f} "
            f"h2d_cuda_s={h2d_s:.6f} kernel_cuda_s={kernel_s:.6f} "
            f"h2d_gib_s={result['h2d_gib_s']:.3f}",
            flush=True,
        )

        gathered = comm.gather(result, root=0) if comm is not None else [result]
        if rank == 0:
            total_wall = max(item["wall_s"] for item in gathered)
            total_kernel = sum(item["kernel_cuda_s"] for item in gathered)
            total_h2d = sum(item["h2d_cuda_s"] for item in gathered)
            total_io = sum(item["io_s"] for item in gathered)
            total_workloads_all_ranks = sum(item["total_workloads"] for item in gathered)
            print(
                "[Rank 0] aggregate "
                f"ranks={len(gathered)} max_wall_s={total_wall:.6f} "
                f"total_workloads={total_workloads_all_ranks} "
                f"sum_io_s={total_io:.6f} sum_h2d_cuda_s={total_h2d:.6f} "
                f"sum_kernel_cuda_s={total_kernel:.6f} "
                f"cuda_work_over_wall={(total_h2d + total_kernel) / total_wall if total_wall > 0 else 0.0:.3f}",
                flush=True,
            )
            if args.json_out:
                with open(args.json_out, "w", encoding="utf-8") as stream:
                    json.dump({"ranks": gathered}, stream, indent=2, sort_keys=True)
                    stream.write("\n")
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == "__main__":
    main()
