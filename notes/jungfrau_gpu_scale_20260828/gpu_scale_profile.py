#!/usr/bin/env python3
"""Profile multi-BD/multi-GPU psana throughput without detector access or D2H."""

import argparse
import json
import os
import threading
import time

# CUDA-aware MPI may initialise CUDA as soon as mpi4py imports MPI.  Derive the
# Open MPI rank from its launcher environment and restrict device visibility
# first, so a multi-GPU run cannot silently pin every worker to physical GPU 0.
RANK_HINT = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
SIZE_HINT = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
N_EB = int(os.environ.get("PS_EB_NODES", "1"))
N_SRV = int(os.environ.get("PS_SRV_NODES", "0"))
FIRST_BD = N_EB + 1
LAST_BD_HINT = SIZE_HINT - N_SRV
IS_BD_HINT = FIRST_BD <= RANK_HINT < LAST_BD_HINT
N_GPUS = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))

# Pin before importing MPI, psana.gpu, or CuPy. MPIDataSource repeats this
# mapping after MPI initialisation.
PHYS_GPU = None
if IS_BD_HINT:
    PHYS_GPU = (RANK_HINT - FIRST_BD) % N_GPUS
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYS_GPU)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mpi4py import MPI


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
LAST_BD = SIZE - N_SRV
IS_BD = FIRST_BD <= RANK < LAST_BD
if RANK != RANK_HINT or SIZE != SIZE_HINT or IS_BD != IS_BD_HINT:
    raise RuntimeError(
        "Open MPI launcher rank hints disagree with MPI.COMM_WORLD: "
        f"hint=({RANK_HINT},{SIZE_HINT},{IS_BD_HINT}) "
        f"actual=({RANK},{SIZE},{IS_BD})"
    )


LOCK = threading.Lock()
STATS = {
    "issue_s": 0.0,
    "issue_calls": 0,
    "io_wait_s": 0.0,
    "io_latency_s": 0.0,
    "io_bytes": 0,
    "io_batches": 0,
    "submit_s": 0.0,
    "submit_calls": 0,
    "retire_finish_s": 0.0,
    "retire_finish_calls": 0,
    "gpu_used_peak_bytes": 0,
    "cupy_pool_peak_bytes": 0,
}
PENDING = {}


def add(**values):
    with LOCK:
        for key, value in values.items():
            STATS[key] += value


def sample_gpu_memory():
    if not IS_BD:
        return
    import cupy as cp

    free_bytes, total_bytes = cp.cuda.Device().mem_info
    used_bytes = total_bytes - free_bytes
    pool_bytes = cp.get_default_memory_pool().used_bytes()
    with LOCK:
        STATS["gpu_used_peak_bytes"] = max(
            STATS["gpu_used_peak_bytes"], int(used_bytes)
        )
        STATS["cupy_pool_peak_bytes"] = max(
            STATS["cupy_pool_peak_bytes"], int(pool_bytes)
        )


def install_hooks():
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader
    from psana.gpu.gpu_stream import EventPool

    original_issue = KvikioGpuReader.issue_batch
    original_wait = KvikioGpuReader.wait_batch
    original_submit = EventPool.submit
    original_retire_finish = EventPool.finish_retire_next

    def issue(self, *args, **kwargs):
        start = time.perf_counter()
        pending = original_issue(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        nbytes = sum(read_size for _, read_size, _ in pending.futures)
        with LOCK:
            PENDING[id(pending)] = (time.perf_counter(), nbytes)
        add(issue_s=elapsed, issue_calls=1)
        sample_gpu_memory()
        return pending

    def wait(self, pending):
        start = time.perf_counter()
        result = original_wait(self, pending)
        elapsed = time.perf_counter() - start
        end = time.perf_counter()
        with LOCK:
            issued, nbytes = PENDING.pop(id(pending), (start, 0))
        add(
            io_wait_s=elapsed,
            io_latency_s=end - issued,
            io_bytes=nbytes,
            io_batches=1,
        )
        sample_gpu_memory()
        return result

    def submit(self, *args, **kwargs):
        start = time.perf_counter()
        result = original_submit(self, *args, **kwargs)
        add(submit_s=time.perf_counter() - start, submit_calls=1)
        sample_gpu_memory()
        return result

    def retire_finish(self):
        start = time.perf_counter()
        result = original_retire_finish(self)
        add(
            retire_finish_s=time.perf_counter() - start,
            retire_finish_calls=1,
        )
        sample_gpu_memory()
        return result

    KvikioGpuReader.issue_batch = issue
    KvikioGpuReader.wait_batch = wait
    EventPool.submit = submit
    EventPool.finish_retire_next = retire_finish


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--pool-depth", type=int, default=1)
    parser.add_argument("--case", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if IS_BD:
        install_hooks()

    import psutil
    from psana import DataSource

    ds = DataSource(
        exp="mfx101210926",
        run=387,
        dir=args.dir,
        max_events=args.max_events,
        batch_size=args.batch_size,
        n_gpu_streams=args.pool_depth,
        gpu_det="jungfrau",
        gpu_d2h_chunk_size=0,
    )
    run = next(ds.runs())

    COMM.Barrier()
    start = MPI.Wtime()
    timestamps = []
    for ctx in run.events():
        # Timestamp metadata is safe; deliberately do not call ctx.get(),
        # ctx.raw(), or any detector method in this throughput benchmark.
        timestamps.append(int(ctx.timestamp))
    loop_s = MPI.Wtime() - start

    record = {
        "rank": RANK,
        "is_bd": IS_BD,
        "events": len(timestamps),
        "timestamps": timestamps,
        "loop_s": loop_s,
        "phys_gpu": PHYS_GPU,
        "rss_bytes": psutil.Process(os.getpid()).memory_info().rss,
    }
    if IS_BD:
        sample_gpu_memory()
        import kvikio
        import kvikio.defaults

        record.update(STATS)
        record.update(
            gds_available=bool(kvikio.DriverProperties().is_gds_available),
            compat_mode=bool(kvikio.defaults.compat_mode()),
            kvikio_nthreads=int(kvikio.defaults.get_num_threads()),
            kvikio_task_size=int(kvikio.defaults.task_size()),
        )

    gathered = COMM.gather(record, root=0)
    if RANK != 0:
        return

    bd_records = [item for item in gathered if item["is_bd"]]
    all_timestamps = [
        timestamp
        for item in bd_records
        for timestamp in item.pop("timestamps")
    ]
    for item in gathered:
        item.pop("timestamps", None)

    total_events = len(all_timestamps)
    loop_max = max(item["loop_s"] for item in gathered)
    io_bytes = sum(item.get("io_bytes", 0) for item in bd_records)
    valid = total_events == args.max_events and len(set(all_timestamps)) == total_events
    result = {
        "case": args.case,
        "valid": valid,
        "events": total_events,
        "unique_timestamps": len(set(all_timestamps)),
        "loop_s": loop_max,
        "rate_hz": total_events / loop_max if loop_max else 0.0,
        "payload_gbps": io_bytes / loop_max / 1e9 if loop_max else 0.0,
        "io_bytes": io_bytes,
        "n_bds": len(bd_records),
        "n_gpus": N_GPUS,
        "batch_size": args.batch_size,
        "pool_depth": args.pool_depth,
        "ranks": bd_records,
    }
    print("GPU_SCALE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
