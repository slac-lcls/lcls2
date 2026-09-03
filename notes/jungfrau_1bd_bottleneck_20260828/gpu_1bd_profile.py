#!/usr/bin/env python3
"""Profile I/O wait and GPU retirement in the one-BD/one-GPU pipeline."""

import argparse
import json
import threading
import time

from mpi4py import MPI


_lock = threading.Lock()
_stats = {
    "issue_s": 0.0,
    "issue_calls": 0,
    "io_wait_s": 0.0,
    "io_latency_s": 0.0,
    "io_bytes": 0,
    "io_batches": 0,
    "retire_s": 0.0,
    "retire_calls": 0,
    "retire_finish_s": 0.0,
    "retire_finish_calls": 0,
    "submit_s": 0.0,
    "submit_calls": 0,
}
_pending = {}


def add(**values):
    with _lock:
        for key, value in values.items():
            _stats[key] += value


def install_hooks():
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader
    from psana.gpu.gpu_stream import EventPool

    original_issue = KvikioGpuReader.issue_batch
    original_wait = KvikioGpuReader.wait_batch
    original_retire = EventPool.begin_retire_next
    original_retire_finish = EventPool.finish_retire_next
    original_submit = EventPool.submit

    def issue(self, *args, **kwargs):
        start = time.perf_counter()
        pending = original_issue(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        nbytes = sum(read_size for _, read_size, _ in pending.futures)
        with _lock:
            _pending[id(pending)] = (time.perf_counter(), nbytes)
        add(issue_s=elapsed, issue_calls=1)
        return pending

    def wait(self, pending):
        start = time.perf_counter()
        result = original_wait(self, pending)
        elapsed = time.perf_counter() - start
        end = time.perf_counter()
        with _lock:
            issued, nbytes = _pending.pop(id(pending), (start, 0))
        add(
            io_wait_s=elapsed,
            io_latency_s=end - issued,
            io_bytes=nbytes,
            io_batches=1,
        )
        return result

    def retire(self):
        start = time.perf_counter()
        result = original_retire(self)
        add(retire_s=time.perf_counter() - start, retire_calls=1)
        return result

    def retire_finish(self):
        start = time.perf_counter()
        result = original_retire_finish(self)
        add(
            retire_finish_s=time.perf_counter() - start,
            retire_finish_calls=1,
        )
        return result

    def submit(self, *args, **kwargs):
        start = time.perf_counter()
        result = original_submit(self, *args, **kwargs)
        add(submit_s=time.perf_counter() - start, submit_calls=1)
        return result

    KvikioGpuReader.issue_batch = issue
    KvikioGpuReader.wait_batch = wait
    EventPool.begin_retire_next = retire
    EventPool.finish_retire_next = retire_finish
    EventPool.submit = submit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--pool-depth", type=int, default=1)
    args = parser.parse_args()

    install_hooks()
    from psana import DataSource

    comm = MPI.COMM_WORLD
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
    comm.Barrier()
    start = MPI.Wtime()
    local_events = sum(1 for _ in run.events())
    loop_local = MPI.Wtime() - start

    total_events = comm.reduce(local_events, op=MPI.SUM, root=0)
    loop_s = comm.reduce(loop_local, op=MPI.MAX, root=0)
    gathered = comm.gather(dict(_stats), root=0)
    if comm.rank == 0:
        bd_stats = max(gathered, key=lambda item: item["io_bytes"])
        result = {
            "events": int(total_events),
            "loop_s": loop_s,
            "rate_hz": total_events / loop_s,
            "batch_size": args.batch_size,
            "pool_depth": args.pool_depth,
            **bd_stats,
        }
        result["payload_gbps"] = bd_stats["io_bytes"] / loop_s / 1e9
        result["io_wait_gbps"] = (
            bd_stats["io_bytes"] / bd_stats["io_wait_s"] / 1e9
            if bd_stats["io_wait_s"]
            else 0.0
        )
        print("GPU_1BD_PROFILE " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
