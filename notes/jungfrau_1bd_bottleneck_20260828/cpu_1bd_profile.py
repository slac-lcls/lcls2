#!/usr/bin/env python3
"""Profile experimental read-side variants of the CPU one-BD path."""

import argparse
import concurrent.futures
import json
import os
import threading
import time

from mpi4py import MPI


_lock = threading.Lock()
_stats = {
    "pread_s_sum": 0.0,
    "buffer_copy_s_sum": 0.0,
    "read_s_sum": 0.0,
    "read_bytes": 0,
    "read_calls": 0,
    "smd_parse_s": 0.0,
    "next_dgrams_s": 0.0,
    "parallel_refill_s": 0.0,
    "parallel_refill_groups": 0,
}


def add_stats(**values):
    with _lock:
        for key, value in values.items():
            _stats[key] += value


def install_profile_hooks(read_threads, use_preadv):
    from psana.psexp.event_manager import EventManager, ExitId

    def profiled_read(self, fd, size, offset):
        request_size = int(size)
        read_start = time.perf_counter()
        pread_s = 0.0
        copy_s = 0.0

        if use_preadv:
            copy_start = time.perf_counter()
            chunk = bytearray(request_size)
            copy_s += time.perf_counter() - copy_start
            position = 0
            for i_retry in range(self.max_retries + 1):
                call_start = time.perf_counter()
                got = os.preadv(fd, [memoryview(chunk)[position:]], offset + position)
                pread_s += time.perf_counter() - call_start
                position += got
                if position == request_size:
                    break
                if i_retry == self.max_retries:
                    self.exit_id = ExitId.BdReadFail
                    break
                time.sleep(1)
        else:
            chunk = bytearray()
            remaining = request_size
            position = offset
            for i_retry in range(self.max_retries + 1):
                call_start = time.perf_counter()
                data = os.pread(fd, remaining, position)
                pread_s += time.perf_counter() - call_start
                copy_start = time.perf_counter()
                chunk.extend(data)
                copy_s += time.perf_counter() - copy_start
                got = len(data)
                if len(chunk) == request_size:
                    break
                if i_retry == self.max_retries:
                    self.exit_id = ExitId.BdReadFail
                    break
                position += got
                remaining -= got
                time.sleep(1)

        elapsed = time.perf_counter() - read_start
        nbytes = len(chunk)
        self._bd_read_bytes += nbytes
        self._bd_read_time += elapsed
        add_stats(
            pread_s_sum=pread_s,
            buffer_copy_s_sum=copy_s,
            read_s_sum=elapsed,
            read_bytes=nbytes,
            read_calls=1,
        )
        return chunk

    EventManager._read = profiled_read

    original_parse = EventManager._get_offset_and_size

    def profiled_parse(self):
        start = time.perf_counter()
        try:
            return original_parse(self)
        finally:
            add_stats(smd_parse_s=time.perf_counter() - start)

    EventManager._get_offset_and_size = profiled_parse

    original_next = EventManager._get_next_dgrams
    executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=read_threads)
        if read_threads > 1
        else None
    )

    def maybe_parallel_next(self):
        if executor is not None:
            refill_streams = []
            for i_smd in range(self.n_smd_files):
                needs_bigdata = (
                    self.dm.n_files > 0
                    and self.isEvent(self.service_array[self.i_evt, i_smd])
                    and not self.use_smds[i_smd]
                    and not self.smd_mode
                )
                if not needs_bigdata:
                    continue
                needs_refill = (
                    self.bd_buf_offsets[i_smd]
                    + self.bd_size_array[self.i_evt, i_smd]
                    > memoryview(self.bd_bufs[i_smd]).nbytes
                )
                if needs_refill:
                    if self.chunk_indices[i_smd] >= len(self.cutoff_indices[i_smd]):
                        return None
                    refill_streams.append(i_smd)

            if len(refill_streams) > 1:
                start = time.perf_counter()
                futures = [executor.submit(self._fill_bd_chunk, i) for i in refill_streams]
                for future in futures:
                    future.result()
                for i_smd in refill_streams:
                    self.chunk_indices[i_smd] += 1
                add_stats(
                    parallel_refill_s=time.perf_counter() - start,
                    parallel_refill_groups=1,
                )
        return original_next(self)

    def profiled_next(self):
        start = time.perf_counter()
        try:
            return maybe_parallel_next(self)
        finally:
            add_stats(next_dgrams_s=time.perf_counter() - start)

    EventManager._get_next_dgrams = profiled_next


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--read-threads", type=int, default=1)
    parser.add_argument("--preadv", action="store_true")
    args = parser.parse_args()

    install_profile_hooks(args.read_threads, args.preadv)

    from psana import DataSource

    comm = MPI.COMM_WORLD
    ds = DataSource(
        exp="mfx101210926",
        run=387,
        dir=args.dir,
        max_events=args.max_events,
        batch_size=args.batch_size,
        detectors=[],
        skip_calib_load="all",
    )
    run = next(ds.runs())
    comm.Barrier()
    start = MPI.Wtime()
    local_events = sum(1 for _ in run.events())
    loop_s_local = MPI.Wtime() - start

    total_events = comm.reduce(local_events, op=MPI.SUM, root=0)
    loop_s = comm.reduce(loop_s_local, op=MPI.MAX, root=0)
    gathered = comm.gather(dict(_stats), root=0)
    if comm.rank == 0:
        bd_stats = max(gathered, key=lambda item: item["read_bytes"])
        result = {
            "events": int(total_events),
            "loop_s": loop_s,
            "rate_hz": total_events / loop_s,
            "chunk_bytes": int(os.environ.get("PS_BD_CHUNKSIZE", 0x1000000)),
            "batch_size": args.batch_size,
            "read_threads": args.read_threads,
            "preadv": args.preadv,
            **bd_stats,
        }
        result["payload_gbps"] = bd_stats["read_bytes"] / loop_s / 1e9
        result["pread_sum_gbps"] = (
            bd_stats["read_bytes"] / bd_stats["pread_s_sum"] / 1e9
            if bd_stats["pread_s_sum"]
            else 0.0
        )
        print("CPU_1BD_PROFILE " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
