#!/usr/bin/env python3
"""Read-only KvikIO compatibility/GDS baseline over Jungfrau streams."""

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--bytes-per-file-gib", type=float, default=4.0)
    parser.add_argument("--task-mib", type=int, default=4)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()

    import cupy as cp
    import kvikio
    import kvikio.defaults

    files = [path.resolve() for path in args.files]
    chunk_bytes = args.chunk_mib * 2**20
    task_bytes = args.task_mib * 2**20
    requested_per_file = int(args.bytes_per_file_gib * 2**30)
    per_file = [min(path.stat().st_size, requested_per_file) for path in files]
    per_file = [size // chunk_bytes * chunk_bytes for size in per_file]
    handles = [kvikio.CuFile(str(path), "r") for path in files]
    buffers = [cp.empty(chunk_bytes, dtype=cp.uint8) for _ in files]

    dp = kvikio.DriverProperties()
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    total = 0
    offsets = [0] * len(files)
    while True:
        futures = []
        for i, (handle, limit) in enumerate(zip(handles, per_file)):
            if offsets[i] >= limit:
                continue
            future = handle.pread(
                buffers[i],
                size=chunk_bytes,
                file_offset=offsets[i],
                task_size=task_bytes,
            )
            futures.append((i, future))
            offsets[i] += chunk_bytes
        if not futures:
            break
        for _, future in futures:
            got = int(future.get())
            if got != chunk_bytes:
                raise RuntimeError(f"short KvikIO read: {got} != {chunk_bytes}")
            total += got
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - start

    for handle in handles:
        handle.close()
    result = {
        "bytes": total,
        "elapsed_s": elapsed,
        "gbps": total / elapsed / 1e9,
        "files": len(files),
        "chunk_mib": args.chunk_mib,
        "task_mib": args.task_mib,
        "nthreads": int(os.environ.get("KVIKIO_NTHREADS", "-1")),
        "gds_available": bool(dp.is_gds_available),
        "compat_mode": bool(kvikio.defaults.compat_mode()),
    }
    print("KVIKIO_READ_BASELINE " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
