#!/usr/bin/env python3
"""Measure pageable and pinned host-to-device copy ceilings."""

import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pageable", "pinned"), required=True)
    parser.add_argument("--buffer-mib", type=int, default=256)
    parser.add_argument("--total-gib", type=float, default=20.0)
    args = parser.parse_args()

    import cupy as cp
    import cupyx
    import numpy as np

    nbytes = args.buffer_mib * 2**20
    count = max(1, int(args.total_gib * 2**30) // nbytes)
    if args.mode == "pinned":
        host = cupyx.empty_pinned(nbytes, dtype=np.uint8)
    else:
        host = np.empty(nbytes, dtype=np.uint8)
    host.fill(17)
    device = cp.empty(nbytes, dtype=cp.uint8)
    stream = cp.cuda.Stream(non_blocking=True)

    device.set(host, stream=stream)
    stream.synchronize()
    start = time.perf_counter()
    for _ in range(count):
        device.set(host, stream=stream)
    stream.synchronize()
    elapsed = time.perf_counter() - start
    total = count * nbytes
    print(
        "H2D_BASELINE "
        + json.dumps(
            {
                "mode": args.mode,
                "bytes": total,
                "copies": count,
                "buffer_mib": args.buffer_mib,
                "elapsed_s": elapsed,
                "gbps": total / elapsed / 1e9,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
