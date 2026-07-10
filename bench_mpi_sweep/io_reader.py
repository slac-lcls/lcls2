#!/usr/bin/env python3
"""Run one node's share of an io_matrix config: N reader threads, each doing
os.pread over its own pre-assigned disjoint window (from plan.tsv).

seq  pattern: 8 MiB preads, ascending offsets.
rand pattern: 6 MiB blocks of the window read exactly once in shuffled order
              (random access without ever letting page cache serve a re-read).

Prints:  READER <cfg> <id> <bytes> <wall_s> <MBps>
         NODE_TOTAL <cfg> node=<n> host=<h> readers=<k> bytes=<b> wall_s=<w> GBps=<g>
"""
import argparse, os, random, socket, sys, time
from concurrent.futures import ThreadPoolExecutor

MiB = 1024 * 1024

def read_window(path, offset, length, pattern):
    if pattern == "seq":
        bs = 8 * MiB
        offs = list(range(offset, offset + length, bs))
    else:
        bs = 6 * MiB
        offs = list(range(offset, offset + length - bs + 1, bs))
        random.Random(12345 + offset).shuffle(offs)
    fd = os.open(path, os.O_RDONLY)
    total = 0
    t0 = time.perf_counter()
    try:
        for o in offs:
            n = min(bs, offset + length - o)
            got = os.pread(fd, n, o)
            total += len(got)
            if len(got) < n:
                break
    finally:
        os.close(fd)
    return total, time.perf_counter() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--node-index", type=int, default=None,
                    help="default: SLURM_PROCID")
    args = ap.parse_args()

    node = args.node_index
    if node is None:
        node = int(os.environ.get("SLURM_PROCID", "0"))

    mine = []
    with open(args.plan) as f:
        next(f)
        for line in f:
            cfg, rid, nidx, pat, path, off, length = line.rstrip("\n").split("\t")
            if cfg == args.config and int(nidx) == node:
                mine.append((int(rid), pat, path, int(off), int(length)))

    if not mine:
        print(f"NODE_TOTAL {args.config} node={node} host={socket.gethostname()} "
              f"readers=0 bytes=0 wall_s=0 GBps=0", flush=True)
        return

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(mine)) as ex:
        futs = {ex.submit(read_window, p, o, l, pat): rid
                for rid, pat, p, o, l in mine}
        total = 0
        for fut, rid in futs.items():
            b, w = fut.result()
            total += b
            print(f"READER {args.config} {rid} {b} {w:.3f} {b/w/1e6:.1f}", flush=True)
    wall = time.perf_counter() - t0
    print(f"NODE_TOTAL {args.config} node={node} host={socket.gethostname()} "
          f"readers={len(mine)} bytes={total} wall_s={wall:.3f} "
          f"GBps={total/wall/1e9:.3f}", flush=True)

if __name__ == "__main__":
    main()
