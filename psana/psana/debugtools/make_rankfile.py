#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import List

def parse_nodelist(nodelist: str) -> List[str]:
    """
    Supports formats like:
      sdfmilan[122,137,140,161]
      sdfmilan[122-124,130]
      sdfmilan122
    """
    if "[" not in nodelist:
        return [nodelist]

    m = re.match(r"^([A-Za-z0-9._-]+)\[(.+)\]$", nodelist)
    if not m:
        raise ValueError(f"Invalid nodelist: {nodelist}")

    prefix, inside = m.group(1), m.group(2)
    nodes = []
    for part in inside.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            width = max(len(start_s), len(end_s))
            for i in range(int(start_s), int(end_s) + 1):
                nodes.append(f"{prefix}{str(i).zfill(width)}")
        else:
            if part.isdigit():
                nodes.append(f"{prefix}{part.zfill(len(part))}")
            else:
                nodes.append(f"{prefix}{part}")
    return nodes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srv_cores", type=int, required=True)
    ap.add_argument("--cores", type=int, required=True)
    ap.add_argument("--nodelist", type=str, required=True)
    ap.add_argument("--out", type=str, default="rankfile.txt")
    args = ap.parse_args()

    nodes = parse_nodelist(args.nodelist)
    n_nodes = len(nodes)

    if args.srv_cores != n_nodes:
        print(f"WARNING: srv_cores ({args.srv_cores}) != number of nodes ({n_nodes})")

    if args.cores < args.srv_cores:
        raise ValueError("cores must be >= srv_cores")

    bd_cores = args.cores - args.srv_cores

    # Spread BD ranks across nodes (round-robin with remainder)
    base = bd_cores // n_nodes
    extra = bd_cores % n_nodes
    bd_counts = [base + (1 if i < extra else 0) for i in range(n_nodes)]

    lines = []
    rank = 0

    # BD ranks first
    for node_idx, node in enumerate(nodes):
        for slot in range(bd_counts[node_idx]):
            lines.append(f"rank {rank}={node} slot={slot}")
            rank += 1

    # SRV ranks last, round-robin across nodes (next slots after BD ranks)
    srv_counts = [0] * n_nodes
    for i in range(args.srv_cores):
        node_idx = i % n_nodes
        node = nodes[node_idx]
        slot = bd_counts[node_idx] + srv_counts[node_idx]
        lines.append(f"rank {rank}={node} slot={slot}")
        srv_counts[node_idx] += 1
        rank += 1

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out} with {rank} ranks")

if __name__ == "__main__":
    main()
