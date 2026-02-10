#!/usr/bin/env python3
"""
Generate an OpenMPI rankfile while skipping specific NUMA nodes.

Default behavior:
  - Detect NUMA layout from /sys/devices/system/node/node*/cpulist
  - Optionally detect NUMA nodes used by wekanode processes and skip them
  - Optionally respect the current cpuset (Cpus_allowed_list)
  - Emit a rankfile mapping rank -> host + core

Example:
  python -m psana.debugtools.make_rankfile --nranks 100 --out rankfile
  mpirun --rankfile rankfile -n 100 python your_script.py
"""

from __future__ import annotations

import argparse
import os
import socket
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def parse_cpulist(s: str) -> List[int]:
    cpus: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def build_cpu_to_numa() -> Dict[int, int]:
    cpu_to_numa: Dict[int, int] = {}
    node_paths = sorted(glob("/sys/devices/system/node/node*/cpulist"))
    for path in node_paths:
        node_str = path.split("node")[-1].split("/")[0]
        try:
            node_id = int(node_str)
        except ValueError:
            continue
        try:
            with open(path, "r") as f:
                cpus = parse_cpulist(f.read().strip())
        except Exception:
            continue
        for cpu in cpus:
            cpu_to_numa[cpu] = node_id
    return cpu_to_numa


def read_cpus_allowed() -> Set[int]:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list:"):
                    _, val = line.split(":", 1)
                    return set(parse_cpulist(val.strip()))
    except Exception:
        pass
    return set()


def get_cpu_from_stat(pid: str) -> int:
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            parts = f.read().split()
        return int(parts[38])
    except Exception:
        return -1


def find_weka_numa(cpu_to_numa: Dict[int, int]) -> Tuple[Set[int], Set[int], List[int]]:
    weka_cpus: Set[int] = set()
    weka_pids: List[int] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/comm", "r") as f:
                comm = f.read().strip()
        except Exception:
            continue
        if comm != "wekanode":
            continue
        cpu = get_cpu_from_stat(pid)
        if cpu >= 0:
            weka_cpus.add(cpu)
        weka_pids.append(int(pid))
    weka_numa = {cpu_to_numa[cpu] for cpu in weka_cpus if cpu in cpu_to_numa}
    return weka_numa, weka_cpus, weka_pids


def build_allowed_cpus(
    cpu_to_numa: Dict[int, int],
    skip_numa: Set[int],
    respect_cpuset: bool,
) -> List[int]:
    cpus = set(cpu_to_numa.keys())
    if respect_cpuset:
        allowed = read_cpus_allowed()
        if allowed:
            cpus &= allowed
    if skip_numa:
        cpus = {cpu for cpu in cpus if cpu_to_numa.get(cpu, -1) not in skip_numa}
    return sorted(cpus)


def build_cpu_sequence(
    cpus: List[int],
    cpu_to_numa: Dict[int, int],
    assign_mode: str,
) -> List[int]:
    if assign_mode == "pack":
        return sorted(cpus)

    # Round-robin across NUMA nodes (default).
    by_numa: Dict[int, List[int]] = {}
    for cpu in sorted(cpus):
        nid = cpu_to_numa.get(cpu, -1)
        by_numa.setdefault(nid, []).append(cpu)

    seq: List[int] = []
    numa_ids = sorted(by_numa)
    idx = {nid: 0 for nid in numa_ids}
    remaining = sum(len(v) for v in by_numa.values())
    while remaining > 0:
        for nid in numa_ids:
            lst = by_numa[nid]
            i = idx[nid]
            if i < len(lst):
                seq.append(lst[i])
                idx[nid] = i + 1
                remaining -= 1
    return seq


def parse_hosts(hosts_arg: str | None) -> List[str]:
    if not hosts_arg:
        return [socket.gethostname()]
    hosts = []
    for part in hosts_arg.split(","):
        part = part.strip()
        if part:
            hosts.append(part)
    return hosts or [socket.gethostname()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate an OpenMPI rankfile skipping NUMA nodes.")
    ap.add_argument("--nranks", type=int, required=True, help="Total ranks to map.")
    ap.add_argument("--out", required=True, help="Rankfile output path.")
    ap.add_argument("--hosts", help="Comma-separated host list (default: local hostname).")
    ap.add_argument("--skip-numa", help="Comma-separated NUMA ids to skip.")
    ap.add_argument(
        "--auto-skip-weka",
        action="store_true",
        default=True,
        help="Detect wekanode CPUs and skip their NUMA nodes (default: on).",
    )
    ap.add_argument(
        "--no-auto-skip-weka",
        action="store_false",
        dest="auto_skip_weka",
        help="Disable auto skip of wekanode NUMA nodes.",
    )
    ap.add_argument(
        "--respect-cpuset",
        action="store_true",
        default=True,
        help="Respect /proc/self/status Cpus_allowed_list (default: on).",
    )
    ap.add_argument(
        "--no-respect-cpuset",
        action="store_false",
        dest="respect_cpuset",
        help="Ignore current cpuset.",
    )
    ap.add_argument(
        "--oversubscribe",
        action="store_true",
        help="Allow more ranks than available CPUs (reuse cores).",
    )
    ap.add_argument(
        "--assign",
        choices=["round-robin", "pack"],
        default="round-robin",
        help="CPU assignment strategy (default: round-robin across NUMA nodes).",
    )
    args = ap.parse_args()

    cpu_to_numa = build_cpu_to_numa()
    if not cpu_to_numa:
        raise SystemExit("No NUMA topology found under /sys/devices/system/node")

    skip_numa: Set[int] = set()
    if args.skip_numa:
        skip_numa |= {int(x) for x in args.skip_numa.split(",") if x.strip()}

    weka_numa: Set[int] = set()
    weka_cpus: Set[int] = set()
    weka_pids: List[int] = []
    if args.auto_skip_weka:
        weka_numa, weka_cpus, weka_pids = find_weka_numa(cpu_to_numa)
        skip_numa |= weka_numa

    hosts = parse_hosts(args.hosts)
    cpus = build_allowed_cpus(cpu_to_numa, skip_numa, args.respect_cpuset)
    cpu_seq = build_cpu_sequence(cpus, cpu_to_numa, args.assign)

    if not cpus:
        raise SystemExit("No CPUs available after filtering; check skip NUMA/cpuset.")

    capacity = len(cpu_seq) * len(hosts)
    if args.nranks > capacity and not args.oversubscribe:
        raise SystemExit(
            f"nranks={args.nranks} exceeds capacity={capacity} "
            f"(cpus={len(cpu_seq)} hosts={len(hosts)}). Use --oversubscribe to allow."
        )

    out_path = Path(args.out)
    lines: List[str] = []
    rank = 0
    for host in hosts:
        for cpu in cpu_seq:
            if rank >= args.nranks:
                break
            lines.append(f"rank {rank}={host} slot={cpu}")
            rank += 1
        if rank >= args.nranks:
            break

    if rank < args.nranks and args.oversubscribe:
        # reuse CPUs in round-robin
        idx = 0
        while rank < args.nranks:
            host = hosts[rank % len(hosts)]
            cpu = cpu_seq[idx % len(cpu_seq)]
            lines.append(f"rank {rank}={host} slot={cpu}")
            rank += 1
            idx += 1

    out_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote rankfile: {out_path} (ranks={args.nranks})")
    print(f"Hosts: {', '.join(hosts)}")
    print(f"CPUs used: {len(cpu_seq)}")
    print(f"Assign mode: {args.assign}")
    print(f"Skipped NUMA: {sorted(skip_numa) if skip_numa else 'none'}")
    if weka_pids:
        print(f"Weka pids: {len(weka_pids)}; weka CPUs: {sorted(weka_cpus)}")
        print(f"Weka NUMA: {sorted(weka_numa)}")


if __name__ == "__main__":
    main()
