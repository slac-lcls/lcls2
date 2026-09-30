#!/usr/bin/env python3
"""Summarize cold-cache Jungfrau GPU sweep logs as CSV and Markdown."""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Optional


TOTAL_BIGDATA_BYTES = 33_557_176_000
TOTAL_EVENTS = 1_000
CPU_ONE_BD_HZ = TOTAL_EVENTS / 21.47
NVME_RAID_RATED_GBPS = 2 * 6.2
NVME_RAID_PCIE_GBPS = 2 * (16e9 * (128 / 130) * 4 / 8) / 1e9
GPU_PCIE_GBPS = (16e9 * (128 / 130) * 16 / 8) / 1e9

NAME_RE = re.compile(
    r"gpu_bs(?P<batch>\d+)_pd(?P<pool>\d+)_nt(?P<threads>\d+)_"
    r"evt(?P<events>\d+)_cold_(?P<job>\d+)\.log$"
)
SUMMARY_RE = re.compile(
    r"n_ebnodes=\d+ n_bdnodes=\d+ Load time=(?P<load>[0-9.]+)s "
    r"Loop time=(?P<loop>[0-9.]+)s Total events: (?P<events>\d+) "
    r"Rate=(?P<reported>[0-9.]+) Hz"
)


def parse_log(path: Path) -> Optional[Dict[str, object]]:
    name_match = NAME_RE.match(path.name)
    if name_match is None:
        return None
    matches = list(SUMMARY_RE.finditer(path.read_text(errors="replace")))
    if not matches:
        return None
    summary = matches[-1]
    events = int(summary.group("events"))
    loop_time = float(summary.group("loop"))
    rate = events / loop_time
    bytes_per_event = TOTAL_BIGDATA_BYTES / TOTAL_EVENTS
    payload_gbps = bytes_per_event * rate / 1e9
    return {
        "job": int(name_match.group("job")),
        "batch_size": int(name_match.group("batch")),
        "pool_depth": int(name_match.group("pool")),
        "kvikio_threads": int(name_match.group("threads")),
        "events": events,
        "load_time_s": float(summary.group("load")),
        "loop_time_s": loop_time,
        "rate_hz": rate,
        "reported_rate_hz": float(summary.group("reported")),
        "speedup_vs_cpu_1bd": rate / CPU_ONE_BD_HZ,
        "xtc_payload_gbps": payload_gbps,
        "pct_nvme_rated": 100 * payload_gbps / NVME_RAID_RATED_GBPS,
        "pct_nvme_pcie": 100 * payload_gbps / NVME_RAID_PCIE_GBPS,
        "pct_gpu_pcie": 100 * payload_gbps / GPU_PCIE_GBPS,
        "log": path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path, nargs="?", default=Path(__file__).parent)
    parser.add_argument("--job", type=int)
    args = parser.parse_args()

    rows = []
    for path in args.log_dir.glob("gpu_*.log"):
        row = parse_log(path)
        if row is not None:
            rows.append(row)
    if args.job is not None:
        rows = [row for row in rows if row["job"] == args.job]
    rows.sort(key=lambda row: (row["kvikio_threads"], row["pool_depth"], row["batch_size"]))
    if not rows:
        raise SystemExit("no completed GPU case summaries found")

    fieldnames = list(rows[0])
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    print("\n| threads | batch | pool | loop (s) | rate (Hz) | vs CPU | XTC GB/s | SSD rated | GPU PCIe |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['kvikio_threads']} | {row['batch_size']} | {row['pool_depth']} "
            f"| {row['loop_time_s']:.2f} | {row['rate_hz']:.1f} "
            f"| {row['speedup_vs_cpu_1bd']:.2f}x | {row['xtc_payload_gbps']:.2f} "
            f"| {row['pct_nvme_rated']:.1f}% | {row['pct_gpu_pcie']:.1f}% |"
        )


if __name__ == "__main__":
    main()
