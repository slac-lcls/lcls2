#!/usr/bin/env python3
"""Summarize and validate a 10,000-event Jungfrau GPU scale matrix."""

import argparse
import json
import pathlib
import re
import statistics


PREFIX = "GPU_SCALE_RESULT "
O_DIRECT_GBPS = 11.714
EXPECTED_EVENTS = 10_000
EXPECTED_BYTES = 335_571_760_000
EXPECTED_CASES = {
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 6),
    (1, 8),
    (2, 2),
    (2, 4),
    (2, 6),
    (4, 4),
    (4, 8),
    (4, 12),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument(
        "--complete",
        action="store_true",
        help="require two valid repetitions of the complete 11-point matrix",
    )
    args = parser.parse_args()

    records = []
    case_status = {}
    gpu_checks = {}
    for line in args.log.read_text(errors="replace").splitlines():
        if line.startswith(PREFIX):
            records.append(json.loads(line[len(PREFIX) :]))
        elif line.startswith("CASE_END "):
            match = re.search(r"tag=(\S+) status=(\d+)", line)
            if match:
                case_status[match.group(1)] = int(match.group(2))
        elif line.startswith("PHYSICAL_GPU_CHECK "):
            match = re.search(r"tag=(\S+) expected=(\d+) observed=(\d+)", line)
            if match:
                gpu_checks[match.group(1)] = (
                    int(match.group(2)),
                    int(match.group(3)),
                )

    if not records:
        raise RuntimeError(f"no {PREFIX.strip()} records in {args.log}")

    for item in records:
        if not item["valid"]:
            raise RuntimeError(f"invalid record: {item['case']}")
        if item["events"] != EXPECTED_EVENTS:
            raise RuntimeError(f"wrong event count: {item['case']}")
        if item["unique_timestamps"] != EXPECTED_EVENTS:
            raise RuntimeError(f"wrong unique timestamp count: {item['case']}")
        if item["io_bytes"] != EXPECTED_BYTES:
            raise RuntimeError(f"wrong payload byte count: {item['case']}")
        if case_status.get(item["case"]) != 0:
            raise RuntimeError(f"missing or failed CASE_END: {item['case']}")
        expected_gpu_check = (item["n_gpus"], item["n_gpus"])
        if gpu_checks.get(item["case"]) != expected_gpu_check:
            raise RuntimeError(f"failed physical GPU check: {item['case']}")

    grouped = {}
    for item in records:
        grouped.setdefault((item["n_gpus"], item["n_bds"]), []).append(item)

    if args.complete:
        if set(grouped) != EXPECTED_CASES:
            raise RuntimeError(
                f"wrong matrix: found={sorted(grouped)} expected={sorted(EXPECTED_CASES)}"
            )
        wrong_repetitions = {
            key: len(items) for key, items in grouped.items() if len(items) != 2
        }
        if wrong_repetitions:
            raise RuntimeError(f"wrong repetition counts: {wrong_repetitions}")

    print(
        "| GPUs | BDs | Reps | Loop range (s) | Rate median (Hz) | "
        "Payload median (GB/s) | O_DIRECT fraction | Max GPU used (GiB) |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (n_gpus, n_bds), items in sorted(grouped.items()):
        loops = [item["loop_s"] for item in items]
        rates = [item["rate_hz"] for item in items]
        payloads = [item["payload_gbps"] for item in items]
        gpu_bytes = [
            rank["gpu_used_peak_bytes"]
            for item in items
            for rank in item["ranks"]
        ]
        median_payload = statistics.median(payloads)
        print(
            f"| {n_gpus} | {n_bds} | {len(items)} | "
            f"{min(loops):.3f}–{max(loops):.3f} | "
            f"{statistics.median(rates):.1f} | "
            f"{median_payload:.3f} | "
            f"{median_payload / O_DIRECT_GBPS:.1%} | "
            f"{max(gpu_bytes) / 1024**3:.1f} |"
        )

    print(f"\nvalidated_records={len(records)}")


if __name__ == "__main__":
    main()
