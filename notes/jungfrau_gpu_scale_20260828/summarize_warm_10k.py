#!/usr/bin/env python3
"""Validate and compare the warm 10k Jungfrau GPU scaling matrix."""

import argparse
import json
import pathlib
import re
import statistics


RESULT_PREFIX = "GPU_SCALE_RESULT "
RESIDENCY_PREFIX = "CACHE_RESIDENCY "
PRIME_PREFIX = "CACHE_PRIME "
EXPECTED_EVENTS = 10_000
EXPECTED_BYTES = 335_571_760_000
O_DIRECT_GBPS = 11.714
GBPS_200GBIT = 25.0
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
COLD_GBPS = {
    (1, 1): 5.857,
    (1, 2): 8.307,
    (1, 4): 9.197,
    (1, 6): 9.140,
    (1, 8): 9.115,
    (2, 2): 8.849,
    (2, 4): 9.286,
    (2, 6): 9.202,
    (4, 4): 9.011,
    (4, 8): 9.043,
    (4, 12): 8.885,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()

    records = []
    residencies = {}
    primes = []
    case_status = {}
    gpu_checks = {}
    for line in args.log.read_text(errors="replace").splitlines():
        if line.startswith(RESULT_PREFIX):
            records.append(json.loads(line[len(RESULT_PREFIX) :]))
        elif line.startswith(RESIDENCY_PREFIX):
            item = json.loads(line[len(RESIDENCY_PREFIX) :])
            residencies[item["tag"]] = item
        elif line.startswith(PRIME_PREFIX):
            primes.append(json.loads(line[len(PRIME_PREFIX) :]))
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

    grouped = {}
    before_fractions = []
    after_fractions = []
    initial = residencies.get("after_initial_eviction")
    if initial is None:
        raise RuntimeError("missing initial post-eviction cache residency")
    if initial["resident_pages"] != 0:
        raise RuntimeError(f"initial eviction was incomplete: {initial}")
    for item in records:
        tag = item["case"]
        if not tag.startswith("warm_"):
            raise RuntimeError(f"non-warm result in warm log: {tag}")
        if not item["valid"]:
            raise RuntimeError(f"invalid result: {tag}")
        if item["events"] != EXPECTED_EVENTS:
            raise RuntimeError(f"wrong event count: {tag}")
        if item["unique_timestamps"] != EXPECTED_EVENTS:
            raise RuntimeError(f"wrong unique timestamp count: {tag}")
        if item["io_bytes"] != EXPECTED_BYTES:
            raise RuntimeError(f"wrong payload byte count: {tag}")
        if case_status.get(tag) != 0:
            raise RuntimeError(f"missing or failed CASE_END: {tag}")
        expected_gpu_check = (item["n_gpus"], item["n_gpus"])
        if gpu_checks.get(tag) != expected_gpu_check:
            raise RuntimeError(f"failed physical GPU check: {tag}")

        before = residencies.get(f"{tag}_before")
        if before is None:
            raise RuntimeError(f"missing pre-case cache residency: {tag}")
        before_fractions.append(before["resident_fraction"])
        if before["resident_fraction"] < 0.99:
            raise RuntimeError(f"cache not warm before {tag}: {before}")
        after = residencies.get(f"{tag}_after")
        if after is None:
            raise RuntimeError(f"missing post-case cache residency: {tag}")
        after_fractions.append(after["resident_fraction"])
        if after["resident_fraction"] < 0.99:
            raise RuntimeError(f"cache was not retained after {tag}: {after}")
        grouped.setdefault((item["n_gpus"], item["n_bds"]), []).append(item)

    if not records:
        raise RuntimeError(f"no warm results in {args.log}")
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
        "| GPUs | BDs | Reps | Loop range (s) | Warm rate median (Hz) | "
        "Warm payload median (GB/s) | Cold payload (GB/s) | Warm/cold |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, items in sorted(grouped.items()):
        loops = [item["loop_s"] for item in items]
        rates = [item["rate_hz"] for item in items]
        payloads = [item["payload_gbps"] for item in items]
        warm_payload = statistics.median(payloads)
        print(
            f"| {key[0]} | {key[1]} | {len(items)} | "
            f"{min(loops):.3f}–{max(loops):.3f} | "
            f"{statistics.median(rates):.1f} | {warm_payload:.3f} | "
            f"{COLD_GBPS[key]:.3f} | {warm_payload / COLD_GBPS[key]:.2f}x |"
        )

    print(f"\nvalidated_records={len(records)}")
    print(f"initial_resident_pages={initial['resident_pages']}")
    print(f"minimum_pre_case_residency={min(before_fractions):.9f}")
    print(f"minimum_post_case_residency={min(after_fractions):.9f}")
    if primes:
        print(f"prime_count={len(primes)}")
        print(
            "prime_gbps_range="
            f"{min(x['gbps'] for x in primes):.3f}–"
            f"{max(x['gbps'] for x in primes):.3f}"
        )
    best = max(
        statistics.median([x["payload_gbps"] for x in items])
        for items in grouped.values()
    )
    print(f"best_warm_gbps={best:.6f}")
    print(f"best_warm_vs_o_direct={best / O_DIRECT_GBPS:.6f}x")
    print(f"best_warm_vs_200gbit={best / GBPS_200GBIT:.6f}x")


if __name__ == "__main__":
    main()
