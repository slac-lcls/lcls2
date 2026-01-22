#!/usr/bin/env python3
import argparse
import re
import sys
import statistics

RANK_PATTERNS = (
    re.compile(r'^\[DEBUG\]\s+Rank(\d+)\s+(.*)$'),
    re.compile(r'^\[DEBUG Rank(\d+)\]\s+(.*)$'),
    re.compile(r'^\[DEBUG\]\s+rank\s+(\d+)\s+(.*)$'),
    re.compile(r'^\[DEBUG\]\s+(.*)$'),  # no rank
)

def match_rank(line):
    for pat in RANK_PATTERNS:
        m = pat.match(line)
        if m:
            if pat.groups == 2:
                return int(m.group(1)), m.group(2).strip()
            return "global", m.group(1).strip()
    return None, None

def parse_since_start(msg):
    m = re.search(r'since_start=([0-9]+(?:\.[0-9]+)?)s', msg)
    return float(m.group(1)) if m else None

def parse_delta(msg):
    m = re.search(r'delta=([0-9]+(?:\.[0-9]+)?)s', msg)
    return float(m.group(1)) if m else None

def parse_any_time(msg):
    val = parse_delta(msg)
    if val is not None:
        return val
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)s', msg)
    return float(m.group(1)) if m else None

def add_value(stage_map, stage, val, rank):
    if val is None:
        return
    stage_map.setdefault(stage, []).append((val, rank))

def record_single(stage_map, stage, rank, val, warnings, seen):
    if val is None:
        return
    if stage in seen:
        warnings.append(f"WARNING multiple {stage} {val:.6f} {rank}")
    else:
        seen.add(stage)
    add_value(stage_map, stage, val, rank)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", help="Path to log file")
    parser.add_argument(
        "--ignore-rank",
        type=int,
        action="append",
        default=[],
        help="Exclude this rank from max calculations (can be repeated).",
    )
    args = parser.parse_args()

    logfile = args.logfile
    stage_values = {}
    warnings = []
    seen_single = set()

    with open(logfile, "r", errors="replace") as f:
        for line in f:
            rank, msg = match_rank(line)
            if rank is None:
                continue
            if isinstance(rank, int) and rank in args.ignore_rank:
                continue

            # single-occurrence stages (delta)
            if msg.startswith("args parsed"):
                record_single(stage_values, "args_parsed", rank, parse_delta(msg), warnings, seen_single)
                continue
            if msg.startswith("ds args setup"):
                record_single(stage_values, "ds_args_setup", rank, parse_delta(msg), warnings, seen_single)
                continue
            if msg.startswith("ds init"):
                record_single(stage_values, "ds_init", rank, parse_delta(msg), warnings, seen_single)
                continue
            if msg.startswith("run init"):
                record_single(stage_values, "run_init", rank, parse_delta(msg), warnings, seen_single)
                continue
            if msg.startswith("smd init"):
                add_value(stage_values, "smd_init", parse_delta(msg), rank)
                continue

            # detector setup (delta)
            if "total setup time since job start" in msg:
                add_value(stage_values, "det_setup", parse_delta(msg), rank)
                continue
            if msg.startswith("define_dets ") and "delta=" in msg:
                add_value(stage_values, "det_setup", parse_delta(msg), rank)
                continue
            if msg.startswith("default det setup"):
                add_value(stage_values, "det_setup", parse_delta(msg), rank)
                continue

            # first/last event (delta)
            if msg.startswith("first evt"):
                add_value(stage_values, "first_evt", parse_delta(msg), rank)
                continue
            if msg.startswith("evt 0 start"):
                add_value(stage_values, "first_evt", parse_delta(msg), rank)
                continue
            if msg.startswith("first event since_start"):
                add_value(stage_values, "first_evt", parse_delta(msg), rank)
                continue

            if msg.startswith("last evt"):
                add_value(stage_values, "last_evt", parse_delta(msg), rank)
                continue
            if msg.startswith("evt ") and " start " in msg:
                add_value(stage_values, "last_evt", parse_delta(msg), rank)
                continue

            # total (epics uses since_start)
            if msg.startswith("epics data done"):
                add_value(stage_values, "total", parse_since_start(msg), rank)
                continue
            if msg.startswith("smalldata.done"):
                add_value(stage_values, "total", parse_delta(msg), rank)
                continue

            # loop sub-stages (delta)
            if "det process time" in msg:
                add_value(stage_values, "det_process_time", parse_any_time(msg), rank)
                continue
            if "event store time" in msg:
                add_value(stage_values, "event_store_time", parse_any_time(msg), rank)
                continue
            if "arp update time" in msg:
                add_value(stage_values, "arp_update_time", parse_any_time(msg), rank)
                continue
            if "jungfrau getData time" in msg:
                add_value(stage_values, "jungfrau_getData_time", parse_any_time(msg), rank)
                continue
            if "jungfrau processFuncs time" in msg:
                add_value(stage_values, "jungfrau_processFuncs_time", parse_any_time(msg), rank)
                continue
            if "jungfrau getUserData time" in msg:
                add_value(stage_values, "jungfrau_getUserData_time", parse_any_time(msg), rank)
                continue
            if "jungfrau getUserEnvData time" in msg:
                add_value(stage_values, "jungfrau_getUserEnvData_time", parse_any_time(msg), rank)
                continue
            if "jungfrau processSums time" in msg:
                add_value(stage_values, "jungfrau_processSums_time", parse_any_time(msg), rank)
                continue
    print("units seconds")
    for key in (
        "args_parsed",
        "ds_args_setup",
        "ds_init",
        "run_init",
        "smd_init",
        "det_setup",
        "first_evt",
        "last_evt",
        "det_process_time",
        "event_store_time",
        "arp_update_time",
        "jungfrau_getData_time",
        "jungfrau_processFuncs_time",
        "jungfrau_getUserData_time",
        "jungfrau_getUserEnvData_time",
        "jungfrau_processSums_time",
        "total",
    ):
        entries = stage_values.get(key)
        if entries:
            vals = [v for v, _ in entries]
            avg = statistics.mean(vals)
            vmin = min(vals)
            vmax = max(vals)
            med = statistics.median(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            max_rank = next(r for v, r in entries if v == vmax)
            print(f"{key} {avg:.6f} {vmin:.6f} {vmax:.6f} {med:.6f} {std:.6f} {max_rank}")
        else:
            print(f"{key} NA NA NA NA NA NA")

    for w in warnings:
        print(w)

if __name__ == "__main__":
    main()
