#!/usr/bin/env python3
import argparse
import re
import sys
import statistics

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
EVENT_RE = re.compile(r"\bevt\s+(\d+)\b")

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

def count_events_from_stage(stage_map, stage):
    counts = {}
    for _, rank in stage_map.get(stage, []):
        if isinstance(rank, int):
            counts[rank] = counts.get(rank, 0) + 1
    return counts

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
    parser.add_argument(
        "--no-det",
        action="store_true",
        help="Skip detector-specific keywords (jungfrau_*).",
    )
    parser.add_argument(
        "--show-max",
        action="store_true",
        help="Only print max value and max rank for each keyword.",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Do not print the label column or the max-rank column.",
    )
    args = parser.parse_args()

    logfile = args.logfile
    stage_values = {}
    warnings = []
    seen_single = set()
    event_numbers_by_rank = {}

    with open(logfile, "r", errors="replace") as f:
        for line in f:
            line = ANSI_ESCAPE_RE.sub("", line)
            rank, msg = match_rank(line)
            if rank is None:
                continue
            if isinstance(rank, int) and rank in args.ignore_rank:
                continue

            if isinstance(rank, int):
                m_evt = EVENT_RE.search(msg)
                if m_evt:
                    event_numbers_by_rank.setdefault(rank, set()).add(int(m_evt.group(1)))

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

            # per-event loop marks (delta)
            if msg.startswith("evt ") and " default detData" in msg:
                add_value(stage_values, "evt_default_detdata", parse_delta(msg), rank)
                continue
            if msg.startswith("evt ") and " user dets" in msg:
                add_value(stage_values, "evt_user_dets", parse_delta(msg), rank)
                continue
            if msg.startswith("evt ") and " intg dets" in msg:
                # deprecated: evt_intg_dets
                continue
            if msg.startswith("evt ") and " event store" in msg:
                add_value(stage_values, "evt_event_store", parse_delta(msg), rank)
                continue

            # total (all done uses since_start)
            if msg.startswith("srv node done"):
                val = parse_delta(msg)
                if val is not None:
                    add_value(stage_values, "srv_node_done", val, rank)
                continue
            if msg.startswith("psana all done"):
                record_single(stage_values, "total", rank, parse_since_start(msg), warnings, seen_single)
                continue

            # loop sub-stages (delta)
            if msg.startswith("smd done save sum"):
                add_value(stage_values, "smd_done_save_sum", parse_delta(msg), rank)
                continue
            if msg.startswith("smd done save cfg"):
                add_value(stage_values, "smd_done_save_cfg", parse_delta(msg), rank)
                continue
            if msg.startswith("smd done all"):
                add_value(stage_values, "smd_done_all", parse_delta(msg), rank)
                continue
            if msg.startswith("smalldata cls init start"):
                add_value(stage_values, "smalldata_cls_init_start", parse_delta(msg), rank)
                continue
            if msg.startswith("smalldata cls init done"):
                add_value(stage_values, "smalldata_cls_init_done", parse_delta(msg), rank)
                continue
            if msg.startswith("smalldata cls event done"):
                add_value(stage_values, "smalldata_cls_event_done", parse_delta(msg), rank)
                continue
            if msg.startswith("smalldata cls sum done"):
                add_value(stage_values, "smalldata_cls_sum_done", parse_delta(msg), rank)
                continue
            if msg.startswith("smalldata cls save summary done"):
                add_value(stage_values, "smalldata_cls_save_summary_done", parse_delta(msg), rank)
                continue
            if msg.startswith("smalldata cls done"):
                add_value(stage_values, "smalldata_cls_done", parse_since_start(msg), rank)
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
    keys = [
        "args_parsed",
        "ds_args_setup",
        "ds_init",
        "run_init",
        "det_setup",
        "first_evt",
        "last_evt",
        "evt_default_detdata",
        "evt_user_dets",
        "evt_event_store",
        "jungfrau_getData_time",
        "jungfrau_processFuncs_time",
        "jungfrau_getUserData_time",
        "jungfrau_getUserEnvData_time",
        "jungfrau_processSums_time",
        "srv_node_done",
        "smd_done_save_sum",
        "smd_done_save_cfg",
        "smd_done_all",
        "smalldata_cls_init_start",
        "smalldata_cls_init_done",
        "smalldata_cls_event_done",
        "smalldata_cls_sum_done",
        "smalldata_cls_save_summary_done",
        "smalldata_cls_done",
        "total",
    ]
    if args.no_det:
        keys = [k for k in keys if not k.startswith("jungfrau_")]

    for key in keys:
        entries = stage_values.get(key)
        if entries:
            vals = [v for v, _ in entries]
            vmax = max(vals)
            max_rank = next(r for v, r in entries if v == vmax)
            if args.show_max:
                if args.no_label:
                    print(f"{vmax:.6f}")
                else:
                    print(f"{key} {vmax:.6f} {max_rank}")
            else:
                avg = statistics.mean(vals)
                vmin = min(vals)
                med = statistics.median(vals)
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                if args.no_label:
                    print(f"{avg:.6f} {vmin:.6f} {vmax:.6f} {med:.6f} {std:.6f}")
                else:
                    print(f"{key} {avg:.6f} {vmin:.6f} {vmax:.6f} {med:.6f} {std:.6f} {max_rank}")
        else:
            if args.show_max:
                if args.no_label:
                    print("NA")
                else:
                    print(f"{key} NA NA")
            else:
                if args.no_label:
                    print("NA NA NA NA NA")
                else:
                    print(f"{key} NA NA NA NA NA NA")

    counts = {r: len(evts) for r, evts in event_numbers_by_rank.items()}
    if not counts:
        counts = count_events_from_stage(stage_values, "smalldata_cls_event_done")

    if counts:
        vals = list(counts.values())
        vmax = max(vals)
        max_rank = next(r for r, v in counts.items() if v == vmax)
        print("units events")
        if args.show_max:
            if args.no_label:
                print(f"{float(vmax):.6f}")
            else:
                print(f"events_per_rank {float(vmax):.6f} {max_rank}")
        else:
            avg = statistics.mean(vals)
            vmin = min(vals)
            med = statistics.median(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            if args.no_label:
                print(f"{avg:.6f} {float(vmin):.6f} {float(vmax):.6f} {med:.6f} {std:.6f}")
                print(f"{sum(vals)}")
            else:
                print(f"events_per_rank {avg:.6f} {float(vmin):.6f} {float(vmax):.6f} {med:.6f} {std:.6f} {max_rank}")
                print(f"total_events {sum(vals)}")

    for w in warnings:
        print(w)

if __name__ == "__main__":
    main()
