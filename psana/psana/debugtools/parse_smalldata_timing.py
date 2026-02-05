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
        "--show-azav",
        action="store_true",
        help="Include azav_* keywords in output.",
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
    azav_keys = set()

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
            if msg.startswith("generic module imports"):
                record_single(stage_values, "generic_module_imports", rank, parse_delta(msg), warnings, seen_single)
                continue
            if msg.startswith("custom module imports"):
                record_single(stage_values, "custom_module_imports", rank, parse_delta(msg), warnings, seen_single)
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
            if msg.startswith("azav "):
                if " delta=" in msg:
                    label = msg.split(" delta=")[0].strip()
                    add_value(stage_values, label, parse_delta(msg), rank)
                    azav_keys.add(label)
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
        "generic_module_imports",
        "custom_module_imports",
        "ds_args_setup",
        "ds_init",
        "run_init",
        "smd_init",
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
    if args.show_azav and azav_keys:
        azav_order = [
            "azav initialized",
            "azav setFromDet start",
            "azav setFromDet after mask",
            "azav setFromDet after mask flatten",
            "azav setFromDet after x/y flatten",
            "azav setFromDet after z flatten",
            "azav setFromDet after z default",
            "azav setFromDet before _init_shared_cache",
            "azav after _init_shared_cache",
            "azav cache_enabled: True, key: True",
            "azav shared cache hit det=jungfrau",
            "azav computing binning",
            "azav computing binning done",
            "azav storing azav cache for key",
            "azav bcast meta done",
            "azav shared cache leader wrote det=jungfrau",
            "azav shared cache retrieved det=jungfrau",
            "azav storing azav cache done",
            "azav setFromFunc",
            "azav after _setup",
            "azav doCake np.bincount start",
            "azav doCake np.bincount done",
        ]
        ordered = []
        for label in azav_order:
            for key in azav_keys:
                if key == label or key.startswith(label):
                    ordered.append(key)
        remaining = sorted(azav_keys.difference(ordered))
        azav_list = ordered + remaining
        # de-dup while preserving order
        seen = set()
        azav_list = [k for k in azav_list if not (k in seen or seen.add(k))]
        keys.extend(azav_list)
    if args.no_det:
        keys = [k for k in keys if not k.startswith("jungfrau_")]

    section_width = max(len("section"), max((len(k) for k in keys), default=0))
    num_width = 10  # room for numbers with 2 decimals
    max_rank_width = len("max_rank")
    if not args.no_label:
        if args.show_max:
            header = (
                f"{'section':<{section_width}} "
                f"{'max':>{num_width}} "
                f"{'max_rank':>{max_rank_width}}"
            )
        else:
            header = (
                f"{'section':<{section_width}} "
                f"{'avg':>{num_width}} "
                f"{'min':>{num_width}} "
                f"{'max':>{num_width}} "
                f"{'med':>{num_width}} "
                f"{'std.':>{num_width}} "
                f"{'max_rank':>{max_rank_width}}"
            )
        print(header)

    for key in keys:
        entries = stage_values.get(key)
        if entries:
            vals = [v for v, _ in entries]
            vmax = max(vals)
            max_rank = next(r for v, r in entries if v == vmax)
            if args.show_max:
                if args.no_label:
                    print(f"{vmax:.2f}")
                else:
                    print(
                        f"{key:<{section_width}} "
                        f"{vmax:>{num_width}.2f} "
                        f"{max_rank:>{max_rank_width}}"
                    )
            else:
                avg = statistics.mean(vals)
                vmin = min(vals)
                med = statistics.median(vals)
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                if args.no_label:
                    print(
                        f"{avg:.2f} {vmin:.2f} {vmax:.2f} {med:.2f} {std:.2f}"
                    )
                else:
                    print(
                        f"{key:<{section_width}} "
                        f"{avg:>{num_width}.2f} "
                        f"{vmin:>{num_width}.2f} "
                        f"{vmax:>{num_width}.2f} "
                        f"{med:>{num_width}.2f} "
                        f"{std:>{num_width}.2f} "
                        f"{max_rank:>{max_rank_width}}"
                    )
        else:
            if args.show_max:
                if args.no_label:
                    print("NA")
                else:
                    print(
                        f"{key:<{section_width}} "
                        f"{'NA':>{num_width}} "
                        f"{'NA':>{max_rank_width}}"
                    )
            else:
                if args.no_label:
                    print("NA NA NA NA NA")
                else:
                    print(
                        f"{key:<{section_width}} "
                        f"{'NA':>{num_width}} "
                        f"{'NA':>{num_width}} "
                        f"{'NA':>{num_width}} "
                        f"{'NA':>{num_width}} "
                        f"{'NA':>{num_width}} "
                        f"{'NA':>{max_rank_width}}"
                    )

    counts = {r: len(evts) for r, evts in event_numbers_by_rank.items()}
    if not counts:
        counts = count_events_from_stage(stage_values, "smalldata_cls_event_done")

    if counts:
        vals = list(counts.values())
        vmax = max(vals)
        max_rank = next(r for r, v in counts.items() if v == vmax)
        print("units events")
        if not args.no_label:
            if args.show_max:
                header = (
                    f"{'section':<{section_width}} "
                    f"{'max':>{num_width}} "
                    f"{'max_rank':>{max_rank_width}}"
                )
            else:
                header = (
                    f"{'section':<{section_width}} "
                    f"{'avg':>{num_width}} "
                    f"{'min':>{num_width}} "
                    f"{'max':>{num_width}} "
                    f"{'med':>{num_width}} "
                    f"{'std.':>{num_width}} "
                    f"{'max_rank':>{max_rank_width}}"
                )
            print(header)
        if args.show_max:
            if args.no_label:
                print(f"{float(vmax):.2f}")
            else:
                print(
                    f"{'events_per_rank':<{section_width}} "
                    f"{float(vmax):>{num_width}.2f} "
                    f"{max_rank:>{max_rank_width}}"
                )
        else:
            avg = statistics.mean(vals)
            vmin = min(vals)
            med = statistics.median(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            if args.no_label:
                print(f"{avg:.2f} {float(vmin):.2f} {float(vmax):.2f} {med:.2f} {std:.2f}")
                print(f"{sum(vals)}")
            else:
                print(
                    f"{'events_per_rank':<{section_width}} "
                    f"{avg:>{num_width}.2f} "
                    f"{float(vmin):>{num_width}.2f} "
                    f"{float(vmax):>{num_width}.2f} "
                    f"{med:>{num_width}.2f} "
                    f"{std:>{num_width}.2f} "
                    f"{max_rank:>{max_rank_width}}"
                )
                print(f"total_events {sum(vals)}")

    for w in warnings:
        print(w)

if __name__ == "__main__":
    main()
