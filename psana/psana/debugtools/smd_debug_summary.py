#!/usr/bin/env python3
import argparse
import re


RANK_PATTERNS = (
    re.compile(r'^\[DEBUG Rank(\d+)\]\s+(.*)$'),
    re.compile(r'^\[DEBUG\]\s+Rank(\d+)\s+(.*)$'),
)


def match_rank(line):
    for pat in RANK_PATTERNS:
        m = pat.match(line)
        if m:
            return int(m.group(1)), m.group(2).strip()
    return None, None


def parse_first_float(text):
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)s', text)
    return float(m.group(1)) if m else None


def parse_kv_time(text, key):
    m = re.search(rf'{re.escape(key)}=([0-9]+(?:\.[0-9]+)?)s', text)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description="Summarize smalldata_tools DEBUG timings by rank.")
    parser.add_argument("logfile", help="Path to log file")
    parser.add_argument("--rank", type=int, default=None, help="Rank to summarize")
    args = parser.parse_args()

    data = {
        "arg_setup_time": None,
        "det_setup_delta": None,
        "det_setup_since": None,
        "event_loop_start": None,
        "first_event_since": None,
        "first_event_delta": None,
        "finalize_since": None,
        "finalize_delta": None,
    }
    dets = {}
    rank_seen = None
    loop_end_since = None

    with open(args.logfile, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            rank, msg = match_rank(line)
            if rank is None:
                continue
            if args.rank is None:
                rank_seen = rank
            elif rank != args.rank:
                continue

            if "marker before begin setup time since job start" in msg:
                data["arg_setup_time"] = parse_first_float(msg)
                continue

            if "smalldata detector setup time" in msg:
                data["det_setup_delta"] = parse_first_float(msg)
                continue

            if "total setup time since job start" in msg:
                data["det_setup_since"] = parse_first_float(msg)
                continue

            if msg.startswith("smalldata init"):
                since = parse_kv_time(msg, "since_start")
                if data["arg_setup_time"] is None and since is not None:
                    data["arg_setup_time"] = since

            if msg.startswith("define_dets ") and " since_start=" in msg:
                if data["det_setup_since"] is None:
                    data["det_setup_since"] = parse_kv_time(msg, "since_start")
                if data["det_setup_delta"] is None:
                    data["det_setup_delta"] = parse_kv_time(msg, "delta")

            if msg.startswith("Event loop start"):
                data["event_loop_start"] = parse_kv_time(msg, "since_start")

            if msg.startswith("evt 0 start"):
                data["first_event_since"] = parse_kv_time(msg, "since_start")

            if msg.startswith("first event since_start"):
                data["first_event_since"] = parse_kv_time(msg, "since_start")

            if msg.startswith("smalldata.done"):
                data["finalize_delta"] = parse_kv_time(msg, "delta")
                data["finalize_since"] = parse_kv_time(msg, "since_start")
                loop_end_since = data["finalize_since"]

            if msg.startswith("define_dets pre-loop config"):
                dets.setdefault("_preloop", {})["delta"] = parse_first_float(msg)
                continue

            if msg.startswith("define_dets det="):
                parts = msg.split()
                det_name = parts[1].split("=", 1)[1]
                section = parts[2]
                delta = parse_kv_time(msg, "delta")
                total = parse_kv_time(msg, "total")
                det_entry = dets.setdefault(det_name, {})
                if section == "DetObject":
                    det_entry["detobject_total"] = total if total is not None else delta
                else:
                    det_entry.setdefault("define_dets_steps", []).append((section, delta, total))
                continue

            if msg.startswith("DetObject det="):
                parts = msg.split()
                det_name = parts[1].split("=", 1)[1]
                section = parts[2]
                delta = parse_kv_time(msg, "delta")
                total = parse_kv_time(msg, "total")
                det_entry = dets.setdefault(det_name, {})
                det_entry.setdefault("detobject_steps", []).append((section, delta, total))

    rank_label = args.rank if args.rank is not None else rank_seen
    print(f"Rank {rank_label} summary")

    if data["arg_setup_time"] is not None:
        print(f"- Arg setup time: {data['arg_setup_time']:.6f}s since_start")
    else:
        print("- Arg setup time: not found")

    if data["det_setup_delta"] is not None or data["det_setup_since"] is not None:
        delta = data["det_setup_delta"]
        since = data["det_setup_since"]
        delta_str = f"{delta:.6f}s" if delta is not None else "n/a"
        since_str = f"{since:.6f}s" if since is not None else "n/a"
        print(f"- Detector setup total: delta={delta_str} since_start={since_str}")
    else:
        print("- Detector setup total: not found")

    preloop = dets.get("_preloop", {}).get("delta")
    if preloop is not None:
        print(f"- Detector setup pre-loop: delta={preloop:.6f}s")

    for det_name, info in sorted(dets.items()):
        if det_name == "_preloop":
            continue
        det_total = info.get("detobject_total")
        if det_total is not None:
            print(f"- {det_name} DetObject total: {det_total:.6f}s")
        steps = info.get("detobject_steps", [])
        if steps:
            step_str = ", ".join(
                f"{name}={delta:.6f}s" for name, delta, _ in steps if delta is not None
            )
            if step_str:
                print(f"- {det_name} DetObject steps (delta): {step_str}")
        define_steps = info.get("define_dets_steps", [])
        if define_steps:
            step_str = ", ".join(
                f"{name}={delta:.6f}s" for name, delta, _ in define_steps if delta is not None
            )
            if step_str:
                print(f"- {det_name} define_dets steps (delta): {step_str}")

    if data["event_loop_start"] is not None and data["first_event_since"] is not None:
        data["first_event_delta"] = data["first_event_since"] - data["event_loop_start"]

    if data["first_event_since"] is not None:
        delta_str = f"{data['first_event_delta']:.6f}s" if data["first_event_delta"] is not None else "n/a"
        print(f"- First event: delta={delta_str} since_start={data['first_event_since']:.6f}s")
    else:
        print("- First event: not found")

    if loop_end_since is not None and data["first_event_since"] is not None:
        loop_delta = loop_end_since - data["first_event_since"]
        print(f"- Loop total: delta={loop_delta:.6f}s since_start={loop_end_since:.6f}s")
    else:
        print("- Loop total: not found")

    if data["finalize_delta"] is not None or data["finalize_since"] is not None:
        delta_str = f"{data['finalize_delta']:.6f}s" if data["finalize_delta"] is not None else "n/a"
        since_str = f"{data['finalize_since']:.6f}s" if data["finalize_since"] is not None else "n/a"
        print(f"- Finalize: delta={delta_str} since_start={since_str}")
    else:
        print("- Finalize: not found")


if __name__ == "__main__":
    main()
