#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


_XTC_LINE_RE = re.compile(
    r"^event\s+\d+,\s+([A-Za-z0-9_]+)\s+transition:.*\sextent\s+(\d+)\s*$"
)
_CHUNK_RE = re.compile(r"^(?P<prefix>.+-s\d{3})-(?P<chunk>c\d{3})(?P<smd>\.smd)?\.xtc2$")


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Stage the minimum XTC2 prefix needed for the first N L1Accept events "
            "of a run, including smalldata files."
        )
    )
    parser.add_argument("--xtc-dir", required=True, help="Run xtc directory.")
    parser.add_argument("--run", required=True, type=int, help="Run number, e.g. 125.")
    parser.add_argument(
        "--events",
        type=int,
        default=1000,
        help="Number of L1Accept events to stage per stream (default: 1000).",
    )
    parser.add_argument(
        "--max-extra-records",
        type=int,
        default=512,
        help="Additional records beyond --events to scan with xtcreader (default: 512).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Destination root for staged prefixes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the staging plan without copying files.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path to write JSON manifest. Defaults to <out-dir>/stage-manifest.json.",
    )
    parser.add_argument(
        "--annotate-detectors",
        action="store_true",
        help="Call detnames once per stream to annotate detector names. Off by default for speed.",
    )
    parser.add_argument(
        "--bd-chunksize",
        type=int,
        default=0x1000000,
        help="Assumed PS_BD_CHUNKSIZE for conservative max_events recommendation (default: 16777216).",
    )
    parser.add_argument(
        "--recommend-min-extent",
        type=int,
        default=5_000_000,
        help=(
            "Only use streams with first_l1_extent >= this many bytes when computing "
            "recommended_max_events (default: 5000000)."
        ),
    )
    return parser


def _run_cmd(args):
    completed = subprocess.run(args, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(args)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


def _detnames(path: Path):
    output = _run_cmd(["detnames", str(path)])
    names = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("-") or line.startswith("Name"):
            continue
        if "|" not in line:
            continue
        name, dtype = [part.strip() for part in line.split("|", 1)]
        names.append({"name": name, "dtype": dtype})
    return names


def _xtcreader_prefix(path: Path, target_l1: int, max_extra_records: int):
    if target_l1 <= 0:
        return {"bytes": 0, "l1_events": 0, "records": 0, "first_l1_extent": None}

    max_records = target_l1 + max_extra_records
    output = _run_cmd(["xtcreader", "-f", str(path), "-n", str(max_records)])

    total_bytes = 0
    l1_events = 0
    records = 0
    first_l1_extent = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = _XTC_LINE_RE.match(line)
        if not match:
            continue

        service = match.group(1)
        extent = int(match.group(2))
        total_bytes += extent
        records += 1
        if service.startswith("L1Accept"):
            l1_events += 1
            if first_l1_extent is None:
                first_l1_extent = extent
            if l1_events >= target_l1:
                break

    return {
        "bytes": total_bytes,
        "l1_events": l1_events,
        "records": records,
        "first_l1_extent": first_l1_extent,
    }


def _group_files(paths):
    grouped = defaultdict(list)
    for path in sorted(paths):
        match = _CHUNK_RE.match(path.name)
        if match is None:
            continue
        prefix = match.group("prefix")
        chunk = match.group("chunk")
        smd_suffix = match.group("smd") or ""
        key = f"{prefix}{smd_suffix}"
        grouped[key].append((chunk, path))

    ordered = []
    for key in sorted(grouped):
        ordered.append([path for _, path in sorted(grouped[key])])
    return ordered


def _copy_prefix(src: Path, dst: Path, nbytes: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    remaining = int(nbytes)
    with src.open("rb") as fin, dst.open("wb") as fout:
        while remaining > 0:
            chunk = fin.read(min(16 * 1024 * 1024, remaining))
            if not chunk:
                break
            fout.write(chunk)
            remaining -= len(chunk)


def _relpath_for_stage(src: Path, xtc_dir: Path):
    try:
        return src.relative_to(xtc_dir)
    except ValueError:
        return src.name


def _human_bytes(nbytes: int):
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{nbytes} B"


def _log(message: str):
    print(message, flush=True)


def _events_per_bd_chunk(first_l1_extent: int | None, bd_chunksize: int):
    if not first_l1_extent or first_l1_extent <= 0:
        return None
    if bd_chunksize <= 0:
        return 1
    return max(1, (int(bd_chunksize) - 1) // int(first_l1_extent))


def _plan_stream(group_paths, target_events: int, max_extra_records: int, annotate_detectors: bool):
    remaining = int(target_events)
    plan = []
    total_l1 = 0
    stream_detectors = _detnames(group_paths[0]) if (annotate_detectors and group_paths) else []
    for src in group_paths:
        if remaining <= 0:
            break
        _log(f"scan {src.name} remaining_l1={remaining}")
        prefix = _xtcreader_prefix(src, remaining, max_extra_records)
        stat_size = src.stat().st_size
        if prefix["l1_events"] < remaining:
            copy_bytes = stat_size
        else:
            copy_bytes = min(prefix["bytes"], stat_size)

        plan.append(
            {
                "source": str(src),
                "bytes": int(copy_bytes),
                "bytes_human": _human_bytes(int(copy_bytes)),
                "file_size": int(stat_size),
                "file_size_human": _human_bytes(int(stat_size)),
                "l1_events": int(prefix["l1_events"]),
                "records": int(prefix["records"]),
                "first_l1_extent": prefix["first_l1_extent"],
            }
        )
        remaining -= int(prefix["l1_events"])
        total_l1 += int(prefix["l1_events"])

    return {
        "detectors": stream_detectors,
        "files": plan,
        "requested_l1_events": int(target_events),
        "covered_l1_events": int(total_l1),
    }


def _recommended_max_events(bigdata_streams, bd_chunksize: int, recommend_min_extent: int):
    safe_limits = []
    details = []
    for stream_plan in bigdata_streams:
        covered = int(stream_plan["covered_l1_events"])
        if covered <= 0 or not stream_plan["files"]:
            continue
        first_extent = stream_plan["files"][0]["first_l1_extent"]
        if first_extent is None or int(first_extent) < int(recommend_min_extent):
            continue
        events_per_chunk = _events_per_bd_chunk(first_extent, bd_chunksize)
        if events_per_chunk is None:
            continue
        safe_limit = (covered // events_per_chunk) * events_per_chunk
        names = ",".join(d["name"] for d in stream_plan.get("detectors", [])) or Path(stream_plan["files"][0]["source"]).name
        details.append(
            {
                "stream": names,
                "covered_l1_events": covered,
                "first_l1_extent": first_extent,
                "events_per_bd_chunk": events_per_chunk,
                "recommended_max_events": safe_limit,
            }
        )
        safe_limits.append(safe_limit)

    recommended = min(safe_limits) if safe_limits else 0
    return recommended, details


def main():
    args = _build_parser().parse_args()
    xtc_dir = Path(args.xtc_dir).resolve()
    smd_dir = xtc_dir / "smalldata"
    run_tag = f"r{args.run:04d}"
    out_dir = Path(args.out_dir).resolve()

    bigdata_paths = sorted(xtc_dir.glob(f"*{run_tag}*.xtc2"))
    bigdata_paths = [p for p in bigdata_paths if p.is_file()]
    smd_paths = sorted(smd_dir.glob(f"*{run_tag}*.smd.xtc2"))

    if not bigdata_paths:
        raise SystemExit(f"No bigdata XTC2 files found under {xtc_dir} for {run_tag}.")
    if not smd_paths:
        raise SystemExit(f"No smalldata XTC2 files found under {smd_dir} for {run_tag}.")

    manifest = {
        "xtc_dir": str(xtc_dir),
        "out_dir": str(out_dir),
        "run": int(args.run),
        "events": int(args.events),
        "max_extra_records": int(args.max_extra_records),
        "bigdata": [],
        "smalldata": [],
    }

    _log(f"planning bigdata streams for run {args.run}")
    for group in _group_files(bigdata_paths):
        stream_plan = _plan_stream(group, args.events, args.max_extra_records, args.annotate_detectors)
        manifest["bigdata"].append(stream_plan)

    _log(f"planning smalldata streams for run {args.run}")
    for group in _group_files(smd_paths):
        stream_plan = _plan_stream(group, args.events, args.max_extra_records, args.annotate_detectors)
        manifest["smalldata"].append(stream_plan)

    total_stage_bytes = 0
    for section in ("bigdata", "smalldata"):
        for stream_plan in manifest[section]:
            for item in stream_plan["files"]:
                total_stage_bytes += int(item["bytes"])

    recommended_max_events, recommendation_details = _recommended_max_events(
        manifest["bigdata"], args.bd_chunksize, args.recommend_min_extent
    )
    manifest["recommended_max_events"] = int(recommended_max_events)
    manifest["recommended_max_events_details"] = recommendation_details
    manifest["bd_chunksize"] = int(args.bd_chunksize)
    manifest["recommend_min_extent"] = int(args.recommend_min_extent)

    print(f"run={args.run} events={args.events} total_stage_bytes={total_stage_bytes} ({_human_bytes(total_stage_bytes)})")
    print(
        f"recommended_max_events={recommended_max_events} "
        f"(assuming PS_BD_CHUNKSIZE={args.bd_chunksize}; use PS_BD_CHUNKSIZE=1 to read up to the nominal staged event count)"
    )
    for section in ("bigdata", "smalldata"):
        print(section)
        for stream_plan in manifest[section]:
            detectors = ",".join(f"{d['name']}:{d['dtype']}" for d in stream_plan["detectors"]) or "unknown"
            for item in stream_plan["files"]:
                print(
                    f"  {Path(item['source']).name}: copy={item['bytes']} ({item['bytes_human']}) "
                    f"l1={item['l1_events']} first_l1_extent={item['first_l1_extent']} detectors={detectors}"
                )

    manifest_path = Path(args.manifest).resolve() if args.manifest else out_dir / "stage-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")

    if args.dry_run:
        return

    for section in ("bigdata", "smalldata"):
        for stream_plan in manifest[section]:
            for item in stream_plan["files"]:
                src = Path(item["source"])
                dst = out_dir / _relpath_for_stage(src, xtc_dir)
                _copy_prefix(src, dst, int(item["bytes"]))
                print(f"staged {dst} bytes={item['bytes']}")


if __name__ == "__main__":
    main()
