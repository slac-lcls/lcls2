#!/usr/bin/env python3
"""Read-only helpers for DAQ log sessions: list sessions overlapping a
window, and search/count/excerpt a session's lines — in both cases
transparently handling the mix of plain `.log` and zstd-compressed
`.log.zst` rotations that live side by side in every session directory.

This exists because that format mix is an easy thing to get wrong by
example-copying: a grep pattern written against `*.log` silently produces
false negatives against `*.log.zst` (compressed bytes don't match a text
pattern), and "which sessions overlap this window" needs real interval
arithmetic, not a same-day mtime check — see `psana-daq-logs/SKILL.md`'s
"One canonical log read" and "Session selection" sections, which this
script implements. Cite those sections rather than re-deriving this
script's behavior elsewhere.

Read-only: opens files for reading and spawns `zstd -dc` (decompress to
stdout) as a subprocess. Never writes, moves, or deletes anything, and
never invokes `zstdcat` — `zstdcat` is `zstd -dcf`, and `-f` makes it pass
non-zstd input through unchanged at exit 0, so a corrupt or mislabeled
`.log.zst` looks like a successful, empty read. `zstd -dc` fails loudly
instead.

Usage:
    daq-logs.py list <log-dir> --start <iso> --end <iso> [--component GLOB]
    daq-logs.py grep <log-dir> <session-prefix> [options]

Examples:
    # Sessions overlapping a window, latest first, with error counts
    daq-logs.py list /sdf/home/x/xppopr/daq/logs/2026/09 \\
        --start 2026-09-17T00:00:00 --end 2026-09-18T00:00:00

    # Count Critical/Error lines (RTPRIO startup noise excluded) for a session
    daq-logs.py grep /sdf/home/x/xppopr/daq/logs/2026/09 15_15:31:17 --count

    # First matching <C>/<E> line per file
    daq-logs.py grep /sdf/home/x/xppopr/daq/logs/2026/09 15_15:31:17 --first

    # Full filtered output, restricted to teb* component files, as JSON
    daq-logs.py grep /sdf/home/x/xppopr/daq/logs/2026/09 15_15:31:17 \\
        --component 'teb*' --full --json

`list` only ever considers the single `<log-dir>` passed in (one `<YYYY>/
<MM>` directory) — if the requested window is near a month boundary, run
it again against the adjacent month directory and merge by sorting on
`(YYYY, MM, prefix)`, not prefix text alone (the 11-char `DD_HH:MM:SS`
prefix carries no month/year information).
"""
import argparse
import fnmatch
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_PATTERN = r"<[EC]>"
DEFAULT_EXCLUDE = r"Inadequate RTPRIO"
PREFIX_RE = re.compile(r"^(\d{2})_(\d{2}):(\d{2}):(\d{2})_")


# ---------------------------------------------------------------------------
# shared file access
# ---------------------------------------------------------------------------


def iter_session_files(log_dir: Path, prefix: str, component_glob):
    """Yield Path objects for every file belonging to this session prefix,
    optionally restricted to a component glob. Both .log and .log.zst are
    included — this is the whole point of this script."""
    candidates = sorted(log_dir.glob(f"{prefix}_*"))
    for p in candidates:
        if not p.is_file():
            continue
        if component_glob is not None:
            tail = p.name[len(prefix) :]
            if not fnmatch.fnmatch(tail, f"*{component_glob}*") and not fnmatch.fnmatch(
                tail, component_glob
            ):
                continue
        yield p


def read_lines(path: Path):
    """Yield decoded text lines from a log file, decompressing .log.zst via
    `zstd -dc` and failing loudly (not silently-empty) on a bad archive."""
    if path.name.endswith(".zst"):
        proc = subprocess.run(["zstd", "-dc", "--", str(path)], capture_output=True)
        if proc.returncode != 0:
            sys.stderr.write(
                f"warning: zstd -dc failed on {path} (rc={proc.returncode}); "
                f"treating as UNAVAILABLE, not an empty read: "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}\n"
            )
            return
        text = proc.stdout.decode("utf-8", "replace")
        yield from text.splitlines()
    else:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\n")


def filtered_lines(path: Path, pattern: re.Pattern, exclude):
    """Apply pattern first, exclude second — filter-then-exclude, never the
    reverse. (`grep -c PATTERN | grep -v EXCLUDE` counts first and filters
    a single number second — a no-op bug, not an equivalent formulation.)"""
    for line in read_lines(path):
        if pattern.search(line) is None:
            continue
        if exclude is not None and exclude.search(line) is not None:
            continue
        yield line


def session_start(prefix: str, dir_year: int, dir_month: int):
    """Parse a DD_HH:MM:SS prefix into a UTC-naive datetime, anchored on the
    log directory's own YYYY/MM (never the invoking agent's wall clock)."""
    m = re.match(r"^(\d{2})_(\d{2}):(\d{2}):(\d{2})$", prefix)
    if not m:
        return None
    day, hh, mm, ss = (int(x) for x in m.groups())
    try:
        return datetime(dir_year, dir_month, day, hh, mm, ss)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# grep subcommand
# ---------------------------------------------------------------------------


def cmd_grep(args):
    mode = "full"
    if args.count:
        mode = "count"
    elif args.first:
        mode = "first"

    pattern = re.compile(args.pattern)
    exclude = re.compile(args.exclude) if args.exclude else None

    results = []
    total = 0
    for path in iter_session_files(args.log_dir, args.session_prefix, args.component):
        matches = list(filtered_lines(path, pattern, exclude))
        if mode == "count":
            total += len(matches)
            if matches:
                results.append({"file": path.name, "count": len(matches)})
        elif mode == "first":
            if matches:
                results.append({"file": path.name, "line": matches[0]})
        else:
            for line in matches:
                results.append({"file": path.name, "line": line})

    if args.json:
        out = {"mode": mode, "results": results}
        if mode == "count":
            out["total"] = total
        print(json.dumps(out, indent=2))
        return

    if mode == "count":
        for r in results:
            print(f"{r['count']}\t{r['file']}")
        print(f"{total}\tTOTAL")
    elif mode == "first":
        for r in results:
            print(f"{r['file']}: {r['line']}")
    else:
        for r in results:
            print(f"{r['file']}: {r['line']}")


# ---------------------------------------------------------------------------
# list subcommand
# ---------------------------------------------------------------------------


def cmd_list(args):
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    if start.tzinfo is not None:
        start = start.astimezone(timezone.utc).replace(tzinfo=None)
    if end.tzinfo is not None:
        end = end.astimezone(timezone.utc).replace(tzinfo=None)

    log_dir = args.log_dir
    try:
        dir_year = int(log_dir.parent.name)
        dir_month = int(log_dir.name)
    except ValueError:
        sys.stderr.write(
            f"warning: {log_dir} doesn't look like a <YYYY>/<MM> directory; "
            "pass the exact <YYYY>/<MM> path, e.g. .../daq/logs/2026/09\n"
        )
        dir_year, dir_month = start.year, start.month

    # group files by session prefix
    sessions = {}  # prefix -> {files: [...], mtimes: [...]}
    for p in sorted(log_dir.glob("*")):
        if not p.is_file():
            continue
        m = PREFIX_RE.match(p.name)
        if not m:
            continue  # straggler file with a non-conforming name — skip
        prefix = f"{m.group(1)}_{m.group(2)}:{m.group(3)}:{m.group(4)}"
        if args.component is not None:
            tail = p.name[len(prefix) :]
            if not fnmatch.fnmatch(tail, f"*{args.component}*") and not fnmatch.fnmatch(
                tail, args.component
            ):
                continue
        s = sessions.setdefault(prefix, {"files": [], "mtime": 0.0})
        s["files"].append(p)
        s["mtime"] = max(s["mtime"], p.stat().st_mtime)

    rows = []
    for prefix, info in sessions.items():
        s_start = session_start(prefix, dir_year, dir_month)
        if s_start is None:
            continue
        s_end = datetime.fromtimestamp(info["mtime"])
        # interval-overlap test: session_start <= requested_end AND
        # session_last_mtime >= requested_start
        if s_start <= end and s_end >= start:
            pattern = re.compile(DEFAULT_PATTERN)
            exclude = re.compile(DEFAULT_EXCLUDE)
            err_count = 0
            first_err = None
            for f in info["files"]:
                for line in filtered_lines(f, pattern, exclude):
                    err_count += 1
                    if first_err is None:
                        first_err = line
            rows.append(
                {
                    "prefix": prefix,
                    "start": s_start.isoformat(),
                    "last_mtime": s_end.isoformat(),
                    "lifetime_minutes": round((s_end - s_start).total_seconds() / 60, 1),
                    "file_count": len(info["files"]),
                    "error_count": err_count,
                    "first_error": first_err,
                    "spans_outside_dir_date": s_start.day != int(prefix[:2]),
                }
            )

    rows.sort(key=lambda r: (dir_year, dir_month, r["prefix"]), reverse=True)

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    if not rows:
        print(f"No sessions in {log_dir} overlap {args.start} .. {args.end}")
        return
    print(f"Sessions overlapping {args.start} .. {args.end} (latest first):")
    for i, r in enumerate(rows, 1):
        err = f"{r['error_count']} errors"
        excerpt = f"  {r['first_error']}" if r["first_error"] else ""
        print(
            f"  {i}. {r['prefix']}   {r['lifetime_minutes']}m   {err}{excerpt}"
        )


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grep", help="search/count/excerpt one session")
    g.add_argument("log_dir", type=Path)
    g.add_argument("session_prefix")
    g.add_argument("--component", default=None)
    g.add_argument("--pattern", default=DEFAULT_PATTERN)
    g.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    g.add_argument("--count", action="store_true")
    g.add_argument("--first", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--json", action="store_true")
    g.set_defaults(func=cmd_grep)

    l = sub.add_parser("list", help="list sessions overlapping a window")
    l.add_argument("log_dir", type=Path)
    l.add_argument("--start", required=True, help="ISO 8601, e.g. 2026-09-17T00:00:00")
    l.add_argument("--end", required=True, help="ISO 8601, e.g. 2026-09-18T00:00:00")
    l.add_argument("--component", default=None)
    l.add_argument("--json", action="store_true")
    l.set_defaults(func=cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
