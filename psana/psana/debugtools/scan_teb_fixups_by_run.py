#!/usr/bin/env python3
"""
scan_teb_fixups_by_run.py

Find every TEB “Fixup L1Accept, <KEY>, … source (<SRC>)” warning, then
in the corresponding DAQ log locate the closest pulse ID match to
<KEY> (using pulse counts at 929 kHz), and print:
  - Run duration (from BeginRun to EndRun)
  - If no EndRun: mark as crashed
  - For each Fixup: pulse ID, match details, offsets from start/end, and context

Usage:
    python -m psana.debugtools.scan_teb_fixups_by_run <teb_glob1> [<teb_glob2> ...]

Example:
    python -m psana.debugtools.scan_teb_fixups_by_run ~/2025/04/24*:teb*.log
"""
import os
import sys
import glob
import re
import argparse
from collections import defaultdict

# Patterns
FIXUP_RE = re.compile(
    r'<W>\s+Fixup\s+L1Accept,\s*'
    r'(?P<key>[0-9A-Fa-f]+),'                # pulse ID in hex
    r'.*?source\s+\d+\s*\((?P<src>[^)]+)\)',  # name in parens
    re.IGNORECASE
)
PARENS_HEX = re.compile(r'\(([0-9A-Fa-f]+)\)')
BEGIN_RE  = re.compile(r'BeginRun.*?\(([0-9A-Fa-f]+)\)', re.IGNORECASE)
END_RE    = re.compile(r'EndRun.*?\(([0-9A-Fa-f]+)\)', re.IGNORECASE)
PULSE_RATE = 929_000  # pulses per second


def group_by_run(files):
    runs = defaultdict(list)
    for p in files:
        base = os.path.basename(p)
        parts = base.split('_', 2)
        run = "_".join(parts[:2]) if len(parts) >= 2 else "UNKNOWN"
        runs[run].append(p)
    return runs


def find_closest_pulse(path, search_key):
    """
    Scan `path` for all (hex) pulse IDs, compute their difference
    from search_key in pulses and seconds, and pick the closest one.
    Returns (best_line_idx, diff_pulses, diff_seconds, all_lines).
    """
    target = int(search_key, 16)
    best = None  # (diff_pulses, line_idx)
    lines = []

    try:
        with open(path, 'r', errors='ignore') as fh:
            for idx, raw in enumerate(fh):
                lines.append(raw.rstrip())
                for m in PARENS_HEX.finditer(raw):
                    val = int(m.group(1), 16)
                    diff = abs(val - target)
                    if best is None or diff < best[0]:
                        best = (diff, idx)
    except Exception:
        return None, None, None, lines

    if best is None:
        return None, None, None, lines
    diff_pulses, line_idx = best
    diff_seconds = diff_pulses / PULSE_RATE
    return line_idx, diff_pulses, diff_seconds, lines


def print_context(lines, idx, before=1, after=7):
    start = max(0, idx - before)
    end   = min(len(lines), idx + after + 1)
    for i in range(start, end):
        prefix = "  ->" if i == idx else "    "
        print(f"{prefix} {i+1:5}: {lines[i]}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan TEB Fixups and report run duration and pulse offsets."
    )
    parser.add_argument('globs', nargs='+', help="Glob patterns for your TEB logs")
    args = parser.parse_args()

    teb_files = sorted(sum((glob.glob(os.path.expanduser(g)) for g in args.globs), []))
    if not teb_files:
        print("❌ No TEB files found. Check your globs!", file=sys.stderr)
        sys.exit(1)

    runs = group_by_run(teb_files)
    for run, tebs in sorted(runs.items()):
        # Determine run boundaries
        begin_pulse = None
        end_pulse   = None
        # scan for BeginRun and EndRun
        for teb in sorted(tebs):
            with open(teb, 'r', errors='ignore') as fh:
                for raw in fh:
                    if begin_pulse is None:
                        mb = BEGIN_RE.search(raw)
                        if mb:
                            begin_pulse = int(mb.group(1), 16)
                    if end_pulse is None:
                        me = END_RE.search(raw)
                        if me:
                            end_pulse = int(me.group(1), 16)
        # print run header
        if begin_pulse is None:
            print(f"\n=== Run: {run} (no BeginRun found) ===")
        elif end_pulse is None:
            print(f"\n=== Run: {run} (crashed) ===")
        else:
            run_pulses = end_pulse - begin_pulse
            run_secs   = run_pulses / PULSE_RATE
            print(f"\n=== Run: {run} (duration: {run_secs:.6f} s) ===")

        # Scan for Fixups
        for teb in sorted(tebs):
            with open(teb, 'r', errors='ignore') as fh:
                for lineno, raw in enumerate(fh, 1):
                    m = FIXUP_RE.search(raw)
                    if not m:
                        continue
                    key, src = m.group('key'), m.group('src')
                    print(f"\n[{os.path.basename(teb)}:{lineno}] Fixup pulse_id={key}, source={src}")

                    # offsets from start
                    if begin_pulse is not None:
                        key_int = int(key, 16)
                        off_start_p = key_int - begin_pulse
                        off_start_s = off_start_p / PULSE_RATE
                        print(f"  -> offset from start: {off_start_p} pulses ({off_start_s:.6f} s)")
                        # only print offset to end if Run ended cleanly
                        if end_pulse is not None:
                            off_end_p = end_pulse - key_int
                            off_end_s = off_end_p / PULSE_RATE
                            print(f"  -> offset to end:   {off_end_p} pulses ({off_end_s:.6f} s)")

                    # locate DAQ log
                    dirpath = os.path.dirname(teb)
                    prefix  = "_".join(os.path.basename(teb).split("_", 2)[:2])
                    pattern = os.path.join(dirpath, f"{prefix}_*:{src}.log")
                    cands   = sorted(glob.glob(pattern))
                    if not cands:
                        print(f"  !! No DAQ log found for `{src}` (tried {pattern})")
                        continue
                    target = cands[0]
                    print(f"  -> searching {os.path.basename(target)} for closest pulse match…")

                    best_idx, diff_p, diff_s, all_lines = find_closest_pulse(target, key)
                    if best_idx is None:
                        print("  !! No pulse ID entries found in DAQ log.")
                    else:
                        print(f"  -> best match at line {best_idx+1}: delta={diff_p} pulses ({diff_s:.6f} s)")
                        print("  -> context around best match:")
                        print_context(all_lines, best_idx)
    print()

if __name__ == '__main__':
    main()
