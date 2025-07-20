#!/usr/bin/env python3
"""
scan_daq_logs_by_run.py

Scan a set of log files for errors, segmentation faults, or
high-priority warnings (!!), and group output by the run timestamp.

Example:
    python -m psana.debugtools.scan_daq_logs_by_run ~/2025/04/24*
"""

import os
import sys
import glob
import re
import argparse
from collections import defaultdict

# --- patterns to look for ---
ERROR_PATTERNS = [
    re.compile(r'(?i)\berror\b'),
    re.compile(r'(?i)\bsegmentation fault\b'),
    re.compile(r'(?i)\bDuplicate\b'),
    re.compile(r'(?i)\bcore dumped\b'),
    re.compile(r'!{2,}'),        # double-bang (!!) warnings
]

IGNORE_PATTERNS = [
    re.compile(r'slurmstepd: error', re.IGNORECASE),
    re.compile(r'srun: error', re.IGNORECASE),
    re.compile(r'Error connecting myTrFd socket', re.IGNORECASE),
]

def find_errors_in_file(path):
    """
    Scan `path` for ERROR_PATTERNS (skipping IGNORE_PATTERNS),
    but only suppress *consecutive* duplicates of the same normalized prefix.
    """
    errors = []
    last_norm = None

    try:
        with open(path, 'r', errors='ignore') as fh:
            for lineno, line in enumerate(fh, 1):
                # 1) skip ignored noise
                if any(p.search(line) for p in IGNORE_PATTERNS):
                    continue

                # 2) check if this line is one of our errors
                if any(p.search(line) for p in ERROR_PATTERNS):
                    # normalize out the variable bits
                    norm = line
                    norm = re.sub(r'\[\d+\]', '', norm)                                  # strip [PID]
                    norm = norm.split('TimeStamp', 1)[0]                                # drop timestamp+rest
                    norm = re.sub(r'0x[0-9A-Fa-f]+(?:\.[0-9A-Fa-f]+)*', '', norm)       # remove hex+tails
                    norm = re.sub(r'[\d\.]+', '', norm)                                 # drop any leftover digits/dots
                    norm = re.sub(r'\s+', ' ', norm).strip()                            # collapse spaces

                    # if it’s *not* the same as the last one we emitted, print it
                    if norm != last_norm:
                        errors.append((lineno, line.rstrip()))
                        last_norm = norm
                    # if it *is* the same, we skip it silently
                else:
                    # any non-error line resets the “last_norm” guard
                    last_norm = None

    except Exception as e:
        errors.append(("<file-open-error>", str(e)))

    return errors

def group_by_run(files):
    """
    Group file paths by the timestamp prefix in their basename.
    Expects basenames like '24_10:26:29_host:proc.log'
    """
    groups = defaultdict(list)
    for path in files:
        base = os.path.basename(path)
        parts = base.split('_', 2)
        if len(parts) >= 2:
            run_key = "_".join(parts[:2])   # e.g. '24_10:26:29'
        else:
            run_key = "UNKNOWN"
        groups[run_key].append(path)
    return groups

def main():
    p = argparse.ArgumentParser(
        description="Scan log files for errors and group them by run timestamp."
    )
    p.add_argument('files', nargs='+',
                   help="Glob pattern(s) for log files, e.g. ~/2025/04/24*")
    args = p.parse_args()

    # expand globs
    all_files = []
    for pat in args.files:
        expanded = glob.glob(os.path.expanduser(pat))
        if not expanded:
            print(f"⚠️  No files match: {pat}", file=sys.stderr)
        all_files.extend(sorted(expanded))

    if not all_files:
        print("❌ No files to scan. Check your glob pattern.", file=sys.stderr)
        sys.exit(1)

    # group by run timestamp
    runs = group_by_run(all_files)

    # scan each group
    for run_key, paths in sorted(runs.items()):
        print(f"\n=== Run: {run_key} ===")
        any_err = False
        for path in sorted(paths):
            errs = find_errors_in_file(path)
            if not errs:
                continue
            any_err = True
            name = os.path.basename(path)
            print(f"\n  ➤ {name}")
            for lineno, text in errs:
                print(f"    {lineno:>5}: {text}")
        if not any_err:
            print("  (no errors found)")

if __name__ == '__main__':
    main()
