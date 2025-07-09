"""
elog_error_scanner.py

This tool automates scanning of Slurm output files for runtime errors that occurred
during experiment data acquisition. It identifies relevant logs based on the creation
timestamps of base HDF5 run files in the smalldata directory of a specified experiment.

Functionality:
- Determines the appropriate experiment path based on experiment name (e.g. 'rixl1032923')
- Lists all base Run*.h5 files (excluding _part files) and collects their modification timestamps
- Matches each timestamp to Slurm output files in the corresponding user's scratch directory
- Scans matching Slurm files for lines containing error keywords such as 'error', 'exception', 'traceback'
- Extracts and annotates logs with experiment name, run number, and modification time for context

Usage:
    python -m psana.debugtools.elog_error_scanner <experiment_name>

Example:
    python -m psana.debugtools.elog_error_scanner rixl1032923

Dependencies:
    - Python 3.6+
    - Access to LCLS file systems (/sdf/data/lcls/ds/, /sdf/scratch/users/)

Author:
    LCLS DAQ Runtime Debugging Toolkit
"""

import re
from datetime import datetime
from pathlib import Path
import subprocess

# Patterns for matching error lines in slurm logs
ERROR_PATTERNS = [r'\berror\b', r'\bexception\b', r'\btraceback\b']
# Time format used in `ls -l`-like output
TIME_FMT = "%b %d %H:%M"

def get_file_listing_from_exp(exp_name):
    """
    Given an experiment name like 'rixl1032923', returns the output of `ls -lt` as list of lines.
    """
    inst = exp_name[:3]
    smd_path = f"/sdf/data/lcls/ds/{inst}/{exp_name}/hdf5/smalldata"
    try:
        result = subprocess.run(["ls", "-lt", smd_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Failed to list files in: {smd_path}")
        print(e.stderr)
        return []

def extract_exp_and_run(file_path):
    """Extract exp and run from the log file if available."""
    exp, run = None, None
    pattern = re.compile(r"Instantiated data source with arguments:.*['\"]exp['\"]:\s*['\"](\w+)['\"].*['\"]run['\"]:\s*(\d+)")
    try:
        with open(file_path, "r", errors="ignore") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    exp, run = match.groups()
                    break
    except Exception:
        pass
    return exp, run

def parse_input_lines(lines):
    """
    Extract timestamps for each user based on base Run*.h5 files (excluding _part files).
    Returns a dict: {username: [datetime1, datetime2, ...]}
    """
    times_by_user = {}
    now = datetime.now()
    file_pattern = re.compile(r'Run\d+\.h5$')  # Only base HDF5 files

    for line in lines:
        parts = line.split()
        if len(parts) < 9:
            continue

        filename = parts[-1]
        if not file_pattern.search(filename):
            continue

        username = parts[2]
        date_str = ' '.join(parts[5:8])
        try:
            if len(parts[6]) == 1:
                date_str = f"{parts[5]} 0{parts[6]} {parts[7]}"
            date = datetime.strptime(date_str, TIME_FMT)
            date = date.replace(year=now.year)
            if date > now:
                date = date.replace(year=now.year - 1)

            times_by_user.setdefault(username, []).append(date)
        except ValueError:
            continue

    return times_by_user

def find_log_files(user, mod_times, margin_minutes=10):
    """Find log files in the user's scratch dir within ±margin of any reference time."""
    log_files = []
    log_dir = Path(f"/sdf/scratch/users/{user[0]}/{user}")
    if not log_dir.exists():
        print(f"User directory not found: {log_dir}")
        return []

    for path in log_dir.glob("slurm-*.out"):
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        for ref_time in mod_times:
            if abs((mtime - ref_time).total_seconds()) <= margin_minutes * 60:
                log_files.append(path)
                break
    return log_files

def scan_file_for_errors(file_path):
    """Scan a single log file for error-related patterns."""
    found = []
    try:
        with open(file_path, "r", errors="ignore") as f:
            for lineno, line in enumerate(f, 1):
                for pattern in ERROR_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        found.append((lineno, line.strip()))
                        break
    except Exception as e:
        found.append((0, f"Could not read file: {e}"))
    return found

def scan_logs(exp_name):
    """Main scan routine for all users found in the input file."""
    lines = get_file_listing_from_exp(exp_name)

    times_by_user = parse_input_lines(lines)
    if not times_by_user:
        raise ValueError("No valid Run*.h5 files found in input.")

    results = {}
    for user, mod_times in times_by_user.items():
        print(f"\n==> Scanning logs for user: {user} ({len(mod_times)} timestamps)")
        log_files = find_log_files(user, mod_times)
        for log in log_files:
            errors = scan_file_for_errors(log)
            if errors:
                exp, run = extract_exp_and_run(log)
                results[str(log)] = (log.stat().st_mtime, exp, run, errors)
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scan slurm logs for experiment runtime errors.")
    parser.add_argument("experiment", help="Experiment name (e.g. rixl1032923)")
    args = parser.parse_args()

    results = scan_logs(args.experiment)
    if not results:
        print("No errors found in any matched slurm log files.")
    else:
        for log_path_str, (mtime_ts, exp, run, errors) in results.items():
            try:
                mod_time_str = datetime.fromtimestamp(mtime_ts).strftime("%Y-%m-%d %H:%M")
            except Exception:
                mod_time_str = "unknown"
            tag = f"(modified: {mod_time_str})"
            if exp and run:
                tag += f"  [exp: {exp}, run: {run}]"
            print(f"==> {log_path_str} {tag}")
            for lineno, msg in errors:
                print(f"    Line {lineno}: {msg}")
            print()

if __name__ == "__main__":
    main()
