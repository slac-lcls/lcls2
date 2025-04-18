#!/usr/bin/env python3

"""
simulate_smalldata_write.py

Simulates concurrent smalldata writing by linking real bigdata XTC2 files and launching
parallel `smdwriter` processes. This is useful for testing tools that read `.smd.xtc2.inprogress` files.

Features:
1. Automatically detects number of streams (s000, s001, ...) by inspecting the experiment's xtc directory.
2. Creates symlinks to bigdata xtc2 files in a user-specified output directory.
3. Launches parallel `smdwriter` commands to write `.smd.xtc2.inprogress` files into `output_path/smalldata/`.
4. Upon completion, `.inprogress` files are renamed to finalized `.smd.xtc2` files.

Usage:
    python -m psana.debugtools.simulate_smalldata_write <exp> <run> <output_path> [-m SECONDS] [-n EVENTS]

Arguments:
    exp         Experiment name, e.g. rixl1032923
    run         Run number, e.g. 22
    output_path Directory to create symlinks and write smalldata files

Options:
    -m          Sleep time per `smdwriter` batch (default: 10 seconds)
    -n          Number of events to write per stream (default: 200)

Example:
    python -m psana.debugtools.simulate_smalldata_write rixl1032923 22 /cds/home/myuser/tmp/debugtest -m 5 -n 100
"""

import os
import argparse
import subprocess
import shutil
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def detect_streams(exp, run):
    exp_prefix = exp[:3]
    run_str = f"r{run:04d}"
    xtc_dir = f"/sdf/data/lcls/ds/{exp_prefix}/{exp}/xtc"
    stream_nums = []

    if not os.path.isdir(xtc_dir):
        raise FileNotFoundError(f"XTC directory not found: {xtc_dir}")

    for fname in os.listdir(xtc_dir):
        match = re.match(rf"{exp}-{run_str}-s(\d+)-c000\.xtc2", fname)
        if match:
            stream_nums.append(int(match.group(1)))

    if not stream_nums:
        raise RuntimeError("No matching stream files found.")

    return max(stream_nums) + 1  # Streams are 0-indexed

def create_symlinks(base_path, exp, run, num_streams):
    exp_prefix = exp[:3]
    run_str = f"r{run:04d}"
    xtc_dir = f"/sdf/data/lcls/ds/{exp_prefix}/{exp}/xtc"
    os.makedirs(base_path, exist_ok=True)

    for i in range(num_streams):
        stream = f"s{str(i).zfill(3)}"
        src = f"{xtc_dir}/{exp}-{run_str}-{stream}-c000.xtc2"
        dest = f"{base_path}/{exp}-{run_str}-{stream}-c000.xtc2"
        try:
            if os.path.islink(dest) or os.path.exists(dest):
                os.unlink(dest)
            os.symlink(src, dest)
            print(f"Linked: {dest} -> {src}")
        except OSError as e:
            print(f"Failed to create symlink {dest}: {e}")

def run_smdwriter(output_dir, exp, run, stream_id, m, n):
    run_str = f"r{run:04d}"
    stream = f"s{str(stream_id).zfill(3)}"
    input_file = f"{exp}-{run_str}-{stream}-c000.xtc2"
    output_basename = f"{exp}-{run_str}-{stream}-c000.smd.xtc2"
    inprogress_file = output_basename + ".inprogress"

    input_path = os.path.join(output_dir, input_file)
    output_dir_path = os.path.join(output_dir, "smalldata")
    inprogress_path = os.path.join(output_dir_path, inprogress_file)
    final_path = os.path.join(output_dir_path, output_basename)

    cmd = [
        "smdwriter",
        "-f", input_path,
        "-o", inprogress_path,
        "-m", str(m),
        "-n", str(n)
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

    # Rename .inprogress to final .smd.xtc2
    if os.path.exists(inprogress_path):
        os.rename(inprogress_path, final_path)
        print(f"Renamed {inprogress_path} -> {final_path}")
    else:
        print(f"Warning: {inprogress_path} not found after smdwriter finished.")

def main():
    parser = argparse.ArgumentParser(description="Simulate smalldata writing with symlinks.")
    parser.add_argument("exp", help="Experiment name, e.g. rixl1032923")
    parser.add_argument("run", type=int, help="Run number, e.g. 22")
    parser.add_argument("output_path", help="Directory to create links and smalldata")
    parser.add_argument("-m", type=int, default=10, help="Sleep time for writer")
    parser.add_argument("-n", type=int, default=200, help="Number of events")
    args = parser.parse_args()

    base_path = Path(args.output_path).resolve()
    smd_path = base_path / "smalldata"

    # Detect number of streams
    num_streams = detect_streams(args.exp, args.run)
    print(f"Detected {num_streams} streams")

    # Create symlinks
    create_symlinks(base_path, args.exp, args.run, num_streams)

    # Clean and recreate smalldata output directory
    if smd_path.exists():
        shutil.rmtree(smd_path)
    smd_path.mkdir(parents=True)

    # Run smdwriters in parallel
    with ThreadPoolExecutor(max_workers=num_streams) as executor:
        for i in range(num_streams):
            executor.submit(run_smdwriter, base_path, args.exp, args.run, i, args.m, args.n)

if __name__ == "__main__":
    main()
