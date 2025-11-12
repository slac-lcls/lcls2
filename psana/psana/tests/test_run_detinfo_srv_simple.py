#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal SRV/NullRun repro: touch run.detinfo on every rank.

To run (example):
  mpirun -n 3 python test_run_detinfo_srv_simple.py
"""

try:
    from mpi4py import MPI
    mpi_available = True
except ImportError:
    mpi_available = False

from psana import DataSource
from setup_input_files import setup_input_files

def test_run_detinfo_srv_simple(tmp_path):
    xtc_dir = prepare_xtc_dir(tmp_path)

    # Keep the script metadata-only (no event loop needed)
    ds = DataSource(exp='xpptut15', run=14, dir=str(xtc_dir), log_level='DEBUG')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    for run in ds.runs():
        print(f"Rank {rank} {run=}")
        _ = run.Detector("xppcspad")

        # Check if detector is available in this run using detinfo
        available_detectors = {}
        for (det_name, det_type), methods in run.detinfo.items():
            if det_name not in available_detectors:
                available_detectors[det_name] = []
            available_detectors[det_name].append((det_type, methods))

        if not available_detectors:
            print(f"Rank {rank} missing {available_detectors=}")

def prepare_xtc_dir(tmp_path):
    comm = MPI.COMM_WORLD if mpi_available else None
    rank = comm.Get_rank() if comm else 0

    if rank == 0:
        xtc_dir = setup_input_files(tmp_path)
    else:
        xtc_dir = tmp_path  # Other ranks will use the same path without setup

    # Barrier to ensure rank 0 completes setup before others proceed
    if comm:
        comm.Barrier()

    return xtc_dir

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_run_detinfo_srv_simple(tmp_path)
