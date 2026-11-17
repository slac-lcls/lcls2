"""Multi-run MPI: a DataSource given run=[14, 15] must deliver every event
of both runs on the BD ranks and every rank must return (#209).

Run 14 comes from setup_input_files; run 15 is written here the same way.
xtcwriter stamps run number 14 into every BeginRun, so runs are identified
by their order in ds.runs(), not by run.runnum.
"""
import os
import subprocess

import numpy as np
from mpi4py import MPI

from psana import DataSource

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
N_EVENTS = 10


def write_run15(xtc_dir):
    smd_dir = os.path.join(xtc_dir, "smalldata")
    os.makedirs(smd_dir, exist_ok=True)
    for i in range(2):
        fname = f"xpptut15-r0015-s{i:03d}-c000.xtc2"
        sfile = os.path.join(xtc_dir, fname)
        if os.path.exists(sfile):
            continue
        subprocess.check_call(
            ["xtcwriter", "-f", sfile, "-t", "-n", str(N_EVENTS), "-s", str(i * 2), "-e", "4", "-m", "1"]
        )
        subprocess.check_call(
            ["smdwriter", "-f", sfile, "-o", os.path.join(smd_dir, fname.replace(".xtc2", ".smd.xtc2"))]
        )


def run_test_multirun():
    xtc_dir = os.path.join(os.environ.get("TEST_XTC_DIR", os.getcwd()), ".tmp")
    if rank == 0:
        write_run15(xtc_dir)
    comm.Barrier()

    ds = DataSource(exp="xpptut15", run=[14, 15], dir=xtc_dir, batch_size=1)
    counts = []
    for run in ds.runs():
        run.Detector("xppcspad")
        counts.append(sum(1 for _ in run.events()))

    assert len(counts) == 2, f"rank {rank} saw {len(counts)} runs"
    totals = comm.allreduce(np.array(counts), op=MPI.SUM).tolist()
    if rank == 0:
        print(f"events per run over all ranks: {totals}")
    assert totals == [N_EVENTS, N_EVENTS], totals


if __name__ == "__main__":
    run_test_multirun()
