"""Manual single-node MPI/GPU transport smoke check.

Topology: one SMD0, one EB, and at least two BD ranks, one per GPU.
MPIDataSource pins BD workers using their BD-local rank before CuPy import.
This checks event delivery, result availability, actual CUDA device placement,
and completion. Pixel correctness belongs to the pytest acceptance suite.

Exit status: 0 = PASS, 1 = failure, 2 = incomplete BD/GPU participation.
Run through run_multi_gpu_test.sh so Slurm bounds hangs and propagates failures.
"""

import argparse
from collections import Counter
import glob
import os
from pathlib import Path
import re
import socket


def summarize(reports, expected_events):
    """Report all ranks, including idle BDs; return a collective exit status."""
    failures = []
    bds = [report for report in reports if report["role"] == "bd"]
    timestamps = [ts for report in bds for ts in report["timestamps"]]
    counts = Counter(timestamps)
    duplicates = {ts: count for ts, count in counts.items() if count > 1}
    if len(timestamps) != expected_events:
        failures.append(f"total events: expected {expected_events}, got {len(timestamps)}")
    if duplicates:
        failures.append(f"duplicate timestamps: {duplicates}")
    if len(bds) < 2:
        failures.append("need at least two BD ranks")
    if any(report["timestamps"] for report in reports if report["role"] != "bd"):
        failures.append("a non-BD rank yielded events")

    devices = [(report["host"], report["gpu"]) for report in bds]
    if any(not report["gpu"] for report in bds):
        failures.append("a BD rank has no measured CUDA device identity")
    if len(set(devices)) != len(devices):
        failures.append("BD ranks share a CUDA device; this check requires one BD per GPU")

    for report in sorted(reports, key=lambda entry: entry["rank"]):
        print(
            f'rank={report["rank"]} role={report["role"]} host={report["host"]} '
            f'cuda_pci_bus={report["gpu"] or "-"} events={len(report["timestamps"])}',
            flush=True,
        )
    active = sum(bool(report["timestamps"]) for report in bds)
    print(f"Total events: {len(timestamps)} / {expected_events}", flush=True)
    print(f"Active GPU BDs: {active} / {len(bds)}", flush=True)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", flush=True)
        return 1

    print("PASS: event delivery (count and unique timestamps)", flush=True)
    if active != len(bds):
        print(
            "INCOMPLETE: not every requested BD/GPU processed events; "
            "rerun with more events or smaller batches.",
            flush=True,
        )
        return 2
    print("PASS: all requested BD/GPU workers exercised; MPI run completed", flush=True)
    return 0


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _dataset(smd_glob):
    files = sorted(glob.glob(smd_glob))
    if not files:
        raise ValueError(f"no SMD files found: {smd_glob}")
    datasets = set()
    for filename in files:
        path = Path(filename)
        match = re.fullmatch(r"(.+)-r(\d+)-s\d+-c\d+\.smd\.xtc2", path.name)
        if match is None:
            raise ValueError(f"invalid SMD filename: {filename}")
        datasets.add((match[1], int(match[2]), str(path.parent.parent.resolve())))
    if len(datasets) != 1:
        raise ValueError("SMD glob must identify exactly one experiment/run/directory")
    return datasets.pop()


def main():
    # Parse on every rank before collectives: all receive identical argv from
    # the launcher, so --help/argument errors cannot strand peers in bcast().
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-events", type=_positive_int, default=50)
    parser.add_argument("--batch-size", type=_positive_int, default=5)
    parser.add_argument("--pool-depth", type=_positive_int, default=2)
    args = parser.parse_args()

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    try:
        hosts = comm.allgather(socket.gethostname())
        if len(set(hosts)) != 1 or comm.Get_size() < 4:
            raise ValueError("requires one node with SMD0 + one EB + at least two BDs")
        if int(os.environ.get("PS_EB_NODES", "1")) != 1:
            raise ValueError("this smoke check supports PS_EB_NODES=1 only")
        if int(os.environ.get("PS_SRV_NODES", "0")) != 0:
            raise ValueError("this smoke check requires PS_SRV_NODES=0")
        if os.environ.get("PS_PARALLEL", "mpi") != "mpi":
            raise ValueError("this smoke check requires PS_PARALLEL=mpi")

        dataset = None
        if rank == 0:
            dataset = _dataset(os.environ.get(
                "PSANA_GPU_TEST_SMD_GLOB",
                "/sdf/data/lcls/ds/prj/public01/xtc/smalldata/"
                "mfx100852324-r0077*.smd.xtc2",
            ))
        exp, run_number, directory = comm.bcast(dataset, root=0)

        # Do not import CuPy before MPIDataSource has established GPU ownership.
        import psana
        from psana.psexp.mpi_ds import MPIDataSource
        from psana.psexp.node import Communicators

        if rank == 0:
            print(f"psana={psana.__file__}", flush=True)
            print(
                f"dataset={exp} run={run_number} dir={directory} "
                f"batch_size={args.batch_size} pool_depth={args.pool_depth}",
                flush=True,
            )
        ds = MPIDataSource(
            Communicators(), exp=exp, run=run_number, dir=directory,
            gpu_det="jungfrau", batch_size=args.batch_size,
            max_events=args.max_events, n_gpu_streams=args.pool_depth,
        )
        report = dict(rank=rank, role=ds.comms.node_type(), host=hosts[rank],
                      gpu=None, timestamps=[])
        if ds.is_bd():
            import cupy as cp

            report["gpu"] = cp.cuda.runtime.deviceGetPCIBusId(
                cp.cuda.runtime.getDevice()
            )

        for run in ds.runs():
            for evt in run.events():
                # Ensure delivery includes the GPU result, without copying
                # pixels or doing another calibration/benchmark check.
                evt.gpu.get("jungfrau.raw")
                report["timestamps"].append(int(evt.timestamp))

        reports = comm.gather(report, root=0)
        status = summarize(reports, args.max_events) if rank == 0 else None
        return comm.bcast(status, root=0)
    except Exception:
        # A rank-local failure must not leave peers blocked in psana or gather.
        import traceback

        print(f"[rank {rank}] smoke check failed:", flush=True)
        traceback.print_exc()
        comm.Abort(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
