#!/usr/bin/env python
"""
ds_count_events.py

A simple MPI-enabled script for counting total events processed by all ranks
for a given experiment and run using psana2.

This is useful for verifying data integrity, debugging incomplete processing,
and sanity-checking total event counts across distributed ranks.

---

Example usage:

    PS_EB_NODES=12 mpirun -n 181 --hostfile slurm_host_test \
        python ds_count_events.py \
        --exp rix100818424 \
        --run 52 \
        --detectors q_atmopal rix_fim0 \
        --max_events 10000 \
        --log_level INFO

Arguments:
    -e, --exp         Experiment name (e.g., rix100818424)
    -r, --run         Run number (e.g., 52)
    -d, --detectors   (Optional) List of detectors to configure psana with
    --max_events      (Optional) Maximum number of events to process per rank (default: 0 = all)
    --log_level       (Optional) psana log level (default: INFO)

The script prints the total number of events processed across all MPI ranks.

Note:
    - Setting max_events > 0 can be helpful for quick tests or debugging.
    - The example runs on 4 nodes with slots=1 for the fist node and slots=60
      for the other three nodes.
"""

import argparse
from psana import DataSource
import numpy as np
from mpi4py import MPI

def parse_args():
    parser = argparse.ArgumentParser(
        description="MPI tool to count total number of events processed in a given experiment and run."
    )
    parser.add_argument('-e', '--exp', required=True, help='Experiment name, e.g., rix100818424')
    parser.add_argument('-r', '--run', type=int, required=True, help='Run number')
    parser.add_argument('-d', '--detectors', nargs='*', default=[], help='Optional list of detector names')
    parser.add_argument('--max_events', type=int, default=0, help='Maximum number of events to process (default: 0 = all)')
    parser.add_argument('--log_level', default='INFO', help='Log level (default: INFO)')
    return parser.parse_args()

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ds = DataSource(
        exp=args.exp,
        run=args.run,
        max_events=args.max_events,
        log_level=args.log_level,
        detectors=args.detectors
    )
    run = next(ds.runs())

    local_count = 0
    for i_evt, evt in enumerate(run.events()):
        local_count += 1

    sendbuf = np.array([local_count], dtype="i")
    recvbuf = np.empty([size, 1], dtype="i") if rank == 0 else None
    comm.Gather(sendbuf, recvbuf, root=0)

    if rank == 0:
        total = np.sum(recvbuf)
        print(f"[{args.log_level}] Total events for {args.exp} run {args.run}: {total}")

if __name__ == '__main__':
    main()
