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
import time
import os
import psutil

def print_memory_usage(rank, i_evt, interval=10):
    if i_evt % interval == 0:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_gb = mem_info.rss / (1024 ** 3)
        print(f"[Rank {rank}] Event {i_evt}: RSS Memory = {rss_gb:.2f} GB")

def check_mode_bits(raw, print_coords=False, max_print=20):
    # Ensure logical (unsigned) shifts even if raw came in as int16
    r = raw.astype(np.uint16, copy=False)

    modes = (r >> 14) & 0x3      # upper two bits
    adc14 = r & 0x3FFF           # lower 14 bits

    # Frame-level summary
    u, c = np.unique(modes, return_counts=True)
    hist = dict(zip(u.tolist(), c.tolist()))
    print(f"shape={r.shape}, unique modes={u.tolist()}, counts={c.tolist()}")
    print("mode histogram:", hist)
    print("any top bits set?", bool((r & 0xC000).any()))
    print("max raw value:", int(r.max()))
    print("adc14 mean/std:", float(adc14.mean()), float(adc14.std()))

    if print_coords and 0 in u:
        coords = np.argwhere(modes == 0)
        print(f"Found {coords.shape[0]} pixels with mode=0")
        for idx, (seg, row, col) in enumerate(coords[:max_print]):
            print(f"  seg={seg}, row={row}, col={col}")
        if coords.shape[0] > max_print:
            print(f"  ... (showing {max_print} of {coords.shape[0]} total)")

    return modes, adc14

def parse_args():
    parser = argparse.ArgumentParser(
        description="MPI tool to count total number of events processed in a given experiment and run."
    )
    parser.add_argument('-e', '--exp', required=True, help='Experiment name, e.g., rix100818424')
    parser.add_argument('-r', '--run', type=int, required=True, help='Run number')
    parser.add_argument('-d', '--detectors', nargs='*', default=[], help='Optional list of detector names')
    parser.add_argument('-c', '--cached_detectors', nargs='*', default=[], help='Optional list of detector names with cached pixel coords')
    parser.add_argument('--max_events', type=int, default=0, help='Maximum number of events to process (default: 0 = all)')
    parser.add_argument('--batch_size', type=int, default=1000, help='No. of events per batch (default: 1000)')
    parser.add_argument('--log_level', default='INFO', help='Log level (default: INFO)')
    parser.add_argument('--debug_detector', default=None, help='Detector name for debug prints (default: None)')
    parser.add_argument('--use_calib_cache', action='store_true',
                        help='Use cached calibration constants if available (default: False)')
    parser.add_argument('--monitor', action='store_true', help='Enable monitoring mode (default: False)')
    return parser.parse_args()

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()  # Synchronize before starting
    start = MPI.Wtime()
    t0 = time.time()

    ds = DataSource(
        exp=args.exp,
        run=args.run,
        max_events=args.max_events,
        batch_size=args.batch_size,
        log_level=args.log_level,
        detectors=args.detectors,
        use_calib_cache=args.use_calib_cache,
        cached_detectors=args.cached_detectors,
        monitor=args.monitor
    )
    run = next(ds.runs())
    det = None
    if args.debug_detector:
        det = run.Detector(args.debug_detector)
        det_name = args.debug_detector.lower()
        if rank == 0:
            print(f"Debugging detector: {args.debug_detector}")

    local_count = 0

    ti0 = time.time()
    interval = 10
    for i_evt, evt in enumerate(run.events()):
        if det:
            if det_name == 'epix10ka':
                raw = det.raw.raw(evt)  # Example access to detector data
                #check_mode_bits(raw, print_coords=True)
            elif det_name == 'jungfrau':
                img = det.raw.image(evt)
                #print_memory_usage(rank, i_evt, interval=interval)
                if i_evt % interval == 0 and i_evt > 0:
                    print(f"[Rank {rank}] Event {i_evt}: Rate: {interval/(time.time()-ti0):.1f} Hz")
                    ti0 = time.time()

        local_count += 1
    t1 = time.time()

    sendbuf = np.array([local_count], dtype="i")
    recvbuf = np.empty([size, 1], dtype="i") if rank == 0 else None
    comm.Gather(sendbuf, recvbuf, root=0)

    end = MPI.Wtime()
    if rank == 0:
        total = np.sum(recvbuf)
        n_ebnodes = int(os.environ.get('PS_EB_NODES', '1'))
        n_bdnodes = size - n_ebnodes - 1
        print(f"[{args.log_level}] {n_ebnodes=} {n_bdnodes=} Total events for {args.exp} run {args.run}: {total} rate {total/(end-start):.1f} Hz")
    else:
        print(f"[{args.log_level}] Rank {rank} processed {local_count} events in {t1-t0:.2f}s Rate {local_count/(t1-t0):.1f} Hz")

if __name__ == '__main__':
    main()
