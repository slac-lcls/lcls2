#!/usr/bin/env python
"""
ds_count_events.py

MPI-enabled script for counting total events processed by all ranks
for a given experiment/run or explicit list of XTC2 files using psana2.

---

Example usage:

    # Using experiment/run
    mpirun -n 16 python ds_count_events.py \
        --exp rix100818424 --run 52

    # Using explicit files (only support single core)
    python ds_count_events.py \
        --xtc_files /path/to/run52.xtc2 /path/to/run52.smd.xtc2

Other options:
    --detectors q_atmopal rix_fim0
    --max_events 10000
    --log_level INFO
    --dir /sdf/data/lcls/ds/rix/rix100818424/xtc
"""

import argparse
import os
import time
import numpy as np
import psutil
from mpi4py import MPI
from psana import DataSource


def print_memory_usage(rank, i_evt, interval=10):
    if i_evt % interval == 0:
        process = psutil.Process(os.getpid())
        rss_gb = process.memory_info().rss / (1024 ** 3)
        print(f"[Rank {rank}] Event {i_evt}: RSS Memory = {rss_gb:.2f} GB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count total number of events from an experiment/run or explicit XTC2 files."
    )
    parser.add_argument('-e', '--exp', help='Experiment name, e.g. rix100818424')
    parser.add_argument('-r', '--run', type=int, help='Run number, e.g. 52')
    parser.add_argument('--xtc_files', nargs='+', help='Explicit list of XTC2 files to process')
    parser.add_argument('--dir', help='Path to directory containing XTC2 files')
    parser.add_argument('-d', '--detectors', nargs='*', default=[], help='List of detector names')
    parser.add_argument('-c', '--cached_detectors', nargs='*', default=[], help='Detectors with cached pixel coords')
    parser.add_argument('--max_events', type=int, default=0, help='Max number of events per rank (0=all)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Events per batch (default: 1000)')
    parser.add_argument('--log_level', default='INFO', help='Log level (default: INFO)')
    parser.add_argument('--debug_detector', default=None, help='Detector name for debug prints')
    parser.add_argument('--use_calib_cache', action='store_true', help='Use cached calibration constants')
    parser.add_argument('--monitor', action='store_true', help='Enable monitoring mode')
    parser.add_argument('--live', action='store_true', help='Enable live mode')
    return parser.parse_args()


def create_datasource(args, rank):
    """Construct DataSource depending on whether xtc_files are provided."""
    if args.xtc_files:
        if rank == 0:
            print(f"Using explicit XTC2 files ({len(args.xtc_files)}):")
            for f in args.xtc_files:
                print(f"  {f}")
        ds = DataSource(
            files=args.xtc_files,
            max_events=args.max_events,
            batch_size=args.batch_size,
            log_level=args.log_level,
            detectors=args.detectors,
            use_calib_cache=args.use_calib_cache,
            cached_detectors=args.cached_detectors,
            monitor=args.monitor,
        )
    else:
        if args.exp is None or args.run is None:
            raise ValueError("Either --xtc_files or both --exp and --run must be provided.")

        # Determine xtc directory path
        if args.dir:
            dir_path = args.dir
        elif args.live:
            # Construct default FFB path for live mode
            dir_path = f"/sdf/data/lcls/drpsrcf/ffb/{args.exp[:3]}/{args.exp}/xtc"
        else:
            dir_path = None

        if rank == 0:
            print(f"Using experiment={args.exp}, run={args.run}, dir={dir_path}")

        ds = DataSource(
            exp=args.exp,
            run=args.run,
            max_events=args.max_events,
            batch_size=args.batch_size,
            log_level=args.log_level,
            detectors=args.detectors,
            use_calib_cache=args.use_calib_cache,
            cached_detectors=args.cached_detectors,
            monitor=args.monitor,
            live=args.live,
            dir=dir_path,
        )
    return ds


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()
    start = MPI.Wtime()
    t0 = time.time()

    ds = create_datasource(args, rank)

    smd = None
    ps_srv_nodes = int(os.environ.get('PS_SRV_NODES', '0'))
    if ps_srv_nodes > 0:
        smd = ds.smalldata(batch_size=args.batch_size)
        if rank == 0:
            print(f"[DEBUG] Exercising smalldata path with PS_SRV_NODES={ps_srv_nodes}")

    run = next(ds.runs())

    det = run.Detector(args.debug_detector) if args.debug_detector else None
    if rank == 0 and det:
        print(f"Debugging detector: {args.debug_detector}")

    local_count = 0
    ti0 = time.time()
    interval = 1000

    det_accessed = False
    for i_evt, evt in enumerate(run.events()):
        if det and args.debug_detector.lower() == 'epix10ka':
            _ = det.raw.raw(evt)
        elif det and args.debug_detector.lower() == 'jungfrau':
            _ = det.raw.image(evt)
        elif det and args.debug_detector.lower() == 'dream_hsd_lmcp':
            _ = det.raw.peaks(evt)  
            _ = det.raw.waveforms(evt) 
            _ = det.raw.padded(evt) 
            det_accessed = True

        if smd:
            smd.event(evt, mydata=42.0)

        if i_evt % interval == 0 and i_evt > 0:
            rate = interval / (time.time() - ti0)
            print(f"[Rank {rank}] Event {i_evt}: Rate = {rate:.1f} Hz {det_accessed=}")
            ti0 = time.time()

        local_count += 1

    if smd:
        smd.done()

    sendbuf = np.array([local_count], dtype="i")
    recvbuf = np.empty([size, 1], dtype="i") if rank == 0 else None
    comm.Gather(sendbuf, recvbuf, root=0)
    end = MPI.Wtime()

    if rank == 0:
        total = np.sum(recvbuf)
        n_ebnodes = int(os.environ.get('PS_EB_NODES', '1'))
        n_bdnodes = size - n_ebnodes - 1
        print(f"[{args.log_level}] {n_ebnodes=} {n_bdnodes=} Total events: {total} Rate={total/(end-start):.1f} Hz")
    else:
        rate = local_count / (time.time() - t0)
        print(f"[{args.log_level}] Rank {rank} processed {local_count} events Rate={rate:.1f} Hz")


if __name__ == '__main__':
    main()
