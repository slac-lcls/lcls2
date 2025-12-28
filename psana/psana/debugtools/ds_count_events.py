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
try:
    from psana.psexp import parallel_pread as _parallel_pread
except ImportError:
    _parallel_pread = None


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
    parser.add_argument('--log_file', help='Path to log file for DataSource (optional)')
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
            log_file=args.log_file,
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
            log_file=args.log_file,
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
    n_ebnodes = int(os.environ.get('PS_EB_NODES', '1'))
    if ps_srv_nodes > 0:
        smd = ds.smalldata(batch_size=args.batch_size)
        if rank == 0:
            print(f"[DEBUG] Exercising smalldata path with PS_SRV_NODES={ps_srv_nodes}")

    if _parallel_pread is not None:
        _parallel_pread.reset_parallel_pread_stats()

    run = next(ds.runs())

    det = run.Detector(args.debug_detector) if args.debug_detector else None
    if rank == 0 and det:
        print(f"Debugging detector: {args.debug_detector}")

    local_count = 0
    event_loop_start = time.time()
    ti0 = event_loop_start
    last_pread_seconds = 0.0
    last_pread_bytes = 0
    last_pread_calls = 0
    interval = 1000

    det_accessed = False
    for i_evt, evt in enumerate(run.events()):
        if det and args.debug_detector.lower() == 'epix10ka':
            _ = det.raw.raw(evt)
            det_accessed = True
        elif det and args.debug_detector.lower() == 'jungfrau':
            _ = det.raw.raw(evt)
            det_accessed = True
        elif det and args.debug_detector.lower() == 'dream_hsd_lmcp':
            _ = det.raw.peaks(evt)  
            _ = det.raw.waveforms(evt) 
            _ = det.raw.padded(evt) 
            det_accessed = True

        if smd:
            smd.event(evt, mydata=42.0)

        if i_evt % interval == 0 and i_evt > 0:
            now = time.time()
            interval_time = now - ti0
            rate = interval / interval_time if interval_time > 0 else 0.0
            if _parallel_pread is not None:
                pread_sec, pread_bytes, pread_calls = _parallel_pread.parallel_pread_stats()
                delta_bytes = pread_bytes - last_pread_bytes
                delta_sec = pread_sec - last_pread_seconds
                delta_calls = pread_calls - last_pread_calls
                io_rate = (delta_bytes / delta_sec) if delta_sec > 1e-12 else 0.0
                io_msg = (
                    f" IO={io_rate / (1024 * 1024):.2f} MiB/s "
                    f"bytes={delta_bytes / (1024 * 1024):.2f} MiB calls={delta_calls}"
                )
                last_pread_seconds = pread_sec
                last_pread_bytes = pread_bytes
                last_pread_calls = pread_calls
            else:
                io_msg = ""
            print(
                f"[Rank {rank}] Event {i_evt}: Rate = {rate:.1f} Hz "
                f"Interval={interval_time:.2f}s {det_accessed=} {io_msg}"
            )
            ti0 = now

        local_count += 1

    if smd:
        smd.done()

    event_loop_end = time.time()
    loop_elapsed = max(event_loop_end - event_loop_start, 1e-12)
    load_time = event_loop_start - t0
    if _parallel_pread is not None:
        pread_sec, pread_bytes, pread_calls = _parallel_pread.parallel_pread_stats()
    else:
        pread_sec = pread_bytes = pread_calls = 0.0

    sendbuf = np.array([local_count], dtype="i")
    recvbuf = np.empty([size, 1], dtype="i") if rank == 0 else None
    comm.Gather(sendbuf, recvbuf, root=0)
    loop_elapsed_max = comm.reduce(loop_elapsed, op=MPI.MAX, root=0)
    load_time_max = comm.reduce(load_time, op=MPI.MAX, root=0)
    end = MPI.Wtime()

    total_pread_bytes = comm.reduce(pread_bytes, op=MPI.SUM, root=0)
    total_pread_sec = comm.reduce(pread_sec, op=MPI.SUM, root=0)
    total_pread_calls = comm.reduce(pread_calls, op=MPI.SUM, root=0)

    if rank == 0:
        total = np.sum(recvbuf)
        n_bdnodes = size - n_ebnodes - 1
        elapsed = end - start
        loop_elapsed_print = loop_elapsed_max if loop_elapsed_max else elapsed
        total_rate = total / loop_elapsed_print if loop_elapsed_print > 0 else 0.0
        print(
            f"[{args.log_level}] {n_ebnodes=} {n_bdnodes=} "
            f"Load time={load_time_max:.2f}s Loop time={loop_elapsed_print:.2f}s "
            f"Total events: {total} Rate={total_rate:.1f} Hz"
        )
        if total_pread_bytes > 0 and total_pread_sec > 0:
            agg_io_rate = (total_pread_bytes / (1024 * 1024)) / total_pread_sec
            print(
                f"[{args.log_level}] TOTAL_IO bytes={total_pread_bytes / (1024 * 1024):.2f} MiB "
                f"time={total_pread_sec:.2f}s rate={agg_io_rate:.2f} MiB/s "
                f"calls={int(total_pread_calls)}"
            )
    else:
        first_bd_rank = n_ebnodes + 1
        last_bd_rank = size - ps_srv_nodes
        if first_bd_rank <= rank < last_bd_rank:
            rate = local_count / loop_elapsed if loop_elapsed > 0 else 0.0
            print(
                f"[{args.log_level}] Rank {rank} processed {local_count} events "
                f"Rate={rate:.1f} Hz Load time={load_time:.2f}s Loop time={loop_elapsed:.2f}s"
            )
            if pread_calls > 0 and pread_sec > 0:
                io_rate = (pread_bytes / (1024 * 1024)) / pread_sec
                print(
                    f"[{args.log_level}] Rank {rank} IO bytes={pread_bytes / (1024 * 1024):.2f} MiB "
                    f"time={pread_sec:.2f}s rate={io_rate:.2f} MiB/s calls={int(pread_calls)}"
                )


if __name__ == '__main__':
    main()
