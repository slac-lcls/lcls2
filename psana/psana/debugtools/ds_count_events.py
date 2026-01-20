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
    parser.add_argument('--print_interval', type=int, default=1000,
                        help='Interval (events) for per-rank progress prints (default: 1000)')
    parser.add_argument('--log_level', default='INFO', help='Log level (default: INFO)')
    parser.add_argument('--debug_detector', default=None, help='Detector name for debug prints')
    parser.add_argument('--calib', action='store_true', help='Use calib data for jungfrau debug detector')
    parser.add_argument('--use_calib_cache', action='store_true', help='Use cached calibration constants')
    parser.add_argument('--skip_calib_load', nargs='+', default=None,
                        help="Detectors to skip calibration loading, or 'all'")
    parser.add_argument('--monitor', action='store_true', help='Enable monitoring mode')
    parser.add_argument('--live', action='store_true', help='Enable live mode')
    parser.add_argument('--log_file', help='Path to log file for DataSource (optional)')
    parser.add_argument('--show_rank_stats', action='store_true',
                        help='Print per-rank statistics (default: only rank 0 summary)')
    parser.add_argument('--test_pixel_coords', action='store_true',
                        help='Call det.raw._pixel_coords() once and report timing')
    args = parser.parse_args()
    if args.skip_calib_load is not None:
        if any(det.lower() == "all" for det in args.skip_calib_load):
            args.skip_calib_load = "all"
    return args


def create_datasource(args, rank):
    """Construct DataSource depending on whether xtc_files are provided."""
    common_kwargs = dict(
        max_events=args.max_events,
        batch_size=args.batch_size,
        log_level=args.log_level,
        detectors=args.detectors,
        use_calib_cache=args.use_calib_cache,
        cached_detectors=args.cached_detectors,
        monitor=args.monitor,
        log_file=args.log_file,
    )
    if args.skip_calib_load is not None:
        common_kwargs["skip_calib_load"] = args.skip_calib_load

    if args.xtc_files:
        if rank == 0:
            print(f"Using explicit XTC2 files ({len(args.xtc_files)}):")
            for f in args.xtc_files:
                print(f"  {f}")
        ds = DataSource(
            files=args.xtc_files,
            **common_kwargs,
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
            **common_kwargs,
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
        print(f"[INFO] Debugging detector: {args.debug_detector}")
    if det and args.test_pixel_coords:
        det_t0 = time.perf_counter()
        coords = det.raw._pixel_coords()
        det_dt = time.perf_counter() - det_t0
        shape_msg = "None"
        if coords is not None:
            try:
                shape_msg = ", ".join(str(a.shape) for a in coords)
            except Exception:
                shape_msg = "unavailable"
        print(f"[Rank {rank}] det.raw._pixel_coords() time={det_dt:.6f}s shapes={shape_msg}")

    local_count = 0
    event_loop_start = time.time()
    ti0 = event_loop_start
    last_pread_seconds = 0.0
    last_pread_bytes = 0
    last_pread_calls = 0
    interval = max(1, args.print_interval)

    det_accessed = False
    det_call_seconds = 0.0
    det_call_count = 0
    det_seg_seconds = 0.0
    det_stack_seconds = 0.0
    for i_evt, evt in enumerate(run.events()):
        if det:
            evt._det_raw_timing = {'segments': 0.0, 'stack': 0.0}
        if det and args.debug_detector.lower() == 'epix10ka':
            det_t0 = time.perf_counter()
            _ = det.raw.raw(evt)
            det_call_seconds += time.perf_counter() - det_t0
            det_call_count += 1
            det_accessed = True
        elif det and args.debug_detector.lower() == 'jungfrau':
            det_t0 = time.perf_counter()
            if args.calib:
                _ = det.raw.calib(evt)
            else:
                _ = det.raw.raw(evt)
            det_call_seconds += time.perf_counter() - det_t0
            det_call_count += 1
            det_accessed = True
        elif det and args.debug_detector.lower() == 'dream_hsd_lmcp':
            det_t0 = time.perf_counter()
            _ = det.raw.peaks(evt)
            _ = det.raw.waveforms(evt)
            _ = det.raw.padded(evt)
            det_call_seconds += time.perf_counter() - det_t0
            det_call_count += 3
            det_accessed = True

        if det and hasattr(evt, '_det_raw_timing'):
            det_seg_seconds += evt._det_raw_timing.get('segments', 0.0)
            det_stack_seconds += evt._det_raw_timing.get('stack', 0.0)
            delattr(evt, '_det_raw_timing')

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
            process = psutil.Process(os.getpid())
            rss_gb = process.memory_info().rss / (1024 ** 3)
            print(
                f"[Rank {rank}] Event {i_evt}: Rate = {rate:.1f} Hz "
                f"Interval={interval_time:.2f}s RSS={rss_gb:.2f} GB {det_accessed=} {io_msg}"
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
    total_pread_calls = comm.reduce(pread_calls, op=MPI.SUM, root=0)
    max_pread_sec = comm.reduce(pread_sec, op=MPI.MAX, root=0)
    det_calls_total = comm.reduce(det_call_count, op=MPI.SUM, root=0)
    det_time_max = comm.reduce(det_call_seconds, op=MPI.MAX, root=0)
    det_seg_max = comm.reduce(det_seg_seconds, op=MPI.MAX, root=0)
    det_stack_max = comm.reduce(det_stack_seconds, op=MPI.MAX, root=0)

    if rank == 0:
        total = np.sum(recvbuf)
        n_bdnodes = size - n_ebnodes - 1
        elapsed = end - start
        loop_elapsed_print = loop_elapsed_max if loop_elapsed_max else elapsed
        total_rate = total / loop_elapsed_print if loop_elapsed_print > 0 else 0.0
        def _env(name, default=""):
            return os.environ.get(name, default)
        march_vars = {k: _env(k, "unset") for k in sorted(os.environ) if k.startswith("PS_MARCH")}
        bd_chunk = _env("PS_BD_CHUNKSIZE", "unset")
        print(f"[{args.log_level}] Marching env: " +
              ", ".join(f"{k}={v}" for k, v in march_vars.items()) +
              f", PS_BD_CHUNKSIZE={bd_chunk}")
        print(
            f"[{args.log_level}] {n_ebnodes=} {n_bdnodes=} "
            f"Load time={load_time_max:.2f}s Loop time={loop_elapsed_print:.2f}s "
            f"Total events: {total} Rate={total_rate:.1f} Hz"
        )
        if n_bdnodes > 0:
            first_bd_rank = n_ebnodes + 1
            last_bd_rank = size - ps_srv_nodes
            bd_counts = recvbuf[first_bd_rank:last_bd_rank, 0]
            if bd_counts.size:
                bd_avg = float(np.mean(bd_counts))
                bd_min = int(np.min(bd_counts))
                bd_max = int(np.max(bd_counts))
                bd_med = float(np.median(bd_counts))
                bd_zero = int(np.sum(bd_counts == 0))
                print(
                    f"[{args.log_level}] BD_EVENTS avg={bd_avg:.1f} "
                    f"min={bd_min} max={bd_max} med={bd_med:.1f} zero={bd_zero}"
                )
        if total_pread_bytes > 0 and max_pread_sec > 0:
            agg_io_rate = (total_pread_bytes / (1024 * 1024)) / max_pread_sec
            print(
                f"[{args.log_level}] TOTAL_IO bytes={total_pread_bytes / (1024 * 1024):.2f} MiB "
                f"time={max_pread_sec:.2f}s rate={agg_io_rate:.2f} MiB/s "
                f"calls={int(total_pread_calls)}"
            )
        if det_calls_total > 0 and det_time_max > 0:
            det_rate = det_calls_total / det_time_max
            seg_msg = f" segments={det_seg_max:.2f}s" if det_seg_max > 0 else ""
            stack_msg = f" stack={det_stack_max:.2f}s" if det_stack_max > 0 else ""
            print(
                f"[{args.log_level}] TOTAL_DET calls={int(det_calls_total)} "
                f"time={det_time_max:.2f}s rate={det_rate:.2f} calls/s{seg_msg}{stack_msg}"
            )
    elif args.show_rank_stats:
        first_bd_rank = n_ebnodes + 1
        last_bd_rank = size - ps_srv_nodes
        if first_bd_rank <= rank < last_bd_rank:
            rate = local_count / loop_elapsed if loop_elapsed > 0 else 0.0
            msg = (
                f"[{args.log_level}] Rank {rank} processed {local_count} events "
                f"Rate={rate:.1f} Hz Load time={load_time:.2f}s Loop time={loop_elapsed:.2f}s"
            )
            if pread_calls > 0 and pread_sec > 0:
                io_rate = (pread_bytes / (1024 * 1024)) / pread_sec
                msg += (
                    f" | IO bytes={pread_bytes / (1024 * 1024):.2f} MiB "
                    f"time={pread_sec:.2f}s rate={io_rate:.2f} MiB/s calls={int(pread_calls)}"
                )
            if det_call_count > 0 and det_call_seconds > 0:
                det_rate = det_call_count / det_call_seconds
                seg_msg = f" segments={det_seg_seconds:.2f}s" if det_seg_seconds > 0 else ""
                stack_msg = f" stack={det_stack_seconds:.2f}s" if det_stack_seconds > 0 else ""
                msg += (
                    f" | DET calls={int(det_call_count)} "
                    f"time={det_call_seconds:.2f}s rate={det_rate:.2f} calls/s"
                    f"{seg_msg}{stack_msg}"
                )
            print(msg)


if __name__ == '__main__':
    main()
