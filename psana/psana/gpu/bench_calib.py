"""
Benchmark: GPU vs CPU Jungfrau calibration throughput.

Measures events/sec and per-stage timing (H->D, kernel, D->H) for the GPU
path compared to the CPU baseline.

Usage (MPI: 1 smd0 + 1 EB + N BD ranks, all sharing the visible GPU(s)):
    mpirun -n <N+2> python psana/psana/gpu/bench_calib.py \\
        -e mfx101572426 -r 47 -n 200 --warmup 10 \\
        --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc

    -n is the per-BD-rank event target. Rank 0 prints an aggregate rate
    (sum of BD-rank rates) at the end. --compare-cpu is serial-only.

Usage (single process, no MPI ranks):
    python psana/psana/gpu/bench_calib.py \\
        -e mfx101572426 -r 47 -n 500 --compare-cpu \\
        --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc

Notes on MPI structure: all ranks must construct the SAME DataSource and
iterate run.events() — smd0/EB ranks do their serving inside that loop and
yield no events. Termination comes from max_events (computed from -n and
the BD rank count); a BD rank must NOT break out early in MPI mode, since
a rank that stops requesting batches can hang smd0/EB.
"""

import argparse
import os
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="GPU Jungfrau calib benchmark")
    p.add_argument("-e", "--exp",      required=True,       help="Experiment name")
    p.add_argument("-r", "--run",      required=True, type=int, help="Run number")
    p.add_argument("-n", "--nevents",  default=500,   type=int, help="Events to time (per BD rank)")
    p.add_argument("--det",            default="jungfrau",  help="Detector name")
    p.add_argument("--dir",            default=None,        help="XTC directory")
    p.add_argument("--warmup",         default=20,    type=int, help="Warmup events")
    p.add_argument("--d2h",            action="store_true", help="Include D->H in timing")
    p.add_argument("--profile",        action="store_true",
                   help="GPU path with per-event WALL-time attribution: buckets "
                        "wait (generator advance = bigdata read + EB/MPI) / read / "
                        "h2d / kernel, to attribute the 32-BD ceiling.")
    p.add_argument("--cpu",            action="store_true",
                   help="CPU-only mode: BD ranks run det.raw.calib() on the shared "
                        "collective DataSource (no GPU). MPI-capable — measures B-CPU "
                        "at the same rank layout as the GPU path.")
    p.add_argument("--compare-cpu",    action="store_true", help="Also run CPU calib path (serial only)")
    p.add_argument("--smd0-debug",     action="store_true",
                   help="DEBUG logging on rank 0 only: emits smd0 per-chunk stats "
                        "(eb_wait vs work time) to attribute the serving-chain ceiling")
    return p.parse_args()


def _rank():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except Exception:
        return 0


def _world_size():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_size()
    except Exception:
        return 1


def _n_eb():
    """Number of EventBuilder ranks (psana2 default: 1). srv nodes unsupported."""
    return int(os.environ.get("PS_EB_NODES", "1"))


def _is_bd_rank():
    """True if this rank should run the benchmark body.

    psana2 MPI layout: rank 0 = smd0, ranks 1..PS_EB_NODES = EB, rest = BD.
    A single-process run does everything itself, so it is a BD rank even
    though mpi4py imports fine and reports rank 0 of a size-1 world.
    """
    size = _world_size()
    if size == 1:
        return True
    return _rank() >= 1 + _n_eb()


def run_gpu_bench(args, run, det_obj, peds_gpu, gmask_gpu, allow_break):
    """Time the GPU path over run.events(). Does NOT create a DataSource —
    the caller's DataSource is shared by all ranks (collective)."""
    import cupy as cp
    from psana.gpu import fused_calib_gpu

    h2d_times, kernel_times, d2h_times = [], [], []
    warmup_done = False
    n_measured = 0
    t_start = None

    for evt in run.events():
        raw = det_obj.raw.raw(evt)
        if raw is None:
            continue

        # H->D
        t0 = time.perf_counter()
        raw_gpu = cp.asarray(raw)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()

        # Kernel
        calib_gpu = fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu)
        cp.cuda.Device().synchronize()
        t2 = time.perf_counter()

        # D->H (optional)
        if args.d2h:
            _ = calib_gpu.get()
            t3 = time.perf_counter()
        else:
            t3 = t2

        if not warmup_done:
            if n_measured >= args.warmup:
                warmup_done = True
                t_start = time.perf_counter()
                h2d_times.clear(); kernel_times.clear(); d2h_times.clear()
            n_measured += 1
            continue

        h2d_times.append((t1 - t0) * 1e3)
        kernel_times.append((t2 - t1) * 1e3)
        d2h_times.append((t3 - t2) * 1e3)
        n_measured += 1

        if allow_break and len(h2d_times) >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0
    n = len(h2d_times)
    return {
        "n": n,
        "wall_s": t_wall,
        "rate_hz": n / t_wall if t_wall > 0 else 0,
        "h2d_ms_mean": np.mean(h2d_times) if h2d_times else 0,
        "kernel_ms_mean": np.mean(kernel_times) if kernel_times else 0,
        "d2h_ms_mean": np.mean(d2h_times) if d2h_times else 0,
    }


def run_gpu_bench_profile(args, run, det_obj, peds_gpu, gmask_gpu, allow_break):
    """Like run_gpu_bench, but also buckets the per-event WALL time into
    wait / read / h2d / kernel so the 32-BD ceiling can be attributed.

    The critical extra bucket is `wait` = the time to advance run.events()
    to the next event. In psana2 the BD rank reads its bigdata during that
    generator advance (smd0 -> EB batch -> BD reads xtc), so `wait` folds
    together the bigdata read + EB/MPI serving latency; `det.raw.raw(evt)`
    itself (the `read` bucket) is only the cheap array-locate inside the
    already-read dgram. wait + read + h2d + kernel should sum to ~wall,
    which is the sanity check that nothing is unattributed.

    Uses the same intra-event syncs as run_gpu_bench (valid: this is the
    synchronous path). Kept separate so the established B-MVP timing loop
    stays byte-identical.
    """
    import cupy as cp
    from psana.gpu import fused_calib_gpu

    wait_times, read_times, h2d_times, kernel_times = [], [], [], []
    warmup_done = False
    n_measured = 0
    t_start = None
    t_prev_end = None

    events = run.events()
    while True:
        t_wait0 = time.perf_counter()
        try:
            evt = next(events)
        except StopIteration:
            break
        t_got = time.perf_counter()
        wait = (t_got - t_prev_end) * 1e3 if t_prev_end is not None \
            else (t_got - t_wait0) * 1e3

        t0 = time.perf_counter()
        raw = det_obj.raw.raw(evt)
        t1 = time.perf_counter()
        if raw is None:
            t_prev_end = time.perf_counter()
            continue

        raw_gpu = cp.asarray(raw)
        cp.cuda.Device().synchronize()
        t2 = time.perf_counter()

        calib_gpu = fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu)
        cp.cuda.Device().synchronize()
        t3 = time.perf_counter()

        if not warmup_done:
            if n_measured >= args.warmup:
                warmup_done = True
                t_start = time.perf_counter()
                wait_times.clear(); read_times.clear()
                h2d_times.clear(); kernel_times.clear()
            n_measured += 1
            t_prev_end = time.perf_counter()
            continue

        wait_times.append(wait)
        read_times.append((t1 - t0) * 1e3)
        h2d_times.append((t2 - t1) * 1e3)
        kernel_times.append((t3 - t2) * 1e3)
        n_measured += 1
        t_prev_end = time.perf_counter()

        if allow_break and len(h2d_times) >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0
    n = len(h2d_times)
    return {
        "n": n,
        "wall_s": t_wall,
        "rate_hz": n / t_wall if t_wall > 0 else 0,
        "wait_ms_mean": np.mean(wait_times) if wait_times else 0,
        "read_ms_mean": np.mean(read_times) if read_times else 0,
        "h2d_ms_mean": np.mean(h2d_times) if h2d_times else 0,
        "kernel_ms_mean": np.mean(kernel_times) if kernel_times else 0,
    }


def run_cpu_bench_mpi(args, run, det_obj, allow_break):
    """Time the CPU calib path over the shared run.events(). Mirrors
    run_gpu_bench's loop structure (collective DataSource, warmup, no early
    break in MPI mode) so B-CPU and B-MVP are measured at IDENTICAL rank
    layouts. Does NOT create a DataSource — the caller's is shared by all
    ranks.

    det.raw.calib(evt) does the raw read + pedestal/gain/common-mode on CPU;
    this is the end-to-end CPU analogue of the GPU path's raw()+H2D+kernel.
    """
    calib_times = []
    warmup_done = False
    n_measured = 0
    t_start = None

    for evt in run.events():
        t0 = time.perf_counter()
        calib = det_obj.raw.calib(evt)
        t1 = time.perf_counter()
        if calib is None:
            continue

        if not warmup_done:
            if n_measured >= args.warmup:
                warmup_done = True
                t_start = time.perf_counter()
                calib_times.clear()
            n_measured += 1
            continue

        calib_times.append((t1 - t0) * 1e3)
        n_measured += 1

        if allow_break and len(calib_times) >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0
    n = len(calib_times)
    return {
        "n": n,
        "wall_s": t_wall,
        "rate_hz": n / t_wall if t_wall > 0 else 0,
        "calib_ms_mean": np.mean(calib_times) if calib_times else 0,
    }


def run_cpu_bench(args, max_events):
    """Serial-only: creates its own second DataSource, which is safe only
    when there are no server ranks to coordinate with."""
    from psana import DataSource

    kwargs = dict(exp=args.exp, run=args.run, max_events=max_events)
    if args.dir:
        kwargs["dir"] = args.dir
    ds  = DataSource(**kwargs)
    run = next(ds.runs())
    det_obj = run.Detector(args.det)

    n_warmup = 0
    n_measured = 0
    t_start = None

    for evt in run.events():
        calib = det_obj.raw.calib(evt)
        if calib is None:
            continue
        if n_warmup < args.warmup:
            n_warmup += 1
            continue
        if t_start is None:
            t_start = time.perf_counter()
        n_measured += 1
        if n_measured >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0
    return {
        "n": n_measured,
        "wall_s": t_wall,
        "rate_hz": n_measured / t_wall if t_wall > 0 else 0,
    }


def _print_occupancy(rank, peds_gpu):
    import cupy as cp

    npixels = int(peds_gpu.size // 3)
    blocks = (npixels + 255) // 256
    try:
        attrs = cp.cuda.Device(0).attributes
        n_sm  = attrs.get("MultiProcessorCount", 108)
        bpsm  = min(attrs.get("MaxBlocksPerMultiprocessor", 32),
                    attrs.get("MaxThreadsPerMultiProcessor", 2048) // 256)
        sat   = n_sm * bpsm
        print(f"[rank {rank}] npixels={npixels:,}  blocks/event={blocks:,}  "
              f"saturation={sat}  occupancy={blocks/sat*100:.0f}%")
    except Exception:
        print(f"[rank {rank}] npixels={npixels:,}  blocks/event={blocks:,}")


def _report_aggregate(result, size):
    """Gather per-BD-rank results on rank 0 and print the aggregate rate.

    Collective over COMM_WORLD: every rank must call this. smd0/EB ranks
    pass result=None; they arrive here after their serving loop ends.
    """
    if size <= 1:
        return
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    results = comm.gather(result, root=0)
    if comm.Get_rank() != 0:
        return
    bd = [r for r in results if r and r.get("n")]
    if not bd:
        print("\n[aggregate] no BD rank produced measurements")
        return
    agg = sum(r["rate_hz"] for r in bd)
    n_total = sum(r["n"] for r in bd)
    print(f"\n[aggregate] BD ranks: {len(bd)}   events measured: {n_total}")
    print(f"  aggregate rate:   {agg:.1f} Hz")
    print(f"  per-rank rate:    {agg / len(bd):.2f} Hz (mean)")
    if any("calib_ms_mean" in r for r in bd):
        print(f"  CPU calib:        {np.mean([r.get('calib_ms_mean', 0) for r in bd]):.3f} ms/event (mean)")
    elif any("wait_ms_mean" in r for r in bd):
        wait = np.mean([r.get('wait_ms_mean', 0) for r in bd])
        read = np.mean([r.get('read_ms_mean', 0) for r in bd])
        h2d  = np.mean([r.get('h2d_ms_mean', 0) for r in bd])
        ker  = np.mean([r.get('kernel_ms_mean', 0) for r in bd])
        per_rank_wall = 1000.0 * len(bd) / agg  # ms/event per rank
        print(f"  wait:             {wait:.3f} ms/event (gen advance = bigdata read + EB/MPI)")
        print(f"  read:             {read:.3f} ms/event (det.raw.raw array-locate)")
        print(f"  H->D:             {h2d:.3f} ms/event")
        print(f"  kernel:           {ker:.3f} ms/event")
        print(f"  sum:              {wait + read + h2d + ker:.3f} ms/event  "
              f"(vs {per_rank_wall:.3f} ms/event per-rank wall)")
    else:
        print(f"  H->D:             {np.mean([r.get('h2d_ms_mean', 0) for r in bd]):.3f} ms/event (mean)")
        print(f"  kernel:           {np.mean([r.get('kernel_ms_mean', 0) for r in bd]):.3f} ms/event (mean)")


def main():
    args = parse_args()
    rank = _rank()
    size = _world_size()
    is_bd = _is_bd_rank()

    from psana.gpu import init_gpu_rank, prep_calib_constants

    if is_bd and not args.cpu:
        init_gpu_rank()

    from psana import DataSource

    # Global event budget: in MPI mode the stream must end on its own
    # (see module docstring), so size it to feed every BD rank its warmup
    # + measurement quota, with 10% slack for skipped/None events.
    n_bd = max(1, size - 1 - _n_eb()) if size > 1 else 1
    max_events = int(1.1 * (args.warmup + args.nevents) * n_bd)

    kwargs = dict(exp=args.exp, run=args.run, max_events=max_events)
    if args.dir:
        kwargs["dir"] = args.dir
    if args.smd0_debug and rank == 0:
        # Rank 0 alone runs at DEBUG so smd0's per-chunk stats (eb_wait vs
        # work time) are emitted without 33 other ranks flooding stdout and
        # distorting the timing. DataSource kwargs may differ across ranks.
        kwargs["log_level"] = "DEBUG"
    ds  = DataSource(**kwargs)
    run = next(ds.runs())
    det = run.Detector(args.det)

    result = None
    if is_bd and args.cpu:
        result = run_cpu_bench_mpi(args, run, det, allow_break=(size == 1))
        print(f"\n[rank {rank}] CPU results ({result['n']} events, "
              f"warmup={args.warmup}):")
        print(f"  rate:        {result['rate_hz']:.1f} Hz")
        print(f"  CPU calib:   {result['calib_ms_mean']:.3f} ms/event")
    elif is_bd:
        peds_gpu, gmask_gpu = prep_calib_constants(det)

        if size == 1 or rank == 1 + _n_eb():   # first BD rank only
            _print_occupancy(rank, peds_gpu)

        if args.profile:
            result = run_gpu_bench_profile(args, run, det, peds_gpu, gmask_gpu,
                                           allow_break=(size == 1))
        else:
            result = run_gpu_bench(args, run, det, peds_gpu, gmask_gpu,
                                   allow_break=(size == 1))

        print(f"\n[rank {rank}] GPU results ({result['n']} events, "
              f"warmup={args.warmup}, d2h={'yes' if args.d2h else 'no'}):")
        print(f"  rate:        {result['rate_hz']:.1f} Hz")
        if args.profile:
            tot = (result['wait_ms_mean'] + result['read_ms_mean']
                   + result['h2d_ms_mean'] + result['kernel_ms_mean'])
            print(f"  wait:        {result['wait_ms_mean']:.3f} ms/event "
                  f"(gen advance = bigdata read + EB/MPI)")
            print(f"  read:        {result['read_ms_mean']:.3f} ms/event "
                  f"(det.raw.raw array-locate)")
            print(f"  H->D:        {result['h2d_ms_mean']:.3f} ms/event")
            print(f"  kernel:      {result['kernel_ms_mean']:.3f} ms/event")
            print(f"  sum:         {tot:.3f} ms/event  "
                  f"(vs {1000.0/result['rate_hz']:.3f} ms/event wall)")
        else:
            print(f"  H->D:        {result['h2d_ms_mean']:.3f} ms/event")
            print(f"  kernel:      {result['kernel_ms_mean']:.3f} ms/event")
        if args.d2h:
            print(f"  D->H:        {result['d2h_ms_mean']:.3f} ms/event")

        if args.compare_cpu:
            if size == 1:
                cpu = run_cpu_bench(args, max_events)
                print(f"\n[rank {rank}] CPU results ({cpu['n']} events):")
                print(f"  rate:        {cpu['rate_hz']:.1f} Hz")
                if cpu["rate_hz"] > 0:
                    print(f"  GPU speedup: {result['rate_hz'] / cpu['rate_hz']:.2f}x")
            elif rank == 1 + _n_eb():
                print(f"[rank {rank}] --compare-cpu is serial-only; skipped in MPI mode")
    else:
        # smd0/EB ranks serve data from inside the events() generator and
        # yield nothing; they must drive the same loop or BD ranks starve.
        for _ in run.events():
            pass

    _report_aggregate(result, size)


if __name__ == "__main__":
    main()
