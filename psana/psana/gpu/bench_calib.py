"""
Benchmark: GPU vs CPU Jungfrau calibration throughput.

Measures events/sec and per-stage timing (H->D, kernel, D->H) for the GPU
path compared to the CPU baseline.

Usage (MPI: 1 smd0 + 1 EB + 1 BD):
    mpirun -n 3 python psana/psana/gpu/bench_calib.py \\
        -e mfx101210926 -r 387 -n 500 --compare-cpu

Usage (GPU only, no MPI):
    python psana/psana/gpu/bench_calib.py \\
        -e mfx101210926 -r 387 -n 500

Outputs per-rank timing and aggregate events/sec.
"""

import argparse
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="GPU Jungfrau calib benchmark")
    p.add_argument("-e", "--exp",      required=True,       help="Experiment name")
    p.add_argument("-r", "--run",      required=True, type=int, help="Run number")
    p.add_argument("-n", "--nevents",  default=500,   type=int, help="Events to time")
    p.add_argument("--det",            default="jungfrau",  help="Detector name")
    p.add_argument("--dir",            default=None,        help="XTC directory")
    p.add_argument("--warmup",         default=20,    type=int, help="Warmup events")
    p.add_argument("--d2h",            action="store_true", help="Include D->H in timing")
    p.add_argument("--compare-cpu",    action="store_true", help="Also run CPU calib path")
    return p.parse_args()


def _rank():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except Exception:
        return 0


def _is_bd_rank():
    """True if this is not rank 0 (smd0) or rank 1 (EB) in a 3-rank job."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() >= 2
    except Exception:
        return True  # single process → treat as BD


def run_gpu_bench(args, det, peds_gpu, gmask_gpu):
    import cupy as cp
    from psana import DataSource

    kwargs = dict(exp=args.exp, run=args.run)
    if args.dir:
        kwargs["dir"] = args.dir
    ds  = DataSource(**kwargs)
    run = next(ds.runs())
    det_obj = run.Detector(args.det)

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

        if len(h2d_times) >= args.nevents:
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


def run_cpu_bench(args):
    from psana import DataSource

    kwargs = dict(exp=args.exp, run=args.run)
    if args.dir:
        kwargs["dir"] = args.dir
    ds  = DataSource(**kwargs)
    run = next(ds.runs())
    det_obj = run.Detector(args.det)

    n_warmup = 0
    n_measured = 0
    t_start = None
    times = []

    for evt in run.events():
        calib = det_obj.raw.calib(evt)
        if calib is None:
            continue
        if n_warmup < args.warmup:
            n_warmup += 1
            continue
        if t_start is None:
            t_start = time.perf_counter()
        times.append(1)
        n_measured += 1
        if n_measured >= args.nevents:
            break

    t_wall = time.perf_counter() - t_start if t_start else 0
    return {
        "n": n_measured,
        "wall_s": t_wall,
        "rate_hz": n_measured / t_wall if t_wall > 0 else 0,
    }


def main():
    args = parse_args()
    rank = _rank()

    from psana.gpu import init_gpu_rank, prep_calib_constants, fused_calib_gpu

    if _is_bd_rank():
        init_gpu_rank()

    import cupy as cp
    from psana import DataSource

    kwargs = dict(exp=args.exp, run=args.run)
    if args.dir:
        kwargs["dir"] = args.dir
    ds  = DataSource(**kwargs)
    run = next(ds.runs())
    det = run.Detector(args.det)

    if _is_bd_rank():
        peds_gpu, gmask_gpu = prep_calib_constants(det)

        # Occupancy estimate
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

        result = run_gpu_bench(args, det, peds_gpu, gmask_gpu)

        print(f"\n[rank {rank}] GPU results ({result['n']} events, "
              f"warmup={args.warmup}, d2h={'yes' if args.d2h else 'no'}):")
        print(f"  rate:        {result['rate_hz']:.1f} Hz")
        print(f"  H->D:        {result['h2d_ms_mean']:.3f} ms/event")
        print(f"  kernel:      {result['kernel_ms_mean']:.3f} ms/event")
        if args.d2h:
            print(f"  D->H:        {result['d2h_ms_mean']:.3f} ms/event")

        if args.compare_cpu:
            cpu = run_cpu_bench(args)
            print(f"\n[rank {rank}] CPU results ({cpu['n']} events):")
            print(f"  rate:        {cpu['rate_hz']:.1f} Hz")
            if cpu["rate_hz"] > 0:
                print(f"  GPU speedup: {result['rate_hz'] / cpu['rate_hz']:.2f}x")


if __name__ == "__main__":
    main()
