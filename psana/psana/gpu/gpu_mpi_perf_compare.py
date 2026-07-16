"""
gpu_mpi_perf_compare.py
═══════════════════════
Multi-node psana2 MPI performance benchmark: CPU vs GPU calibration.

Runs both the CPU (numpy) and GPU (CUDA) calibration paths in the same MPI
topology on the same dataset and reports aggregate throughput, per-rank
breakdown, and GPU speedup over CPU.

MPI topology  (PS_EB_NODES=1)
─────────────────────────────
  rank 0            — SMD0:  reads SMD files, distributes EB chunks
  rank 1            — EB:    builds events; CPU path sends full smd_batch,
                              GPU path packs (smd_batch + GPUBAT1 stubs)
  ranks 2 … N-1     — BD:    CPU path runs numpy calib;
                              GPU path runs fused_calib_gpu on A100

Measurements
────────────
  CPU path  DataSource(exp=, run=, dir=)
              BD rank time includes: bigdata DramManager I/O + numpy calib
  GPU path  DataSource(exp=, run=, dir=, gpu_det=)
              BD rank time includes: GDS NVMe→GPU + fused_calib_gpu kernel

The CPU run executes first (establishes the production baseline), followed by
one or more GPU runs at configurable batch sizes.  Both runs use the same
number of BD ranks, the same data, and the same max_events count so the
comparison is apples-to-apples.

Run
───
  sh psana/psana/gpu/scripts/run_mpi_perf_compare.sh

  # Or directly on a GPU node with MPI:
  PS_EB_NODES=1 mpirun -n 4 --bind-to none \\
      python3 gpu_mpi_perf_compare.py --n-events 50 --batch-size 10
"""

import argparse
import os
import time

from mpi4py import MPI
import numpy as np

_COMM = MPI.COMM_WORLD
_WORLD_RANK = _COMM.Get_rank()
_WORLD_SIZE = _COMM.Get_size()

_EXP = "mfx100852324"
_RUN = 77
_DIR = "/sdf/data/lcls/ds/prj/public01/xtc"
_DET = "jungfrau"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar(val, max_val, width=36):
    if max_val <= 0:
        return ""
    filled = max(0, min(width, int(round(val / max_val * width))))
    return "█" * filled + "░" * (width - filled)


def _khz(eps: float) -> str:
    """Format an events/second rate as kHz with 3 decimal places."""
    return f"{eps / 1000:.3f} kHz"


def _flush_gpu_pool():
    """Force GC then release all cached CuPy pool blocks."""
    import gc

    gc.collect()
    try:
        import cupy as cp

        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CPU path: numpy calibration on BD ranks
# ---------------------------------------------------------------------------


def run_cpu(exp, run, xtc_dir, det_name, n_warmup, n_events):
    """Time BD-rank numpy calibration via the standard MPI CPU path.

    Each BD rank receives its share of events from EB (round-robin) and
    calibrates them via det.raw.calib(evt) — the same numpy path used in
    single-process production today.
    """
    from psana.psexp.mpi_ds import MPIDataSource
    from psana.psexp.node import Communicators

    comms = Communicators()
    ds = MPIDataSource(
        comms,
        exp=exp,
        run=run,
        dir=xtc_dir,
        max_events=n_warmup + n_events,
    )

    if not ds.is_bd():
        for r in ds.runs():
            for _ in r.events():
                pass
        return {}

    bd_rank = comms.bd_rank
    det = None
    wall_t_start = wall_t_end = None
    evt_ms = []
    n = 0

    for r in ds.runs():
        if det is None:
            det = r.Detector(det_name)
        for evt in r.events():
            if n == n_warmup:
                wall_t_start = time.perf_counter()
            t0 = time.perf_counter()
            calib = det.raw.calib(evt)  # numpy calibration
            dt = (time.perf_counter() - t0) * 1000.0
            n += 1
            if n > n_warmup:
                evt_ms.append(dt)
            if n == n_warmup + n_events:
                wall_t_end = time.perf_counter()

    # Capture end time even if max_events was not reached (e.g. run ended early
    # or events were distributed unevenly across BD ranks).
    if wall_t_start is not None and wall_t_end is None:
        wall_t_end = time.perf_counter()

    wall_ms = (wall_t_end - wall_t_start) * 1000.0 if (wall_t_end and wall_t_start) else 0.0
    cnt = len(evt_ms)
    return {
        "rank": _WORLD_RANK,
        "bd_rank": bd_rank,
        "n_events": cnt,
        "mean_ms": float(np.mean(evt_ms)) if evt_ms else 0.0,
        "p95_ms": float(np.percentile(evt_ms, 95)) if evt_ms else 0.0,
        "wall_total_ms": wall_ms,
        "evt_per_sec": cnt / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# GPU path: CUDA calibration on BD ranks
# ---------------------------------------------------------------------------


def run_gpu(exp, run, xtc_dir, det_name, batch_size, n_gpu_streams, n_warmup, n_events, d2h_chunk_size=0):
    """Time BD-rank GPU calibration via the MPI GPU path.

    d2h_chunk_size=0  → call .on_gpu  (no D→H, ceiling throughput)
    d2h_chunk_size=N  → activate _D2hPipeline; call .on_cpu (pre-done D→H)
    """
    from psana.psexp.mpi_ds import MPIDataSource
    from psana.psexp.node import Communicators

    _flush_gpu_pool()
    comms = Communicators()
    ds = MPIDataSource(
        comms,
        exp=exp,
        run=run,
        dir=xtc_dir,
        gpu_det=det_name,
        batch_size=batch_size,
        n_gpu_streams=n_gpu_streams,
        gpu_d2h_chunk_size=d2h_chunk_size,
        max_events=n_warmup + n_events,
    )

    if not ds.is_bd():
        for r in ds.runs():
            for _ in r.events():
                pass
        return {}

    import cupy as cp

    phys_gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    bd_rank = comms.bd_rank
    wall_t_start = wall_t_end = None
    hot_ms = []
    n = 0

    # Detect I/O path directly from kvikio — works for both serial and MPI.
    # compat_mode=True  → CPU fallback: NVMe → CPU DRAM → GPU VRAM
    # compat_mode=False → GDS:          NVMe → GPU VRAM direct (DMA)
    try:
        import kvikio as _kv

        _dp = _kv.DriverProperties()
        io_path = "GDS" if _dp.is_gds_available else "CPU-fallback"
    except Exception:
        io_path = "unknown"

    for r in ds.runs():
        for ctx in r.events():
            if n == n_warmup:
                wall_t_start = time.perf_counter()
            t0 = time.perf_counter()
            res = ctx.get(det_name + ".calib")
            _ = res.on_cpu if d2h_chunk_size > 0 else res.on_gpu
            dt = (time.perf_counter() - t0) * 1000.0
            n += 1
            if n > n_warmup:
                hot_ms.append(dt)
            if n == n_warmup + n_events:
                wall_t_end = time.perf_counter()

    if wall_t_start is not None and wall_t_end is None:
        wall_t_end = time.perf_counter()

    wall_ms = (wall_t_end - wall_t_start) * 1000.0 if (wall_t_end and wall_t_start) else 0.0
    cnt = len(hot_ms)
    del ds, comms
    _flush_gpu_pool()
    return {
        "rank": _WORLD_RANK,
        "bd_rank": bd_rank,
        "phys_gpu": phys_gpu,
        "n_events": cnt,
        "hot_mean_ms": float(np.mean(hot_ms)) if hot_ms else 0.0,
        "wall_total_ms": wall_ms,
        "wall_per_evt_ms": wall_ms / cnt if cnt > 0 else 0.0,
        "evt_per_sec": cnt / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
        "io_path": io_path,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _agg_eps(stats_list):
    """Aggregate events/sec = total_events / max(wall_time_s) across BD ranks."""
    if not stats_list:
        return 0.0
    total = sum(s["n_events"] for s in stats_list)
    max_w = max(s["wall_total_ms"] for s in stats_list) / 1000.0
    return total / max_w if max_w > 0 else 0.0


def _bd_rank_count(cpu_stats, gpu_runs):
    """Return the number of BD ranks represented by the benchmark."""
    for _, stats in gpu_runs:
        if stats:
            return len(stats)
    if cpu_stats:
        return len(cpu_stats)
    return _WORLD_SIZE - 1 - int(os.environ.get("PS_EB_NODES", "1"))


def print_report(args, cpu_stats, gpu_runs):
    """
    cpu_stats  : list of per-BD-rank dicts from run_cpu()
    gpu_runs   : list of (batch_size, [per-BD-rank dicts]) from run_gpu()
    """
    n_bd = _bd_rank_count(cpu_stats, gpu_runs)
    sep = "═" * 90
    sep2 = "─" * 90
    ps_eb = os.environ.get("PS_EB_NODES", "1")

    print(f"\n{sep}")
    print(f"  psana2 Multi-Node CPU vs GPU Performance Benchmark")
    print(f"  exp={args.exp}  run={args.run}  det={args.det}")
    print(f"  MPI: {_WORLD_SIZE} total ranks  PS_EB_NODES={ps_eb}  BD ranks={n_bd}")
    print(f"  warmup={args.n_warmup}  timed_events/rank≈{args.n_events}  pool_depth={args.pool_depth}")
    print(sep)

    # ── CPU baseline ──────────────────────────────────────────────────────
    cpu_agg = _agg_eps(cpu_stats)
    if cpu_stats:
        print(f"\n  ── CPU baseline (numpy calibration, {n_bd} BD rank(s)) ──")
        print(f"  {sep2}")
        print(f"  {'Rank':>5}  {'events':>7}  {'mean ms':>9}  {'p95 ms':>8}  {'kHz':>9}")
        for s in sorted(cpu_stats, key=lambda x: x["bd_rank"]):
            print(f"  {s['rank']:>5}  {s['n_events']:>7}  {s['mean_ms']:>9.1f}  {s['p95_ms']:>8.1f}  {_khz(s['evt_per_sec']):>9}")
        print(f"  {sep2}")
        total_cpu = sum(s["n_events"] for s in cpu_stats)
        print(f"  AGGREGATE  {total_cpu:>7}  {'':>9}  {'':>8}  {_khz(cpu_agg):>9}  ← total_events / max_wall")
    else:
        print(f"\n  ── CPU baseline skipped (--skip-cpu) ──")

    # ── GPU runs ──────────────────────────────────────────────────────────
    best_gpu_agg = 0.0
    best_bs = None

    for bs, gpu_stats in gpu_runs:
        if not gpu_stats:
            continue
        gpu_agg = _agg_eps(gpu_stats)
        speedup = gpu_agg / cpu_agg if cpu_agg > 0 else 0.0
        all_speedups = [_agg_eps(gs) / cpu_agg for _, gs in gpu_runs if cpu_agg > 0 and _agg_eps(gs) > 0]
        max_speedup = max(all_speedups) if all_speedups else max(speedup, 1.0)
        bar = _bar(speedup, max_speedup)

        print(f"\n  ── GPU  batch_size={bs} ──")
        print(f"  {sep2}")
        print(f"  {'Rank':>5}  {'GPU':>4}  {'events':>7}  {'hot ms':>8}  {'wall ms':>9}  {'kHz':>9}")
        for s in sorted(gpu_stats, key=lambda x: x["bd_rank"]):
            print(
                f"  {s['rank']:>5}  {s['phys_gpu']:>4}  {s['n_events']:>7}  {s['hot_mean_ms']:>8.3f}  {s['wall_per_evt_ms']:>9.1f}  {_khz(s['evt_per_sec']):>9}"
            )
        print(f"  {sep2}")
        total_gpu = sum(s["n_events"] for s in gpu_stats)
        print(f"  AGGREGATE  {'':>4}  {total_gpu:>7}  {'':>8}  {'':>9}  {_khz(gpu_agg):>9}")
        print(f"\n  CPU aggregate:  {_khz(cpu_agg):>9}")
        print(f"  GPU aggregate:  {_khz(gpu_agg):>9}")
        print(f"  GPU speedup:    {speedup:>8.2f}×  {bar}")

        # I/O path — crucial for interpreting throughput numbers.
        io_paths = list({s.get("io_path", "unknown") for s in gpu_stats if s.get("n_events", 0) > 0})
        io_path = io_paths[0] if len(io_paths) == 1 else str(io_paths)
        if io_path == "GDS":
            print(f"\n  I/O: GDS — NVMe → GPU VRAM direct (DMA, bypasses CPU DRAM)")
        else:
            print(f"\n  I/O: {io_path} — NVMe → CPU DRAM → GPU VRAM via cudaMemcpy")
            print(f"  WARNING: True GDS not available. Likely cause: Lustre/GPFS filesystem")
            print(f"           or cuFile kernel module not loaded on this node.")

        if gpu_agg > best_gpu_agg:
            best_gpu_agg = gpu_agg
            best_bs = bs

    # ── Single-rank comparison (valid even with uneven distribution) ─────
    # When running with --oversubscribe (all MPI ranks on one Slurm task),
    # one BD rank may process all events before the other starts — giving
    # skewed per-rank numbers.  The single-rank CPU vs GPU comparison is
    # still valid: compare the rank that received the most events on each path.
    cpu_active = [s for s in cpu_stats if s["n_events"] > 0]
    if cpu_active and gpu_runs:
        _, gpu_stats_best = max(gpu_runs, key=lambda x: _agg_eps(x[1]) if x[1] else 0)
        gpu_active = [s for s in gpu_stats_best if s["n_events"] > 0]
        if cpu_active and gpu_active:
            best_cpu = max(cpu_active, key=lambda s: s["n_events"])
            best_gpu = max(gpu_active, key=lambda s: s["n_events"])
            rank_speedup = best_gpu["evt_per_sec"] / best_cpu["evt_per_sec"] if best_cpu["evt_per_sec"] > 0 else 0.0
            print(f"\n  Single-rank CPU vs GPU (rank {best_cpu['rank']} processing {best_cpu['n_events']} events):")
            print(f"  CPU rank {best_cpu['rank']}:  {_khz(best_cpu['evt_per_sec']):>9}  ({best_cpu['mean_ms']:.1f} ms/evt)")
            print(f"  GPU rank {best_gpu['rank']}:  {_khz(best_gpu['evt_per_sec']):>9}  ({best_gpu['wall_per_evt_ms']:.2f} ms/evt)")
            print(f"  Single-rank GPU speedup:  {rank_speedup:.1f}×")

        uneven = any(s["n_events"] == 0 for s in cpu_stats if True)
        if uneven:
            print(
                f"\n  NOTE: uneven event distribution detected "
                f"(one BD rank received 0 events).  A small init-timing "
                f"difference between BD ranks caused one to send its first "
                f"EB request before the other, consuming all batches.  "
                f"This is most visible in the CPU path where init is faster "
                f"(no _setup_gpu_geometry overhead).  A _COMM.Barrier() "
                f"before the event loop mitigates but may not fully eliminate "
                f"this on nodes with asymmetric NUMA topology."
            )

    # ── Summary ───────────────────────────────────────────────────────────
    if len(gpu_runs) > 1:
        print(f"\n  {sep2}")
        print(f"  Batch-size scaling (GPU aggregate, {n_bd} BD rank(s)):")
        base_eps = _agg_eps(gpu_runs[0][1]) if gpu_runs else 1.0
        for bs, gpu_stats in gpu_runs:
            agg = _agg_eps(gpu_stats)
            rel = agg / base_eps if base_eps > 0 else 0.0
            bar = _bar(rel, 5.0, width=30)
            print(f"  bs={str(bs):>5}  {_khz(agg):>9}  {rel:.2f}× vs bs=1  {bar}")
        if best_bs:
            cpu_str = f"{best_gpu_agg / cpu_agg:.1f}× over CPU" if cpu_agg > 0 else "CPU skipped"
            print(f"\n  Best batch size: {best_bs}  ({_khz(best_gpu_agg)}, {cpu_str})")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="psana2 multi-node CPU vs GPU MPI performance benchmark")
    p.add_argument("--exp", default=_EXP)
    p.add_argument("--run", type=int, default=_RUN)
    p.add_argument("--dir", default=_DIR)
    p.add_argument("--det", default=_DET)
    p.add_argument("--n-events", type=int, default=50, help="Events to time per BD rank (default 50)")
    p.add_argument("--n-warmup", type=int, default=5, help="Warmup events excluded from timing (default 5)")
    p.add_argument("--batch-size", type=int, default=10, help="GPU batch size for a single run (default 10)")
    p.add_argument("--pool-depth", type=int, default=4, help="EventPool depth / n_gpu_streams (default 4)")
    p.add_argument("--batch-sizes", help="Comma-separated batch sizes to sweep, e.g. 10,20,30,50")
    p.add_argument("--pool-depths", help="Comma-separated pool depths to sweep, e.g. 2,4")
    p.add_argument("--scaling", action="store_true", help="GPU batch-size scaling sweep (bs = 1, 2, 5, 10, 20)")
    p.add_argument("--skip-cpu", action="store_true", help="Skip CPU baseline (GPU-only mode)")
    p.add_argument(
        "--gpu-d2h-chunk-sizes",
        help="Comma-separated chunk sizes for _D2hPipeline sweep, "
        "e.g. 1,10,50.  Each runs .on_cpu with the internal "
        "pipeline active.  0 (default) runs .on_gpu only.",
    )
    args = p.parse_args()

    kw = dict(exp=args.exp, run=args.run, xtc_dir=args.dir, det_name=args.det, n_warmup=args.n_warmup, n_events=args.n_events)

    n_bd = _WORLD_SIZE - 1 - int(os.environ.get("PS_EB_NODES", "1"))
    if _WORLD_RANK == 0:
        print(f"\npsana2 CPU vs GPU benchmark — {_WORLD_SIZE} MPI ranks, {n_bd} BD rank(s)", flush=True)

    # ── CPU baseline ──────────────────────────────────────────────────────
    cpu_stats = []
    if not args.skip_cpu:
        if _WORLD_RANK == 0:
            print("  [CPU] running numpy calibration baseline...", flush=True)
        local = run_cpu(**kw)
        all_ = _COMM.gather(local, root=0)
        if _WORLD_RANK == 0:
            cpu_stats = [s for s in (all_ or []) if s]
            total = sum(s["n_events"] for s in cpu_stats)
            agg = _agg_eps(cpu_stats)
            print(f"  [CPU] total={total} events  aggregate={_khz(agg)}", flush=True)
        _COMM.Barrier()

    # ── Build (batch_size, pool_depth) grid ──────────────────────────────
    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    elif args.scaling:
        batch_sizes = [1, 2, 5, 10, 20]
    else:
        batch_sizes = [args.batch_size]

    pool_depths = [int(x) for x in args.pool_depths.split(",")] if args.pool_depths else [args.pool_depth]

    gpu_grid = []  # list of (bs, pd, gpu_stats)

    for pd in pool_depths:
        for bs in batch_sizes:
            if _WORLD_RANK == 0:
                print(f"  [GPU] batch_size={bs}  pool_depth={pd}...", flush=True)

            _flush_gpu_pool()
            local = run_gpu(batch_size=bs, n_gpu_streams=pd, **kw)
            all_ = _COMM.gather(local, root=0)
            if _WORLD_RANK == 0:
                gpu_stats = [s for s in (all_ or []) if s]
                gpu_grid.append((bs, pd, gpu_stats))
                agg = _agg_eps(gpu_stats)
                print(f"  [GPU] bs={bs} pd={pd}  aggregate={_khz(agg)}", flush=True)
            _COMM.Barrier()

    # ── _D2hPipeline sweep (.on_cpu with internal chunked D→H) ────────────
    if args.gpu_d2h_chunk_sizes:
        d2h_chunks = [int(x) for x in args.gpu_d2h_chunk_sizes.split(",")]
        if _WORLD_RANK == 0:
            print(f"\n  ── _D2hPipeline sweep (bs={batch_sizes[0]} pd={pool_depths[0]}) ──", flush=True)
        for chunk in d2h_chunks:
            if _WORLD_RANK == 0:
                print(f"  [D2H chunk={chunk}] on_cpu...", flush=True)
            _flush_gpu_pool()
            local = run_gpu(batch_size=batch_sizes[0], n_gpu_streams=pool_depths[0], d2h_chunk_size=chunk, **kw)
            all_ = _COMM.gather(local, root=0)
            if _WORLD_RANK == 0:
                stats = [s for s in (all_ or []) if s]
                gpu_grid.append((f"d2h={chunk}", pool_depths[0], stats))
                print(f"  [D2H chunk={chunk}] aggregate={_khz(_agg_eps(stats))}", flush=True)
            _COMM.Barrier()

    # ── Report ─────────────────────────────────────────────────────────────
    if _WORLD_RANK == 0:
        # Flatten to (bs, stats) pairs for print_report (uses first pd seen)
        gpu_runs = [(bs, stats) for bs, pd, stats in gpu_grid if pd == pool_depths[0]]
        print_report(args, cpu_stats, gpu_runs)

        # Grid summary when sweeping both batch sizes and pool depths
        if len(pool_depths) > 1 or len(batch_sizes) > 1:
            cpu_agg = _agg_eps(cpu_stats)
            sep2 = "─" * 90
            sep = "═" * 90
            print(f"\n{sep}")
            print(f"  Grid summary: GPU aggregate kHz  (CPU baseline: {_khz(cpu_agg)})")
            print(f"  {sep2}")
            # Header
            pd_header = "".join(f"  pd={pd:2d}  {'kHz':>9}  ×CPU" for pd in pool_depths)
            print(f"  {'bs':>4}{pd_header}")
            print(f"  {sep2}")
            for bs in batch_sizes:
                row = f"  {bs:>4}"
                for pd in pool_depths:
                    match = [(s, g) for b, p, g in gpu_grid if b == bs and p == pd for s in [_agg_eps(g)]]
                    if match:
                        agg, _ = match[0]
                        spd = agg / cpu_agg if cpu_agg > 0 else 0.0
                        row += f"  {_khz(agg):>9}  {spd:5.1f}×"
                    else:
                        row += f"  {'OOM':>9}  {'—':>5} "
                print(row)
            print(f"  {sep2}")
            best = max(gpu_grid, key=lambda x: _agg_eps(x[2]))
            best_speedup = _agg_eps(best[2]) / cpu_agg if cpu_agg > 0 else 0.0
            cpu_str = f"{best_speedup:.1f}× over CPU" if cpu_agg > 0 else "CPU skipped"
            print(f"  Best: bs={best[0]} pd={best[1]}  {_khz(_agg_eps(best[2]))}  {cpu_str}")
            print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
