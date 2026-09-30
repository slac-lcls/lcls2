"""
gpu_mpi_benchmark.py
════════════════════
Multi-GPU multi-node psana2 MPI performance benchmark.

Measures GPU calibration throughput with N GPU BD ranks distributed across
one or more nodes, and compares against the single-GPU baseline from
gpu_performance_benchmark.py.

MPI topology  (PS_EB_NODES=1)
─────────────────────────────
  rank 0             — SMD0:  reads SMD files, distributes EB chunks
  rank 1             — EB:    builds events, sends packed (smd+GPUBAT1) to BD
  ranks 2 … N-1      — BD:    each pinned to one GPU, runs GPU calibration

Each BD rank measures:
  hot_loop_ms    Python attribute access time per event (EventPool result ready)
  wall_ms        True amortised per-event wall time (GDS read + CUDA kernel)
  n_events       Number of L1Accept events processed by this rank

Rank 0 aggregates:
  aggregate_evt_s      = total_events / max(all_BD_wall_times_s)
  single_gpu_baseline  = from --single-gpu-baseline arg (ms/event from
                         gpu_performance_benchmark.py, bs=10 path)
  scaling_eff          = aggregate_evt_s / (n_bd_ranks × baseline_evt_s)

Batch-size scaling sweep
────────────────────────
  Optionally repeats the MPI run for batch_sizes = (1, 2, 5, 10, 20) and
  reports how aggregate throughput scales with batch size.  Requires running
  the same Slurm allocation multiple times (one per batch size); use
  --scaling for this mode.

Run
───
  sh psana/psana/gpu/scripts/run_mpi_performance_benchmark.sh
  # or on a node with MPI + GPUs:
  PS_EB_NODES=1 mpirun -n 6 --bind-to none python3 gpu_mpi_benchmark.py
"""

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Step 1 — MPI init BEFORE any CuPy import.
# GPU pinning (CUDA_VISIBLE_DEVICES) is set inside MPIDataSource.__init__()
# via init_gpu_rank(bd_rank - 1) before the CuPy import is triggered.
# ---------------------------------------------------------------------------

from mpi4py import MPI
import numpy as np

_COMM       = MPI.COMM_WORLD
_WORLD_RANK = _COMM.Get_rank()
_WORLD_SIZE = _COMM.Get_size()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_EXP = 'mfx100852324'
_RUN = 77
_DIR = '/sdf/data/lcls/ds/prj/public01/xtc'
_DET = 'jungfrau'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats(times_ms):
    t = np.array(times_ms, dtype=np.float64)
    return dict(
        mean=float(np.mean(t)),
        p95=float(np.percentile(t, 95)),
        throughput=1000.0 / float(np.mean(t)),
    )


def _bar(val, max_val, width=40):
    if max_val <= 0:
        return ''
    filled = max(0, min(width, int(round(val / max_val * width))))
    return '█' * filled + '░' * (width - filled)


def _abort(msg):
    if _WORLD_RANK == 0:
        print(f'\nFAIL: {msg}', flush=True)
    _COMM.Abort(1)


# ---------------------------------------------------------------------------
# Single MPI run: all ranks participate; BD ranks time their events.
# ---------------------------------------------------------------------------

def run_mpi_gpu(exp, run, xtc_dir, det_name,
                batch_size, n_gpu_streams, n_warmup, n_events):
    """Run one complete MPI GPU calibration pass.

    Returns on BD ranks: dict with per-rank timing.
    Returns on non-BD ranks: empty dict.
    """
    from psana.psexp.mpi_ds import MPIDataSource
    from psana.psexp.node import Communicators

    comms = Communicators()

    # max_events is distributed across ALL BD ranks by the pipeline.
    # Each BD rank sees roughly (n_warmup + n_events) / n_bd_ranks events,
    # so we multiply by n_bd_ranks to ensure every rank gets enough events
    # to complete both warmup and the timed window.
    n_bd_ranks = max(1, _WORLD_SIZE - 1 - int(os.environ.get('PS_EB_NODES', '1')))
    max_events_global = (n_warmup + n_events) * n_bd_ranks

    ds    = MPIDataSource(
        comms,
        exp=exp,
        run=run,
        dir=xtc_dir,
        gpu_det=det_name,
        batch_size=batch_size,
        n_gpu_streams=n_gpu_streams,
        max_events=max_events_global,
    )

    if not ds.is_bd():
        # SMD0 / EB: drive the pipeline, no timing.
        for r in ds.runs():
            for _ in r.events():
                pass
        return {}

    # BD rank: import CuPy now that CUDA_VISIBLE_DEVICES is pinned.
    import cupy as cp

    phys_gpu = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
    bd_rank  = comms.bd_rank   # 1-indexed within bd_comm (0 = EB)

    # Hot-loop timings (Python overhead for accessing pre-computed result).
    hot_times_ms  = []
    # Wall timings (amortised per-event: GDS reads + CUDA kernel).
    wall_t_start  = wall_t_end = None
    n             = 0

    for r in ds.runs():
        for ctx in r.events():
            if n == n_warmup:
                wall_t_start = time.perf_counter()

            t0 = time.perf_counter()
            _  = ctx.get(det_name + '.calib').on_gpu
            dt = (time.perf_counter() - t0) * 1000.0

            n += 1
            if n > n_warmup:
                hot_times_ms.append(dt)

            if n == n_warmup + n_events:
                wall_t_end = time.perf_counter()

    # Capture end time if the run finished before reaching the exact target
    # count (e.g. the data has fewer events than n_warmup + n_events).
    if wall_t_start is not None and wall_t_end is None:
        wall_t_end = time.perf_counter()

    wall_total_ms = (wall_t_end - wall_t_start) * 1000.0 if (
        wall_t_end is not None and wall_t_start is not None
    ) else 0.0

    counted = len(hot_times_ms)
    return {
        'rank':          _WORLD_RANK,
        'bd_rank':       bd_rank,
        'phys_gpu':      phys_gpu,
        'n_events':      counted,
        'hot_mean_ms':   float(np.mean(hot_times_ms)) if hot_times_ms else 0.0,
        'hot_p95_ms':    float(np.percentile(hot_times_ms, 95)) if hot_times_ms else 0.0,
        'wall_total_ms': wall_total_ms,
        'wall_per_evt_ms': wall_total_ms / counted if counted > 0 else 0.0,
        'evt_per_sec':   counted / (wall_total_ms / 1000.0) if wall_total_ms > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def print_report(args, all_runs):
    """
    all_runs: list of (batch_size, bd_stats_list) tuples.
    bd_stats_list: one dict per BD rank, as returned by run_mpi_gpu().
    """
    sep  = '═' * 90
    sep2 = '─' * 90
    n_bd = len(all_runs[0][1]) if all_runs else 0

    print(f'\n{sep}')
    print(f'  psana2 Multi-GPU MPI Performance Benchmark')
    print(f'  exp={args.exp}  run={args.run}  det={args.det}')
    print(f'  MPI topology: {_WORLD_SIZE} total ranks  '
          f'PS_EB_NODES={os.environ.get("PS_EB_NODES","1")}  '
          f'BD GPU ranks={n_bd}')
    print(f'  warmup={args.n_warmup}  timed_events={args.n_events}  '
          f'pool_depth={args.pool_depth}')
    print(sep)

    for bs, bd_stats in all_runs:
        if not bd_stats:
            continue

        total_events  = sum(s['n_events']      for s in bd_stats)
        max_wall_s    = max(s['wall_total_ms']  for s in bd_stats) / 1000.0
        aggregate_eps = total_events / max_wall_s if max_wall_s > 0 else 0.0
        avg_hot_ms    = float(np.mean([s['hot_mean_ms']     for s in bd_stats]))
        avg_wall_ms   = float(np.mean([s['wall_per_evt_ms'] for s in bd_stats]))

        print(f'\n  batch_size = {bs}')
        print(f'  {sep2}')
        print(f'  {"Rank":>6}  {"GPU":>5}  {"events":>7}  '
              f'{"hot ms":>8}  {"wall ms/evt":>12}  {"evt/s":>8}')
        for s in sorted(bd_stats, key=lambda x: x['bd_rank']):
            print(f'  {s["rank"]:>6}  {s["phys_gpu"]:>5}  {s["n_events"]:>7}  '
                  f'{s["hot_mean_ms"]:>8.3f}  {s["wall_per_evt_ms"]:>12.1f}  '
                  f'{s["evt_per_sec"]:>8.0f}')

        print(f'  {sep2}')
        print(f'  {"AGGREGATE":>6}  {"":>5}  {total_events:>7}  '
              f'{avg_hot_ms:>8.3f}  {avg_wall_ms:>12.1f}  '
              f'{aggregate_eps:>8.0f}  ← total events / max wall time')

        if args.single_gpu_baseline_ms > 0:
            baseline_eps = 1000.0 / args.single_gpu_baseline_ms
            ideal_eps    = n_bd * baseline_eps
            eff          = aggregate_eps / ideal_eps * 100.0
            bar          = _bar(eff, 100.0, width=30)
            print(f'\n  Single-GPU baseline:  {args.single_gpu_baseline_ms:.1f} ms/evt  '
                  f'({baseline_eps:.0f} evt/s)')
            print(f'  Ideal {n_bd}-GPU:       {ideal_eps:.0f} evt/s')
            print(f'  Actual aggregate:     {aggregate_eps:.0f} evt/s')
            print(f'  Scaling efficiency:   {eff:.1f}%  {bar}')

    # Batch-size scaling summary (if multiple runs)
    if len(all_runs) > 1:
        print(f'\n  {sep2}')
        print(f'  Batch-size scaling (aggregate evt/s, {n_bd} GPU BD ranks):')
        base_eps = None
        for bs, bd_stats in all_runs:
            if not bd_stats:
                continue
            total   = sum(s['n_events'] for s in bd_stats)
            max_w   = max(s['wall_total_ms'] for s in bd_stats) / 1000.0
            agg_eps = total / max_w if max_w > 0 else 0.0
            if base_eps is None:
                base_eps = agg_eps
            speedup = agg_eps / base_eps if base_eps > 0 else 0.0
            bar     = _bar(speedup, len(all_runs) * 3.0, width=30)
            print(f'  bs={bs:3d}  {agg_eps:8.0f} evt/s  {speedup:.2f}× vs bs=1  {bar}')

    print(f'\n{sep}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='psana2 multi-GPU MPI calibration benchmark'
    )
    p.add_argument('--exp',        default=_EXP)
    p.add_argument('--run',        type=int, default=_RUN)
    p.add_argument('--dir',        default=_DIR)
    p.add_argument('--det',        default=_DET)
    p.add_argument('--n-events',   type=int, default=50,
                   help='Events to time per BD rank per run (default 50)')
    p.add_argument('--n-warmup',   type=int, default=5,
                   help='Warmup events excluded from timing (default 5)')
    p.add_argument('--batch-size', type=int, default=10,
                   help='GPU batch size (default 10)')
    p.add_argument('--pool-depth', type=int, default=4,
                   help='EventPool depth / n_gpu_streams (default 4)')
    p.add_argument('--single-gpu-baseline-ms', type=float, default=0.0,
                   metavar='MS',
                   help='Single-GPU amortised ms/evt from gpu_performance_benchmark.py '
                        '(used to compute scaling efficiency)')
    p.add_argument('--scaling', action='store_true',
                   help='Run batch-size scaling sweep (bs = 1,2,5,10,20). '
                        'Each sweep entry restarts the MPI DataSource.')
    args = p.parse_args()

    kw = dict(
        exp=args.exp, run=args.run, xtc_dir=args.dir, det_name=args.det,
        n_gpu_streams=args.pool_depth, n_warmup=args.n_warmup,
        n_events=args.n_events,
    )

    if _WORLD_RANK == 0:
        n_bd = _WORLD_SIZE - 1 - int(os.environ.get('PS_EB_NODES', '1'))
        print(f'\nStarting multi-GPU MPI benchmark — '
              f'{_WORLD_SIZE} MPI ranks, {n_bd} GPU BD rank(s)', flush=True)

    batch_sizes = [1, 2, 5, 10, 20] if args.scaling else [args.batch_size]
    all_runs = []

    for bs in batch_sizes:
        if _WORLD_RANK == 0:
            print(f'  batch_size={bs} ...', flush=True)

        local_stats = run_mpi_gpu(batch_size=bs, **kw)

        # Gather all BD stats at rank 0.
        all_stats = _COMM.gather(local_stats, root=0)

        if _WORLD_RANK == 0:
            bd_stats = [s for s in (all_stats or []) if s]
            all_runs.append((bs, bd_stats))
            total = sum(s['n_events'] for s in bd_stats)
            max_w = max(s['wall_total_ms'] for s in bd_stats) if bd_stats else 0.0
            agg_str = (f'{total/(max_w/1000.):.0f} evt/s'
                       if max_w > 0 else 'n/a (no wall time)')
            print(f'    total={total} events  aggregate={agg_str}', flush=True)

    if _WORLD_RANK == 0:
        print_report(args, all_runs)


if __name__ == '__main__':
    main()
