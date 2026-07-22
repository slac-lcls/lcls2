"""
bench_pipeline_stages.py
════════════════════════
Per-stage breakdown of the psana2-gpu pipeline under user-facing knobs.

Answers, per configuration: where does each millisecond of an event go —
pipeline feed (EB wait + GDS/read + H2D), GPU calib kernel, GPU azint
kernel, result access, optional D2H — and how does that compare with the
kernel-only rates?  Run it with the knobs users actually turn:

  --batch_size N        events per EB batch
  --n_gpu_streams N     EventPool depth (stream pooling)
  --azint MODE          off | sorted | atomic   (registry 'azint' kernel)
  --azint-cm            common-mode variant ('azint_cm', needs raw)
  --nbins N             q bins
  --no-share-calib      disable CUDA-IPC constant sharing between BD ranks
  --d2h WHAT            none | azint | calib    (per-event D2H cost probe)

MPI topology (PS_EB_NODES=1): rank 0 SMD0, rank 1 EB, ranks 2+ BD.
Single node, 32 BD ranks on one A100:

  PS_EB_NODES=1 mpirun -n 34 python bench_pipeline_stages.py \\
      -e mfx101572426 -r 47 \\
      --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc \\
      --azint sorted --n_events 500

Stage semantics (per BD rank, mean over timed events)
  feed_ms    wall time from previous event yield to this one — EB wait,
             reads, kernel launches (kernels are async; overlap hides here)
  hot_ms     ctx.get('{det}.calib') / '.azint' access (no sync)
  d2h_ms     optional explicit .get() of the chosen result
  calib_ms   GPU time of the calib kernel   (CUDA-event pairs)
  azint_ms   GPU time of the azint kernel   (CUDA-event pairs)
"""

import argparse
import json
import os
import time

from mpi4py import MPI
import numpy as np

_COMM = MPI.COMM_WORLD
_RANK = _COMM.Get_rank()
_SIZE = _COMM.Get_size()

_EXP = 'mfx101572426'
_RUN = 47
_DIR = '/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc'
_DET = 'jungfrau'


# ---------------------------------------------------------------------------
# GPU-time proxies: wrap a registry kernel, record CUDA-event pairs around
# each dispatch on the dispatch stream.  Resolved after the loop (sync done).
# ---------------------------------------------------------------------------

class _TimedKernel:
    _MAX_PAIRS = 4000

    def __init__(self, kernel):
        object.__setattr__(self, '_kernel', kernel)
        object.__setattr__(self, 'pairs', [])

    def __getattr__(self, attr):
        return getattr(self._kernel, attr)

    def __setattr__(self, attr, value):
        setattr(self._kernel, attr, value)

    def _record(self, fn, stream, args, kwargs):
        import cupy as cp
        s_evt, e_evt = cp.cuda.Event(), cp.cuda.Event()
        strm = stream if stream is not None else cp.cuda.Stream.null
        s_evt.record(strm)
        result = fn(*args, **kwargs)
        e_evt.record(strm)
        if len(self.pairs) < self._MAX_PAIRS:
            self.pairs.append((s_evt, e_evt))
        return result

    def calibrate(self, *args, **kwargs):
        return self._record(self._kernel.calibrate,
                            kwargs.get('stream'), args, kwargs)

    def reduce(self, *args, **kwargs):
        return self._record(self._kernel.reduce,
                            kwargs.get('stream'), args, kwargs)

    def mean_ms(self, skip=20):
        import cupy as cp
        times = []
        for s_evt, e_evt in self.pairs[skip:]:
            try:
                times.append(cp.cuda.get_elapsed_time(s_evt, e_evt))
            except Exception:
                pass
        return float(np.mean(times)) if times else 0.0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('-e', '--exp', default=_EXP)
    ap.add_argument('-r', '--run', type=int, default=_RUN)
    ap.add_argument('--dir', default=_DIR)
    ap.add_argument('-d', '--det', default=_DET)
    ap.add_argument('--batch_size', type=int, default=10)
    ap.add_argument('--n_gpu_streams', type=int, default=4)
    ap.add_argument('--n_warmup', type=int, default=100)
    ap.add_argument('--n_events', type=int, default=500)
    ap.add_argument('--azint', choices=('off', 'sorted', 'atomic'),
                    default='off')
    ap.add_argument('--azint-cm', action='store_true')
    ap.add_argument('--nbins', type=int, default=256)
    ap.add_argument('--no-share-calib', action='store_true')
    ap.add_argument('--d2h', choices=('none', 'azint', 'calib'),
                    default='none')
    ap.add_argument('--json_out', default=None)
    ap.add_argument('--stats-dir', default=None,
                    help='Hang-proof mode: each BD rank writes '
                         'rank<N>.json here as soon as it finishes and no '
                         'MPI gather is attempted (the EB termination bug '
                         'can strand ranks; aggregate afterwards with '
                         '--report-dir).')
    ap.add_argument('--report-dir', default=None,
                    help='Aggregate rank*.json files from a --stats-dir '
                         'run and print the report (no MPI, run serially).')
    return ap.parse_args()


def main():
    args = parse_args()

    if args.report_dir:
        import glob
        ranks = []
        for path in sorted(glob.glob(os.path.join(args.report_dir,
                                                  'rank*.json'))):
            with open(path) as f:
                ranks.append(json.load(f))
        _report(args, ranks)
        return

    # Knob: disable CUDA-IPC constant sharing (A/B against default-on).
    if args.no_share_calib:
        import psana.gpu.gpu_mpi as _gm
        _gm.share_calib_between_gpu_peers = (
            lambda *a, **kw: True)   # leader semantics, no sharing

    # Knob: register the azint reduction kernel(s) before DataSource.
    from psana.gpu.gpu_kernel_registry import default_registry
    reg = default_registry()

    calib_proxy = _TimedKernel(reg.get(args.det, 'calib'))
    reg.register(calib_proxy)

    azint_proxy = None
    azint_key = None
    if args.azint != 'off':
        from psana.gpu.gpu_azint_kernel import JungfrauAzintKernel
        azint_proxy = _TimedKernel(JungfrauAzintKernel(
            nbins=args.nbins,
            method=args.azint,
            with_cm=args.azint_cm,
        ))
        reg.register(azint_proxy)
        azint_key = f'{args.det}.{azint_proxy.name}'

    from psana.psexp.mpi_ds import MPIDataSource
    from psana.psexp.node import Communicators

    comms = Communicators()
    n_bd = max(1, _SIZE - 1 - int(os.environ.get('PS_EB_NODES', '1')))
    ds = MPIDataSource(
        comms,
        exp=args.exp, run=args.run, dir=args.dir,
        gpu_det=args.det,
        batch_size=args.batch_size,
        n_gpu_streams=args.n_gpu_streams,
        max_events=(args.n_warmup + args.n_events) * n_bd,
    )

    if not ds.is_bd():
        for r in ds.runs():
            for _ in r.events():
                pass
        stats = None
    else:
        stats = _bd_loop(ds, args, comms, calib_proxy, azint_proxy,
                         azint_key)

    if args.stats_dir:
        # Hang-proof mode: persist per-rank stats immediately, skip the
        # collective (stranded ranks would deadlock the gather).
        if stats is not None:
            path = os.path.join(args.stats_dir,
                                f'rank{stats["bd_rank"]:03d}.json')
            with open(path, 'w') as f:
                json.dump(stats, f)
            print(f'[stages] rank {_RANK}: wrote {path}', flush=True)
        return

    print(f'[stages] rank {_RANK}: event loop done, entering gather',
          flush=True)
    all_stats = _COMM.gather(stats, root=0)
    if _RANK == 0:
        _report(args, all_stats)


def _bd_loop(ds, args, comms, calib_proxy, azint_proxy, azint_key):
    import cupy as cp

    calib_key = f'{args.det}.calib'
    feed_ms, hot_ms, d2h_ms = [], [], []
    wall_t0 = wall_t1 = None
    n = 0
    done = False
    t_prev = time.perf_counter()

    # Break out at the event target instead of draining the iterator:
    # with max_events truncation the GPU-path BD loop never receives a
    # termination signal and blocks forever waiting for the next batch
    # (framework bug — reproduced with azint off and sharing off).
    for r in ds.runs():
        if done:
            break
        for ctx in r.events():
            t_in = time.perf_counter()
            if n == args.n_warmup:
                wall_t0 = t_in

            t0 = time.perf_counter()
            calib = ctx.get(calib_key).on_gpu
            azint = ctx.get(azint_key).on_gpu if azint_key else None
            t1 = time.perf_counter()

            dt_d2h = 0.0
            if args.d2h != 'none':
                target = azint if (args.d2h == 'azint'
                                   and azint is not None) else calib
                td = time.perf_counter()
                _ = target.get()
                dt_d2h = (time.perf_counter() - td) * 1e3

            n += 1
            if n > args.n_warmup:
                feed_ms.append((t_in - t_prev) * 1e3)
                hot_ms.append((t1 - t0) * 1e3)
                d2h_ms.append(dt_d2h)
            t_prev = time.perf_counter()
            if n % 100 == 0:
                print(f'[stages] rank {_RANK}: {n} events', flush=True)
            if n == args.n_warmup + args.n_events:
                wall_t1 = t_prev
                done = True
                break

    if wall_t0 is not None and wall_t1 is None:
        wall_t1 = time.perf_counter()
    cp.cuda.Device().synchronize()

    counted = len(feed_ms)
    wall_ms = (wall_t1 - wall_t0) * 1e3 if wall_t0 is not None else 0.0
    stats = {
        'rank': _RANK,
        'bd_rank': comms.bd_rank,
        'phys_gpu': int(os.environ.get('CUDA_VISIBLE_DEVICES', '0')
                        .split(',')[0]),
        'n_events': counted,
        'feed_ms': float(np.mean(feed_ms)) if feed_ms else 0.0,
        'hot_ms': float(np.mean(hot_ms)) if hot_ms else 0.0,
        'd2h_ms': float(np.mean(d2h_ms)) if d2h_ms else 0.0,
        'calib_kernel_ms': calib_proxy.mean_ms(),
        'azint_kernel_ms': azint_proxy.mean_ms() if azint_proxy else 0.0,
        'wall_per_evt_ms': wall_ms / counted if counted else 0.0,
        'evt_per_sec': counted / (wall_ms / 1e3) if wall_ms > 0 else 0.0,
    }
    return stats


def _report(args, all_stats):
    bd = [s for s in all_stats if s and s.get('n_events', 0) > 0]
    if not bd:
        print('no BD stats collected')
        return

    cols = ('bd_rank', 'phys_gpu', 'n_events', 'feed_ms', 'hot_ms',
            'd2h_ms', 'calib_kernel_ms', 'azint_kernel_ms',
            'wall_per_evt_ms', 'evt_per_sec')
    hdr = ('bd', 'gpu', 'events', 'feed', 'hot', 'd2h',
           'k_calib', 'k_azint', 'wall/evt', 'Hz')
    print('\nconfig: azint=%s%s nbins=%d bs=%d streams=%d share=%s d2h=%s'
          % (args.azint, '+cm' if args.azint_cm else '', args.nbins,
             args.batch_size, args.n_gpu_streams,
             'off' if args.no_share_calib else 'on', args.d2h))
    print('  '.join(f'{h:>9s}' for h in hdr))
    for s in sorted(bd, key=lambda x: x['bd_rank']):
        print('  '.join(f'{s[c]:9.3f}' if isinstance(s[c], float)
                        else f'{s[c]:9d}' for c in cols))

    total_events = sum(s['n_events'] for s in bd)
    slowest_s = max(s['n_events'] / s['evt_per_sec']
                    for s in bd if s['evt_per_sec'] > 0)
    agg = total_events / slowest_s
    mean = {c: float(np.mean([s[c] for s in bd])) for c in cols[3:]}
    print(f'\naggregate: {total_events} events, {agg:.1f} Hz '
          f'({len(bd)} BD ranks)')
    print('per-event mean: feed %.2f  hot %.3f  d2h %.3f  '
          'k_calib %.3f  k_azint %.3f ms'
          % (mean['feed_ms'], mean['hot_ms'], mean['d2h_ms'],
             mean['calib_kernel_ms'], mean['azint_kernel_ms']))

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump({'config': vars(args), 'ranks': bd,
                       'aggregate_hz': agg}, f, indent=2)
        print(f'wrote {args.json_out}')


if __name__ == '__main__':
    main()
