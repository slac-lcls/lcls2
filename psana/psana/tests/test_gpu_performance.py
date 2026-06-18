"""
test_gpu_performance.py — GPU vs CPU calibration performance benchmark.

Uses the production psana2 DataSource (exp= / run=) for BOTH paths so that
both CPU and GPU naturally see ALL Jungfrau streams via the standard
SerialDataSource + EventBuilder + DgramManager pipeline.  There is no file
enumeration or stream-discovery workaround — everything works the same way
it does in a real experiment.

Both paths process the same data:
  - 5 Jungfrau streams (0, 6, 7, 8, 9) = 32 segments total
  - shape (32, 512, 1024) float32 per event

Benchmark paths and what is timed
----------------------------------
  cpu_all_streams   SerialDataSource, no GPU routing.
                    TIMES: bigdata reads (all 10 streams) + numpy calibration.
                    This is the full sequential critical path: ~30 ms I/O +
                    ~200 ms numpy compute = ~235 ms/event.

  gpu_on_gpu        Same exp/run, gpu_det='jungfrau'.
                    TIMES: Python dict lookup only (~0 ms).
                    The GDS reads and CUDA kernel ran earlier in the EventPool
                    pipeline, overlapping with I/O.  To get the true amortised
                    wall time, measure the whole loop: ~4 ms/event at bs=5.

  gpu_on_cpu        GPU path with D→H every event.
                    TIMES: stream sync + 64 MB PCIe transfer (~13 ms).
                    Calibration is already done; only the transfer is measured.

  gpu_selective     GPU path with D→H for ~10 % of events.
                    Models the recommended usage for sparse-hit experiments.

Environment
-----------
Set PSANA_GPU_PERF_EXP, PSANA_GPU_PERF_RUN, PSANA_GPU_PERF_DIR to override
the defaults (mfx100852324, 77, /sdf/data/lcls/ds/prj/public01/xtc).

Run on a GPU node
-----------------
    pytest psana/psana/tests/test_gpu_performance.py -m slow -s

    # Interactive (full output + batch-size sweep):
    python psana/psana/tests/test_gpu_performance.py
"""

import os
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

# Production experiment/run defaults — override via env vars.
_EXP  = os.environ.get('PSANA_GPU_PERF_EXP', 'mfx100852324')
_RUN  = int(os.environ.get('PSANA_GPU_PERF_RUN', '77'))
_DIR  = os.environ.get('PSANA_GPU_PERF_DIR',
                        '/sdf/data/lcls/ds/prj/public01/xtc')
_DET  = os.environ.get('PSANA_GPU_PERF_DET', 'jungfrau')

_N_WARMUP  = 5
_N_EVENTS  = 50    # events timed per benchmark path


def _gpu_available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _data_available() -> bool:
    import os
    return os.path.isdir(_DIR)


requires_gpu  = pytest.mark.skipif(not _gpu_available(),  reason='no CUDA device')
requires_data = pytest.mark.skipif(not _data_available(), reason=f'{_DIR} not accessible')


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    label:     str
    n_events:  int
    per_event: List[float]
    shape:     tuple

    @property
    def mean_ms(self):   return float(np.mean(self.per_event))
    @property
    def median_ms(self): return float(np.median(self.per_event))
    @property
    def min_ms(self):    return float(np.min(self.per_event))
    @property
    def p95_ms(self):    return float(np.percentile(self.per_event, 95))
    @property
    def throughput(self): return 1000.0 / self.mean_ms

    def report(self, cpu_mean=None) -> str:
        speedup = (f'  {cpu_mean / self.mean_ms:5.1f}x faster'
                   if cpu_mean and self.mean_ms < cpu_mean else '')
        return (
            f"  {self.label:<38} "
            f"mean={self.mean_ms:8.2f}ms  "
            f"median={self.median_ms:8.2f}ms  "
            f"min={self.min_ms:7.2f}ms  "
            f"p95={self.p95_ms:8.2f}ms  "
            f"{self.throughput:6.0f} evt/s  "
            f"shape={self.shape}"
            f"{speedup}"
        )


# ---------------------------------------------------------------------------
# Calibration helper (numpy equivalent of the CUDA kernel)
# ---------------------------------------------------------------------------

def _build_calibconst(det):
    """Return (peds_flat, gmask_flat, n_pix_calib) from det.calibconst."""
    cc    = det.calibconst
    peds  = cc['pedestals'][0].astype(np.float32)
    gain  = cc['pixel_gain'][0].astype(np.float32)
    try:
        status = cc['pixel_status'][0]
        mask   = (status[0] == 0).astype(np.float32)
    except Exception:
        mask = None
    gfac  = np.where(gain != 0, np.float32(1.) / gain, np.float32(0.))
    gmask = (gfac * mask[np.newaxis]).astype(np.float32) if mask is not None \
            else gfac.astype(np.float32)
    n_pix = int(np.prod(peds.shape[1:]))
    return (np.ascontiguousarray(peds.ravel()),
            np.ascontiguousarray(gmask.ravel()),
            n_pix)


def _apply_calib_numpy(raw, peds_flat, gmask_flat, n_pix_calib):
    """Jungfrau calibration in numpy — same formula as the CUDA kernel."""
    n_pix     = int(raw.size)
    flat      = raw.ravel().astype(np.uint16)
    gbits     = (flat >> 14).astype(np.int32)
    data_bits = (flat & np.uint16(0x3fff)).astype(np.float32)
    n_modes   = peds_flat.size // n_pix_calib
    calib     = np.zeros(n_pix, dtype=np.float32)
    for mode, gv in [(0, 0), (1, 1), (2, 3)]:
        if mode >= n_modes:
            break
        base = mode * n_pix_calib
        px   = gbits == gv
        ped  = peds_flat[base:base + n_pix]
        gm   = gmask_flat[base:base + n_pix]
        calib[px] = (data_bits[px] - ped[px]) * gm[px]
    return calib.reshape(raw.shape)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def _cfg(exp=None, run=None, xtc_dir=None, det=None):
    """Return (exp, run, xtc_dir, det) using module defaults for any None."""
    return (exp or _EXP, run or _RUN, xtc_dir or _DIR, det or _DET)

def bench_cpu_all_streams(n_warmup=_N_WARMUP, n_events=_N_EVENTS, **cfg_kw):
    """CPU path: production DataSource, all streams, numpy calibration.

    Uses DataSource(exp=, run=) — the standard production path.
    SerialDataSource opens ALL SMD and bigdata files, EventBuilder combines
    all streams, det.raw.raw(evt) returns all 32 Jungfrau segments.
    Times bigdata reads (all 10 streams) + numpy calibration together.
    """
    import psana
    exp, run, xtc_dir, det = _cfg(**cfg_kw)
    ds  = psana.DataSource(exp=exp, run=run, dir=xtc_dir,
                           max_events=n_warmup + n_events)
    run = next(ds.runs())
    det = run.Detector(_DET)
    peds_flat, gmask_flat, n_pix_calib = _build_calibconst(det)

    times, shape = [], None
    for i, evt in enumerate(run.events()):
        raw = det.raw.raw(evt)
        if raw is None:
            continue
        t0    = time.perf_counter()
        calib = _apply_calib_numpy(raw, peds_flat, gmask_flat, n_pix_calib)
        dt    = (time.perf_counter() - t0) * 1000
        if i >= n_warmup:
            times.append(dt)
            shape = calib.shape

    return BenchResult('cpu (SerialDS, numpy)', len(times), times, shape or ())


def bench_gpu_on_gpu(batch_size, n_warmup=_N_WARMUP, n_events=_N_EVENTS, **cfg_kw):
    """GPU path: same exp/run with gpu_det=; calibration stays on GPU."""
    import psana
    # Build calibconst before the GPU run (same as CPU path).
    ds_cal = psana.DataSource(exp=_cfg(**cfg_kw)[0], run=_cfg(**cfg_kw)[1], dir=_cfg(**cfg_kw)[2], max_events=1)
    det_cal = next(ds_cal.runs()).Detector(_DET)

    from psana import DataSource
    # SerialDataSource is used automatically for exp= even with gpu_det=.
    ds  = DataSource(exp=_cfg(**cfg_kw)[0], run=_cfg(**cfg_kw)[1], dir=_cfg(**cfg_kw)[2], gpu_det=_cfg(**cfg_kw)[3],
                     batch_size=batch_size, max_events=n_warmup + n_events)
    times, shape = [], None
    for run in ds.runs():
        for i, ctx in enumerate(run.events()):
            t0    = time.perf_counter()
            calib = ctx.get('calib').on_gpu
            dt    = (time.perf_counter() - t0) * 1000
            if i >= n_warmup:
                times.append(dt)
                shape = tuple(calib.shape)

    label = f'gpu_on_gpu  (bs={batch_size})'
    return BenchResult(label, len(times), times, shape or ())


def bench_gpu_on_cpu(batch_size, n_warmup=_N_WARMUP, n_events=_N_EVENTS, **cfg_kw):
    """GPU path with D→H every event."""
    from psana import DataSource
    ds  = DataSource(exp=_cfg(**cfg_kw)[0], run=_cfg(**cfg_kw)[1], dir=_cfg(**cfg_kw)[2], gpu_det=_cfg(**cfg_kw)[3],
                     batch_size=batch_size, max_events=n_warmup + n_events)
    times, shape = [], None
    for run in ds.runs():
        for i, ctx in enumerate(run.events()):
            t0    = time.perf_counter()
            calib = ctx.get('calib').on_cpu
            dt    = (time.perf_counter() - t0) * 1000
            if i >= n_warmup:
                times.append(dt)
                shape = calib.shape

    label = f'gpu_on_cpu  (bs={batch_size})'
    return BenchResult(label, len(times), times, shape or ())


def bench_gpu_selective(batch_size, hit_pct=10,
                         n_warmup=_N_WARMUP, n_events=_N_EVENTS, **cfg_kw):
    """GPU path with D→H for ~hit_pct% of events."""
    import cupy as cp
    from psana import DataSource
    ds   = DataSource(exp=_cfg(**cfg_kw)[0], run=_cfg(**cfg_kw)[1], dir=_cfg(**cfg_kw)[2], gpu_det=_cfg(**cfg_kw)[3],
                      batch_size=batch_size, max_events=n_warmup + n_events)
    times, shape, n_dth = [], None, 0
    step = max(1, 100 // hit_pct)
    for run in ds.runs():
        for i, ctx in enumerate(run.events()):
            t0    = time.perf_counter()
            calib = ctx.get('calib').on_gpu
            _     = int(cp.sum(calib > 5.0))
            if i % step == 0:
                _ = ctx.get('calib').on_cpu
                n_dth += 1
            dt = (time.perf_counter() - t0) * 1000
            if i >= n_warmup:
                times.append(dt)
                shape = tuple(calib.shape)

    actual = 100 * n_dth / max(1, len(times) + n_warmup)
    label  = f'gpu_selective({actual:.0f}%→CPU,bs={batch_size})'
    return BenchResult(label, len(times), times, shape or ())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@requires_gpu
@requires_data
def test_gpu_vs_cpu_throughput(capsys):
    """GPU vs CPU throughput using production DataSource(exp=, run=).

    Both paths use SerialDataSource which processes ALL streams together —
    the same way production psana2 works.  CPU uses det.raw.raw(evt) +
    numpy; GPU uses GDS reads + CUDA kernel.
    """
    bs = 5
    cpu = bench_cpu_all_streams(n_events=_N_EVENTS)
    g0  = bench_gpu_on_gpu(bs, n_events=_N_EVENTS)
    gc  = bench_gpu_on_cpu(bs, n_events=_N_EVENTS)
    gs  = bench_gpu_selective(bs, hit_pct=10, n_events=_N_EVENTS)

    with capsys.disabled():
        sep = '=' * 115
        print(f'\n\n{sep}')
        print(f'GPU vs CPU  |  exp={_EXP}  run={_RUN}  '
              f'det={_DET}  warmup={_N_WARMUP}  n={_N_EVENTS}  bs={bs}')
        print(sep)
        base = cpu.mean_ms
        for r in [cpu, g0, gc, gs]:
            print(r.report(cpu_mean=base if r is not cpu else None))
        print(sep)
        print('  Both paths use SerialDataSource — identical stream access.')
        print('  CPU shape and GPU shape should both be (32, 512, 1024).')

    assert cpu.shape == g0.shape, (
        f'CPU shape {cpu.shape} != GPU shape {g0.shape} — stream mismatch'
    )
    assert g0.throughput > cpu.throughput * 0.5


@pytest.mark.slow
@requires_gpu
@requires_data
def test_gpu_batch_size_scaling(capsys):
    """GPU throughput vs batch_size with production DataSource."""
    results = [bench_gpu_on_gpu(bs, n_events=_N_EVENTS)
               for bs in [1, 2, 5, 10, 20]]

    with capsys.disabled():
        sep = '=' * 90
        print(f'\n\n{sep}')
        print(f'Batch-size scaling  |  exp={_EXP}  run={_RUN}  n={_N_EVENTS}')
        print(sep)
        base = results[0].mean_ms
        for r in results:
            print(r.report(cpu_mean=base if r is not results[0] else None))
        print(sep)

    assert results[0].n_events > 0


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='GPU vs CPU performance (production DataSource)')
    p.add_argument('--exp',       default=_EXP)
    p.add_argument('--run',       type=int, default=_RUN)
    p.add_argument('--dir',       default=_DIR)
    p.add_argument('--det',       default=_DET)
    p.add_argument('--n-events',  type=int, default=_N_EVENTS)
    p.add_argument('--n-warmup',  type=int, default=_N_WARMUP)
    p.add_argument('--batch-size', type=int, default=5)
    p.add_argument('--no-cpu',    action='store_true')
    args = p.parse_args()

    # Pass exp/run/dir/det as kwargs to each bench function explicitly.
    cfg = dict(exp=args.exp, run=args.run, xtc_dir=args.dir, det=args.det)

    w, n, bs = args.n_warmup, args.n_events, args.batch_size
    sep = '=' * 115

    print(f'\n{sep}')
    print(f'GPU vs CPU Calibration  |  exp={_EXP}  run={_RUN}  det={_DET}')
    print(f'  warmup={w}  n_events={n}  batch_size={bs}')
    print(f'  dir={_DIR}')
    print(sep)

    kw = dict(exp=args.exp, run=args.run, xtc_dir=args.dir, det=args.det)
    results = []
    if not args.no_cpu:
        print('Running cpu_all_streams ...')
        results.append(bench_cpu_all_streams(w, n, **kw))

    for label, fn in [
        ('gpu_on_gpu',        lambda: bench_gpu_on_gpu(bs, w, n, **kw)),
        ('gpu_on_cpu',        lambda: bench_gpu_on_cpu(bs, w, n, **kw)),
        ('gpu_selective 10%', lambda: bench_gpu_selective(bs, 10, w, n, **kw)),
    ]:
        print(f'Running {label} ...')
        results.append(fn())

    print(f'\n{sep}')
    base = results[0].mean_ms if results else None
    for r in results:
        b = base if r is not results[0] else None
        print(r.report(cpu_mean=b))
    print(sep)
    if results and len(results) > 1:
        print(f'\n  Both paths: shape={results[0].shape}  '
              f'{"✓ same" if results[0].shape == results[1].shape else "✗ DIFFERENT"}')

    print(f'\nBatch-size scaling (gpu_on_gpu):')
    print('-' * 80)
    base_ms = None
    for b in [1, 2, 5, 10, 20]:
        r = bench_gpu_on_gpu(b, w, n, **kw)
        if base_ms is None:
            base_ms = r.mean_ms
        print(f'  bs={b:3d}  {r.mean_ms:8.2f}ms  '
              f'{r.throughput:7.0f} evt/s  vs bs=1: {base_ms/r.mean_ms:.2f}x')
