"""
gpu_jungfrau_calib_example.py
══════════════════════════════
Demonstrates the §1a GPU-accelerated event loop with the full psana2
DataSource integration.

The user writes EXACTLY the same code as standard psana2 with two changes:
  1. Add  gpu_det='jungfrau'  to DataSource (or DataSource(files=..., ...))
  2. Use  ctx  instead of  evt  in the for loop

Before (standard psana2 CPU loop)
----------------------------------
    ds  = DataSource(files=xtc_files)
    run = next(ds.runs())
    det = run.Detector('jungfrau')
    for evt in run.events():
        calib = det.raw.calib(evt)              # NumPy float32, CPU
        n = np.sum(calib > 5.0)
        if n > 100:
            save(calib)

After (GPU-accelerated, two changes highlighted)
-------------------------------------------------
    from psana import DataSource                # unchanged
    import cupy as cp

    ds  = DataSource(files=xtc_files,
                     gpu_det='jungfrau')        # CHANGE 1: add gpu_det=

    for run in ds.runs():
        for ctx in run.events():               # CHANGE 2: ctx, not evt
            calib = ctx.get('jungfrau.calib').on_gpu    # CuPy, GPU
            n     = int(cp.sum(calib > 5.0))            # GPU compute
            if n > 100:
                save(ctx.get('jungfrau.calib').on_cpu)  # D→H only here

Run on a GPU node
-----------------
    export PS_TEST_GPU_STREAM_IDS=6,8,9,10,11
    cd ~/lcls2
    python psana/psana/gpu/gpu_jungfrau_calib_example.py
"""

import argparse
import os

import cupy as cp
import numpy as np

from psana import DataSource   # standard psana2 — unchanged


_DEFAULT_SMD_GLOB = os.environ.get('PSANA_GPU_TEST_SMD_GLOB', '')


def _parse_args():
    p = argparse.ArgumentParser(
        description='§1a GPU calibration — DataSource integration'
    )
    p.add_argument('--smd-glob', default=_DEFAULT_SMD_GLOB,
                   help='Glob for SMD .smd.xtc2 files '
                        '(default: $PSANA_GPU_TEST_SMD_GLOB; required to run)')
    p.add_argument('--threshold',  type=float, default=5.0)
    p.add_argument('--min-hits',   type=int,   default=100)
    p.add_argument('--batch-size', type=int,   default=5)
    p.add_argument('--max-events', type=int,   default=20)
    p.add_argument('--quiet',      action='store_true')
    return p.parse_args()


def main():
    args = _parse_args()

    import glob
    smd_files = sorted(glob.glob(args.smd_glob))
    smd_files = list(dict.fromkeys(smd_files))

    # ── §1a user code ────────────────────────────────────────────────────────
    # CHANGE 1: add gpu_det= to DataSource — everything else is unchanged.
    # Pass the SMD files so the SerialDataSource path is used (required for
    # the GPU BD pipeline which needs SMD-indexed bigdata file offsets).
    ds = DataSource(
        files=smd_files,
        gpu_det='jungfrau',                    # <-- new
        batch_size=args.batch_size,
        max_events=args.max_events,
    )

    n_hit = n_blank = 0

    for run in ds.runs():
        # CHANGE 2: run.events() now yields ctx instead of evt
        for ctx in run.events():
            # ctx.get() resolves 'calib' → '<gpu_det>.calib' via DetectorRouter.
            # The detector name is not repeated here — it's declared once above.
            calib   = ctx.get('calib').on_gpu                # CuPy float32
            n_above = int(cp.sum(calib > args.threshold))    # GPU compute

            if n_above > args.min_hits:
                n_hit += 1
                tag   = 'HIT  '
                # D→H only for confirmed hits
                _ = ctx.get('calib').on_cpu                  # D→H here
            else:
                n_blank += 1
                tag = 'blank'

            if not args.quiet:
                print(f'[{tag}] ts={ctx.timestamp}  '
                      f'shape={tuple(calib.shape)}  '
                      f'bright_pixels={n_above}  '
                      f'mean={float(calib.mean()):.2f}')
    # ── end user code ────────────────────────────────────────────────────────

    total = n_hit + n_blank
    if total == 0:
        print('WARNING: no events — check PS_TEST_GPU_STREAM_IDS and files')
        return
    print(f'\nProcessed {total} events: '
          f'{n_hit} hits ({100*n_hit/total:.1f}%)  '
          f'{n_blank} blanks')


if __name__ == '__main__':
    main()
