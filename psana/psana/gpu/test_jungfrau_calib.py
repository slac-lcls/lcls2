"""
Correctness test: GPU Jungfrau calibration vs psana2 CPU reference.

Compares fused_calib_gpu() output against det.raw.calib(evt) pixel-by-pixel
for N events on a real run.

Usage (single process, no MPI needed):
    python psana/psana/gpu/test_jungfrau_calib.py \\
        -e mfx101210926 -r 387 -n 50

Pass criterion: zero pixel mismatches at atol=1e-3 across all tested events.

Known acceptable divergences:
- Common-mode correction: det.raw.calib() may apply common-mode by default
  depending on calibration store settings; GPU path does not. If mismatches
  appear only in the ~10 ADU range uniformly across a panel, run with
  --no-common-mode to confirm.
- Masked pixels: CPU path may zero-fill pixels that GPU path leaves as 0.0
  from bad gain bits (0b10). These should agree.
"""

import argparse
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="GPU vs CPU Jungfrau calib correctness test")
    p.add_argument("-e", "--exp",     required=True,  help="Experiment name")
    p.add_argument("-r", "--run",     required=True, type=int, help="Run number")
    p.add_argument("-n", "--nevents", default=50,    type=int, help="Events to check")
    p.add_argument("--det",           default="jungfrau", help="Detector name")
    p.add_argument("--atol",          default=1e-3,  type=float, help="Absolute tolerance")
    p.add_argument("--dir",           default=None,  help="XTC directory (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    # GPU pinning before CuPy import (no-op on single-process runs).
    from psana.gpu import init_gpu_rank
    init_gpu_rank()

    import cupy as cp
    from psana import DataSource
    from psana.gpu import prep_calib_constants, fused_calib_gpu

    kwargs = dict(exp=args.exp, run=args.run)
    if args.dir:
        kwargs["dir"] = args.dir
    ds  = DataSource(**kwargs)
    run = next(ds.runs())
    det = run.Detector(args.det)

    peds_gpu, gmask_gpu = prep_calib_constants(det)
    print(f"peds_gpu shape: {peds_gpu.shape}  dtype: {peds_gpu.dtype}")
    print(f"gmask_gpu shape: {gmask_gpu.shape}  dtype: {gmask_gpu.dtype}")
    print(f"Checking {args.nevents} events at atol={args.atol} ...")

    checked = 0
    skipped = 0
    mismatches = 0
    max_diff_seen = 0.0

    for i, evt in enumerate(run.events()):
        if checked >= args.nevents:
            break

        raw = det.raw.raw(evt)
        if raw is None:
            skipped += 1
            continue

        cpu_ref = det.raw.calib(evt)
        if cpu_ref is None:
            skipped += 1
            continue
        cpu_ref = cpu_ref.astype(np.float32)

        gpu_out = fused_calib_gpu(cp.asarray(raw), peds_gpu, gmask_gpu).get()

        diff = np.abs(cpu_ref - gpu_out)
        max_diff = float(diff.max())
        max_diff_seen = max(max_diff_seen, max_diff)

        if not np.allclose(cpu_ref, gpu_out, atol=args.atol, rtol=0):
            mismatches += 1
            bad_pixels = int((diff > args.atol).sum())
            print(f"  evt {i}: MISMATCH — {bad_pixels} pixels > atol, "
                  f"max_diff={max_diff:.6f}")
        else:
            print(f"  evt {i}: OK  max_diff={max_diff:.6f}")

        checked += 1

    print()
    print(f"Result: {checked} events checked, {skipped} skipped, "
          f"{mismatches} mismatches, max_diff_seen={max_diff_seen:.6f}")

    if mismatches:
        print("FAIL")
        sys.exit(1)
    else:
        print("PASS")


if __name__ == "__main__":
    main()
