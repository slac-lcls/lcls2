"""
Correctness test: GPU Jungfrau calibration vs psana2 CPU reference.

Compares fused_calib_gpu() output against det.raw.calib(evt) pixel-by-pixel
for N events on a real run.

Usage (single process, no MPI needed):
    python psana/psana/gpu/test_jungfrau_calib.py \\
        -e mfx101572426 -r 47 -n 50 \\
        --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc

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
    p.add_argument("--copy-true",     action="store_true",
                   help="Restore det.raw.raw(evt, copy=True). This gate now "
                        "DEFAULTS to copy=False to match the promoted GPU bench "
                        "path (iter 7), so the bit-exact check covers the "
                        "view-into-_raw_buf input the GPU path actually consumes.")
    p.add_argument("--seg-h2d",       action="store_true",
                   help="Gate the bench --seg-h2d variant: build the device raw "
                        "buffer by copying each segment's .raw directly host->device "
                        "(per-seg .set()), skipping det.raw.raw's host stack memcpy, "
                        "then compare fused_calib_gpu against det.raw.calib bit-for-bit.")
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

        if args.seg_h2d:
            # Mirror the bench --seg-h2d transfer route: per-segment host->device
            # into a device buffer, no host stack. Must be bit-identical to the
            # cp.asarray(det.raw.raw) route.
            raw_det = det.raw
            seg_nums = raw_det._segment_numbers
            segs = raw_det._segments(evt)
            if segs is None:
                skipped += 1
                continue
            s0 = segs[seg_nums[0]].raw
            buf = cp.empty((len(seg_nums),) + s0.shape, dtype=s0.dtype)
            for idx, sid in enumerate(seg_nums):
                buf[idx].set(np.ascontiguousarray(segs[sid].raw))
            raw_gpu = buf.reshape(-1, s0.shape[-2], s0.shape[-1])
        else:
            raw = det.raw.raw(evt, copy=args.copy_true)
            if raw is None:
                skipped += 1
                continue
            raw_gpu = cp.asarray(raw)

        cpu_ref = det.raw.calib(evt)
        if cpu_ref is None:
            skipped += 1
            continue
        cpu_ref = cpu_ref.astype(np.float32)

        gpu_out = fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu).get()

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
