#!/usr/bin/env python
"""
test_analysis_kernels.py — verify and time cuda/analysis_kernels.cu.

Synthetic Jungfrau-shaped data (no psana, no data source — runs on any GPU
node): random 14-bit ADC values with random gain bits, random pedestals and
gain*mask constants, and a q-bin map computed from a tiled-panel geometry.
Each fused kernel is checked against a numpy float64 reference, then timed.

Usage (on a GPU node):
    python test_analysis_kernels.py                    # 8-segment quick check
    python test_analysis_kernels.py --segs 32          # full 16.8 Mpix JF
    python test_analysis_kernels.py --nbins 512 --iters 200

This file is standalone on purpose: together with cuda/analysis_kernels.cu
and cuda/fused_calib.cuh it can be dropped into any branch's psana/psana/gpu/
to run identical kernels on both GPU data paths.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

CUDA_DIR = Path(__file__).parent / 'cuda'

ROWS, COLS = 512, 1024          # Jungfrau 0.5M panel
BANK_ROWS, BANK_COLS, NBANKS_PER_ROW = 256, 64, 16
NBANKS_TOTAL = (ROWS // BANK_ROWS) * NBANKS_PER_ROW
TPB = 256


# ---------------------------------------------------------------------------
# Synthetic data + geometry
# ---------------------------------------------------------------------------

def make_data(nsegs, seed):
    rng = np.random.default_rng(seed)
    npix = nsegs * ROWS * COLS

    adc = rng.integers(0, 0x4000, npix, dtype=np.uint16)
    gain_bits = rng.choice(np.array([0, 1, 3, 2], dtype=np.uint16),
                           npix, p=[0.90, 0.05, 0.04, 0.01])
    raw = (adc | (gain_bits << 14)).astype(np.uint16)

    peds = rng.normal(2000, 100, (3, npix)).astype(np.float32)
    gain = rng.uniform(0.5, 2.0, (3, npix)).astype(np.float32)
    mask = (rng.random(npix) > 0.02)            # ~2% bad pixels
    gmask = (gain * mask[None, :]).astype(np.float32)
    return raw, peds, gmask, mask


def make_bin_idx(nsegs, nbins, mask, dist=0.1, wavelength=1e-10, pixel=75e-6):
    """Per-pixel q-bin index from a simple tiled-panel geometry (the
    once-per-run CPU precompute; with real data this comes from psana
    pixel coords instead).  Returns (bin_idx int32, q bin centers)."""
    panels_per_row = 4
    iy, ix = np.mgrid[0:ROWS, 0:COLS]
    x = np.concatenate([(ix + (s % panels_per_row) * COLS).ravel()
                        for s in range(nsegs)]).astype(np.float64)
    y = np.concatenate([(iy + (s // panels_per_row) * ROWS).ravel()
                        for s in range(nsegs)]).astype(np.float64)
    x = (x - x.mean()) * pixel
    y = (y - y.mean()) * pixel

    r = np.hypot(x, y)
    q = 4 * np.pi / (wavelength * 1e10) * np.sin(np.arctan2(r, dist) / 2)

    edges = np.linspace(q[mask].min(), q[mask].max(), nbins + 1)
    bin_idx = np.clip(np.digitize(q, edges) - 1, 0, nbins - 1)
    bin_idx = np.where(mask, bin_idx, -1).astype(np.int32)
    return bin_idx, 0.5 * (edges[:-1] + edges[1:])


# ---------------------------------------------------------------------------
# CPU references (float64 accumulation)
# ---------------------------------------------------------------------------

def calib_cpu(raw, peds, gmask):
    """Mirror of psana_gpu::jungfrau_calib_pixel."""
    gb = (raw >> 14).astype(np.intp)
    mode = np.choose(gb, [0, 1, 0, 2])          # 00->0, 01->1, 11->2, 10->bad
    idx = np.arange(raw.size)
    v = ((raw & 0x3FFF).astype(np.float32) - peds[mode, idx]) * gmask[mode, idx]
    v[gb == 2] = 0.0
    return v


def azint_cpu(v, bin_idx, nbins):
    sel = bin_idx >= 0
    sum_I = np.bincount(bin_idx[sel], weights=v[sel].astype(np.float64),
                        minlength=nbins)
    sum_N = np.bincount(bin_idx[sel], minlength=nbins).astype(np.float64)
    return sum_I, sum_N


def common_mode_cpu(v, raw, gmask, nsegs, cormax, min_pixels):
    """Bank-mean common mode, matching fused_calib_cm_azint_kernel pass 1."""
    v = v.reshape(nsegs, ROWS, COLS).copy()
    raw = raw.reshape(nsegs, ROWS, COLS)
    gmask0 = gmask[0].reshape(nsegs, ROWS, COLS)
    for s in range(nsegs):
        for half in range(ROWS // BANK_ROWS):
            for b in range(NBANKS_PER_ROW):
                r0, c0 = half * BANK_ROWS, b * BANK_COLS
                bank = (s, slice(r0, r0 + BANK_ROWS), slice(c0, c0 + BANK_COLS))
                sel = ((raw[bank] < 0x4000) & (gmask0[bank] != 0)
                       & (np.abs(v[bank]) < cormax))
                if sel.sum() >= min_pixels:
                    v[bank] -= np.float32(v[bank][sel].mean(dtype=np.float64))
    return v.ravel()


# ---------------------------------------------------------------------------
# GPU driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--segs', type=int, default=8, help='panels (32 = full JF16M)')
    ap.add_argument('--nbins', type=int, default=256)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--cormax', type=float, default=100.0)
    ap.add_argument('--min-pixels', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import cupy as cp

    nsegs, nbins = args.segs, args.nbins
    npix = nsegs * ROWS * COLS
    print(f'{nsegs} segs = {npix / 1e6:.1f} Mpix, {nbins} q-bins')

    raw, peds, gmask, mask = make_data(nsegs, args.seed)
    bin_idx, _q = make_bin_idx(nsegs, nbins, mask)

    mod = cp.RawModule(code=(CUDA_DIR / 'analysis_kernels.cu').read_text(),
                       options=('--std=c++17', f'-I{CUDA_DIR}'))
    k_azint = mod.get_function('fused_calib_azint_kernel')
    k_cm = mod.get_function('fused_calib_cm_azint_kernel')

    raw_d = cp.asarray(raw)
    peds_d = cp.asarray(peds.ravel())
    gmask_d = cp.asarray(gmask.ravel())
    bin_d = cp.asarray(bin_idx)
    sum_I = cp.zeros(nbins, dtype=cp.float32)
    sum_N = cp.zeros(nbins, dtype=cp.float32)

    grid_1d = ((npix + TPB - 1) // TPB,)
    azint_args = (raw_d, peds_d, gmask_d, bin_d, sum_I, sum_N, np.uint64(npix))
    cm_args = (raw_d, peds_d, gmask_d, bin_d, sum_I, sum_N,
               np.int32(ROWS * COLS), np.int32(COLS),
               np.int32(BANK_ROWS), np.int32(BANK_COLS),
               np.int32(NBANKS_PER_ROW),
               np.float32(args.cormax), np.int32(args.min_pixels),
               np.uint64(npix))

    def launch(which):
        sum_I.fill(0)
        sum_N.fill(0)
        if which == 'azint':
            k_azint(grid_1d, (TPB,), azint_args)
        else:
            k_cm((nsegs, NBANKS_TOTAL), (TPB,), cm_args)

    # --- correctness --------------------------------------------------------
    v = calib_cpu(raw, peds, gmask)
    refs = {
        'azint': azint_cpu(v, bin_idx, nbins),
        'cm':    azint_cpu(common_mode_cpu(v, raw, gmask, nsegs,
                                           args.cormax, args.min_pixels),
                           bin_idx, nbins),
    }

    failed = False
    for which, (ref_I, ref_N) in refs.items():
        launch(which)
        got_I, got_N = sum_I.get().astype(np.float64), sum_N.get()
        ok_N = np.array_equal(got_N, ref_N)
        # float32 atomics reorder sums; per-bin averages agree to ~1e-4
        denom = np.maximum(ref_N, 1)
        err = np.abs(got_I - ref_I) / denom / max(1.0, np.abs(ref_I / denom).max())
        ok_I = err.max() < 1e-4
        print(f'{which:5s}  counts {"exact" if ok_N else "MISMATCH"},  '
              f'intensity max rel err {err.max():.2e}  '
              f'{"PASS" if ok_N and ok_I else "FAIL"}')
        failed |= not (ok_N and ok_I)

    # --- timing -------------------------------------------------------------
    gb_in = npix * 2 / 1e9                       # raw uint16 traffic
    for which in refs:
        for _ in range(10):
            launch(which)
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        for _ in range(args.iters):
            launch(which)
        stop.record()
        stop.synchronize()
        ms = cp.cuda.get_elapsed_time(start, stop) / args.iters
        print(f'{which:5s}  {ms:7.3f} ms/event  {1e3 / ms:8.1f} Hz  '
              f'{gb_in / (ms / 1e3):6.1f} GB/s raw in')

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
