#!/usr/bin/env python
"""
test_azint_kernel.py — verify JungfrauAzintKernel.reduce() against numpy.

Standalone GPU test: stubs the detector objects, synthetic calibrated data,
no data source.  Covers method='sorted', method='atomic', and with_cm=True,
plus per-call timing.

Usage (GPU node, worktree env):
    python test_azint_kernel.py [--segs 8] [--nbins 256] [--iters 100]
"""

import argparse
import sys

import numpy as np

ROWS, COLS = 512, 1024


class _StubRaw:
    def _pixel_coord_indexes(self):
        raise RuntimeError('no geometry in stub')   # forces tiled fallback


class _StubDet:
    raw = _StubRaw()


class _StubGPUDetector:
    def __init__(self, det_shape, gmask_gpu):
        self.det_shape = det_shape
        self.gmask_gpu = gmask_gpu


def cm_reference(v, raw, gmask0, nsegs, cormax, min_pixels):
    v = v.reshape(nsegs, ROWS, COLS).copy()
    raw = raw.reshape(nsegs, ROWS, COLS)
    gmask0 = gmask0.reshape(nsegs, ROWS, COLS)
    for s in range(nsegs):
        for half in range(2):
            for b in range(16):
                r0, c0 = half * 256, b * 64
                bank = (s, slice(r0, r0 + 256), slice(c0, c0 + 64))
                sel = ((raw[bank] < 0x4000) & (gmask0[bank] != 0)
                       & (np.abs(v[bank]) < cormax))
                if sel.sum() >= min_pixels:
                    v[bank] -= np.float32(v[bank][sel].mean(dtype=np.float64))
    return v.ravel()


def azint_reference(v, bin_idx, nbins):
    sel = bin_idx >= 0
    sum_I = np.bincount(bin_idx[sel], weights=v[sel].astype(np.float64),
                        minlength=nbins)
    sum_N = np.bincount(bin_idx[sel], minlength=nbins).astype(np.float64)
    with np.errstate(invalid='ignore'):
        avg = np.where(sum_N > 0, sum_I / np.maximum(sum_N, 1), 0.0)
    return avg, sum_I, sum_N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segs', type=int, default=8)
    ap.add_argument('--nbins', type=int, default=256)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import cupy as cp
    from psana.gpu.gpu_azint_kernel import JungfrauAzintKernel

    nsegs, nbins = args.segs, args.nbins
    npix = nsegs * ROWS * COLS
    rng = np.random.default_rng(args.seed)

    calib = rng.normal(50, 30, npix).astype(np.float32)
    raw = rng.integers(0, 0x4000, npix, dtype=np.uint16)
    raw[rng.random(npix) < 0.05] |= 0x4000            # some non-gain-0 pixels
    mask = rng.random(npix) > 0.02
    gmask = np.tile(mask.astype(np.float32), 3) \
        * rng.uniform(0.5, 2.0, 3 * npix).astype(np.float32)

    gmask_d = cp.asarray(gmask)
    calib_d = cp.asarray(calib)
    raw_d = cp.asarray(raw)
    stub_gpu_det = _StubGPUDetector((nsegs, ROWS, COLS), gmask_d)

    print(f'{nsegs} segs = {npix / 1e6:.1f} Mpix, {nbins} bins')
    failed = False

    configs = [
        dict(method='sorted'),
        dict(method='atomic'),
        dict(method='sorted', with_cm=True),
    ]
    for cfg in configs:
        kern = JungfrauAzintKernel(nbins=nbins, cormax=100.0,
                                   min_pixels=10, **cfg)
        kern.setup(_StubDet(), stub_gpu_det)
        assert kern.result_shape((nsegs, ROWS, COLS)) == (3, nbins)

        # Rebuild the same bin_idx the kernel derived (fallback geometry).
        ix, iy = JungfrauAzintKernel._pixel_indexes(_StubDet(),
                                                    nsegs, ROWS, COLS)
        bin_idx, _ = kern._compute_bins(ix, iy, mask)

        v_ref = calib.copy()
        if cfg.get('with_cm'):
            v_ref = cm_reference(v_ref, raw, gmask[:npix], nsegs, 100.0, 10)
        ref_avg, ref_I, ref_N = azint_reference(v_ref, bin_idx, nbins)

        calib_work = calib_d.copy()   # with_cm mutates in place
        out = kern.reduce(calib_work, raw_gpu=raw_d, gmask_gpu=gmask_d)
        got = out.get().astype(np.float64)

        ok_N = np.array_equal(got[2], ref_N)
        scale = max(1.0, np.abs(ref_avg).max())
        err = np.abs(got[0] - ref_avg).max() / scale
        ok_I = err < 1e-4
        label = kern.name + '/' + cfg.get('method')
        print(f'{label:16s} counts {"exact" if ok_N else "MISMATCH"},  '
              f'I_avg max rel err {err:.2e}  '
              f'{"PASS" if ok_N and ok_I else "FAIL"}')
        failed |= not (ok_N and ok_I)

        # timing
        for _ in range(10):
            kern.reduce(calib_d, raw_gpu=raw_d, gmask_gpu=gmask_d)
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        for _ in range(args.iters):
            kern.reduce(calib_d, raw_gpu=raw_d, gmask_gpu=gmask_d)
        stop.record()
        stop.synchronize()
        ms = cp.cuda.get_elapsed_time(start, stop) / args.iters
        print(f'{"":16s} {ms:7.3f} ms/event  {1e3 / ms:8.1f} Hz')

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
