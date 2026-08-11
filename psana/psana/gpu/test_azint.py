"""
test_azint.py — GPU unit test for the downstream JungfrauAzint operator.

Validates both integration methods against a float64 CPU reference on
synthetic calibrated frames (no psana data or detector required — the
tiled-panel geometry fallback is used).  Requires a GPU.

    python psana/psana/gpu/test_azint.py
"""

import numpy as np


def _cpu_reference(calib, bin_idx, nbins):
    sum_I = np.zeros(nbins, dtype=np.float64)
    sum_N = np.zeros(nbins, dtype=np.float64)
    valid = bin_idx >= 0
    np.add.at(sum_I, bin_idx[valid], calib[valid].astype(np.float64))
    np.add.at(sum_N, bin_idx[valid], 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        I_avg = np.where(sum_N > 0, sum_I / sum_N, 0.0)
    return I_avg, sum_I, sum_N


def main():
    import cupy as cp
    from psana.gpu.gpu_azint import JungfrauAzint

    rng = np.random.default_rng(7)
    nsegs, nrows, ncols = 4, 512, 1024
    npix = nsegs * nrows * ncols
    nbins = 256

    calib = rng.normal(50.0, 12.0, size=npix).astype(np.float32)
    mask = rng.random(npix) > 0.02   # ~2% masked pixels

    ok = True
    for method in ('sorted', 'atomic'):
        az = JungfrauAzint(nbins=nbins, method=method)
        az.setup(det_shape=(nsegs, nrows, ncols), det=None, mask=mask)

        # Recompute the reference from the same bin assignment.
        ix, iy = JungfrauAzint._pixel_indexes(None, nsegs, nrows, ncols)
        bin_idx, _ = az._compute_bins(ix, iy, mask)
        ref_I, ref_sI, ref_sN = _cpu_reference(calib, bin_idx, nbins)

        stream = cp.cuda.Stream()
        calib_gpu = cp.asarray(calib.reshape(nsegs, nrows, ncols))
        out = az(calib_gpu, stream=stream)
        stream.synchronize()
        I_avg, sum_I, sum_N = (cp.asnumpy(out[i]) for i in range(3))

        e_N = np.max(np.abs(sum_N - ref_sN))
        e_I = np.max(np.abs(sum_I - ref_sI) / np.maximum(np.abs(ref_sI), 1))
        e_A = np.max(np.abs(I_avg - ref_I) / np.maximum(np.abs(ref_I), 1e-6))
        passed = e_N == 0 and e_I < 1e-5 and e_A < 1e-5

        # Timing (kernel-only, CUDA events)
        s_evt, e_evt = cp.cuda.Event(), cp.cuda.Event()
        iters = 20
        with stream:
            s_evt.record(stream)
        for _ in range(iters):
            az(calib_gpu, stream=stream, out=out)
        with stream:
            e_evt.record(stream)
        stream.synchronize()
        ms = cp.cuda.get_elapsed_time(s_evt, e_evt) / iters

        status = 'PASS' if passed else 'FAIL'
        print(f'{method:>7s}: {status}  sum_N err {e_N:.0f}  '
              f'sum_I rel {e_I:.2e}  I_avg rel {e_A:.2e}  '
              f'{ms:.3f} ms/frame ({npix/1e6:.1f} Mpix)')
        ok = ok and passed

    print('ALL PASS' if ok else 'FAILURES')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
