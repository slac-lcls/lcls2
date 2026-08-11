"""
gpu_azint.py — downstream azimuthal-integration operator for GPU results.

A CuPy analysis operator that consumes the calibrated frame the pipeline
delivers and produces a per-event (3, nbins) result on device — rows are
I_avg, sum_I, sum_N.  It plugs in downstream of the framework via the
zero-copy view API (the AMI Phase-3 "GPU-native graph node" shape)::

    from psana.gpu.gpu_azint import JungfrauAzint

    az = JungfrauAzint(nbins=256)
    az.setup(det_shape=(32, 512, 1024), det=det)   # once per run

    stream = cp.cuda.Stream()
    for ctx in run.events():
        res = ctx.get('jungfrau.calib')
        with res.on_gpu_view(stream) as calib:      # zero-copy slot view
            hist = az(calib, stream=stream)         # (3, nbins) on GPU
        # slot recycling waits for the done-event recorded at exit

Integration strategies (identical results, very different performance):
    method='sorted'  gather into bin-contiguous order + per-bin tree
                     reduction; no atomics.        ~0.3 ms  (default)
    method='atomic'  one global atomicAdd per pixel; heavy serialization
                     on the hot bins.              ~11 ms solo, dilates
                     linearly with ranks sharing the GPU
The atomic method is kept as a deliberate heavyweight configuration for
pipeline stress-testing.

Geometry
--------
q-bin assignment is precomputed once per run in setup() from the psana
detector geometry (pixel coordinate indexes x pixel size), with
    q = 4*pi/lambda * sin(atan2(r, dist) / 2)
Pass geometry=dict(dist=..., wavelength=..., poni1=..., poni2=...,
pixel_size=...) in meters.  With geometry=None, pixels are binned by
radius in pixel units (equal-radius rings; correct binning structure and
contention behavior, arbitrary q scale).  Pass mask (flat bool, npix) to
exclude pixels from all bins.

The bank common-mode variant from the registry-era module is not ported:
it corrects the frame in place using the raw gain bits, and raw is not
delivered downstream on this pipeline.  Common-mode belongs framework-side
(before the calib result surfaces) if it is needed.
"""

from pathlib import Path

import numpy as np

_CUDA_DIR = Path(__file__).parent / 'cuda'
_TPB = 256


class JungfrauAzint:
    """Per-event azimuthal integration of the calibrated Jungfrau frame.

    Parameters
    ----------
    nbins    : number of radial bins (default 256)
    method   : 'sorted' (no atomics, default) or 'atomic' (global atomics,
               deliberately heavyweight — see module docstring)
    geometry : dict(dist, wavelength, poni1=0, poni2=0, pixel_size=75e-6)
               in meters, or None for radius-unit binning.
    q_range  : optional (qmin, qmax) override.

    After setup(), ``self.q`` holds the (nbins,) bin centers.
    """

    def __init__(self, nbins=256, method='sorted', geometry=None,
                 q_range=None):
        if method not in ('sorted', 'atomic'):
            raise ValueError("method must be 'sorted' or 'atomic'")
        self.nbins = int(nbins)
        self.method = method
        self.geometry = geometry
        self.q_range = q_range
        self.q = None            # (nbins,) bin centers, set in setup()
        self._npix = None

    def setup(self, det_shape, det=None, mask=None):
        """Precompute per-pixel q-bin tables and upload them (once per run).

        Parameters
        ----------
        det_shape : (nsegs, nrows, ncols) of the calibrated frame
        det       : psana Detector or None — when given, panel positions come
                    from det.raw._pixel_coord_indexes(); otherwise a tiled
                    panel layout is used (realistic binning structure,
                    approximate ring positions).
        mask      : flat bool array of npix or None — False pixels are
                    excluded from all bins.
        """
        import cupy as cp

        nsegs, nrows, ncols = det_shape
        self._nsegs, self._nrows, self._ncols = nsegs, nrows, ncols
        npix = nsegs * nrows * ncols
        self._npix = npix

        ix, iy = self._pixel_indexes(det, nsegs, nrows, ncols)
        if mask is None:
            mask = np.ones(npix, dtype=bool)
        else:
            mask = np.asarray(mask).ravel().astype(bool)

        bin_idx, self.q = self._compute_bins(ix, iy, mask)

        if self.method == 'sorted':
            valid_pix = np.nonzero(bin_idx >= 0)[0]
            order = valid_pix[np.argsort(bin_idx[valid_pix],
                                         kind='stable')].astype(np.int32)
            offsets = np.zeros(self.nbins + 1, dtype=np.int32)
            offsets[1:] = np.cumsum(np.bincount(bin_idx[valid_pix],
                                                minlength=self.nbins))
            self._sort_order_d = cp.asarray(order)
            self._bin_offsets_d = cp.asarray(offsets)
            self._n_valid = int(order.size)
            self._sorted_d = cp.empty(self._n_valid, dtype=cp.float32)
        else:
            self._bin_idx_d = cp.asarray(bin_idx)

        src = (_CUDA_DIR / 'analysis_kernels.cu').read_text()
        self._mod = cp.RawModule(code=src,
                                 options=('--std=c++17', f'-I{_CUDA_DIR}'))
        self._k_gather = self._mod.get_function('azint_gather_kernel')
        self._k_sorted = self._mod.get_function('azint_sorted_kernel')
        self._k_atomic = self._mod.get_function('azint_global_kernel')
        self._k_norm = self._mod.get_function('normalize_kernel')

    def __call__(self, calib_gpu, stream=None, out=None):
        """Integrate one calibrated frame.  Returns (3, nbins) float32 on GPU.

        All launches are enqueued on ``stream`` (CuPy null stream when None).
        When calib_gpu is a slot view from on_gpu_view(stream), call this
        inside the ``with`` block on the same stream.
        """
        import cupy as cp

        flat = calib_gpu.ravel()
        if flat.size != self._npix:
            raise ValueError(
                f'{type(self).__name__}: frame has {flat.size} pixels but '
                f'bin tables were built for {self._npix} '
                f'(partial-detector GPU routing is not supported yet)')

        ctx = stream if stream is not None else cp.cuda.Stream.null
        with ctx:
            if out is None:
                out = cp.empty((3, self.nbins), dtype=cp.float32)
            sum_I, sum_N = out[1], out[2]

            if self.method == 'sorted':
                blocks = (self._n_valid + _TPB - 1) // _TPB
                self._k_gather(
                    (blocks,), (_TPB,),
                    (flat, self._sort_order_d, self._sorted_d,
                     np.uint64(self._n_valid)))
                self._k_sorted(
                    (self.nbins,), (_TPB,),
                    (self._sorted_d, self._bin_offsets_d,
                     sum_I, sum_N, np.int32(self.nbins)))
            else:
                sum_I.fill(0)
                sum_N.fill(0)
                blocks = (self._npix + _TPB - 1) // _TPB
                self._k_atomic(
                    (blocks,), (_TPB,),
                    (flat, self._bin_idx_d, sum_I, sum_N,
                     np.uint64(self._npix)))

            nb_blocks = (self.nbins + _TPB - 1) // _TPB
            self._k_norm((nb_blocks,), (_TPB,),
                         (sum_I, sum_N, out[0], np.int64(self.nbins)))
        return out

    # ------------------------------------------------------------------
    # Geometry precompute (CPU, once per run)
    # ------------------------------------------------------------------

    @staticmethod
    def _pixel_indexes(det, nsegs, nrows, ncols):
        """Per-pixel integer image coordinates, flattened.

        Prefers psana geometry (same source the image-assembly path uses);
        falls back to a tiled panel layout when geometry is unavailable.
        """
        if det is not None:
            try:
                ix, iy = det.raw._pixel_coord_indexes()
                return (np.asarray(ix).ravel().astype(np.float64),
                        np.asarray(iy).ravel().astype(np.float64))
            except Exception:
                pass
        panels_per_row = 4
        iy, ix = np.mgrid[0:nrows, 0:ncols]
        x = np.concatenate([(ix + (s % panels_per_row) * ncols).ravel()
                            for s in range(nsegs)]).astype(np.float64)
        y = np.concatenate([(iy + (s // panels_per_row) * nrows).ravel()
                            for s in range(nsegs)]).astype(np.float64)
        return x, y

    def _compute_bins(self, ix, iy, mask):
        """(bin_idx int32 with -1 = excluded, q bin centers)."""
        if self.geometry is not None:
            g = dict(self.geometry)
            px_mm = g.get('pixel_size', 75e-6) * 1e3
            x_mm = (ix - ix.max() / 2) * px_mm + g.get('poni2', 0.0) * 1e3
            y_mm = (iy - iy.max() / 2) * px_mm + g.get('poni1', 0.0) * 1e3
            r_mm = np.hypot(x_mm, y_mm)
            wl_A = g['wavelength'] * 1e10
            q = 4 * np.pi / wl_A * np.sin(
                np.arctan2(r_mm, g['dist'] * 1e3) / 2)
        else:
            # No beamline geometry: bin by radius in pixel units.
            q = np.hypot(ix - ix.max() / 2, iy - iy.max() / 2)

        if self.q_range is not None:
            q_min, q_max = (float(v) for v in self.q_range)
        else:
            q_min, q_max = q[mask].min(), q[mask].max()

        edges = np.linspace(q_min, q_max, self.nbins + 1)
        bin_idx = np.clip(np.digitize(q, edges) - 1, 0, self.nbins - 1)
        bin_idx = np.where(mask, bin_idx, -1).astype(np.int32)
        return bin_idx, 0.5 * (edges[:-1] + edges[1:])
