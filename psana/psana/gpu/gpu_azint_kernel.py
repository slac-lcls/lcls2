"""
gpu_azint_kernel.py — azimuthal-integration reduction kernels for the
GPUKernelRegistry.

Registers under result names 'azint' (and 'azint_cm' when common-mode is
enabled), so the event loop surfaces per-event I(q) as::

    from psana.gpu import default_registry
    from psana.gpu.gpu_azint_kernel import JungfrauAzintKernel

    default_registry().register(JungfrauAzintKernel(nbins=256))

    # ... DataSource(gpu_det='jungfrau', ...) event loop ...
    azint = ctx.get('jungfrau.azint').on_gpu   # (3, nbins) float32
    I_avg, sum_I, sum_N = azint                # rows

Design
------
This is a *reduction* kernel (GPUKernel.reduce, result_name != 'calib'):
it consumes the event's calibrated GPU buffer that GPUDetector has already
materialized into its batch slot buffer, and produces a (3, nbins) result.
The 'calib' product and its contract are untouched.

The CUDA source (cuda/analysis_kernels.cu) also carries fused raw->azint
variants of the same math for kernel-only benchmarking; measured on
A100 / 16.8 Mpix the post-calib path used here is the faster arrangement
(all passes coalesced) given that the pipeline materializes calib anyway.

Integration strategies (identical results, very different performance):
    method='sorted'  gather into bin-contiguous order + per-bin tree
                     reduction; no atomics.        ~0.3 ms  (default)
    method='atomic'  one global atomicAdd per pixel; heavy serialization
                     on the hot bins.              ~11 ms
The atomic method is kept as a deliberate heavyweight configuration for
pipeline stress-testing — it places the kernel below the FFB storage
floor, the sorted method above it.

Geometry
--------
q-bin assignment is precomputed once per run in setup() from the psana
detector geometry (pixel coordinate indexes x pixel size), with the same
q formula as the jungfrau_gpu_azint DirectIntegrate:
    q = 4*pi/lambda * sin(atan2(r, dist) / 2)
Pass geometry=dict(dist=..., wavelength=..., poni1=..., poni2=...,
pixel_size=...) in meters.  With geometry=None, pixels are binned by
radius in pixel units (equal-radius rings; correct binning structure and
contention behavior, arbitrary q scale) — sufficient for performance work
when beamline geometry isn't known.  Masked pixels (gmask mode-0 == 0)
are excluded from all bins.
"""

from pathlib import Path

import numpy as np

from psana.gpu.gpu_kernel_registry import GPUKernel

_CUDA_DIR = Path(__file__).parent / 'cuda'
_TPB = 256


class JungfrauAzintKernel(GPUKernel):
    """Per-event azimuthal integration of the calibrated Jungfrau frame.

    Parameters
    ----------
    nbins      : number of radial bins (default 256)
    method     : 'sorted' (no atomics, default) or 'atomic' (global atomics,
                 deliberately heavyweight — see module docstring)
    with_cm    : apply bank common-mode correction to the calibrated frame
                 (IN PLACE — '{det}.calib' for the event is then the
                 CM-corrected frame) before integrating.  Registers under
                 'azint_cm' and requires raw (needs_raw).
    geometry   : dict(dist, wavelength, poni1=0, poni2=0, pixel_size=75e-6)
                 in meters, or None for radius-unit binning.
    q_range    : optional (qmin, qmax) override.
    cormax, min_pixels : common-mode parameters (with_cm only).
    """

    det_types = ['jungfrau']
    raw_dtype = 'uint16'

    def __init__(self, nbins=256, method='sorted', with_cm=False,
                 geometry=None, q_range=None, cormax=100.0, min_pixels=10):
        if method not in ('sorted', 'atomic'):
            raise ValueError("method must be 'sorted' or 'atomic'")
        self.nbins = int(nbins)
        self.method = method
        self.with_cm = bool(with_cm)
        self.name = 'azint_cm' if with_cm else 'azint'
        self.needs_raw = self.with_cm
        self.geometry = geometry
        self.q_range = q_range
        self.cormax = float(cormax)
        self.min_pixels = int(min_pixels)
        self.q = None            # (nbins,) bin centers, set in setup()
        self._npix = None

    # ------------------------------------------------------------------
    # Registry hooks
    # ------------------------------------------------------------------

    def result_shape(self, det_shape):
        return (3, self.nbins)   # rows: I_avg, sum_I, sum_N

    def setup(self, det, gpu_detector):
        """Precompute per-pixel q-bin tables and upload them (BeginRun)."""
        import cupy as cp

        nsegs, nrows, ncols = gpu_detector.det_shape
        self._nsegs, self._nrows, self._ncols = nsegs, nrows, ncols
        npix = nsegs * nrows * ncols
        self._npix = npix

        ix, iy = self._pixel_indexes(det, nsegs, nrows, ncols)
        mask = (gpu_detector.gmask_gpu[:npix] != 0).get()   # mode-0 gain*mask

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
        self._k_cm = self._mod.get_function('common_mode_bank_kernel')

    def reduce(self, calib_gpu, raw_gpu=None, gmask_gpu=None, stream=None):
        import cupy as cp

        flat = calib_gpu.ravel()
        if flat.size != self._npix:
            raise ValueError(
                f'{type(self).__name__}: event has {flat.size} pixels but '
                f'bin tables were built for {self._npix} '
                f'(partial-detector GPU routing is not supported yet)')

        out = cp.empty((3, self.nbins), dtype=cp.float32)
        sum_I, sum_N = out[1], out[2]
        ctx = stream if stream is not None else cp.cuda.Stream.null

        with ctx:
            if self.with_cm:
                if raw_gpu is None or gmask_gpu is None:
                    raise ValueError(
                        f'{type(self).__name__}: with_cm=True requires '
                        f'raw_gpu and gmask_gpu')
                nbanks = (self._nrows // 256) * 16
                self._k_cm(
                    (self._nsegs, nbanks), (_TPB,),
                    (flat, raw_gpu.ravel(), gmask_gpu,
                     np.int32(self._nrows * self._ncols),
                     np.int32(self._ncols),
                     np.int32(256), np.int32(64), np.int32(16),
                     np.float32(self.cormax), np.int32(self.min_pixels)))

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
        falls back to a tiled panel layout when geometry is unavailable —
        binning structure and contention behavior stay realistic, only the
        ring positions are approximate.
        """
        try:
            ix, iy = det.raw._pixel_coord_indexes()
            return (np.asarray(ix).ravel().astype(np.float64),
                    np.asarray(iy).ravel().astype(np.float64))
        except Exception:
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
