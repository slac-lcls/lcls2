"""
psana/gpu/detector_router.py — DetectorRouter for GPU/CPU detector dispatch.

The DetectorRouter tracks which detectors in a run are GPU-routed (large
area detectors calibrated on GPU) and which are CPU-only (scalars, waveforms,
diagnostics).

Two capabilities:

1. Key resolution  (unqualified → qualified):
       ctx.get('calib')          → resolves to 'jungfrau.calib'
       ctx.get('jungfrau.calib') → passes through unchanged (backward compat)

2. Full calibration combining  (GPU segments + CPU segments → complete array):
       setup_full_routing() — call at BeginRun with both GPU and CPU seg_ids.
       assemble_full_calib() — per-event; scatters both sets into a complete
                               (n_calibconst_segs, nrows, ncols) array.

Guide reference: §5 (DetectorRouter).
"""

import numpy as np


class DetectorRouter:
    """Tracks GPU-routed and CPU-only detectors for a run.

    Created once per run inside gpu_events() and attached to every
    GpuEventContext so that unqualified keys can be resolved without
    the user repeating the detector name, and so that GPU + CPU detector
    segments can be combined into a complete calibrated array.

    Usage — key resolution
    ----------------------
        router = DetectorRouter()
        router.register_gpu('jungfrau')   # large area → GPU path
        router.register_cpu('gmd')        # scalar     → CPU path

        router.resolve_key('calib')           # → 'jungfrau.calib'
        router.resolve_key('jungfrau.calib')  # → 'jungfrau.calib' (unchanged)

    Usage — full combining
    ----------------------
        # At BeginRun:
        router.setup_full_routing(
            det_name          = 'jungfrau',
            gpu_seg_ids       = [0,1,2,3,7, 5,9,...],  # in GPU output row order
            cpu_seg_ids       = [6,10,14,18,22,26,30, 11,15,19,...],
            calibconst_n_segs = 32,
            nrows             = 512,
            ncols             = 1024,
        )

        # Per event (in gpu_events loop):
        full_calib = router.assemble_full_calib(
            'jungfrau',
            gpu_calib_gpu,   # cp.ndarray (19, 512, 1024)
            cpu_calib_np,    # np.ndarray (13, 512, 1024) or None
        )
    """

    def __init__(self):
        self._gpu_det_names: list = []   # ordered; first = default
        self._cpu_det_names: list = []
        # Per-detector full routing info (populated by setup_full_routing).
        self._full_routing: dict = {}    # det_name → _FullRoutingInfo

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_gpu(self, det_name: str) -> None:
        """Register a detector as GPU-routed.

        The first registered GPU detector becomes the *default*: unqualified
        keys like ``'calib'`` resolve to ``f'{default}.calib'``.
        """
        if det_name not in self._gpu_det_names:
            self._gpu_det_names.append(det_name)

    def register_cpu(self, det_name: str) -> None:
        """Register a detector as CPU-only (scalar, waveform, etc.)."""
        if det_name not in self._cpu_det_names:
            self._cpu_det_names.append(det_name)

    # ------------------------------------------------------------------
    # Full combining setup (call once at BeginRun)
    # ------------------------------------------------------------------

    def setup_full_routing(self, det_name: str, gpu_seg_ids: list,
                            cpu_seg_ids: list,
                            calibconst_n_segs: int,
                            nrows: int = 512, ncols: int = 1024,
                            gpu_det_obj=None) -> None:
        """Precompute scatter indices for GPU + CPU segment combining.

        Parameters
        ----------
        det_name          : str  — e.g. 'jungfrau'
        gpu_seg_ids       : list[int]  — calibconst row indices for the GPU
                            segments in the same order as GPUDetector outputs
                            them (i.e. stream_id ascending, L1Accept child-XTC
                            order within each stream).
        cpu_seg_ids       : list[int]  — calibconst row indices for the CPU
                            segments, in the sorted order that
                            det.raw.calib(evt) returns them.
        calibconst_n_segs : int  — total calibconst rows (e.g. 32 for full
                            Jungfrau 16M).  Sets the first dimension of the
                            combined output array.
        nrows, ncols      : int  — panel pixel dimensions (512 × 1024 for
                            Jungfrau 0.5M).
        """
        # Pre-extract CPU calibconst rows from GPU arrays (D2H once at BeginRun)
        # so that per-event CPU calibration only needs numpy arithmetic.
        cpu_peds_flat = None
        cpu_gmask_flat = None
        if gpu_det_obj is not None and cpu_seg_ids:
            try:
                n_modes   = gpu_det_obj._n_modes_calib
                n_pix_seg = gpu_det_obj._n_pix_seg
                n_total   = gpu_det_obj._n_segs_calib * n_pix_seg
                rows_p, rows_g = [], []
                for m in range(n_modes):
                    for sid in cpu_seg_ids:
                        s = m * n_total + sid * n_pix_seg
                        e = s + n_pix_seg
                        rows_p.append(gpu_det_obj.peds_gpu[s:e].get())
                        rows_g.append(gpu_det_obj.gmask_gpu[s:e].get())
                cpu_peds_flat  = np.concatenate(rows_p)
                cpu_gmask_flat = np.concatenate(rows_g)
            except Exception:
                pass   # best-effort; CPU calib falls back to zeros

        self._full_routing[det_name] = _FullRoutingInfo(
            gpu_seg_ids=list(gpu_seg_ids),
            cpu_seg_ids=list(cpu_seg_ids),
            calibconst_n_segs=calibconst_n_segs,
            nrows=nrows,
            ncols=ncols,
            cpu_peds_flat=cpu_peds_flat,
            cpu_gmask_flat=cpu_gmask_flat,
        )

    def has_full_routing(self, det_name: str) -> bool:
        """Return True if setup_full_routing() has been called for det_name."""
        return det_name in self._full_routing

    # ------------------------------------------------------------------
    # Full combining per event
    # ------------------------------------------------------------------

    def assemble_full_calib(self, det_name: str,
                             gpu_calib,    # cp.ndarray (n_gpu_segs, nrows, ncols)
                             cpu_calib):   # np.ndarray (n_cpu_segs, nrows, ncols) or None
        """Scatter GPU + CPU calibrated segments into a complete detector array.

        Returns a CuPy array of shape (calibconst_n_segs, nrows, ncols) with
        each segment placed at its calibconst row index.  Pixels for segments
        not present in either path remain zero.

        Parameters
        ----------
        det_name   : str
        gpu_calib  : cp.ndarray float32, shape (n_gpu_segs, nrows, ncols)
        cpu_calib  : np.ndarray float32, shape (n_cpu_segs, nrows, ncols)
                     or None if CPU path produced no data for this event.

        Returns
        -------
        cp.ndarray float32, shape (calibconst_n_segs, nrows, ncols)
        None if setup_full_routing() was not called for det_name.
        """
        ri = self._full_routing.get(det_name)
        if ri is None:
            return None

        import cupy as cp
        n, nrows, ncols = ri.calibconst_n_segs, ri.nrows, ri.ncols
        full = cp.zeros((n, nrows, ncols), dtype=cp.float32)

        # Scatter GPU-calibrated segments.
        for k, seg_id in enumerate(ri.gpu_seg_ids):
            if k < gpu_calib.shape[0]:
                full[seg_id] = gpu_calib[k]

        # Scatter CPU-calibrated segments (H→D transfer).
        if cpu_calib is not None:
            cpu_gpu = cp.asarray(cpu_calib.astype(np.float32))
            for k, seg_id in enumerate(ri.cpu_seg_ids):
                if k < cpu_gpu.shape[0]:
                    full[seg_id] = cpu_gpu[k]

        return full

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------

    @property
    def default_gpu_det(self):
        """The primary GPU-routed detector name, or None if none registered."""
        return self._gpu_det_names[0] if self._gpu_det_names else None

    def resolve_key(self, key: str) -> str:
        """Expand an unqualified result key to a fully-qualified one.

        Rules
        -----
        - If ``key`` contains ``'.'`` it is already qualified → unchanged.
        - Otherwise the key is prefixed with the default GPU detector name.
        - If no GPU detector is registered, the key is returned unchanged.
        """
        if '.' in key:
            return key
        default = self.default_gpu_det
        if default is not None:
            return f'{default}.{key}'
        return key

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def gpu_det_names(self) -> list:
        return list(self._gpu_det_names)

    @property
    def cpu_det_names(self) -> list:
        return list(self._cpu_det_names)

    def __repr__(self) -> str:
        return (f'DetectorRouter(gpu={self._gpu_det_names}, '
                f'cpu={self._cpu_det_names})')


    # ------------------------------------------------------------------
    # Per-event CPU calibration
    # ------------------------------------------------------------------

    def compute_cpu_calib(self, det_name: str, psana_det, evt):
        """Apply calibration to the CPU-path segments for this event.

        Uses the calibconst rows pre-extracted at BeginRun and the
        raw pixels from psana_det.raw.raw(evt).  Returns None when the
        detector has no data in this event or setup_full_routing was not
        called.

        Returns
        -------
        np.ndarray float32, shape (n_cpu_segs, nrows, ncols), or None.
        """
        ri = self._full_routing.get(det_name)
        if ri is None or not ri.cpu_seg_ids or ri.cpu_peds_flat is None:
            return None

        raw = psana_det.raw.raw(evt)
        if raw is None:
            return None

        raw_flat  = raw.ravel()
        gbits     = (raw_flat >> 14).astype(np.int32)
        data_bits = (raw_flat & 0x3fff).astype(np.float32)
        n_pix     = raw_flat.size
        n_modes   = ri.cpu_peds_flat.size // n_pix

        calib = np.zeros(n_pix, dtype=np.float32)
        for mode, gv in [(0, 0), (1, 1), (2, 3)]:
            if mode >= n_modes:
                break
            px   = gbits == gv
            ped  = ri.cpu_peds_flat[mode * n_pix:(mode + 1) * n_pix]
            gmsk = ri.cpu_gmask_flat[mode * n_pix:(mode + 1) * n_pix]
            calib[px] = (data_bits[px] - ped[px]) * gmsk[px]

        return calib.reshape(raw.shape)


class _FullRoutingInfo:
    """Internal: precomputed segment scatter info for one detector."""

    __slots__ = ('gpu_seg_ids', 'cpu_seg_ids',
                 'calibconst_n_segs', 'nrows', 'ncols',
                 'cpu_peds_flat', 'cpu_gmask_flat')

    def __init__(self, gpu_seg_ids, cpu_seg_ids,
                 calibconst_n_segs, nrows, ncols,
                 cpu_peds_flat=None, cpu_gmask_flat=None):
        self.gpu_seg_ids       = gpu_seg_ids
        self.cpu_seg_ids       = cpu_seg_ids
        self.calibconst_n_segs = calibconst_n_segs
        self.nrows             = nrows
        self.ncols             = ncols
        self.cpu_peds_flat     = cpu_peds_flat    # np.ndarray or None
        self.cpu_gmask_flat    = cpu_gmask_flat   # np.ndarray or None
