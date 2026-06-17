"""
psana/gpu/context.py — Generic per-event GPU result types.

These are the user-visible API objects from the implementation guide §1a/§6:

    for ctx in run.events():                  # or gpu_events(...)
        calib  = ctx.get('jungfrau.calib')    # → GPUResult
        energy = ctx.raw('gmd').energy        # → CPU detector (unchanged)
        peaks  = ctx.get('jungfrau.peaks')    # → GPUResult (future)

GPUResult
    Wraps a GPU-resident CuPy array with .on_gpu / .on_cpu access.
    The D→H transfer is deferred until .on_cpu is accessed so that the caller
    decides when to pay the PCIe cost (e.g. only for confirmed hits).

GpuEventContext
    Per-event container returned by gpu_events().  Holds:
      - GPU results keyed by 'det_name.result_type'  (ctx.get)
      - A reference to the CPU Event for scalar/waveform detectors  (ctx.raw)

This module has no dependency on Jungfrau, MFX, or any specific detector.
"""

from __future__ import annotations


class GPUResult:
    """GPU-resident detector result with lazy D→H transfer.

    Returned by ``GpuEventContext.get('det.result')``.

    Attributes
    ----------
    on_gpu : cp.ndarray
        The calibrated array on device.  Accessing this property never
        triggers a D→H transfer; downstream GPU kernels can use it directly.
    on_cpu : np.ndarray
        A NumPy copy on host.  The first access synchronises the CUDA stream
        the array was produced on, then calls ``.get()``.  Subsequent accesses
        return a new copy each time.

    Example
    -------
    >>> result = ctx.get('jungfrau.calib')
    >>> n = int(cp.sum(result.on_gpu > 5.0))   # stays on GPU
    >>> if n > 100:
    ...     cpu = result.on_cpu                 # D→H only here
    """

    __slots__ = ('_arr', '_stream')

    def __init__(self, arr_gpu, stream=None):
        """
        Parameters
        ----------
        arr_gpu : cp.ndarray
            Calibrated float32 array on device.
        stream  : cp.cuda.Stream or None
            CUDA stream on which arr_gpu was produced.  If not None it is
            synchronised before the D→H copy in .on_cpu.
        """
        self._arr    = arr_gpu
        self._stream = stream

    @property
    def on_gpu(self):
        """Return the CuPy ndarray on device without any transfer."""
        return self._arr

    @property
    def on_cpu(self):
        """Synchronise the production stream and return a NumPy ndarray."""
        if self._stream is not None:
            self._stream.synchronize()
        return self._arr.get()

    def __repr__(self) -> str:
        shape = getattr(self._arr, 'shape', '?')
        dtype = getattr(self._arr, 'dtype', '?')
        return f'GPUResult(shape={shape}, dtype={dtype})'


class GpuEventContext:
    """Per-event context combining GPU results with CPU detector access.

    This is the object the user receives in the ``for ctx in gpu_events(...)``
    loop.  It mirrors the psana2 EventContext described in §6 of the
    implementation guide, providing:

      ctx.get('det.result')  → GPUResult   (calibrated array on GPU)
      ctx.raw('det')         → any          (CPU detector, unchanged API)
      ctx.timestamp          → int          (64-bit LCLS event timestamp)
      ctx.service()          → int          (TransitionId, forwarded from evt)

    The key format for ctx.get() is ``'det_name.result_type'``, e.g.:
        'jungfrau.calib'   calibrated float32 image (prototype: working)
        'jungfrau.peaks'   hit-finder peak list     (future: §9)
        'epix.calib'       ePix calibrated image     (future: §5)

    CPU detectors are accessed via the original psana2 API unchanged:
        energy = ctx.raw('gmd').energy
        xpos   = ctx.raw('ipm').xpos
        ts     = ctx.raw('evr').timestamp
    """

    __slots__ = ('_evt', '_gpu_results', '_cpu_dets', '_stream', '_cache',
                 '_router')

    def __init__(self, evt, gpu_results: dict, cpu_dets: dict | None = None,
                 stream=None, router=None):
        """
        Parameters
        ----------
        evt         : psana2 Event
            The CPU event for this L1Accept (used by ctx.raw()).
        gpu_results : dict  {key: cp.ndarray}
            Mapping of 'det_name.result_type' → CuPy array already on GPU.
            E.g. {'jungfrau.calib': calib_gpu}.
        cpu_dets    : dict  {det_name: psana Detector} or None
            Pre-loaded CPU detectors for ctx.raw() access.
        stream      : cp.cuda.Stream or None
            CUDA stream on which the GPU arrays in gpu_results were produced.
        router      : DetectorRouter or None
            When provided, unqualified keys in ctx.get() are resolved via
            router.resolve_key().  E.g. ctx.get('calib') → ctx.get('det.calib').
        """
        self._evt         = evt
        self._gpu_results = gpu_results   # {key: cp.ndarray}
        self._cpu_dets    = cpu_dets or {}
        self._stream      = stream
        self._router      = router
        self._cache: dict = {}            # GPUResult objects, built on demand

    # ------------------------------------------------------------------
    # GPU detector access
    # ------------------------------------------------------------------

    def get(self, key: str) -> GPUResult:
        """Return the GPU result for key.

        Accepts both qualified and unqualified keys:

            Unqualified (recommended):
                ctx.get('calib')   — resolved to '<gpu_det>.calib' via router
                ctx.get('raw')     — resolved to '<gpu_det>.raw'
                ctx.get('image')   — resolved to '<gpu_det>.image'

            Qualified (backward compatible):
                ctx.get('jungfrau.calib')   — used as-is

        The unqualified form only works when a DetectorRouter was provided at
        construction time (which gpu_events() and DataSource(gpu_det=) do
        automatically).

        The GPUResult is constructed lazily and cached.

        Raises
        ------
        KeyError  if the (resolved) key is not available for this event.
        NotImplementedError  if common-mode correction is requested (Phase F3).
        """
        # Resolve unqualified keys ('calib' → 'jungfrau.calib') via router.
        resolved = (self._router.resolve_key(key)
                    if self._router is not None else key)

        if resolved not in self._cache:
            if resolved not in self._gpu_results:
                available = sorted(self._gpu_results)
                if resolved.endswith('.image'):
                    raise KeyError(
                        f"'{key}' (→ '{resolved}') not available — geometry "
                        f"may not have been loaded.  "
                        f"Available GPU keys: {available}"
                    )
                if resolved == key:
                    raise KeyError(
                        f"'{key}' not available.  "
                        f"Available GPU keys: {available}"
                    )
                raise KeyError(
                    f"'{key}' resolved to '{resolved}' which is not available.  "
                    f"Available GPU keys: {available}"
                )
            self._cache[resolved] = GPUResult(
                self._gpu_results[resolved], self._stream
            )
        return self._cache[resolved]

    def get_async(self, key: str) -> GPUResult:
        """Return the GPU result without synchronising the stream.

        Use when you want to issue subsequent GPU work before blocking.
        Call .on_cpu only after all GPU work on the stream is complete.
        Equivalent to ctx.get() since both return the same GPUResult;
        the distinction is a hint to the caller not to call .on_cpu yet.
        """
        return self.get(key)

    # ------------------------------------------------------------------
    # CPU detector access (guide §5: DetectorRouter.get_cpu)
    # ------------------------------------------------------------------

    def raw(self, det_name: str):
        """Access a CPU detector — identical to original psana2 API.

        Returns whatever the psana2 Detector object returns for this event.
        Scalar detectors (GMD, IPM, EVR, timetool) stay on CPU; no GPU
        stream is involved.

        Example
        -------
        >>> energy = ctx.raw('gmd').energy     # float64
        >>> xpos   = ctx.raw('ipm').xpos       # float32
        >>> ts     = ctx.raw('evr').timestamp  # uint64
        """
        if det_name not in self._cpu_dets:
            available = sorted(self._cpu_dets)
            raise KeyError(
                f"CPU detector '{det_name}' not registered.  "
                f"Available: {available}"
            )
        return self._cpu_dets[det_name](self._evt)

    # ------------------------------------------------------------------
    # Event metadata (forwarded from the underlying CPU Event)
    # ------------------------------------------------------------------

    @property
    def timestamp(self) -> int:
        """64-bit LCLS event timestamp."""
        return self._evt.timestamp

    def service(self) -> int:
        """TransitionId service type (12 = L1Accept)."""
        return self._evt.service()

    def __repr__(self) -> str:
        keys = sorted(self._gpu_results)
        return (f'GpuEventContext(ts={self.timestamp}, '
                f'gpu_keys={keys})')
