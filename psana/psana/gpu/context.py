"""
psana/gpu/context.py — Per-event GPU result types.

GPUResult
    Wraps a GPU-resident CuPy array with lazy .on_gpu / .on_cpu access.
    Carries an optional SlotLease so downstream consumers can release the
    EventPool slot as soon as their D→H is done rather than holding it
    until the Python generator advances.

SlotLease
    Completion token linking one event's calibrated output view to the
    EventPool slot it was produced in.  Created by EventPool.submit(),
    attached to GPUResult, consumed by GpuEvents._D2hPipeline.

GpuEventContext
    Per-event container returned by DataSource(gpu_det=...).
"""

from __future__ import annotations


class SlotLease:
    """Controls when an EventPool slot may be recycled.

    Lifecycle
    ---------
    1. EventPool.submit() queues calibration on ``stream``, records
       ``calib_done`` immediately after all kernels, then creates one
       SlotLease per event (sharing ``calib_done``, unique ``view``).

    2. GpuEvents._D2hPipeline receives the lease via GPUResult._lease.
       It issues cudaMemcpyAsync on a separate D→H stream after
       waiting for calib_done, then records d2h_done and calls
       register_d2h_done(event).

    3. EventPool.retire_next() calls wait_until_safe_to_reuse() on
       every lease in the outgoing slot before synchronising the
       calibration stream and recycling the slot.

    Rule: a slot may be reused only after every consumer of that slot
    has completed — generator advancement alone is not sufficient.
    """

    __slots__ = ('slot_id', 'calib_done', 'view', '_d2h_done')

    def __init__(self, slot_id: int, calib_done, view):
        """
        Parameters
        ----------
        slot_id    : int           — EventPool slot index
        calib_done : cp.cuda.Event — fires after calibration kernel completes
        view       : cp.ndarray   — slice of the slot output buffer
                                    (n_segs, nrows, ncols) float32
        """
        self.slot_id    = slot_id
        self.calib_done = calib_done
        self.view       = view
        self._d2h_done  = None   # set by _D2hPipeline after D→H is issued

    def register_d2h_done(self, event):
        """Record the CUDA event that fires when this slot's D→H is done.

        Called by _D2hPipeline immediately after issuing cudaMemcpyAsync.
        EventPool will wait on this event in retire_next() before
        recycling the slot for a future batch.
        """
        self._d2h_done = event

    def wait_until_safe_to_reuse(self):
        """Block the calling thread until the consumer has completed.

        Called by EventPool.retire_next() before overwriting the slot.
        If no D→H was registered (e.g. the event was dropped or the
        joiner was not used), returns immediately.
        """
        if self._d2h_done is not None:
            self._d2h_done.synchronize()


class GPUResult:
    """GPU-resident detector result with lazy D→H transfer.

    Returned by GpuEventContext.get('det.result').

    Attributes
    ----------
    on_gpu : cp.ndarray
        Calibrated array on device.  Never triggers a D→H transfer.
    on_cpu : np.ndarray
        Host copy.  If GpuEvents has already transferred this result via
        its internal D→H pipeline (gpu_d2h_chunk_size > 0), returns the
        pre-populated pinned numpy array immediately with no synchronisation.
        Otherwise synchronises the production stream on first access.
    _lease : SlotLease | None
        Slot ownership token.  Used by GpuEvents._D2hPipeline to issue
        direct async D→H from the slot view and signal when the slot is
        safe to recycle.  User code should not access _lease directly.
    _pinned_cpu : np.ndarray | None
        Set by GpuEvents._D2hPipeline after D→H completes.  When set,
        on_cpu returns this directly without any further GPU transfer.
    """

    __slots__ = ('_arr', '_stream', '_lease', '_pinned_cpu', '_pending_d2h')

    def __init__(self, arr_gpu, stream=None, lease=None, pinned_cpu=None):
        """
        Parameters
        ----------
        arr_gpu    : cp.ndarray | None
        stream     : cp.cuda.Stream | None — production stream
        lease      : SlotLease | None
        pinned_cpu : np.ndarray | None — pre-done D→H result (set by _D2hPipeline
                     once the transfer is confirmed complete)
        """
        self._arr         = arr_gpu
        self._stream      = stream
        self._lease       = lease
        self._pinned_cpu  = pinned_cpu
        # Set by _D2hPipeline immediately after issuing async D→H.
        # Carries the CUDA done-event + pinned-slot reference so on_cpu
        # can wait lazily rather than blocking inside the generator.
        self._pending_d2h = None   # _PendingD2H | None

    @property
    def on_gpu(self):
        """Return the CuPy ndarray on device without any transfer."""
        return self._arr

    @property
    def on_cpu(self):
        """Return the calibrated result as a NumPy ndarray on the host.

        Three paths in priority order:

        1. _pinned_cpu already set  → return immediately (free).
        2. _pending_d2h set         → wait for the async D→H that
           _D2hPipeline issued before yielding this event, then copy
           from the pinned slot and cache in _pinned_cpu.
        3. Fallback                 → synchronise production stream and
           call arr.get() (blocking D→H at the call site).
        """
        if self._pinned_cpu is not None:
            return self._pinned_cpu
        if self._pending_d2h is not None:
            self._pinned_cpu  = self._pending_d2h.get()
            self._pending_d2h = None
            return self._pinned_cpu
        if self._stream is not None:
            self._stream.synchronize()
        return self._arr.get()

    def __repr__(self) -> str:
        shape = getattr(self._arr, 'shape', '?')
        dtype = getattr(self._arr, 'dtype', '?')
        return f'GPUResult(shape={shape}, dtype={dtype})'


class GpuEventContext:
    """Per-event context combining GPU results with CPU detector access.

    Returned by run.events() when DataSource has gpu_det enabled.

        ctx.get('det.result')  → GPUResult
        ctx.raw('det')         → CPU detector (unchanged API)
        ctx.timestamp          → int (64-bit LCLS timestamp)
        ctx.service()          → int (TransitionId)
    """

    __slots__ = ('_evt', '_gpu_results', '_cpu_dets', '_stream',
                 '_cache', '_router', '_leases', '_pinned_results')

    def __init__(self, evt, gpu_results: dict,
                 cpu_dets: dict | None = None,
                 stream=None, router=None,
                 leases: dict | None = None):
        """
        Parameters
        ----------
        evt         : psana2 Event
        gpu_results : dict  {key: cp.ndarray}
        cpu_dets    : dict  {det_name: psana Detector} | None
        stream      : cp.cuda.Stream | None
        router      : DetectorRouter | None
        leases      : dict  {key: SlotLease} | None
            Per-key slot leases created by EventPool.submit().
            Attached to GPUResult objects in get().
        """
        self._evt             = evt
        self._gpu_results     = gpu_results
        self._cpu_dets        = cpu_dets or {}
        self._stream          = stream
        self._router          = router
        self._leases          = leases or {}
        self._cache: dict     = {}
        # Populated by GpuEvents._D2hPipeline after async D→H completes.
        # Maps resolved key → np.ndarray (pinned host copy, already ready).
        self._pinned_results: dict = {}

    def get(self, key: str) -> GPUResult:
        """Return the GPU result for key, with its SlotLease attached.

        Accepts both qualified ('jungfrau.calib') and unqualified
        ('calib') keys when a DetectorRouter is present.
        """
        resolved = (self._router.resolve_key(key)
                    if self._router is not None else key)

        if resolved not in self._cache:
            if resolved not in self._gpu_results:
                available = sorted(self._gpu_results)
                if resolved.endswith('.image'):
                    raise KeyError(
                        f"'{key}' (→ '{resolved}') not available — "
                        f"geometry may not have been loaded.  "
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
            lease      = self._leases.get(resolved)
            pinned_cpu = self._pinned_results.get(resolved)
            self._cache[resolved] = GPUResult(
                self._gpu_results[resolved], self._stream,
                lease=lease, pinned_cpu=pinned_cpu,
            )
        return self._cache[resolved]

    def raw(self, det_name: str):
        """Access a CPU detector — identical to original psana2 API."""
        if det_name not in self._cpu_dets:
            available = sorted(self._cpu_dets)
            raise KeyError(
                f"CPU detector '{det_name}' not registered.  "
                f"Available: {available}"
            )
        return self._cpu_dets[det_name](self._evt)

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
