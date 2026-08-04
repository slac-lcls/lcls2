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

    3. EventPool.begin_retire_next() exposes the completed result.
       After the caller has had a chance to register its consumer,
       finish_retire_next() calls wait_until_safe_to_reuse() before reuse.

    Rule: a slot may be reused only after every consumer of that slot
    has completed — generator advancement alone is not sufficient.
    """

    __slots__ = ('calib_done', '_d2h_done')

    def __init__(self, calib_done):
        """
        Parameters
        ----------
        calib_done : cp.cuda.Event — fires after calibration kernel completes
        """
        self.calib_done = calib_done
        self._d2h_done  = None   # set by _D2hPipeline or _GpuViewContext.__exit__

    def register_d2h_done(self, event):
        """Record the CUDA event that fires when this slot's consumer is done.

        Called by _D2hPipeline after issuing cudaMemcpyAsync, or by
        _GpuViewContext.__exit__ after the user's downstream GPU kernel.
        EventPool waits on this event in finish_retire_next() before reuse.
        """
        self._d2h_done = event

    def wait_until_safe_to_reuse(self):
        """Block until the consumer has completed during final retirement.

        Two outcomes:
          _d2h_done set  → synchronize then recycle
          neither        → recycle immediately (on_gpu copy or no access)
        """
        if self._d2h_done is not None:
            self._d2h_done.synchronize()


class _GpuViewContext:
    """Context manager returned by GPUResult.on_gpu_view(stream).

    ``__enter__`` returns the raw slot-buffer array (zero-copy).
    ``__exit__`` records a CUDA done-event on *stream* so that
    EventPool.finish_retire_next() knows the slot is safe to recycle once
    that event fires.

    ``__del__`` is a safety fallback: if the caller somehow receives
    this object but never uses it as a ``with`` statement, the done-event
    is recorded on the null stream (conservative — the null stream
    serialises with all default-stream work) so the slot is never
    permanently held.
    """

    __slots__ = ('_arr', '_lease', '_stream', '_exited')

    def __init__(self, arr, lease, stream):
        self._arr    = arr
        self._lease  = lease
        self._stream = stream
        self._exited = False

    def __enter__(self):
        return self._arr

    def __exit__(self, *_):
        import cupy as cp
        stream = self._stream or cp.cuda.Stream.null
        done   = cp.cuda.Event(disable_timing=True)
        stream.record(done)
        self._lease.register_d2h_done(done)
        self._exited = True

    def __del__(self):
        # Safety: if the object is GC'd without having been used as a context
        # manager, record a conservative done-event on the null stream so the
        # slot is not held forever.  Same pattern as _PendingD2H.__del__.
        if not self._exited and self._lease._d2h_done is None:
            try:
                import cupy as cp
                done = cp.cuda.Event(disable_timing=True)
                cp.cuda.Stream.null.record(done)
                self._lease.register_d2h_done(done)
            except Exception:
                pass


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
        Cached independent CPU result.  Set either after GpuEvents' pinned
        D→H completes or by the synchronous fallback.  When set, on_cpu
        returns it without another GPU transfer.
    """

    __slots__ = ('_arr', '_lease', '_pinned_cpu', '_pending_d2h')

    def __init__(self, arr_gpu, lease=None):
        """
        Parameters
        ----------
        arr_gpu : cp.ndarray | None
        lease   : SlotLease | None
        """
        self._arr         = arr_gpu
        self._lease       = lease
        self._pinned_cpu  = None
        # Set by _D2hPipeline immediately after issuing async D→H.
        # Carries the CUDA done-event + pinned-slot reference so on_cpu
        # can wait lazily rather than blocking inside the generator.
        self._pending_d2h = None   # _PendingD2H | None

    @property
    def on_gpu(self):
        """Return an independent D→D copy of the calibrated result.

        The copy is not tied to the EventPool slot buffer — the slot can
        be recycled immediately after this call.  Use when the copy cost
        (~2 ms D→D for Jungfrau) is acceptable and simplicity is preferred.
        """
        return self._arr.copy()

    def on_gpu_view(self, stream=None):
        """Return a context manager that yields a zero-copy view into the slot buffer.

        Fastest GPU path — avoids the D→D copy — but all kernels that read
        the view MUST run on ``stream`` and MUST be enqueued inside the
        ``with`` block.  ``__exit__`` records a CUDA done-event on ``stream``
        automatically so EventPool knows when the slot is safe to recycle.

        Usage::

            with ctx.get('jungfrau.calib').on_gpu_view(stream) as arr:
                my_kernel(arr, stream=stream)
            # done event recorded automatically — nothing else needed

        If ``stream`` is None the CuPy null (default) stream is used.

        Raises RuntimeError if this GPUResult has no SlotLease (i.e. was not
        produced by EventPool.submit()) — use on_gpu (D→D copy) instead.
        """
        if self._lease is None:
            raise RuntimeError(
                "on_gpu_view is not safe: this GPUResult has no SlotLease. "
                "Use on_gpu (D→D copy) instead, which is always safe."
            )
        return _GpuViewContext(self._arr, self._lease, stream)

    @property
    def on_cpu(self):
        """Return the calibrated result as a NumPy ndarray on the host.

        Three paths in priority order:

        1. _pinned_cpu already set  → return immediately (free).
        2. _pending_d2h set         → wait for the async D→H that
           _D2hPipeline issued before yielding this event, then copy
           from the pinned slot and cache in _pinned_cpu.
        3. Fallback                 → call arr.get() (blocking D→H at the
           call site), cache the independent NumPy result, and return it.
        """
        if self._pinned_cpu is not None:
            return self._pinned_cpu
        if self._pending_d2h is not None:
            self._pinned_cpu  = self._pending_d2h.get()
            self._pending_d2h = None
            return self._pinned_cpu
        self._pinned_cpu = self._arr.get()
        return self._pinned_cpu

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

    __slots__ = ('_evt', '_gpu_results', '_cpu_dets',
                 '_cache', '_router', '_leases')

    def __init__(self, evt, gpu_results: dict,
                 cpu_dets: dict | None = None,
                 router=None,
                 leases: dict | None = None):
        """
        Parameters
        ----------
        evt         : psana2 Event
        gpu_results : dict  {key: cp.ndarray}
        cpu_dets    : dict  {det_name: psana Detector} | None
        router      : DetectorRouter | None
        leases      : dict  {key: SlotLease} | None
            Per-key slot leases created by EventPool.submit().
            Attached to GPUResult objects in get().
        """
        self._evt         = evt
        self._gpu_results = gpu_results
        self._cpu_dets    = cpu_dets or {}
        self._router      = router
        self._leases      = leases or {}
        self._cache: dict = {}

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
            self._cache[resolved] = GPUResult(
                self._gpu_results[resolved],
                lease=self._leases.get(resolved),
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
