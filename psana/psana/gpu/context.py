"""
psana/gpu/context.py — Per-event GPU result types.

GPUResult
    Wraps a detector result with explicit GPU / CPU accessors.
    Carries an optional SlotLease so downstream consumers can release the
    EventPool slot as soon as their D→H is done rather than holding it
    until the Python generator advances.

SlotLease
    Completion token linking one event's calibrated output view to the
    EventPool slot it was produced in.  Created by EventPool.submit(),
    consumed immediately by GpuEvents._D2hPipeline for automatic D→H, and
    attached to GPUResult when the event is later delivered.

GpuEventContext
    Per-event container returned by DataSource(gpu_det=...).
"""

from __future__ import annotations


class SlotLease:
    """Controls when an EventPool slot may be recycled.

    Lifecycle
    ---------
    1. EventPool.submit() queues calibration and final routing on ``stream``,
       records ``result_ready`` after that producer work, then creates one
       SlotLease per (timestamp, result key), sharing ``result_ready``.

    2. GpuEvents._D2hPipeline receives the submitted slot record.  It issues
       cudaMemcpyAsync on a separate D→H stream after waiting for result_ready,
       then records a completion event and calls register_consumer_done(event).

    3. EventPool.begin_retire_next() exposes the result and its prepared host
       token.  After the caller has had a chance to register any external GPU
       consumer, finish_retire_next() waits before reuse.

    Rule: a slot may be reused only after every consumer of that slot
    has completed — generator advancement alone is not sufficient.
    """

    __slots__ = ('result_ready', '_consumer_done')

    def __init__(self, result_ready):
        """
        Parameters
        ----------
        result_ready : cp.cuda.Event
            Fires after calibration and final result-routing work completes.
        """
        self.result_ready = result_ready
        self._consumer_done = None

    def register_consumer_done(self, event):
        """Record the CUDA event that fires when this slot's consumer is done.

        Called by _D2hPipeline after issuing cudaMemcpyAsync, or by
        _GpuViewContext.__exit__ after the user's downstream GPU kernel.
        EventPool waits on this event in finish_retire_next() before reuse.
        """
        self._consumer_done = event

    def wait_until_safe_to_reuse(self):
        """Block until the consumer has completed during final retirement.

        Two outcomes:
          _consumer_done set  → synchronize then recycle
          neither        → recycle immediately (on_gpu copy or no access)
        """
        if self._consumer_done is not None:
            self._consumer_done.synchronize()


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
        self._lease.register_consumer_done(done)
        self._exited = True

    def __del__(self):
        # Safety: if the object is GC'd without having been used as a context
        # manager, record a conservative done-event on the null stream so the
        # slot is not held forever.  Same pattern as _PendingD2H.__del__.
        if not self._exited and self._lease._consumer_done is None:
            try:
                import cupy as cp
                done = cp.cuda.Event(disable_timing=True)
                cp.cuda.Stream.null.record(done)
                self._lease.register_consumer_done(done)
            except Exception:
                pass


class GPUResult:
    """Detector result with explicit device and cached host access.

    Returned by GpuEventContext.get('det.result').

    Attributes
    ----------
    on_gpu : cp.ndarray
        Calibrated array on device.  Never triggers a D→H transfer.
    on_cpu : np.ndarray
        Host copy.  If GpuEvents has already transferred this result via
        its internal D→H pipeline (gpu_d2h_chunk_size > 0), waits for that
        token when necessary and caches an independent NumPy result. Otherwise
        performs one blocking D→H on first access and caches the result.
    _lease : SlotLease | None
        Slot ownership token.  Used by GpuEvents._D2hPipeline to issue
        direct async D→H from the slot view and signal when the slot is
        safe to recycle.  User code should not access _lease directly.
    _cpu_cache : np.ndarray | None
        Cached independent CPU result.  Set either after GpuEvents' pinned
        D→H completes or by the synchronous fallback.  When set, on_cpu
        returns it without another GPU transfer.
    """

    __slots__ = ('_arr', '_lease', '_cpu_cache', '_pending_d2h',
                 '_device_released')

    def __init__(self, arr_gpu, lease=None, device_released=False):
        """
        Parameters
        ----------
        arr_gpu : cp.ndarray | None
        lease   : SlotLease | None
        """
        self._arr         = arr_gpu
        self._lease       = lease
        self._cpu_cache   = None
        # Set by _D2hPipeline immediately after issuing async D→H.
        # Carries the CUDA done-event + pinned-slot reference so on_cpu
        # can wait lazily rather than blocking inside the generator.
        self._pending_d2h = None   # _PendingD2H | None
        # Automatic D2H contexts can outlive the EventPool device slot.  Keep
        # that state explicit so stale slot-backed arrays are never exposed.
        self._device_released = device_released

    def _require_device_storage(self, accessor: str):
        if self._device_released or self._arr is None:
            raise RuntimeError(
                f"{accessor} is unavailable because automatic D2H completed "
                "and the EventPool device slot was released before this event "
                "was yielded. Use on_cpu, or set gpu_d2h_chunk_size=0 for a "
                "GPU consumer."
            )

    @property
    def on_gpu(self):
        """Return an independent D→D copy of the calibrated result.

        The copy is not tied to the EventPool slot buffer — the slot can
        be recycled immediately after this call.  Use when the copy cost
        (~2 ms D→D for Jungfrau) is acceptable and simplicity is preferred.
        """
        self._require_device_storage("on_gpu")
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
        self._require_device_storage("on_gpu_view")
        if self._lease is None:
            raise RuntimeError(
                "on_gpu_view is not safe: this GPUResult has no SlotLease. "
                "Use on_gpu (D→D copy) instead, which is always safe."
            )
        if self._pending_d2h is not None:
            raise RuntimeError(
                "on_gpu_view is unavailable after automatic D2H has been "
                "scheduled. Use gpu_d2h_chunk_size=0 for a zero-copy GPU "
                "consumer, or use on_gpu for an independent D→D copy."
            )
        return _GpuViewContext(self._arr, self._lease, stream)

    @property
    def on_cpu(self):
        """Return the calibrated result as a NumPy ndarray on the host.

        Three paths in priority order:

        1. _cpu_cache already set   → return immediately (free).
        2. _pending_d2h set         → wait for the async D→H that
           _D2hPipeline issued before yielding this event, then copy
           from the pinned slot and cache in _cpu_cache.
        3. Fallback                 → call arr.get() (blocking D→H at the
           call site), cache the independent NumPy result, and return it.
        """
        if self._cpu_cache is not None:
            return self._cpu_cache
        if self._pending_d2h is not None:
            self._cpu_cache   = self._pending_d2h.get()
            self._pending_d2h = None
            return self._cpu_cache
        if self._device_released or self._arr is None:
            raise RuntimeError(
                "on_cpu has no host result after the EventPool device slot "
                "was released; this indicates an incomplete automatic-D2H "
                "handoff."
            )
        self._cpu_cache = self._arr.get()
        return self._cpu_cache

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

    __slots__ = ('_evt', '_gpu_results', '_cpu_dets', '_cache', '_router',
                 '_leases', '_pending_d2h', '_cached_cpu_results',
                 '_device_released')

    def __init__(self, evt, gpu_results: dict,
                 cpu_dets: dict | None = None,
                 router=None,
                 leases: dict | None = None,
                 pending_d2h: dict | None = None,
                 cached_cpu_results: dict | None = None,
                 device_released: bool = False):
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
        pending_d2h : dict  {key: _PendingD2H} | None
            Host-result tokens armed immediately after slot submission.
        cached_cpu_results : dict  {key: np.ndarray} | None
            Independent CPU results materialized under pinned-buffer pressure.
        """
        self._evt         = evt
        self._gpu_results = gpu_results
        self._cpu_dets    = cpu_dets or {}
        self._router      = router
        self._leases      = leases or {}
        self._pending_d2h = pending_d2h or {}
        self._cached_cpu_results = cached_cpu_results or {}
        self._device_released = device_released
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
            result = GPUResult(
                self._gpu_results[resolved],
                lease=self._leases.get(resolved),
                device_released=self._device_released,
            )
            result._pending_d2h = self._pending_d2h.get(resolved)
            result._cpu_cache = self._cached_cpu_results.get(resolved)
            self._cache[resolved] = result
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
