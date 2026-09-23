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
    consumed immediately by GpuEventManager._D2hPipeline for automatic D→H, and
    attached to GPUResult when the event is later delivered.

GpuEventState
    Per-event GPU results and parsed detector fields attached to
    :class:`psana.Event`.
"""

from __future__ import annotations


class SlotLease:
    """A result slot stays occupied through every consumer and open view.

    Retirement rejects new acquisitions. An open context reports pressure
    rather than blocking its own caller; its completion permits a later retry.
    """

    __slots__ = ('result_ready', '_consumer_done', '_lock', '_views',
                 '_closing', '_closed', '_callbacks')

    def __init__(self, result_ready):
        from threading import RLock
        self.result_ready = result_ready
        self._consumer_done = []
        self._lock = RLock()
        self._views = 0
        self._closing = False
        self._closed = False
        self._callbacks = []

    def on_retire(self, callback):
        from weakref import WeakMethod
        with self._lock:
            if self._closed:
                callback()
            else:
                self._callbacks.append(WeakMethod(callback))

    def require_active(self):
        if self._closing or self._closed:
            raise RuntimeError("GPU result lease is retiring or released")

    def acquire_view(self):
        with self._lock:
            self.require_active()
            self._views += 1

    def finish_view(self, event):
        with self._lock:
            if not self._views:
                raise RuntimeError("GPU view already released")
            if event is not None:
                self._consumer_done.append(event)
            self._views -= 1

    def register_consumer_done(self, event):
        with self._lock:
            self.require_active()
            self._consumer_done.append(event)

    def wait_until_safe_to_reuse(self):
        with self._lock:
            if self._closed:
                return
            self._closing = True
            if self._views:
                raise RuntimeError("GPU result has an open view; exit its context before retrying retirement")
            for event in ([self.result_ready] if self.result_ready is not None else []) + self._consumer_done:
                event.synchronize()
            self._closed = True
            self.result_ready = None
            self._consumer_done.clear()
            callbacks, self._callbacks = self._callbacks, []
            for ref in callbacks:
                callback = ref()
                if callback is not None:
                    callback()


class _GpuViewContext:
    """Acquire on entry; record completion on the actual consumer stream."""

    __slots__ = ('_result', '_stream', '_entered', '_exited')

    def __init__(self, result, stream):
        self._result = result
        self._stream = stream
        self._entered = False
        self._exited = False

    def __enter__(self):
        import cupy as cp
        if self._entered or self._exited:
            raise RuntimeError("GPU view context cannot be entered twice")
        lease = self._result._lease
        with lease._lock:
            self._result._require_device_storage("on_gpu_view")
            lease.acquire_view()
            self._entered = True
        self._stream = self._stream or cp.cuda.Stream.null
        try:
            if lease.result_ready is not None:
                self._stream.wait_event(lease.result_ready)
            return self._result._arr
        except BaseException:
            lease.finish_view(None)
            self._exited = True
            raise

    def __exit__(self, *_):
        import cupy as cp
        if not self._entered or self._exited:
            return
        # If recording fails, drain this same stream. If that also fails,
        # leave the view acquired so the occupied slot remains retryable.
        try:
            done = cp.cuda.Event(disable_timing=True)
            self._stream.record(done)
        except BaseException:
            self._stream.synchronize()
            self._result._lease.finish_view(None)
            self._exited = True
            raise
        self._result._lease.finish_view(done)
        self._exited = True


class GPUResult:
    """Detector result with explicit device and cached host access.

    Returned by ``evt.gpu.get('det.result')``.

    Attributes
    ----------
    on_gpu : cp.ndarray
        Calibrated array on device.  Never triggers a D→H transfer.
    on_cpu : np.ndarray
        Host copy. If GpuEventManager has already transferred this result via
        its internal D→H pipeline (gpu_d2h_chunk_size > 0), waits for that
        token when necessary and caches an independent NumPy result. Otherwise
        performs one blocking D→H on first access and caches the result.
    _lease : SlotLease | None
        Slot ownership token. Used by GpuEventManager._D2hPipeline to issue
        direct async D→H from the slot view and signal when the slot is
        safe to recycle.  User code should not access _lease directly.
    _cpu_cache : np.ndarray | None
        Cached independent CPU result. Set after GpuEventManager's pinned
        D→H completes or by the synchronous fallback.  When set, on_cpu
        returns it without another GPU transfer.
    """

    __slots__ = ('_arr', '_lease', '_cpu_cache', '_pending_d2h',
                 '_device_released', '__weakref__')

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
        if lease is not None and hasattr(lease, 'on_retire'):
            lease.on_retire(self._retire_device)

    def _retire_device(self):
        self._arr = None

    def _require_device_storage(self, accessor: str):
        if self._lease is not None:
            self._lease.require_active()
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
        be recycled after the copy completes. A completion event is recorded
        on the current CuPy stream and joined during slot retirement.
        """
        self._require_device_storage("on_gpu")
        if self._lease is None:
            return self._arr.copy()
        import cupy as cp
        stream = getattr(cp.cuda, 'get_current_stream', lambda: cp.cuda.Stream.null)()
        with _GpuViewContext(self, stream) as array:
            return array.copy()

    def on_gpu_view(self, stream=None):
        """Return a context manager that yields a zero-copy view into the slot buffer.

        Fastest GPU path — avoids the D→D copy — but all kernels that read
        the view MUST run on ``stream`` and MUST be enqueued inside the
        ``with`` block.  ``__exit__`` records a CUDA done-event on ``stream``
        automatically so EventPool knows when the slot is safe to recycle.

        Usage::

            with evt.gpu.get('jungfrau.calib').on_gpu_view(stream) as arr:
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
        return _GpuViewContext(self, stream)

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
        self._require_device_storage("on_cpu")
        self._cpu_cache = self._arr.get()
        return self._cpu_cache

    def __repr__(self) -> str:
        shape = getattr(self._arr, 'shape', '?')
        dtype = getattr(self._arr, 'dtype', '?')
        return f'GPUResult(shape={shape}, dtype={dtype})'


class GpuEventState:
    """GPU results and leases owned by one :class:`psana.Event`.

    This state intentionally has no reference back to its Event or to the
    run-wide GPU manager. Normal detector access remains ``det.raw.raw(evt)``;
    parsed GPU fields are selected with ``detector(name).field(alg, field)``.
    """

    __slots__ = ('_gpu_results', '_detector_names', '_cache', '_leases',
                 '_pending_d2h', '_cached_cpu_results',
                 '_device_released', '_detector_bindings', '_event_dgrams',
                 '_input_lease', '_detector_cache', '__weakref__')

    def __init__(self, gpu_results: dict, detector_names=None,
                 leases: dict | None = None,
                 pending_d2h: dict | None = None,
                 cached_cpu_results: dict | None = None,
                 detector_bindings: dict | None = None,
                 event_dgrams=None,
                 input_lease=None,
                 device_released: bool = False):
        """
        Parameters
        ----------
        gpu_results : dict  {key: cp.ndarray}
        detector_names : sequence[str] | None
            GPU detectors configured for the run. Used to resolve an
            unqualified result key without a separate routing object.
        leases      : dict  {key: SlotLease} | None
            Per-key slot leases created by EventPool.submit().
            Attached to GPUResult objects in get().
        pending_d2h : dict  {key: _PendingD2H} | None
            Host-result tokens armed immediately after slot submission.
        cached_cpu_results : dict  {key: np.ndarray} | None
            Independent CPU results materialized under pinned-buffer pressure.
        """
        self._gpu_results = dict(gpu_results)
        if detector_names is None:
            detector_names = dict.fromkeys(
                key.split('.', 1)[0]
                for key in gpu_results
                if '.' in key
            )
        self._detector_names = tuple(detector_names)
        self._leases      = leases or {}
        self._pending_d2h = pending_d2h or {}
        self._cached_cpu_results = cached_cpu_results or {}
        self._device_released = device_released
        self._detector_bindings = detector_bindings or {}
        self._event_dgrams = event_dgrams
        self._input_lease = input_lease
        self._detector_cache = {}
        self._cache: dict = {}
        for lease in self._leases.values():
            if hasattr(lease, 'on_retire'):
                lease.on_retire(self._retire_device)

    def _retire_device(self):
        if all(getattr(lease, '_closed', False) for lease in self._leases.values()):
            self._gpu_results = dict.fromkeys(self._gpu_results)

    def detector(self, det_name):
        """Return Configure-backed field access for one GPU detector.

        Use ``evt.gpu.detector(name).field(alg, field, segment=...)`` for
        detector-independent access to fields decoded by the GPU XTC parser.
        """
        det_name = str(det_name)
        try:
            binding = self._detector_bindings[det_name]
        except KeyError:
            raise KeyError(
                f"GPU detector {det_name!r} is not configured; available: "
                f"{sorted(self._detector_bindings)}"
            ) from None
        if det_name not in self._detector_cache:
            from psana.gpu.gpu_input import GpuDetectorEvent

            self._detector_cache[det_name] = GpuDetectorEvent(
                binding,
                self._event_dgrams,
                self._input_lease,
                device_released=self._device_released,
            )
        return self._detector_cache[det_name]

    def get(self, key: str) -> GPUResult:
        """Return the GPU result for key, with its SlotLease attached.

        Accepts a qualified key such as ``'jungfrau.calib'``. An unqualified
        key such as ``'calib'`` is accepted when exactly one available result
        has that suffix.
        """
        resolved = key
        if '.' not in key:
            if len(self._detector_names) == 1:
                resolved = f'{self._detector_names[0]}.{key}'
            elif len(self._detector_names) > 1:
                raise KeyError(
                    f"'{key}' is ambiguous. Use a detector-qualified key. "
                    f"GPU detectors: {sorted(self._detector_names)}"
                )

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

    def __repr__(self) -> str:
        keys = sorted(self._gpu_results)
        detectors = sorted(self._detector_bindings)
        return f'GpuEventState(gpu_keys={keys}, detectors={detectors})'
