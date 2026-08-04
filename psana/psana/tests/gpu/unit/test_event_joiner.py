"""
Unit tests for SlotLease, EventPool lease tracking, and the internal
GpuEvents._D2hPipeline.

All tests run on CPU only — CuPy is replaced with a lightweight fake that:
  - alloc_pinned_memory → bytearray (supports np.frombuffer)
  - cuda.Event         → immediately-done fake event (synchronous semantics)
  - cuda.Stream        → records wait_event / synchronize calls
  - runtime.memcpyAsync → ctypes.memmove so actual data is copied

Tests cover the design requirements from
gpu_memory_backpressure_and_async_join.md §Validation:

  - A slot cannot be recycled while D→H is in flight.
  - A downstream CUDA completion token controls release.
  - Generator advancement alone does not release a lease.
  - Multiple D→H chunks produce one correctly ordered logical join.
  - BeginStep / EndRun flush partial joins correctly.
"""

import ctypes
import sys
from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake CuPy infrastructure
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Fake cp.cuda.Event.  Immediately-done (synchronous fake)."""

    def __init__(self, disable_timing=False):
        self.done = False  # tests can set this to True to signal
        self._synced = False
        self._sync_calls = 0

    def record(self, stream=None):
        """Mark as done when recorded."""
        self.done = True

    def synchronize(self):
        self._sync_calls += 1
        self._synced = True
        self.done = True

    def signal(self):
        """Test helper: mark as done without recording."""
        self.done = True


class _PendingEvent(_FakeEvent):
    """A fake event that is NOT done until explicitly signalled."""

    def __init__(self):
        super().__init__()
        self.done = False  # override: starts not done

    def record(self, stream=None):
        pass  # recording does NOT auto-mark done


class _FakeStream:
    """Fake cp.cuda.Stream."""

    def __init__(self, non_blocking=True):
        self.ptr = 0
        self.synchronize_calls = 0
        self.wait_events: list = []
        self.recorded_events: list = []

    def synchronize(self):
        self.synchronize_calls += 1

    def wait_event(self, event):
        self.wait_events.append(event)

    def record(self, event):
        self.recorded_events.append(event)


def _fake_memcpy(dst_ptr, src_ptr, nbytes, kind, stream_ptr):
    """Synchronous CPU memcpy — makes test data actually land in pinned buf."""
    ctypes.memmove(dst_ptr, src_ptr, nbytes)


FAKE_CUPY = SimpleNamespace(
    cuda=SimpleNamespace(
        Stream=_FakeStream,
        Event=_FakeEvent,
        alloc_pinned_memory=bytearray,  # bytearray(nbytes) is buffer-protocol compatible
        runtime=SimpleNamespace(
            memcpyDeviceToHost=2,
            memcpyAsync=_fake_memcpy,
        ),
    )
)


@pytest.fixture(autouse=True)
def patch_cupy(monkeypatch):
    """Replace 'cupy' for every test in this file."""
    monkeypatch.setitem(sys.modules, "cupy", FAKE_CUPY)


# ---------------------------------------------------------------------------
# Fake GPU array  (stands in for a cp.ndarray)
# ---------------------------------------------------------------------------


class _FakeGPUArr:
    """Fake CuPy ndarray — backed by a numpy array, exposes .data.ptr."""

    def __init__(self, data: np.ndarray):
        self._np = np.ascontiguousarray(data, dtype=np.float32)
        self.get_calls = 0
        self.shape = self._np.shape
        self.dtype = self._np.dtype
        self.nbytes = self._np.nbytes
        self.data = SimpleNamespace(ptr=self._np.ctypes.data)

    def copy(self) -> "_FakeGPUArr":
        return _FakeGPUArr(self._np.copy())

    def get(self) -> np.ndarray:
        self.get_calls += 1
        return self._np.copy()


def _make_arr(n_segs=4, nrows=8, ncols=8, fill=None) -> _FakeGPUArr:
    if fill is not None:
        data = np.full((n_segs, nrows, ncols), fill, dtype=np.float32)
    else:
        data = np.random.randn(n_segs, nrows, ncols).astype(np.float32)
    return _FakeGPUArr(data)


# ---------------------------------------------------------------------------
# Fake GPUResult  (wraps _FakeGPUArr + optional SlotLease)
# ---------------------------------------------------------------------------


# ===========================================================================
# SlotLease tests
# ===========================================================================


class TestSlotLease:
    def test_no_d2h_registered_passes_immediately(self):
        """wait_until_safe_to_reuse() with no D→H registered should return
        without blocking."""
        from psana.gpu.context import SlotLease

        event = _FakeEvent()
        lease = SlotLease(calib_done=event)
        # No register_d2h_done — should be a no-op
        lease.wait_until_safe_to_reuse()  # must not raise or hang

    def test_d2h_registered_calls_synchronize(self):
        """wait_until_safe_to_reuse() must call synchronize() on the D→H event."""
        from psana.gpu.context import SlotLease

        calib = _FakeEvent()
        d2h = _FakeEvent()
        lease = SlotLease(calib_done=calib)
        lease.register_d2h_done(d2h)
        lease.wait_until_safe_to_reuse()
        assert d2h._sync_calls == 1

    def test_generator_advancement_alone_does_not_release(self):
        """A lease with a pending (not-done) D→H event must block until
        synchronize() is explicitly called."""
        from psana.gpu.context import SlotLease

        calib = _FakeEvent()
        pending = _PendingEvent()  # starts not done
        lease = SlotLease(calib_done=calib)
        lease.register_d2h_done(pending)

        assert not pending.done  # not done yet
        # Calling wait_until_safe_to_reuse() must call synchronize()
        lease.wait_until_safe_to_reuse()
        assert pending._sync_calls == 1  # synchronize was called


# ===========================================================================
# EventPool lease-tracking tests
# ===========================================================================


class TestEventPoolLeases:
    def test_finish_retire_observes_consumer_registered_after_begin(self, monkeypatch):
        """A consumer registered while the result is exposed must be joined."""
        from psana.gpu.context import SlotLease
        from psana.gpu.gpu_stream import EventPool

        pool = EventPool(n=1)
        detectors = {}  # no detectors → no events, but leases_by_ts={}

        # Manually inject a slot that has a lease with a pending D→H.
        pending_d2h = _PendingEvent()
        calib_done = _FakeEvent()
        lease = SlotLease(calib_done)
        stream = pool._streams[0]
        pool._slots[0] = ({}, [], stream, [lease], {})
        pool._write_idx = 1  # pretend one batch was submitted

        assert not pending_d2h._synced
        pool.begin_retire_next()
        assert pool._slots[0] is not None, "begin must retain slot ownership"

        # Models on_gpu_view().__exit__ running after the context was yielded.
        lease.register_d2h_done(pending_d2h)
        pool.finish_retire_next()
        assert pending_d2h._synced, "finish must join the late-registered consumer"
        assert pool._slots[0] is None


# ===========================================================================
# GPUResult._pinned_cpu / GpuEventContext._pinned_results tests
# ===========================================================================


class TestOnGpuAndView:
    """Tests for on_gpu (D→D copy) and on_gpu_view (context-manager zero-copy)."""

    def test_on_gpu_returns_independent_copy(self):
        """on_gpu must return a D→D copy — not a view — so the slot can be
        recycled immediately without data corruption."""
        from psana.gpu.context import GPUResult

        arr = _make_arr(fill=5.0)
        result = GPUResult(arr_gpu=arr)
        copy = result.on_gpu
        assert copy is not arr, "on_gpu must return a copy, not the original array"

    def test_on_gpu_copy_value(self):
        """Data in the copy must match the source array."""
        from psana.gpu.context import GPUResult

        arr = _make_arr(fill=3.0)
        result = GPUResult(arr_gpu=arr)
        np.testing.assert_allclose(result.on_gpu._np, arr._np)

    def test_on_gpu_view_yields_original_array(self):
        """__enter__ must return the original array (no copy)."""
        from psana.gpu.context import GPUResult, SlotLease

        arr = _make_arr()
        lease = SlotLease(calib_done=_FakeEvent())
        result = GPUResult(arr_gpu=arr, lease=lease)
        with result.on_gpu_view(_FakeStream()) as view:
            assert view is arr, "on_gpu_view must yield the original array, not a copy"

    def test_on_gpu_view_records_done_event_on_exit(self):
        """__exit__ must record a done-event on the provided stream so
        EventPool.finish_retire_next() knows when the slot is safe to recycle."""
        from psana.gpu.context import GPUResult, SlotLease

        arr = _make_arr()
        lease = SlotLease(calib_done=_FakeEvent())
        result = GPUResult(arr_gpu=arr, lease=lease)
        stream = _FakeStream()
        with result.on_gpu_view(stream):
            pass
        assert lease._d2h_done is not None, "__exit__ must register a done event on the lease"
        assert lease._d2h_done in stream.recorded_events, \
            "done event must be recorded on the provided stream"

    def test_on_gpu_view_retire_safe_after_context_exit(self):
        """After the with block exits, wait_until_safe_to_reuse() must
        synchronize the done event without raising."""
        from psana.gpu.context import GPUResult, SlotLease

        arr = _make_arr()
        lease = SlotLease(calib_done=_FakeEvent())
        result = GPUResult(arr_gpu=arr, lease=lease)
        with result.on_gpu_view(_FakeStream()):
            pass
        lease.wait_until_safe_to_reuse()   # must not raise
        assert lease._d2h_done._synced, "final retirement must synchronize the done event"

    def test_on_gpu_view_raises_without_lease(self):
        """on_gpu_view must raise RuntimeError when the GPUResult has no lease."""
        from psana.gpu.context import GPUResult

        arr = _make_arr()
        result = GPUResult(arr_gpu=arr)   # no lease
        with pytest.raises(RuntimeError):
            result.on_gpu_view()


class TestGpuBudget:
    """Tests for _GpuBudget committed-bytes counter."""

    def test_reserve_within_budget(self):
        """reserve() within budget increments committed bytes."""
        from psana.gpu.gpu_budget import _GpuBudget

        b = _GpuBudget(limit_bytes=1000)
        b.reserve(400)
        assert b.committed() == 400
        assert b.available() == 600

    def test_reserve_exceeds_budget_raises(self):
        """reserve() over budget raises GpuMemoryPressureError."""
        from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError

        b = _GpuBudget(limit_bytes=1000)
        b.reserve(800)
        with pytest.raises(GpuMemoryPressureError):
            b.reserve(300)  # 800 + 300 > 1000

    def test_release_decrements_committed(self):
        """release() returns bytes to the available pool."""
        from psana.gpu.gpu_budget import _GpuBudget

        b = _GpuBudget(limit_bytes=1000)
        b.reserve(600)
        b.release(600)
        assert b.committed() == 0
        assert b.available() == 1000

    def test_reserve_after_release(self):
        """After release, previously over-budget reservation succeeds."""
        from psana.gpu.gpu_budget import _GpuBudget

        b = _GpuBudget(limit_bytes=1000)
        b.reserve(800)
        b.release(800)
        b.reserve(900)  # should now succeed
        assert b.committed() == 900

    def test_release_does_not_go_negative(self):
        """release() clamps at zero — no negative committed bytes."""
        from psana.gpu.gpu_budget import _GpuBudget

        b = _GpuBudget(limit_bytes=1000)
        b.reserve(100)
        b.release(500)  # releasing more than committed
        assert b.committed() == 0


class TestPinnedCpu:
    """Tests for the GpuEvents internal D→H path where _pinned_cpu is set
    on GPUResult and _pinned_results is set on GpuEventContext before the
    context is yielded to the user."""

    def test_on_cpu_returns_pinned_immediately(self):
        """When _pinned_cpu is set, on_cpu must return it without touching _arr."""
        from psana.gpu.context import GPUResult

        pinned = np.ones((4, 8, 8), dtype=np.float32) * 7.0
        result = GPUResult(arr_gpu=None)
        result._pinned_cpu = pinned   # set directly (as _D2hPipeline does via on_cpu)
        out = result.on_cpu
        np.testing.assert_array_equal(out, pinned)

    def test_on_gpu_unaffected_by_pinned_cpu(self):
        """on_gpu returns a copy of _arr even when pinned_cpu is set.
        The copy must have the same values as _arr, not pinned_cpu."""
        from psana.gpu.context import GPUResult

        arr = _make_arr(fill=3.0)
        pinned = np.zeros((4, 8, 8), dtype=np.float32)
        result = GPUResult(arr_gpu=arr)
        result._pinned_cpu = pinned
        copy = result.on_gpu
        assert copy is not arr, "on_gpu must return a copy, not the original"
        np.testing.assert_allclose(copy._np, arr._np)

    def test_sync_fallback_caches_first_d2h_result(self):
        """Repeated on_cpu access must not read a reused GPU slot again."""
        from psana.gpu.context import GPUResult

        arr = _make_arr(fill=3.0)
        result = GPUResult(arr_gpu=arr)

        first = result.on_cpu
        arr._np.fill(9.0)  # model the execution slot being overwritten
        second = result.on_cpu

        assert first is second
        assert arr.get_calls == 1
        np.testing.assert_allclose(second, 3.0)

    def test_gpu_event_context_get_returns_gpu_result(self):
        """GpuEventContext.get() must return a GPUResult with the correct array."""
        from psana.gpu.context import GpuEventContext, GPUResult
        from types import SimpleNamespace

        arr = _make_arr(fill=5.0)
        fake_evt = SimpleNamespace(timestamp=42)

        ctx = GpuEventContext(
            evt=fake_evt,
            gpu_results={"jungfrau.calib": arr},
        )
        result = ctx.get("jungfrau.calib")
        assert isinstance(result, GPUResult)
        assert result._arr is arr
        # on_cpu falls back to arr.get() when no _pending_d2h is set
        np.testing.assert_allclose(result.on_cpu, arr._np)


# ===========================================================================
# _D2hPipeline tests (GpuEvents internal class)
# ===========================================================================


class TestD2hPipeline:
    """Tests for the internal GpuEvents._D2hPipeline class.

    With the lazy-sync design, contexts are yielded IMMEDIATELY after
    D→H is issued.  GPUResult._pending_d2h carries the CUDA done-event;
    on_cpu waits lazily at the call site.
    """

    def _make_ctx(self, fill=1.0, key="jungfrau.calib", n_segs=4, nrows=8, ncols=8):
        from psana.gpu.context import GpuEventContext, SlotLease
        from types import SimpleNamespace

        arr = _make_arr(n_segs=n_segs, nrows=nrows, ncols=ncols, fill=fill)
        calib = _FakeEvent()
        lease = SlotLease(calib_done=calib)
        fake_evt = SimpleNamespace(timestamp=int(fill * 100))
        ctx = GpuEventContext(
            evt=fake_evt,
            gpu_results={key: arr},
            leases={key: lease},
        )
        return ctx

    def test_pipeline_yields_immediately(self):
        """Checks that the generator does not hold events waiting for D→H
        to complete.  Adds 2 events with chunk_size=2.  After the first
        add() the chunk is not full so nothing is returned.  After the
        second add() the chunk fires and both events are returned — before
        the transfer has finished (because the fake done_event is not yet
        polled)."""
        from psana.gpu.gpu_events import _D2hPipeline

        pipe = _D2hPipeline(det_key="jungfrau.calib", chunk_size=2)
        ctx0 = self._make_ctx(fill=1.0)
        ctx1 = self._make_ctx(fill=2.0)

        ready = pipe.add(ctx0)
        assert ready == [], "should not yield before chunk is full"
        ready = pipe.add(ctx1)
        assert len(ready) == 2, "both contexts yielded after chunk filled"

    def test_pipeline_sets_pending_d2h(self):
        """Checks that the lazy-sync token is attached before yielding.
        After a chunk fires, each context's GPUResult must have
        _pending_d2h set.  This token is what allows on_cpu to wait for
        the transfer lazily at the call site."""
        from psana.gpu.gpu_events import _D2hPipeline

        pipe = _D2hPipeline(det_key="jungfrau.calib", chunk_size=2)
        ctx0 = self._make_ctx(fill=1.0)
        ctx1 = self._make_ctx(fill=2.0)
        for ctx in [ctx0, ctx1]:
            pipe.add(ctx)
        # After yielding, the GPUResult in the cache must have _pending_d2h.
        for ctx in [ctx0, ctx1]:
            result = ctx._cache.get("jungfrau.calib")
            assert result is not None
            assert result._pending_d2h is not None, "_pending_d2h must be set so on_cpu can sync lazily"

    def test_on_cpu_returns_correct_data(self):
        """Checks the full data path end-to-end.  Two fake GPU arrays are
        filled with known values (3.0 and 7.0) and run through the
        pipeline.  on_cpu is called on each yielded context and the
        returned numpy arrays are compared to the originals.  This
        exercises memcpyAsync → done_event.synchronize() → .copy() in
        sequence and confirms no data corruption or row mix-up."""
        from psana.gpu.gpu_events import _D2hPipeline

        pipe = _D2hPipeline(det_key="jungfrau.calib", chunk_size=2)
        fills = [3.0, 7.0]
        ctxs = [self._make_ctx(fill=v) for v in fills]
        ready = []
        for ctx in ctxs:
            ready.extend(pipe.add(ctx))
        for ctx, expected in zip(ready, fills):
            result = ctx.get("jungfrau.calib")
            np.testing.assert_allclose(result.on_cpu, expected)

    def test_pipeline_flush_partial(self):
        """Checks the batch-boundary drain path.  Adds 1 event to a
        pipeline with chunk_size=3 — the chunk is not full so add()
        returns nothing.  Calls flush() and checks the stranded event is
        returned.  Without this flush, events whose count does not reach
        chunk_size (e.g. the last batch of a run) would never be
        yielded to the user."""
        from psana.gpu.gpu_events import _D2hPipeline

        pipe = _D2hPipeline(det_key="jungfrau.calib", chunk_size=3)
        ctx = self._make_ctx(fill=1.0)
        assert pipe.add(ctx) == [], "no yield before chunk_size reached"
        ready = pipe.flush()
        assert len(ready) == 1
        # The context is yielded with _pending_d2h set.
        result = ready[0].get("jungfrau.calib")
        assert result._pending_d2h is not None or result._pinned_cpu is not None

    def test_pipeline_unknown_key_passthrough(self):
        """Checks that the pipeline does not buffer events it has no data
        for.  A context whose gpu_results dict does not contain the
        pipeline's det_key (e.g. a BeginStep transition or a detector
        that was absent in this event) must be returned immediately by
        add() without entering the chunk buffer."""
        from psana.gpu.gpu_events import _D2hPipeline
        from psana.gpu.context import GpuEventContext
        from types import SimpleNamespace

        pipe = _D2hPipeline(det_key="jungfrau.calib", chunk_size=2)
        fake_evt = SimpleNamespace(timestamp=0)
        ctx = GpuEventContext(
            evt=fake_evt,
            gpu_results={},  # no calib key
            leases={},
        )
        ready = pipe.add(ctx)
        assert ready == [ctx], "context without matching key must pass through"

    def test_calib_done_waited_before_d2h(self):
        """Checks the ordering guarantee between calibration and transfer.
        After adding one event, inspects the D→H stream's wait_events
        list and confirms that the calib_done CUDA event from the slot
        lease appears in it.  This proves the memcpy cannot start until
        the calibration kernel has finished writing to the slot buffer —
        reading stale data would cause silent correctness errors."""
        from psana.gpu.gpu_events import _D2hPipeline

        pipe = _D2hPipeline(det_key="jungfrau.calib", chunk_size=1)
        ctx = self._make_ctx(fill=5.0)
        pipe.add(ctx)
        calib = ctx._leases["jungfrau.calib"].calib_done
        assert calib in pipe._d2h_stream.wait_events, "D→H stream must call wait_event(calib_done) before memcpyAsync"

    def test_get_free_slot_returns_none_when_all_slots_busy(self):
        """Non-blocking fallback: _flush_chunk returns events without
        _pending_d2h when all pinned slots are occupied.

        This prevents the generator from deadlocking when the user
        accumulates contexts before calling on_cpu (e.g. list(run.events())).

        Sequence:
          1. chunk_size=1 — each add() flushes immediately.
          2. Add two events without calling on_cpu — both slots are now
             claimed; _available queue is empty.
          3. Add a third event.  _get_free_slot() must return None
             (SimpleQueue.get_nowait() raises Empty), so the event is
             yielded WITHOUT _pending_d2h (sync fallback path).
          4. Verify ctx2's GPUResult has _pending_d2h=None.
          5. Now call on_cpu for ctx0 — dec_ref() returns slot to queue.
          6. Add a fourth event — slot available, async D→H resumes.
        """
        from psana.gpu.gpu_events import _D2hPipeline

        pipe = _D2hPipeline(det_key="jungfrau.calib", chunk_size=1)

        # Warm up: two events — both slots claimed; _available queue empty.
        ctx0 = self._make_ctx(fill=1.0)
        ctx1 = self._make_ctx(fill=2.0)
        pipe.add(ctx0)
        pipe.add(ctx1)
        assert pipe._available.empty(), "both slots should be claimed"

        # Third event: queue empty → sync fallback (no _pending_d2h).
        ctx2 = self._make_ctx(fill=3.0)
        ready = pipe.add(ctx2)
        assert len(ready) == 1, "ctx2 should be yielded immediately via sync fallback"
        result2 = ready[0].get("jungfrau.calib")
        assert result2._pending_d2h is None, (
            "sync fallback should not attach _pending_d2h"
        )

        # Free a slot: dec_ref() puts the slot back into _available.
        ctx0.get("jungfrau.calib").on_cpu
        assert not pipe._available.empty(), "slot should be back in the free queue"

        # Fourth event: slot available → async D→H resumes.
        ctx3 = self._make_ctx(fill=4.0)
        ready = pipe.add(ctx3)
        assert len(ready) == 1, "ctx3 should be yielded"
        result3 = ready[0].get("jungfrau.calib")
        assert result3._pending_d2h is not None, (
            "async D→H should resume once a slot is returned to the queue"
        )

    def test_dec_ref_race_lock_prevents_lost_update(self):
        """_refs_lock ensures that concurrent dec_ref() calls produce the
        correct final count and return the slot to the queue exactly once.

        With chunk_size=2 a single _PinnedSlot holds _refs=2.  Two threads
        each call dec_ref() simultaneously — without the lock the lost-update
        race could leave _refs=1 and the slot would never be returned to
        _available.  With the lock only one thread decrements at a time, so
        _refs reaches 0 and the slot is put() into _available exactly once.
        """
        import threading
        from psana.gpu.gpu_events import _D2hPipeline

        pipe = _D2hPipeline(det_key="jungfrau.calib", chunk_size=1)

        # Prime: two events with chunk_size=1 → each flushes into its own slot.
        # After both adds the _available queue must be empty (_refs=1 each).
        ctx0 = self._make_ctx(fill=1.0)
        ctx1 = self._make_ctx(fill=2.0)
        pipe.add(ctx0)
        pipe.add(ctx1)
        assert pipe._available.empty(), "both slots should be claimed (queue empty)"

        # Pick the slot that ctx0 is using — it has _refs=1.
        # We will call dec_ref twice simultaneously to exercise the lock.
        slot = pipe._pinned_pool[0]
        # Force _refs=2 so that two concurrent dec_ref() calls are needed
        # to trigger the put() — mirroring chunk_size=2 semantics.
        slot._refs = 2

        barrier = threading.Barrier(2)
        errors = []

        def call_dec_ref():
            try:
                barrier.wait()  # both threads start dec_ref simultaneously
                slot.dec_ref()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=call_dec_ref)
        t2 = threading.Thread(target=call_dec_ref)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors, f"unexpected exception in thread: {errors}"
        assert slot._refs == 0, f"_refs should be 0 after two dec_ref calls, got {slot._refs}"
        # Slot should appear in the queue exactly once.
        returned = []
        while not pipe._available.empty():
            returned.append(pipe._available.get_nowait())
        assert len(returned) == 1, (
            f"slot should be returned to queue exactly once, got {len(returned)}"
        )
        assert returned[0] is slot, "the returned object should be the slot itself"
