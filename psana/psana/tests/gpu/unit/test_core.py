"""Fast CPU-only invariants for the psana GPU event path."""

import os
import sys
from types import SimpleNamespace

import pytest

import psana.gpu.gpu_events as gpu_events_module
from psana.gpu.detector_router import DetectorRouter
from psana.gpu.gpu_calib import _segment_ids_in_l1_order
from psana.gpu.gpu_events import GpuEvents
from psana.gpu.gpu_stream import EventPool
from psana.event import Event
from psana.psexp import TransitionId
from psana.psexp.packet_footer import PacketFooter


def test_public_gpu_api_is_minimal():
    import psana.gpu as gpu

    # D→H join is now internal to GpuEvents — no join class in public API.
    assert "EventJoiner" not in gpu.__all__, "EventJoiner was made internal"
    assert "CalibJoiner" not in gpu.__all__, "CalibJoiner was renamed then made internal"
    # These implementation-detail names must never be public.
    internal_names = {
        "gpu_error_handler",
        "share_calib_between_gpu_peers",
        "verify_gpu_pinning",
    }
    assert internal_names.isdisjoint(gpu.__all__)


def test_gpu_only_event_preserves_l1_metadata_without_detector_segments():
    """A GPU-only event must not require a CPU BigData dgram."""
    timestamp = (1_234_567 << 32) | 890
    dgram = gpu_events_module._GpuOnlyDgram(timestamp)
    evt = Event([dgram, None])

    assert evt.timestamp == timestamp
    assert evt.service() == TransitionId.L1Accept
    assert evt.env == TransitionId.L1Accept << 24
    assert evt._det_segments == {}


def test_segment_ids_preserve_l1_child_order():
    dgram = SimpleNamespace(
        jungfrau={
            17: object(),
            13: object(),
            9: object(),
            5: object(),
            29: object(),
            25: object(),
            21: object(),
        }
    )

    assert _segment_ids_in_l1_order(dgram, "jungfrau") == [
        17,
        13,
        9,
        5,
        29,
        25,
        21,
    ]
    assert _segment_ids_in_l1_order(object(), "jungfrau") == []


class _FakeEvent:
    def __init__(self, disable_timing=False):
        self.done = True

    def record(self, stream=None):
        pass

    def synchronize(self):
        pass


class _FakeStream:
    def __init__(self, non_blocking=True):
        self.non_blocking = non_blocking
        self.synchronize_calls = 0
        self.ptr = 0

    def synchronize(self):
        self.synchronize_calls += 1

    def wait_event(self, event):
        pass


class _FakeDetector:
    def process_batch(self, *args, **kwargs):
        return iter(())


class _FakeFlushPool:
    def __init__(self, log, pending=()):
        self.log = log
        self.pending = list(pending)
        self.flush_calls = 0
        self.yield_count = 0

    def flush(self):
        self.flush_calls += 1
        self.log.append("flush")
        pending, self.pending = self.pending, []
        for item in pending:
            self.yield_count += 1
            if hasattr(item, "gpu_results_by_ts"):
                yield item
                continue
            results, evts = item[:2]
            leases = item[2] if len(item) > 2 else {}
            yield SimpleNamespace(
                gpu_results_by_ts=results,
                cpu_evts=evts,
                leases_by_ts=leases,
                pending_d2h_by_ts={},
                cached_cpu_results_by_ts={},
            )


@pytest.fixture
def fake_transition_decode(monkeypatch):
    monkeypatch.setattr(
        gpu_events_module,
        "_iter_step_events",
        lambda transition_batch, configs: iter(transition_batch),
    )


def _transition_batch(*services):
    transitions = [(service, [service]) for service in services]
    return {0: (transitions, None)}


def _new_gpu_events(log, pending=()):
    events = GpuEvents.__new__(GpuEvents)
    events.configs = []
    events.event_pool = _FakeFlushPool(log, pending=pending)
    events.gpu_detectors = {}
    events.router = None
    events.cpu_dets = {}
    events._d2h_pipelines = {}
    events._high_water = {}
    events._first_batch_logged = True  # suppress first-batch log in tests
    from psana.gpu.gpu_budget import _GpuBudget

    events._gpu_budget = _GpuBudget(limit_bytes=1024**4)  # 1 TiB sentinel
    events.run = SimpleNamespace(_handle_transition=lambda dgrams: log.append(("transition", dgrams[0])))
    return events


def test_event_pool_retires_slot_before_reuse(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(cuda=SimpleNamespace(Stream=_FakeStream, Event=_FakeEvent)),
    )

    pool = EventPool(n=1)
    detectors = {"jungfrau": (None, _FakeDetector())}

    pool.submit(None, None, ["event-0"], detectors)
    with pytest.raises(RuntimeError, match="before retirement finished"):
        pool.submit(None, None, ["event-1"], detectors)

    record = pool.begin_retire_next()
    assert record.gpu_results_by_ts == {}
    assert record.cpu_evts == ["event-0"]
    assert record.leases_by_ts == {}
    assert pool._streams[0].synchronize_calls == 1
    with pytest.raises(RuntimeError, match="before retirement finished"):
        pool.submit(None, None, ["event-1"], detectors)

    pool.finish_retire_next()
    pool.submit(None, None, ["event-1"], detectors)


def test_beginstep_flushes_before_calib_update(monkeypatch, fake_transition_decode):
    log = []
    events = _new_gpu_events(log)
    events.gpu_detectors = {
        "jungfrau": (
            object(),
            SimpleNamespace(beginstep=lambda peds, gmask: log.append(("beginstep", peds, gmask))),
        )
    }

    def fake_constants(det):
        log.append("constants")
        return "peds", "gmask"

    monkeypatch.setattr(gpu_events_module, "_compute_calib_constants_cpu", fake_constants)

    step_dict = _transition_batch(
        TransitionId.Enable,
        TransitionId.BeginStep,
        TransitionId.Disable,
    )
    assert list(events._handle_steps(step_dict)) == []
    assert log == [
        "flush",
        ("transition", TransitionId.Enable),
        "constants",
        ("beginstep", "peds", "gmask"),
        ("transition", TransitionId.BeginStep),
        ("transition", TransitionId.Disable),
    ]


def test_non_boundary_transitions_do_not_flush(fake_transition_decode):
    log = []
    events = _new_gpu_events(log)
    step_dict = _transition_batch(
        TransitionId.Enable,
        TransitionId.Disable,
        TransitionId.EndStep,
    )

    assert list(events._handle_steps(step_dict)) == []
    assert log == [
        ("transition", TransitionId.Enable),
        ("transition", TransitionId.Disable),
        ("transition", TransitionId.EndStep),
    ]
    assert events.event_pool.flush_calls == 0


def test_endrun_flushes_pending_result_once_and_stops(fake_transition_decode):
    log = []
    timestamp = 123
    import numpy as np

    # Use a real ndarray so on_gpu (which now returns a copy) works correctly.
    gpu_result = np.ones((4, 8, 8), dtype=np.float32) * 42.0
    cpu_evt = SimpleNamespace(timestamp=timestamp)
    events = _new_gpu_events(
        log,
        pending=[({timestamp: {"jungfrau.calib": gpu_result}}, [cpu_evt])],
    )
    events.gpu_reader = SimpleNamespace(close=lambda: log.append("close"))

    request_count = 0

    def next_batch():
        nonlocal request_count
        request_count += 1
        if request_count > 1:
            raise AssertionError("GpuEvents requested a batch after EndRun")
        return {}, {}, _transition_batch(TransitionId.EndRun)

    events._next_batch = next_batch

    results = list(events._events())

    assert request_count == 1
    assert len(results) == 1
    assert results[0].timestamp == timestamp
    # on_gpu returns a copy — verify the value not identity
    copy = results[0].get("jungfrau.calib").on_gpu
    np.testing.assert_array_equal(copy, gpu_result)
    assert events.event_pool.yield_count == 1
    assert ("transition", TransitionId.EndRun) in log
    assert log[-1] == "close"


def _pack_transport(smd_bytes, gpu_bytes):
    footer = PacketFooter(2)
    footer.set_size(0, len(smd_bytes))
    footer.set_size(1, len(gpu_bytes))
    return bytearray(smd_bytes) + bytearray(gpu_bytes) + bytearray(footer.footer)


def _unpack_transport(chunk):
    from psana.psexp.node import BigDataNode

    receiver = SimpleNamespace()
    unpack = BigDataNode._unpack_batch.__get__(receiver, SimpleNamespace)
    return unpack(chunk)


def test_mpi_transport_unpacking():
    cases = [
        (bytearray(), b"", b""),
        (_pack_transport(b"smd", b"GPUBAT1\0gpu"), b"smd", b"GPUBAT1\0gpu"),
        (_pack_transport(b"cpu-only", b""), b"cpu-only", b""),
        # A legacy two-packet step batch is not a GPU transport envelope.
        (_pack_transport(b"step-one", b"step-two"),
         bytes(_pack_transport(b"step-one", b"step-two")), b""),
        (bytearray(b"legacy-without-footer"), b"legacy-without-footer", b""),
    ]

    for packed, expected_smd, expected_gpu in cases:
        smd, gpu = _unpack_transport(packed)
        assert bytes(smd) == expected_smd
        assert bytes(gpu) == expected_gpu


@pytest.mark.parametrize(
    "local_rank,n_gpus,expected",
    [(0, 1, 0), (0, 4, 0), (3, 4, 3), (5, 4, 1), (3, 2, 1)],
)
def test_gpu_rank_mapping(monkeypatch, local_rank, n_gpus, expected):
    from psana.gpu.gpu_mpi import init_gpu_rank

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert init_gpu_rank(local_rank=local_rank, n_gpus=n_gpus) == expected
    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(expected)


def test_gpu_io_error_aborts_mpi_job():
    from psana.gpu.gpu_mpi import gpu_error_handler

    abort_calls = []

    class _FakeComm:
        def Get_rank(self):
            return 0

        def Abort(self, code):
            abort_calls.append(code)

    error = RuntimeError("KvikIO pread failed")
    handler = gpu_error_handler(_FakeComm())
    assert handler.__exit__(type(error), error, None) is True
    assert abort_calls == [1]


def test_default_result_routing():
    router = DetectorRouter()
    router.register_gpu("jungfrau")
    assert router.resolve_key("calib") == "jungfrau.calib"
    assert router.resolve_key("jungfrau.calib") == "jungfrau.calib"


# ---------------------------------------------------------------------------
# Phase 3: GpuSubbatchView, estimate_subbatch_bytes, _split_subbatches
# ---------------------------------------------------------------------------

import struct

from psana.gpu.gpu_batch import (
    GPU_BATCH_MAGIC,
    GPU_BATCH_VERSION,
    GPU_DESC_FLAG_VALID,
    GPU_DESC_NBYTES,
    GPU_EVENT_NBYTES,
    GPU_HEADER_NBYTES,
    GpuBatchView,
    GpuSubbatchView,
)
from psana.gpu.gpu_calib import GPUDetector


def _make_batch(n_events, descs_per_event=2, bd_size=1024, stream_ids=None):
    """Build a minimal valid GPUBAT1 binary for unit testing.

    All events share the same layout: ``descs_per_event`` descriptors each,
    with bd_size bytes of bigdata per descriptor.  Stream IDs default to
    [0, 1, ..., descs_per_event-1].
    """
    if stream_ids is None:
        stream_ids = list(range(descs_per_event))
    n_desc     = n_events * descs_per_event
    evt_offset = GPU_HEADER_NBYTES
    dsc_offset = evt_offset + n_events * GPU_EVENT_NBYTES
    total      = dsc_offset + n_desc * GPU_DESC_NBYTES
    mask       = sum(1 << s for s in stream_ids)

    buf = bytearray(total)
    struct.pack_into('<11Q', buf, 0,
        GPU_BATCH_MAGIC, GPU_BATCH_VERSION, GPU_HEADER_NBYTES,
        GPU_EVENT_NBYTES, GPU_DESC_NBYTES, n_events, n_desc,
        mask, evt_offset, dsc_offset, total,
    )
    for i in range(n_events):
        struct.pack_into('<5Q', buf, evt_offset + i * GPU_EVENT_NBYTES,
            i, 1000 + i, i * descs_per_event, descs_per_event, 0)
    for i in range(n_desc):
        stream_i = stream_ids[i % descs_per_event]
        struct.pack_into('<7Q', buf, dsc_offset + i * GPU_DESC_NBYTES,
            i // descs_per_event, stream_i, 0, bd_size, 0, GPU_DESC_FLAG_VALID, 0)
    return bytes(buf)


class _FakeDetForEstimate:
    """Minimal stand-in for GPUDetector used in estimate_subbatch_bytes tests."""
    _passthrough = False   # normal (uint16) mode — matches GPUDetector default

    def __init__(self, n_segs, nrows, ncols, stream_seg_map=None):
        self._stream_seg_map = stream_seg_map
        self._n_segs_calib   = n_segs
        self._nrows          = nrows
        self._ncols          = ncols

    def estimate_subbatch_bytes(self, n_events):
        return GPUDetector.estimate_subbatch_bytes(self, n_events)


def _new_splitting_gpu_events(det, budget_bytes):
    """Create a minimal GpuEvents with just enough state for _split_subbatches."""
    events = GpuEvents.__new__(GpuEvents)
    events.gpu_detectors         = {'jungfrau': (None, det, {})}
    events._subbatch_budget_bytes = budget_bytes
    return events


# -- GpuSubbatchView tests ---------------------------------------------------

class TestGpuSubbatchView:

    def test_n_events(self):
        gv = GpuBatchView(_make_batch(5))
        sb = GpuSubbatchView(gv, 1, 4)
        assert sb.n_events == 3

    def test_has_work(self):
        gv = GpuBatchView(_make_batch(5))
        assert GpuSubbatchView(gv, 0, 3).has_work is True

    def test_empty_subbatch_raises(self):
        gv = GpuBatchView(_make_batch(5))
        with pytest.raises(ValueError, match="empty range"):
            GpuSubbatchView(gv, 2, 2)

    def test_out_of_range_raises(self):
        gv = GpuBatchView(_make_batch(3))
        with pytest.raises(ValueError):
            GpuSubbatchView(gv, 0, 4)   # event_end > n_events

    def test_timestamps(self):
        gv = GpuBatchView(_make_batch(5))
        sb = GpuSubbatchView(gv, 2, 5)
        assert sb.timestamps == frozenset({1002, 1003, 1004})

    def test_iter_events_first_desc_reindexed(self):
        """first_desc must be relative to the subbatch's own desc_table."""
        # 4 events, 3 descs each.  Subbatch [1, 3) covers events 1 and 2.
        gv   = GpuBatchView(_make_batch(4, descs_per_event=3, stream_ids=[0, 1, 2]))
        sb   = GpuSubbatchView(gv, 1, 3)
        evts = list(sb.iter_events())

        # Event 1 → first_desc=0,  n_desc=3
        # Event 2 → first_desc=3,  n_desc=3
        assert [e.first_desc for e in evts] == [0, 3]
        assert [e.n_desc     for e in evts] == [3, 3]

    def test_iter_events_preserves_timestamps_and_batch_event_index(self):
        gv   = GpuBatchView(_make_batch(5))
        sb   = GpuSubbatchView(gv, 2, 5)
        evts = list(sb.iter_events())
        assert [e.timestamp         for e in evts] == [1002, 1003, 1004]
        assert [e.batch_event_index for e in evts] == [2,    3,    4]

    def test_total_read_bytes(self):
        # 3 events, 2 descs each, 2048 bytes per desc
        # subbatch [0, 2): 2 events × 2 descs × 2048 = 8192
        gv = GpuBatchView(_make_batch(3, descs_per_event=2, bd_size=2048))
        sb = GpuSubbatchView(gv, 0, 2)
        assert sb.total_read_bytes == 2 * 2 * 2048

    def test_whole_batch_subbatch(self):
        """A subbatch covering the entire batch is identical to the parent."""
        gv   = GpuBatchView(_make_batch(4))
        sb   = GpuSubbatchView(gv, 0, 4)
        full = list(gv.iter_events())
        sub  = list(sb.iter_events())
        assert [e.timestamp for e in sub] == [e.timestamp for e in full]
        # first_desc for subbatch-0 must equal the parent's first_desc
        # (both start from 0 for the first event)
        assert sub[0].first_desc == full[0].first_desc == 0


# -- GPUDetector.estimate_subbatch_bytes tests --------------------------------

class TestEstimateSubbatchBytes:

    def test_returns_zero_for_n_events_zero(self):
        det = _FakeDetForEstimate(4, 512, 1024)
        assert det.estimate_subbatch_bytes(0) == 0

    def test_linear_in_n_events(self):
        det = _FakeDetForEstimate(4, 512, 1024)
        e1  = det.estimate_subbatch_bytes(1)
        e10 = det.estimate_subbatch_bytes(10)
        assert e10 == 10 * e1

    def test_formula_with_stream_seg_map(self):
        # stream_seg_map: 2 GPU streams, 5 and 7 segs respectively → 12 total GPU segs
        det = _FakeDetForEstimate(
            n_segs=32, nrows=512, ncols=1024,
            stream_seg_map={6: list(range(5)), 8: list(range(7))},
        )
        n_segs_gpu = 5 + 7
        expected   = 1 * n_segs_gpu * 512 * 1024 * (4 + 2)
        assert det.estimate_subbatch_bytes(1) == expected

    def test_formula_without_stream_seg_map_uses_n_segs_calib(self):
        det = _FakeDetForEstimate(n_segs=8, nrows=256, ncols=512)
        expected = 1 * 8 * 256 * 512 * (4 + 2)
        assert det.estimate_subbatch_bytes(1) == expected


# -- _split_subbatches tests --------------------------------------------------

class TestSplitSubbatches:

    def _events_and_det(self, n_segs, nrows, ncols, budget_bytes):
        det    = _FakeDetForEstimate(n_segs, nrows, ncols)
        events = _new_splitting_gpu_events(det, budget_bytes)
        return events, det

    def test_no_split_when_budget_large(self):
        events, det = self._events_and_det(4, 512, 1024, budget_bytes=10 * 1024**3)
        gv = GpuBatchView(_make_batch(6))
        sbs = events._split_subbatches(gv)
        assert len(sbs) == 1
        assert sbs[0]._start == 0 and sbs[0]._end == 6

    def test_splits_into_equal_halves(self):
        # 4 events, bd_size=0 (no raw input cost).
        # Budget = exactly 2 events of calib cost.
        det    = _FakeDetForEstimate(4, 512, 1024)
        per_ev = det.estimate_subbatch_bytes(1)
        events = _new_splitting_gpu_events(det, per_ev * 2)

        # bd_size=0 → raw cost = 0, only calib cost counts
        gv  = GpuBatchView(_make_batch(4, descs_per_event=2, bd_size=0))
        sbs = events._split_subbatches(gv)
        assert len(sbs) == 2
        assert sbs[0]._start == 0 and sbs[0]._end == 2
        assert sbs[1]._start == 2 and sbs[1]._end == 4

    def test_single_oversized_event_not_split(self):
        """An event that alone exceeds budget must still be included."""
        det    = _FakeDetForEstimate(4, 512, 1024)
        events = _new_splitting_gpu_events(det, budget_bytes=1)   # effectively 0
        gv     = GpuBatchView(_make_batch(3, bd_size=0))
        sbs    = events._split_subbatches(gv)
        # Each event must appear in exactly one subbatch (even with tiny budget)
        assert len(sbs) == 3
        for i, sb in enumerate(sbs):
            assert sb._start == i and sb._end == i + 1

    def test_event_order_preserved(self):
        det    = _FakeDetForEstimate(4, 512, 1024)
        per_ev = det.estimate_subbatch_bytes(1)
        events = _new_splitting_gpu_events(det, per_ev * 2)
        gv     = GpuBatchView(_make_batch(6, bd_size=0))
        sbs    = events._split_subbatches(gv)
        all_ts = []
        for sb in sbs:
            all_ts.extend(e.timestamp for e in sb.iter_events())
        assert all_ts == [1000, 1001, 1002, 1003, 1004, 1005]

    def test_empty_batch_returns_empty_list(self):
        det    = _FakeDetForEstimate(4, 512, 1024)
        events = _new_splitting_gpu_events(det, 10 * 1024**3)
        # build a batch with 0 events
        hdr_bytes = GPU_HEADER_NBYTES
        buf = bytearray(hdr_bytes)
        struct.pack_into('<11Q', buf, 0,
            GPU_BATCH_MAGIC, GPU_BATCH_VERSION, GPU_HEADER_NBYTES,
            GPU_EVENT_NBYTES, GPU_DESC_NBYTES, 0, 0, 0,
            hdr_bytes, hdr_bytes, hdr_bytes,
        )
        gv  = GpuBatchView(bytes(buf), validate=True)
        sbs = events._split_subbatches(gv)
        assert sbs == []

    def test_subbatch_estimates_stay_within_budget(self):
        """For each subbatch, estimated bytes <= budget (except single-event overflows)."""
        det    = _FakeDetForEstimate(4, 512, 1024)
        per_ev = det.estimate_subbatch_bytes(1)
        budget = per_ev * 3   # 3 events per subbatch max
        events = _new_splitting_gpu_events(det, budget)
        gv     = GpuBatchView(_make_batch(10, bd_size=0))
        sbs    = events._split_subbatches(gv)
        for sb in sbs:
            sb_est = det.estimate_subbatch_bytes(sb.n_events)
            assert sb_est <= budget or sb.n_events == 1, (
                f"subbatch has {sb.n_events} events, "
                f"estimated {sb_est} bytes > budget {budget}"
            )
