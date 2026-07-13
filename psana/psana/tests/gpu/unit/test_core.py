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
from psana.psexp import TransitionId
from psana.psexp.packet_footer import PacketFooter


def test_public_gpu_api_is_minimal():
    import psana.gpu as gpu

    assert gpu.__all__ == ["GPUResult", "GpuEventContext", "init_gpu_rank"]
    internal_names = {
        "DetectorRouter",
        "EventPool",
        "GPUKernelRegistry",
        "create_gpu_communicators",
        "gpu_error_handler",
        "log_gpu_mem",
        "optimal_kernel_batch_size",
        "share_calib_between_gpu_peers",
        "verify_gpu_pinning",
    }
    assert internal_names.isdisjoint(vars(gpu))


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


class _FakeStream:
    def __init__(self, non_blocking=True):
        self.non_blocking = non_blocking
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


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
            yield item


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
    events.run = SimpleNamespace(
        _handle_transition=lambda dgrams: log.append(("transition", dgrams[0]))
    )
    return events


def test_event_pool_retires_slot_before_reuse(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(cuda=SimpleNamespace(Stream=_FakeStream)),
    )

    pool = EventPool(n=1)
    detectors = {"jungfrau": (None, _FakeDetector())}

    pool.submit(None, None, ["event-0"], detectors)
    with pytest.raises(RuntimeError, match="without retire_next"):
        pool.submit(None, None, ["event-1"], detectors)

    results, events = pool.retire_next()
    assert results == {}
    assert events == ["event-0"]
    assert pool._streams[0].synchronize_calls == 1

    pool.submit(None, None, ["event-1"], detectors)


def test_beginstep_flushes_before_calib_update(monkeypatch, fake_transition_decode):
    log = []
    events = _new_gpu_events(log)
    events.gpu_detectors = {
        "jungfrau": (
            object(),
            SimpleNamespace(
                beginstep=lambda peds, gmask: log.append(
                    ("beginstep", peds, gmask)
                )
            ),
        )
    }

    def fake_constants(det):
        log.append("constants")
        return "peds", "gmask"

    monkeypatch.setattr(
        gpu_events_module, "_compute_calib_constants_cpu", fake_constants
    )

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
    gpu_result = object()
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
    assert results[0].get("jungfrau.calib").on_gpu is gpu_result
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
