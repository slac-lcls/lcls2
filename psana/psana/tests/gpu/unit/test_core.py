"""Fast CPU-only invariants for the psana GPU event path."""

import os
import sys
from types import SimpleNamespace

import pytest

from psana.gpu.detector_router import DetectorRouter
from psana.gpu.gpu_calib import _segment_ids_in_l1_order
from psana.gpu.gpu_kernel_registry import (
    JungfrauCalibKernel,
    SimpleAreaCalibKernel,
    default_registry,
)
from psana.gpu.gpu_stream import EventPool
from psana.psexp.packet_footer import PacketFooter


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


def test_default_kernel_and_result_routing():
    registry = default_registry()
    assert isinstance(registry.get("jungfrau", "calib"), JungfrauCalibKernel)
    assert isinstance(registry.get("epix100", "calib"), SimpleAreaCalibKernel)
    assert registry.get("unknown", "calib") is None

    router = DetectorRouter()
    router.register_gpu("jungfrau")
    assert router.resolve_key("calib") == "jungfrau.calib"
    assert router.resolve_key("jungfrau.calib") == "jungfrau.calib"
