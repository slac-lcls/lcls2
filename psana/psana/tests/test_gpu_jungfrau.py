import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

import psana.pycalgos.utilsdetector as ud
from psana.gpu.backends.jungfrau import GpuJungfrauBackend, GpuJungfrauRaw
from psana.gpu.profiler import GpuProfiler


class _CpuRawStub:
    def __init__(self):
        self.calib_calls = []
        self.raw_calls = []

    def calib(self, evt, **kwargs):
        self.calib_calls.append((evt, kwargs))
        return "cpu-calib"

    def raw(self, evt, copy=True):
        self.raw_calls.append((evt, copy))
        return np.array([1, 2, 3], dtype=np.uint16)


class _BackendStub:
    def _get_cupy(self):
        return None


class _LoggerStub:
    def __init__(self):
        self.messages = []

    def info(self, msg, *args):
        self.messages.append(msg % args if args else msg)

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _dummy_run():
    return SimpleNamespace(
        runnum=17,
        logger=logging.getLogger("test_gpu_jungfrau"),
        profiler=None,
        dsparms=SimpleNamespace(det_classes={"normal": {}}),
    )


def test_gpu_profiler_summary_and_trace(tmp_path):
    logger = _LoggerStub()
    summary_path = tmp_path / "gpu-summary.json"
    profiler = GpuProfiler(mode="summary", output_path=str(summary_path), logger=logger, run_label="exp:r1")
    profiler.record_stage1(0.001)
    profiler.record_queue_wait(0.002)
    profiler.record_transition_drain(0.003)
    profiler.record_event_completed()
    summary = profiler.flush_summary()

    assert summary["events_completed"] == 1
    assert summary["metrics"]["stage1"]["count"] == 1
    assert summary_path.exists()
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["run"] == "exp:r1"
    assert logger.messages

    trace_path = tmp_path / "gpu-trace.jsonl"
    trace = GpuProfiler(mode="trace", output_path=str(trace_path), run_label="exp:r2")
    trace.record_stage1(0.004)
    trace.record_event_completed()
    trace.flush_summary()
    trace.record_stage1(0.005)
    trace.record_event_completed()
    trace.flush_summary()

    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["flush_count"] == 1
    assert json.loads(lines[1])["flush_count"] == 2


def test_gpu_jungfrau_raw_wrapper_prefers_gpu_result_and_falls_back():
    cpu_raw = _CpuRawStub()
    wrapper = GpuJungfrauRaw(cpu_raw, _BackendStub())
    evt = SimpleNamespace(_gpu_calib="gpu-calib", _gpu_raw=np.array([5, 6], dtype=np.uint16), _gpu_raw_storage="host")

    assert wrapper.calib(evt) == "gpu-calib"
    np.testing.assert_array_equal(wrapper.raw(evt), np.array([5, 6], dtype=np.uint16))

    fallback_evt = SimpleNamespace()
    assert wrapper.calib(fallback_evt, cversion=0) == "cpu-calib"
    assert cpu_raw.calib_calls


def test_gpu_jungfrau_backend_transition_invalidates_cache():
    backend = GpuJungfrauBackend(_dummy_run())
    backend.device_cache._entries[("a",)] = np.array([1], dtype=np.float32)
    backend.device_cache._storage_kind[("a",)] = "host"
    backend._host_calib = {"ccons": np.array([1], dtype=np.float32)}
    backend._cache_version = ("jungfrau", 17)

    backend.on_transition(SimpleNamespace(), 1)

    assert backend.device_cache.get(("a",)) is None
    assert backend._host_calib is None
    assert backend._cache_version is None


def test_gpu_jungfrau_backend_cpu_calib_v3_matches_reference():
    backend = GpuJungfrauBackend(_dummy_run())
    raw = np.array(
        [
            [[0, 1, (1 << 14) + 2], [3, (3 << 14) + 4, (1 << 14) + 5]],
            [[(3 << 14) + 6, 7, 8], [9, 10, (1 << 14) + 11]],
        ],
        dtype=np.uint16,
    )
    size = raw.size
    ccons = np.arange(4 * size * 2, dtype=np.float32)

    expected = np.empty(raw.shape, dtype=np.float32)
    ud.calib_jungfrau_v3(raw, ccons, 512 * 1024, expected)

    actual = backend._cpu_calib_v3(raw, {"ccons": ccons})
    np.testing.assert_allclose(actual, expected)
