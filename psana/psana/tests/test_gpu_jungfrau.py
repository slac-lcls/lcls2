import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

import psana.pycalgos.utilsdetector as ud
from psana.gpu.backends.jungfrau import GpuJungfrauBackend, GpuJungfrauRaw
from psana.gpu.execution import CupyExecutionBackend
from psana.gpu.pipeline import GpuPipeline
from psana.gpu.profiler import GpuProfiler
from psana.gpu.runtime import make_gpu_runtime


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


class _ExecutionStub:
    def array_from_evt_data(self, arr, storage, copy=False):
        return np.array(arr, copy=copy)


class _BackendStub:
    def __init__(self):
        self.execution = _ExecutionStub()


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
        expt='xpptut',
        runnum=17,
        logger=logging.getLogger("test_gpu_jungfrau"),
        profiler=None,
        dsparms=SimpleNamespace(
            det_classes={"normal": {}},
            gpu_detector='jungfrau',
            gpu_runtime='default',
            gpu_pipeline='default',
            gpu_queue_depth=2,
        ),
    )


def test_gpu_profiler_summary_and_trace(tmp_path):
    logger = _LoggerStub()
    summary_path = tmp_path / "gpu-summary.json"
    profiler = GpuProfiler(mode="summary", output_path=str(summary_path), logger=logger, run_label="exp:r1")
    profiler.record_initialization(0.01)
    profiler.record_event_loop_wall(0.02)
    profiler.record_stage1(0.001)
    profiler.record_queue_wait(0.002)
    profiler.record_transition_drain(0.003)
    profiler.record_transfer(1024 * 1024, 0.001)
    profiler.record_event_completed()
    summary = profiler.flush_summary()

    assert summary["events_completed"] == 1
    assert summary["metrics"]["initialization"]["count"] == 1
    assert summary["metrics"]["initialization"]["total_s"] == pytest.approx(0.01)
    assert summary["metrics"]["event_loop_wall"]["count"] == 1
    assert summary["metrics"]["event_loop_wall"]["total_s"] == pytest.approx(0.02)
    assert summary["metrics"]["initialization"]["min_s"] == pytest.approx(0.01)
    assert summary["metrics"]["stage1"]["count"] == 1
    assert summary["metrics"]["stage1"]["min_s"] == pytest.approx(0.001)
    assert summary["transfer"]["count"] == 1
    assert summary["transfer"]["total_bytes"] == 1024 * 1024
    assert summary["transfer"]["avg_rate_Bps"] == pytest.approx(1024 * 1024 / 0.001)
    assert summary_path.exists()
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["run"] == "exp:r1"
    assert written["metrics"]["initialization"]["total_s"] == pytest.approx(0.01)
    assert written["metrics"]["event_loop_wall"]["total_s"] == pytest.approx(0.02)
    assert written["transfer"]["total_bytes"] == 1024 * 1024
    assert logger.messages[0] == "gpu profile summary run=exp:r1 events=1"
    assert logger.messages[1] == "gpu profile cpu_wall_s init=0.010 loop=0.020 rate_evt_s=50.000"
    assert logger.messages[2] == "gpu profile avg_s stage1=0.001 queue=0.002 drain=0.003 copy=0.000 kernel=0.000 cache_upload=0.000 transfer_size_mib=1.000 transfer_rate_mib_s=1000.000"
    assert logger.messages[3] == "gpu profile min_s stage1=0.001 queue=0.002 drain=0.003 copy=0.000 kernel=0.000 cache_upload=0.000 transfer_size_mib=1.000 transfer_rate_mib_s=1000.000"
    assert logger.messages[4] == "gpu profile max_s stage1=0.001 queue=0.002 drain=0.003 copy=0.000 kernel=0.000 cache_upload=0.000 transfer_size_mib=1.000 transfer_rate_mib_s=1000.000"
    assert logger.messages[5] == "gpu profile total_s stage1=0.001 queue=0.002 drain=0.003 copy=0.000 kernel=0.000 cache_upload=0.000 transfer_size_mib=1.000 transfer_rate_mib_s=1000.000"

    trace_path = tmp_path / "gpu-trace.jsonl"
    trace = GpuProfiler(mode="trace", output_path=str(trace_path), run_label="exp:r2")
    trace.record_initialization(0.02)
    trace.record_event_loop_wall(0.03)
    trace.record_stage1(0.004)
    trace.record_transfer(2048, 0.002)
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
    assert wrapper.calib_gpu(evt) == "gpu-calib"
    np.testing.assert_array_equal(wrapper.raw(evt), np.array([5, 6], dtype=np.uint16))
    np.testing.assert_array_equal(wrapper.raw_gpu(evt), np.array([5, 6], dtype=np.uint16))

    fallback_evt = SimpleNamespace()
    assert wrapper.raw_gpu(fallback_evt) is None
    assert wrapper.calib_gpu(fallback_evt) is None
    assert wrapper.calib(fallback_evt, cversion=0) == "cpu-calib"
    assert cpu_raw.calib_calls


def test_make_gpu_runtime_builds_default_runtime():
    run = _dummy_run()
    profiler = GpuProfiler(mode='off', run_label='exp:r17')
    run.profiler = profiler

    runtime = make_gpu_runtime(run=run, profiler=profiler)

    assert runtime.describe() == {'runtime': 'cupy', 'pipeline': '3stage'}
    assert isinstance(runtime.execution, CupyExecutionBackend)
    assert isinstance(runtime.backend, GpuJungfrauBackend)
    assert isinstance(runtime.pipeline, GpuPipeline)
    assert runtime.backend.execution is runtime.execution


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


class _AsyncBackendStub:
    def __init__(self):
        self.ready = {}
        self.waited = []
        self.completed = []

    def allocate_slot_buffers(self, slot):
        return None

    def pack_l1_to_host(self, rec, slot):
        return None

    def ensure_device_cache(self, rec, slot):
        return None

    def transfer_to_device(self, rec, slot):
        return None

    def launch_compute(self, rec, slot):
        self.ready[slot.slot_id] = False
        return None

    def slot_is_ready(self, slot):
        return self.ready.get(slot.slot_id, True)

    def wait_slot(self, slot):
        self.waited.append(slot.slot_id)
        self.ready[slot.slot_id] = True
        return None

    def on_slot_ready(self, slot):
        self.completed.append(slot.slot_id)
        return None

    def on_transition(self, rec, state_version):
        return None


def _l1_record(evt):
    return SimpleNamespace(dgrams=[], service=12, event=evt)


def test_gpu_pipeline_preserves_submission_order_for_async_slots():
    backend = _AsyncBackendStub()
    pipeline = GpuPipeline(backend=backend, queue_depth=2, profiler=None)
    evt1 = SimpleNamespace(tag="evt1")
    evt2 = SimpleNamespace(tag="evt2")

    pipeline.submit_l1(_l1_record(evt1))
    pipeline.submit_l1(_l1_record(evt2))

    backend.ready[1] = True
    assert list(pipeline.pop_ready()) == []

    backend.ready[0] = True
    assert list(pipeline.pop_ready()) == [evt1, evt2]
    assert backend.completed == [0, 1]
    assert pipeline.has_free_slot()


def test_gpu_pipeline_wait_ready_and_transition_flush_pending_slots():
    backend = _AsyncBackendStub()
    pipeline = GpuPipeline(backend=backend, queue_depth=1, profiler=None)
    evt = SimpleNamespace(tag="evt")

    pipeline.submit_l1(_l1_record(evt))

    assert not pipeline.has_free_slot()
    assert list(pipeline.wait_ready()) == [evt]
    assert backend.waited == [0]
    assert pipeline.has_free_slot()

    evt2 = SimpleNamespace(tag="evt2")
    pipeline.submit_l1(_l1_record(evt2))
    flushed = pipeline.handle_transition(SimpleNamespace())

    assert flushed == [evt2]
    assert backend.waited == [0, 0]
    assert pipeline.state_version == 1
    assert pipeline.has_free_slot()
