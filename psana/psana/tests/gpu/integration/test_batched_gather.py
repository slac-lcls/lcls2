"""Canonical subbatch gather equivalence, absence, strides, and dependencies."""
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from psana.gpu import gpu_detector as gd
from psana.gpu.gpu_input import GpuDetectorBinding, GpuEventDgrams
from psana.gpu.gpudgram.batch import GpuXtcBatchPool
from psana.gpu.gpudgram.config import GpuStreamConfigTable
from psana.gpu.gpudgram import parser as p
from psana.gpu.gpu_kvikio_read import (
    DESC_NCOLS, DESC_EVENT_INDEX, DESC_STREAM_ID, DESC_DEVICE_OFFSET, DESC_READ_SIZE,
    DESC_TIMESTAMP,
)


def _gpu_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not _gpu_available(), reason="no CUDA device available")]


def _setup(cp, passthrough=False, budget=None, parser_slots=1):
    dtype = np.float32 if passthrough else np.uint16

    def entry(segment, names, det="camera"):
        return dict(det_name=det, det_type="test", det_id=det, segment=segment,
                    alg_name="raw", alg_version=(1, 0, 0), names_id_value=names,
                    fields=[dict(name="counter", type=2, element_size=4, rank=0,
                                 field_index=0, shape_index=-1),
                            dict(name="pixels", type=8 if passthrough else 1,
                                 element_size=np.dtype(dtype).itemsize, rank=2,
                                 field_index=1, shape_index=0)])

    configs = GpuStreamConfigTable({0: [entry(4, 10)],
                                    1: [entry(9, 10), entry(8, 11)],
                                    2: [entry(0, 10, "other")]})
    binding = GpuDetectorBinding("camera", canonical_segment_ids=(9, 4, 8),
                                field_handles_by_segment={
                                    s: configs.resolve("camera", s, "raw", "pixels")
                                    for s in (9, 4, 8)})
    # Handle order deliberately differs from both stream and canonical order.
    pool = GpuXtcBatchPool(configs, field_handles=tuple(reversed(configs.field_handles())),
                           n_slots=parser_slots, budget=budget)
    peds = None if passthrough else cp.full(3 * 3 * 300, 7, dtype=cp.float32)
    gains = None if passthrough else cp.full(3 * 3 * 300, 2, dtype=cp.float32)
    detector = gd.GPUDetector((3, 3, 100), peds, gains, binding, n_slots=1,
                              budget=budget, passthrough=passthrough)
    detector.configure_gather(pool.handle_indices)
    cp.cuda.get_current_stream().synchronize()
    return pool, detector, dtype


def _xtc(kind, payload=b"", src=0):
    return struct.pack("<IHHI", src, 0, kind, 12 + len(payload)) + payload


def _input(cp, pool, producer, selections, dtype):
    pieces, descriptors, events, expected = [], [], [], []
    offset = 0
    for i, streams in enumerate(selections):
        first = len(descriptors)
        canonical = np.zeros((3, 3, 100), dtype=dtype)
        for stream in streams:
            nodes = b""
            for segment, names in ({0: [(4, 10)], 1: [(8, 11), (9, 10)],
                                    2: [(0, 10)]}[stream]):
                values = (np.arange(300).reshape(3, 100) + i * 400 + segment).astype(dtype)
                shape = _xtc(2, struct.pack("<5I", 3, 100, 0, 0, 0))
                data = _xtc(3, struct.pack("<I", i) + values.tobytes())
                nodes += _xtc(1, shape + data, names)
                if segment in (9, 4, 8):
                    canonical[(9, 4, 8).index(segment)] = values
            payload = struct.pack("<QI", 100 + i, 12 << 24) + _xtc(0, nodes)
            row = np.zeros(DESC_NCOLS, dtype=np.uint64)
            row[[DESC_EVENT_INDEX, DESC_STREAM_ID, DESC_DEVICE_OFFSET, DESC_READ_SIZE]] = (
                i * 3 + 7, stream, offset, len(payload))
            row[DESC_TIMESTAMP] = 100 + i
            descriptors.append(row)
            pieces.append(payload)
            offset += len(payload)
        events.append(SimpleNamespace(first_desc=first, n_desc=len(streams),
                                      timestamp=100 + i, batch_event_index=i * 3 + 7))
        if set(streams) & {0, 1}:
            expected.append(canonical)
    with producer:
        data = cp.asarray(np.frombuffer(b"".join(pieces), dtype=np.uint8))
    # Keep the host upload source alive until the copy finishes in this fixture.
    producer.synchronize()
    desc = np.asarray(descriptors, dtype=np.uint64).reshape(-1, DESC_NCOLS)
    batch = pool.parse(0, data, desc, producer)
    batch._test_descriptors = desc
    return batch, tuple(GpuEventDgrams(e, batch) for e in events), expected


def _old_gather(cp, detector, events, stream):
    raw, calib = [], []
    with stream:
        for event in events:
            if not detector.binding.has_sources(event):
                continue
            target = cp.zeros(detector.det_shape, dtype=(cp.float32 if detector._passthrough
                                                       else cp.uint16))
            present = cp.zeros(3, dtype=cp.uint8)
            for dgram, row, _, handle in detector.binding.iter_sources(event):
                locators = dgram.locate(handle).wait_on(stream)
                gd._gather_locator_field_gpu(dgram.data_gpu, locators, dgram.dgram_index,
                                             handle, row, 300, target, present)
            if detector._passthrough:
                out = target
            else:
                out = cp.empty(detector.det_shape, dtype=cp.float32)
                gd.fused_calib_gpu(target, detector.peds_gpu, detector.gmask_gpu, out=out)
                gd._zero_missing_rows_gpu(out, present)
            raw.append(target)
            calib.append(out)
    return raw, calib


@pytest.mark.parametrize("passthrough", [False, True])
@pytest.mark.parametrize("cross_stream", [False, True])
def test_canonical_tail_reuse_missing_sources_and_launch_count(monkeypatch, passthrough, cross_stream):
    import cupy as cp
    from psana.gpu.gpu_budget import _GpuBudget

    budget = _GpuBudget(16 * 1024**2)
    pool, detector, dtype = _setup(cp, passthrough, budget)
    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True) if cross_stream else producer
    launches = []
    kernel = gd._batched_gather_kernel

    def counted(dtype):
        launch = kernel(dtype)
        def run(*args, **kwargs):
            launches.append(1)
            return launch(*args, **kwargs)
        return run

    monkeypatch.setattr(gd, "_batched_gather_kernel", counted)
    original_map_ptr = None
    for selections in ([(1, 0), (0,), (1,), (2,), (1, 0)],
                       [(0,), (1,)], [], [(1, 0)] * 7):
        batch, events, expected = _input(cp, pool, producer, selections, dtype)
        assert batch._locators == {}
        # The new hot path must not request per-segment locator objects.
        def unexpected(*args, **kwargs):
            raise AssertionError("per-handle locate in canonical gather")
        batch.locate = unexpected
        before = len(launches)
        actual = list(detector.process_batch(events, stream=consumer, slot_id=0))
        consumer.synchronize()
        producer.synchronize()
        assert batch._locators == {}
        del batch.locate  # restore class method without a self -> bound-method cycle
        old_raw, old_calib = _old_gather(cp, detector, events, consumer)
        consumer.synchronize()
        assert len(launches) - before == bool(expected)
        assert [a.timestamp for a in actual] == [e.timestamp for e in events
                                               if detector.binding.has_sources(e)]
        for i, result in enumerate(actual):
            target = result.calib_gpu if passthrough else result.raw_gpu
            np.testing.assert_array_equal(target.get(), expected[i])
            np.testing.assert_array_equal(target.get(), old_raw[i].get())
            np.testing.assert_array_equal(result.calib_gpu.get(), old_calib[i].get())
        if actual:
            pointer = detector._gather_maps[0].device.data.ptr
            if original_map_ptr is None:
                original_map_ptr = pointer
            elif len(selections) == 2:
                assert pointer == original_map_ptr
                assert batch.configured_locations().capacity > batch.n_dgrams
        memory = detector.memory_bytes()
        # Constants are an existing separate accounting category at this base.
        assert budget.committed() == pool.memory_bytes()['total'] + memory['total'] - memory['constants']


@pytest.mark.parametrize("column,value", [
    (p.LOC_STATUS, p.STATUS_NOT_PRESENT), (p.LOC_STATUS, p.STATUS_DUPLICATE),
    (p.LOC_STATUS, p.STATUS_CORRUPTED), (p.LOC_STATUS, p.STATUS_BAD_SHAPE),
    (p.LOC_TYPE, 8), (p.LOC_RANK, 1), (p.LOC_NBYTES, 0),
    (p.LOC_NBYTES, 602), (p.LOC_OFFSET, 2**64 - 1), (p.LOC_OFFSET, 1000000),
])
def test_rejected_locator_zeroes_pixels_and_calibration(column, value):
    import cupy as cp
    pool, detector, dtype = _setup(cp)
    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)
    # First fill reusable output/presence buffers with valid nonzero values.
    batch, events, _ = _input(cp, pool, producer, [(0, 1)], dtype)
    list(detector.process_batch(events, stream=consumer, slot_id=0))
    consumer.synchronize()
    batch, events, expected = _input(cp, pool, producer, [(0, 1)], dtype)
    handle = detector.binding.field_handles_by_segment[4]
    with producer:
        batch.locate(handle).rows_gpu[0, column] = value
        batch._configured_ready.record(producer)
    old_raw, old_calib = _old_gather(cp, detector, events, consumer)
    result, = list(detector.process_batch(events, stream=consumer, slot_id=0))
    consumer.synchronize()
    np.testing.assert_array_equal(result.raw_gpu.get(), old_raw[0].get())
    np.testing.assert_array_equal(result.calib_gpu.get(), old_calib[0].get())
    assert not result.raw_gpu[1].get().any()
    assert not result.calib_gpu[1].get().any()
    assert detector._present_slot_bufs[0].get().tolist() == [1, 0, 1]


def test_map_growth_failure_preserves_previous_buffer(monkeypatch):
    import cupy as cp
    from psana.gpu.gpu_budget import _GpuBudget
    pool, detector, dtype = _setup(cp)
    producer = cp.cuda.Stream(non_blocking=True)
    batch, events, _ = _input(cp, pool, producer, [(0, 1)], dtype)
    mapping = gd._GatherMap()
    budget = _GpuBudget(1024**2)
    mapping.prepare(events, (0, 1), producer, budget)
    producer.synchronize()
    old_device, old_host = mapping.device, mapping.host
    committed = budget.committed()
    original = cp.empty
    def fail(*args, **kwargs):
        raise MemoryError("injected device allocation failure")
    monkeypatch.setattr(cp, "empty", fail)
    with pytest.raises(MemoryError):
        mapping.prepare(events * 2, (0, 1), producer, budget)
    assert mapping.device is old_device and mapping.host is old_host
    assert budget.committed() == committed
    monkeypatch.setattr(cp, "empty", original)
    mapping.prepare(events * 2, (0, 1), producer, budget)
    producer.synchronize()
    # Both generations occupy rounded pool blocks while the old alias survives.
    assert budget.committed() == committed * 2
    del old_device
    assert budget.committed() == committed


def test_configured_dependency_waits_only_across_streams():
    import cupy as cp
    pool, _, dtype = _setup(cp)
    producer = cp.cuda.Stream(non_blocking=True)
    batch, _, _ = _input(cp, pool, producer, [(0, 1)], dtype)
    calls = []
    locations = batch.configured_locations()
    locations.wait_on(SimpleNamespace(ptr=producer.ptr, wait_event=calls.append))
    assert not calls
    locations.wait_on(SimpleNamespace(ptr=-1, wait_event=calls.append))
    assert calls == [locations.ready]
    producer.synchronize()


def test_execution_slot_waits_for_delayed_gather_consumer():
    import cupy as cp
    from psana.gpu.gpu_stream import EventPool

    pool, detector, dtype = _setup(cp)
    executions = EventPool(n=1)
    consumer = cp.cuda.Stream(non_blocking=True)
    delay = cp.RawKernel('''extern "C" __global__ void delay() {
        unsigned long long start = clock64();
        while (clock64() - start < 30000000ULL) {}
    }''', "delay")
    delay.compile()
    copied = []
    references = []
    for selections in ([(0, 1), (1,)], [(0,)]):
        producer = executions._streams[0]
        batch, events, expected = _input(cp, pool, producer, selections, dtype)
        producer.synchronize()
        gv = SimpleNamespace(iter_events=lambda: (e.event for e in events))
        read = SimpleNamespace(data_gpu=batch.data_gpu, desc_table=batch._test_descriptors,
                               retain_input=lambda: lambda: None)
        record = executions.submit(gv, read, [], {"camera": (None, detector)}, pool)
        assert executions.begin_retire_next() is record
        lease = record.leases_by_ts[100]['camera.raw']
        with consumer:
            consumer.wait_event(lease.result_ready)
            delay((1,), (1,), ())
            copied.append(record.gpu_results_by_ts[100]['camera.raw'].copy())
            done = cp.cuda.Event(disable_timing=True)
            done.record(consumer)
        lease.register_consumer_done(done)
        # Both output and pinned gather-map storage are still occupied.
        with pytest.raises(RuntimeError, match="before retirement"):
            executions.submit(gv, read, [], {"camera": (None, detector)}, pool)
        executions.finish_retire_next()
        assert done.done
        references.append(expected[0])
    for actual, expected in zip(copied, references):
        np.testing.assert_array_equal(actual.get(), expected)
