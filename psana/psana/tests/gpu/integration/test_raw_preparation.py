"""Device gate for calibration-free dense preparation and borrowed storage."""
import gc
import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu import gpu_calib, gpu_detector as gd, gpu_events
from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError
from psana.gpu.gpu_input import GpuDetectorBinding, GpuEventDgrams, InputSlotLease
from psana.gpu.gpudgram.batch import GpuXtcBatchPool
from psana.gpu.gpudgram.config import GpuStreamConfigTable
from psana.gpu.gpu_kvikio_read import (
    DESC_NCOLS, DESC_EVENT_INDEX, DESC_STREAM_ID, DESC_DEVICE_OFFSET,
    DESC_READ_SIZE, DESC_TIMESTAMP,
)
from test_batched_gather import _gpu_available, _xtc

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not _gpu_available(), reason='no CUDA device available')]


def setup_raw(budget, rank):
    configs = GpuStreamConfigTable({i: [dict(
        det_name='jf', det_type='jungfrau', det_id='jf', segment=segment,
        alg_name='raw', alg_version=(0, 2, 0), names_id_value=10,
        fields=[dict(name='raw', type=1, element_size=2, rank=rank,
                     field_index=0, shape_index=0)],
    )] for i, segment in enumerate((4, 9))})
    binding = GpuDetectorBinding('jf', canonical_segment_ids=(9, 4),
                                field_handles_by_segment={
                                    s: configs.resolve('jf', s, 'raw', 'raw') for s in (9, 4)})
    parser = GpuXtcBatchPool(configs, field_handles=configs.field_handles(),
                           n_slots=1, budget=budget)
    raw = gd.DenseInputPreparer.jungfrau_raw(configs, binding, n_slots=1, budget=budget)
    raw.configure_gather(parser.handle_indices)
    return parser, raw


def make_input(cp, parser, producer, rank, missing=False, malformed=False):
    packed, rows, specs = bytearray(), [], []
    expected = np.zeros((3, 2, 512, 1024), np.uint16)
    presence = np.zeros((3, 2), np.uint8)
    for i in range(4):
        streams = () if i == 3 else ((0,) if missing and i == 1 else (1, 0))
        first = len(rows)
        for sid in streams:
            value = np.uint16(0xC000 + 10 * i + sid)  # preserve gain bits
            pixels = np.full((512, 1024), value, np.uint16)
            shape = (1, 512, 1024) if rank == 3 else (512, 1024)
            if malformed and i == 2 and sid == 1:
                shape = (1, 1024, 512) if rank == 3 else (1024, 512)
            else:
                expected[i, 1 - sid] = pixels
                presence[i, 1 - sid] = 1
            dims = shape + (0,) * (5 - len(shape))
            child = _xtc(1, _xtc(2, struct.pack('<5I', *dims)) +
                         _xtc(3, pixels.tobytes()), 10)
            payload = struct.pack('<QI', 100 + i, 12 << 24) + _xtc(0, child)
            row = np.zeros(DESC_NCOLS, np.uint64)
            row[[DESC_EVENT_INDEX, DESC_STREAM_ID, DESC_DEVICE_OFFSET,
                 DESC_READ_SIZE, DESC_TIMESTAMP]] = (7 + 3 * i, sid, len(packed), len(payload), 100 + i)
            rows.append(row)
            packed.extend(payload)
        specs.append(NS(timestamp=100 + i, batch_event_index=7 + 3 * i,
                        first_desc=first, n_desc=len(streams)))
    host = np.frombuffer(packed, np.uint8)
    with producer:
        data = cp.asarray(host)
    producer.synchronize()  # fixture upload owner, not the preparation hot path
    window = parser.parse_window(NS(data_gpu=data, desc_table=np.asarray(rows, np.uint64),
                                    retain_input=lambda: lambda: None), producer, batch_id=1)
    events = GpuEventDgrams.from_windows(NS(iter_events=lambda: iter(specs)),
                                        (window,), batch_id=1)
    return window, events, expected, presence


@pytest.mark.parametrize('rank', [2, 3])
def test_raw_only_pixels_presence_reuse_owners_and_one_gather(monkeypatch, rank):
    import cupy as cp

    def forbidden(*args, **kwargs):
        raise AssertionError('legacy calibration/geometry entered raw preparation')
    for module, names in ((gpu_calib, ('_compute_calib_constants_cpu', 'prep_calib_constants',
                                       'prepare_geometry', 'prepare_geometry_from_arrays',
                                       'fused_calib_gpu')),
                          (gd, ('fused_calib_gpu', 'prepare_geometry', 'prepare_geometry_from_arrays')),
                          (gpu_events, ('prep_calib_constants', '_compute_calib_constants_cpu'))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    budget = _GpuBudget(64 * 1024**2)
    parser, raw = setup_raw(budget, rank)
    producer, execution, consumer = [cp.cuda.Stream(non_blocking=True) for _ in range(3)]
    cp.cuda.get_current_stream().synchronize()
    launches = []
    kernel = gd._batched_gather_kernel
    def counted(dtype):
        launch = kernel(dtype)
        def invoke(*args, **kwargs):
            launches.append(1)
            return launch(*args, **kwargs)
        return invoke
    monkeypatch.setattr(gd, '_batched_gather_kernel', counted)
    for iteration in range(2):
        window, events, expected, present = make_input(
            cp, parser, producer, rank, missing=bool(iteration), malformed=bool(iteration))
        lease = InputSlotLease(None, (window,))
        window.wait_ready(execution)
        window.batch.locate = forbidden
        prepared = raw.prepare_batch(events, stream=execution, slot_id=0)
        done = cp.cuda.Event(disable_timing=True)
        done.record(execution)
        lease.result_ready = done
        assert [e.event.batch_event_index for e in prepared.events] == [7, 10, 13]
        with consumer:
            consumer.wait_event(done)
            copied = prepared.data.copy()
            copied_presence = prepared.present.copy()
            terminal = cp.cuda.Event(disable_timing=True)
            terminal.record(consumer)
        lease.register_consumer_done(terminal)
        assert window.batch._locators == {}
        assert not window.close()  # prepared input users still hold the owner
        lease.wait_until_safe_to_reuse()
        assert window.released
        np.testing.assert_array_equal(copied.get(), expected)
        np.testing.assert_array_equal(copied_presence.get(), present)
        assert len(launches) == iteration + 1
        assert set(raw.memory_bytes()) == {'raw_slots', 'routing', 'total'}
        assert all(need == (0, 0) for need in raw.allocation_requirements(3, 0))
        del prepared, events, window, lease, copied, copied_presence
    parser.trim_free_buffers()
    raw.trim_slot_buffers()
    gc.collect()
    assert budget.committed() == parser.memory_bytes()['config'] + raw.memory_bytes()['routing']


def test_raw_allocation_is_budgeted_before_growth():
    import cupy as cp
    budget = _GpuBudget(64 * 1024**2)
    parser, raw = setup_raw(budget, 3)
    stream = cp.cuda.Stream(non_blocking=True)
    window, events, _, _ = make_input(cp, parser, stream, 3)
    stream.synchronize()
    budget._limit = budget.committed()  # no room for raw/presence/map allocations
    with pytest.raises(GpuMemoryPressureError):
        raw.prepare_batch(events, stream=stream, slot_id=0)
    assert raw._raw_slot_bufs == [None]
    assert raw.memory_bytes()['raw_slots'] == 0
    window.close()
