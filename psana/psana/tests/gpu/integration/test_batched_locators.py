"""Batched locator equivalence, capacity strides, and stream dependencies."""
import struct

import numpy as np
import pytest

from psana.gpu.gpudgram.batch import GpuXtcBatchPool, build_dgram_records
from psana.gpu.gpudgram.config import GpuStreamConfigTable
from psana.gpu.gpudgram import parser as p
from psana.gpu.gpu_kvikio_read import (
    DESC_NCOLS, DESC_EVENT_INDEX, DESC_STREAM_ID, DESC_DEVICE_OFFSET, DESC_READ_SIZE,
)

def _gpu_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not _gpu_available(), reason='no CUDA device available')]


def _configs():
    def entry(det, names_id):
        return dict(det_name=det, det_type='test', det_id=det, segment=0,
                    alg_name='raw', alg_version=(1, 0, 0), names_id_value=names_id,
                    fields=[dict(name='counter', type=2, element_size=4, rank=0,
                                 field_index=0, shape_index=-1),
                            dict(name='pixels', type=1, element_size=2, rank=2,
                                 field_index=1, shape_index=0)])
    return GpuStreamConfigTable({0: [entry('camera', 10), entry('other', 11)],
                                1: [entry('camera', 10)],
                                2: [entry('absent', 10)]})


def _xtc(kind, payload=b'', src=0, damage=0):
    return struct.pack('<IHHI', src, damage, kind, 12 + len(payload)) + payload


def _event(case='ok'):
    shape = _xtc(2, struct.pack('<5I', 2, 3, 0, 0, 0))
    data = _xtc(3, struct.pack('<I6H', 42, 1, 2, 3, 4, 5, 6))
    if case == 'missing_shapes':
        shape = b''
    elif case == 'bad_shape':
        shape = _xtc(2, b'\0' * 4)
    elif case == 'overflow':
        data = _xtc(3, b'\0' * 5)
    elif case == 'missing_data':
        data = b''
    node = _xtc(1, shape + data, src=99 if case == 'unknown' else 10,
                damage=8 if case == 'corrupted' else 0)
    if case == 'duplicate':
        node += node
    elif case == 'bad_xtc':
        node = struct.pack('<IHHI', 10, 0, 1, 99999)
    elif case == 'absent':
        node = b''
    return struct.pack('<QI', 123, 12 << 24) + _xtc(0, node)


def _input(cp, cases, streams):
    events = [_event(case) for case in cases]
    desc = np.zeros((len(events), DESC_NCOLS), dtype=np.uint64)
    desc[:, DESC_EVENT_INDEX] = np.arange(len(events)) * 3 + 7
    desc[:, DESC_STREAM_ID] = streams
    desc[:, DESC_READ_SIZE] = [len(e) for e in events]
    desc[:, DESC_DEVICE_OFFSET] = np.cumsum([0] + [len(e) for e in events[:-1]])[:len(events)]
    data = cp.asarray(np.frombuffer(b''.join(events), dtype=np.uint8))
    cp.cuda.get_current_stream().synchronize()
    return data, desc


def _compare(cp, pool, data, desc, stream):
    batch = pool.parse(0, data, desc, stream)
    # Independent submission through the original single-handle path, sharing
    # only the unchanged field-offset algorithm with batched decoding.
    with stream:
        reference = p.GpuEventBatch(data, pool.device_configs,
                                    cp.asarray(build_dgram_records(desc)),
                                    max_shapes_per_dgram=pool.max_shapes_per_dgram,
                                    stream=stream)
        expected = [reference.locate(h) for h in pool.field_handles]
    consumer = cp.cuda.Stream(non_blocking=True)
    copied = []
    with consumer:
        for handle in pool.field_handles:
            copied.append(batch.locate(handle).wait_on(consumer).copy())
    consumer.synchronize()
    stream.synchronize()
    for got, want in zip(copied, expected):
        got, want = cp.asnumpy(got), cp.asnumpy(want.rows_gpu)
        np.testing.assert_array_equal(got[:, p.LOC_STATUS], want[:, p.LOC_STATUS])
        # The winner of a duplicate CAS is unspecified in both implementations.
        mask = want[:, p.LOC_STATUS] != p.STATUS_DUPLICATE
        np.testing.assert_array_equal(got[mask], want[mask])
    return batch, copied


def test_stream_grouping_tail_growth_empty_and_lazy():
    import cupy as cp
    from psana.gpu.gpu_budget import _GpuBudget

    configs = _configs()
    handles = configs.field_handles()
    lazy = handles[1]
    eager = tuple(reversed([h for h in handles if h != lazy]))
    budget = _GpuBudget(limit_bytes=1024**2)
    pool = GpuXtcBatchPool(configs, field_handles=eager, n_slots=1, budget=budget)
    stream = cp.cuda.Stream(non_blocking=True)
    original_ptr = None
    for streams in ([1, 0, 1, 0, 0], [0, 1], [], [1] * 7):
        data, desc = _input(cp, ['ok'] * len(streams), streams)
        batch, rows = _compare(cp, pool, data, desc, stream)
        assert len({id(batch.locate(h).ready) for h in eager}) == 1
        assert all(batch.locate(h).rows_gpu.flags.c_contiguous for h in eager)
        ptr = pool._slots[0].locator_backing.data.ptr
        if len(streams) == 5:
            original_ptr = ptr
        elif len(streams) <= 2:
            assert ptr == original_ptr
        assert budget.committed() == pool.memory_bytes()['total']
        if len(streams) == 2:
            # Omitted handles still support lazy lookup on another stream.
            consumer = cp.cuda.Stream(non_blocking=True)
            result = batch.locate(lazy, stream=consumer)
            consumer.synchronize()
            assert result.ready is not batch.locate(eager[0]).ready
            actual = cp.asnumpy(result.rows_gpu)
            assert actual[:, p.LOC_STATUS].tolist() == [p.STATUS_FOUND, p.STATUS_NOT_PRESENT]
            assert actual[0, p.LOC_DIM0:p.LOC_DIM0 + 2].tolist() == [2, 3]
            assert actual[0, p.LOC_NBYTES] == 12
            offset = int(actual[0, p.LOC_OFFSET])
            np.testing.assert_array_equal(cp.asnumpy(data[offset:offset+12]).view(np.uint16),
                                          np.arange(1, 7, dtype=np.uint16))


@pytest.mark.parametrize('case,status', [
    ('ok', p.STATUS_FOUND), ('absent', p.STATUS_NOT_PRESENT),
    ('missing_shapes', p.STATUS_MISSING_SHAPES), ('missing_data', p.STATUS_MISSING_DATA),
    ('bad_shape', p.STATUS_BAD_SHAPE), ('overflow', p.STATUS_DATA_OVERFLOW),
    ('duplicate', p.STATUS_DUPLICATE), ('corrupted', p.STATUS_CORRUPTED),
    ('bad_xtc', p.STATUS_BAD_XTC), ('unknown', p.STATUS_UNKNOWN_NAMES),
    ('bad_stream', p.STATUS_BAD_STREAM), ('bad_dgram', p.STATUS_BAD_DGRAM),
    ('capacity', p.STATUS_CAPACITY),
])
def test_batched_matches_lazy_errors(case, status):
    import cupy as cp
    configs = _configs()
    pool = GpuXtcBatchPool(configs, field_handles=configs.field_handles(), n_slots=1,
                           max_shapes_per_dgram=1 if case == 'capacity' else 4)
    streams = [99 if case == 'bad_stream' else 0, 1, 0]
    data, desc = _input(cp, ['duplicate' if case == 'capacity' else case, 'ok', 'absent'], streams)
    if case == 'bad_dgram':
        desc[0, DESC_READ_SIZE] = 10
    batch, _ = _compare(cp, pool, data, desc, cp.cuda.Stream(non_blocking=True))
    handle = configs.resolve('camera', 0, 'raw', 'pixels', stream_id=0)
    assert int(cp.asnumpy(batch.locate(handle).rows_gpu)[0, p.LOC_STATUS]) == status


def test_empty_and_no_handles_avoid_location_launches(monkeypatch):
    import cupy as cp
    configs = _configs()
    def unexpected():
        raise AssertionError('unexpected locator launch')
    monkeypatch.setattr(p, '_init_locators_kernel', unexpected)
    monkeypatch.setattr(p, '_locate_fields_kernel', unexpected)
    stream = cp.cuda.Stream(non_blocking=True)
    for handles, cases, streams in [((), ['ok'], [0]), (configs.field_handles(), [], [])]:
        pool = GpuXtcBatchPool(configs, field_handles=handles, n_slots=1)
        data, desc = _input(cp, cases, streams)
        batch = pool.parse(0, data, desc, stream)
        stream.synchronize()
        assert len(batch._locators) == len(handles)
