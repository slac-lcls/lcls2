"""Public field alignment, cache identity, accounting and fault injection."""
import ctypes
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from field_view_fixture import DTYPES, field_case, retire
from psana.gpu.gpu_budget import GpuMemoryPressureError, _GpuBudget
from psana.gpu.gpu_stream import EventPool, _EventSlot
from psana.gpu.gpudgram.batch import LOC_NBYTES


class Array(np.ndarray):
    @property
    def data(self):
        return NS(ptr=self.ctypes.data)

    def get(self):
        return np.asarray(self).copy()


class Event:
    def __init__(self, **kwargs):
        self.waits = 0
        self.done = True

    def synchronize(self):
        self.waits += 1


class Stream:
    def __init__(self):
        self.ptr = id(self)
        self.waited, self.recorded = [], []
        self.fail_record = self.fail_drain = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def wait_event(self, event):
        self.waited.append(event)

    def record(self, event):
        if self.fail_record:
            raise RuntimeError('record failed')
        self.recorded.append(event)

    def synchronize(self):
        if self.fail_drain:
            raise RuntimeError('drain failed')

    @property
    def done(self):
        return not self.fail_drain


@pytest.fixture
def backend(monkeypatch):
    stream = Stream()
    copies = []
    def memcpy(dst, src, size, kind, stream_ptr):
        copies.append((dst, src, size, stream_ptr))
        ctypes.memmove(dst, src, size)
    cp = NS(asarray=lambda a: np.asarray(a).copy().view(Array),
            empty=lambda shape, dtype: np.empty(shape, dtype).view(Array),
            cuda=NS(Event=Event, Stream=NS(null=stream),
                    get_current_stream=lambda: stream,
                    runtime=NS(memcpyAsync=memcpy, memcpyDeviceToDevice=3)))
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    return NS(cp=cp, stream=stream, copies=copies)


@pytest.mark.parametrize('type_id', range(10))
@pytest.mark.parametrize('shape', [(), (3, 5), (0,)])
@pytest.mark.parametrize('base_shift,offset', [(0, i) for i in range(8)] + [(1, 8), (1, 7)])
def test_actual_address_and_exact_bits(backend, type_id, shape, base_shift, offset):
    case = field_case(backend.cp, type_id=type_id, shapes=(shape,),
                      base_shift=base_shift, offset=offset)
    source_ptr = case.sources[0].data_gpu.data.ptr + offset
    aligned = source_ptr % np.dtype(DTYPES[type_id]).itemsize == 0
    with case.result.on_gpu_view(backend.stream) as values:
        value = values.only()
        assert value.shape == shape and value.dtype == np.dtype(DTYPES[type_id])
        assert value.tobytes() == case.expected[9].tobytes()
        if value.size:
            assert value.data.ptr % value.dtype.itemsize == 0
            assert (value.data.ptr == source_ptr) == aligned
        ptr = value.data.ptr
    copies = len(backend.copies)
    charge = case.budget.committed()
    with case.result.on_gpu_view(backend.stream) as cached:
        assert cached.only().data.ptr == ptr
    assert len(backend.copies) == copies == int(not aligned and value.nbytes != 0)
    assert case.budget.committed() == charge == (0 if aligned else value.nbytes)
    if aligned or value.nbytes == 0:
        assert case.lease._field_copies is case.lease._retained_copies is None
        assert case.result._views_ready == ()
    del values, value, cached
    retire(case)
    assert case.budget.committed() == 0 and case.owner.released


def test_segment_selection_reuses_copy_and_waits_on_second_stream(backend):
    case = field_case(backend.cp, shapes=((2, 3), (5,), ()), absent=(17,))
    with case.detector.field('raw', 'value', segment=4).on_gpu_view(backend.stream) as selected:
        ptr = selected.only().data.ptr
    second = Stream()
    with case.result.on_gpu_view(second) as values:
        assert values.segment_ids == (9, 4)
        assert values[4].data.ptr == ptr
        assert all(values[s].tobytes() == case.expected[s].tobytes() for s in values)
    assert len(backend.copies) == 2
    assert len(case.lease._field_copies) == 2
    assert all(event in second.waited for _, event in case.lease._field_copies.values())
    assert case.budget.committed() == 22
    del selected, values
    retire(case)
    assert case.budget.committed() == 0


@pytest.mark.parametrize('failure', ['budget', 'event', 'allocation', 'copy', 'record', 'drain'])
def test_failed_materialization_never_publishes_and_can_retire(backend, monkeypatch, failure):
    case = field_case(backend.cp, limit=1 if failure == 'budget' else 100)
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')
    if failure == 'allocation':
        monkeypatch.setattr(backend.cp, 'empty', fail)
    if failure == 'event':
        monkeypatch.setattr(backend.cp.cuda, 'Event', fail)
    if failure == 'copy':
        copy = backend.cp.cuda.runtime.memcpyAsync
        def submitted_then_failed(*args):
            copy(*args)
            fail()
        monkeypatch.setattr(backend.cp.cuda.runtime, 'memcpyAsync', submitted_then_failed)
    backend.stream.fail_record = failure in ('record', 'drain')
    backend.stream.fail_drain = failure == 'drain'
    with pytest.raises(GpuMemoryPressureError if failure == 'budget' else RuntimeError):
        with case.result.on_gpu_view(backend.stream):
            pytest.fail('failed materialization was published')
    assert case.result._views is None and case.lease._field_copies is None
    assert case.owner.references == 1
    assert case.budget.committed() == (0 if failure in ('budget', 'event', 'allocation') else 30)
    if failure == 'drain':
        with pytest.raises(RuntimeError, match='drain failed'):
            retire(case)
        assert not case.owner.released and case.budget.committed() == 30
    backend.stream.fail_record = backend.stream.fail_drain = False
    retire(case)
    assert case.budget.committed() == 0 and case.owner.released


def test_invalid_later_segment_does_not_publish_partial_result_or_recopy(backend):
    case = field_case(backend.cp, shapes=((3,), (5,)))
    case.rows[4][0, LOC_NBYTES] += 1
    with pytest.raises(RuntimeError, match='byte count'):
        with case.result.on_gpu_view(backend.stream):
            pass
    assert case.result._views is None and len(backend.copies) == 1
    case.rows[4][0, LOC_NBYTES] -= 1
    with case.result.on_gpu_view(backend.stream) as values:
        assert values.segment_ids == (9, 4)
    assert len(backend.copies) == 2
    del values
    retire(case)
    with pytest.raises(RuntimeError, match='retiring'):
        case.result.on_gpu_view(backend.stream)


def test_new_event_does_not_reuse_retired_copy(backend):
    first = field_case(backend.cp)
    with first.result.on_gpu_view(backend.stream) as values:
        assert values.only().tobytes() == first.expected[9].tobytes()
    del values
    retire(first)
    second = field_case(backend.cp, offset=3)
    with second.result.on_gpu_view(backend.stream) as values:
        assert values.only().tobytes() == second.expected[9].tobytes()
    assert len(backend.copies) == 2
    del values
    retire(second)


def test_empty_unaligned_field_needs_no_copy_resources(backend, monkeypatch):
    case = field_case(backend.cp, type_id=3, shapes=((0, 3),), offset=1, limit=0)
    assert (case.sources[0].data_gpu.data.ptr + 1) % 8 != 0
    def fail(*args, **kwargs):
        pytest.fail('empty field requested copy resources')
    monkeypatch.setattr(backend.cp, 'empty', fail)
    # Isolate materialization: a public view context still records its normal
    # consumer completion event, even when no copy-ready event is needed.
    with monkeypatch.context() as m:
        m.setattr(backend.cp.cuda, 'Event', fail)
        values = case.result._slot_views(backend.stream)
    assert values.only().shape == (0, 3) and values.only().dtype == np.dtype('u8')
    assert not backend.copies and not case.budget.allocation_snapshot()
    assert case.budget.committed() == 0
    assert case.lease._field_copies is case.lease._retained_copies is None
    assert case.result._views_ready == ()
    del values
    retire(case)


def test_many_events_exhaust_shared_budget_until_slot_retirement(backend):
    # Three separate event roots in one delivery slot, each needing 30 bytes.
    # NumPy has no pool rounding; CUDA capacity rounding has device coverage.
    budget = _GpuBudget(60)
    cases = [field_case(backend.cp, budget=budget) for _ in range(3)]
    slot = _EventSlot(0, {}, [], backend.stream, [c.lease for c in cases], {})
    pool = EventPool.__new__(EventPool)
    pool._n, pool._write_idx, pool._retiring = 1, 0, None
    pool._slots = [slot]
    assert pool.begin_retire_next() is slot
    for index, case in enumerate(cases[:2]):
        with case.result.on_gpu_view(backend.stream) as values:
            assert values.only().tobytes() == case.expected[9].tobytes()
        del values
        assert budget.committed() == (index + 1) * 30
        assert not case.lease._closed
    before = budget.allocation_snapshot()
    with pytest.raises(GpuMemoryPressureError, match='GPU memory budget exceeded') as error:
        with cases[2].result.on_gpu_view(backend.stream):
            pytest.fail('over-budget field was published')
    assert 'Reduce batch_size or n_gpu_streams' in str(error.value)
    assert 'gpu_memory_budget_gb' in str(error.value)
    assert budget.committed() == 60 and budget.allocation_snapshot() == before
    assert len(backend.copies) == 2
    assert cases[2].result._views is None
    assert cases[2].lease._field_copies is None
    for case in cases:
        case.owner.close()
    pool.finish_retire_next()
    assert pool._slots == [None] and all(c.lease._closed for c in cases)
    assert budget.committed() == 0 and budget.allocation_snapshot() == ()
    for case in cases:
        case.owner.drain()
        assert case.owner.released
    # Later events can allocate after retirement; the failed event was not
    # automatically retried or published.
    later = field_case(backend.cp, budget=budget)
    with later.result.on_gpu_view(backend.stream) as values:
        assert values.only().tobytes() == later.expected[9].tobytes()
    del values
    retire(later)
    assert budget.committed() == 0
