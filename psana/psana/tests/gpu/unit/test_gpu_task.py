"""Host declarations, exact uploads, transactional refresh and setup boundaries."""
import pickle
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu import GpuTask
from psana.gpu.gpu_task import RequestedConstants, prepare_task_inputs
from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError
from psana.psexp.ds_base import DataSourceBase, DsParms


def noop(evt, stream):
    raise AssertionError('Stage 2 must not invoke callbacks')


def params(**kwargs):
    return DsParms(5, 0, 0, False, None, '', 0, False, [], 0, [], '', **kwargs)


@pytest.fixture
def cpu_uploads(monkeypatch):
    log = []
    def empty(shape, dtype):
        log.append(('allocate', shape, np.dtype(dtype)))
        return np.empty(shape, dtype=dtype)
    stream = NS(synchronize=lambda: log.append('upload-complete'))
    monkeypatch.setitem(sys.modules, 'cupy', NS(empty=empty, cuda=NS(get_current_stream=lambda: stream)))
    return log


def test_declaration_is_host_only_deduplicated_immutable_and_picklable(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    inputs = ['jf.raw', ('jf', 'raw', 'raw'), 'jf.raw']
    task = GpuTask(noop, inputs, [('jf', 'pixel_gain')] * 2)
    inputs.clear()
    assert task.inputs == ('jf.raw', ('jf', 'raw', 'raw'))
    assert task.calibconst == (('jf', 'pixel_gain'),)
    assert pickle.loads(pickle.dumps(task)) == task
    with pytest.raises(AttributeError):
        task.inputs = ()


@pytest.mark.parametrize('kwargs,error', [
    ({'function': None}, TypeError),
    ({'inputs': 'jf.raw'}, TypeError),
    ({'inputs': ['jf.calib']}, ValueError),
    ({'inputs': ['.raw']}, ValueError),
    ({'inputs': [('jf', 'config', 'raw')]}, ValueError),
    ({'inputs': [('jf', 'raw')]}, TypeError),
    ({'calibconst': ['pixel_gain']}, TypeError),
    ({'calibconst': [('jf', '')]}, ValueError),
])
def test_bad_declarations(kwargs, error):
    with pytest.raises(error):
        GpuTask(**dict({'function': noop}, **kwargs))


def test_routing_and_source_validation_before_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    task = GpuTask(noop, ['jf.raw'])
    with pytest.raises(ValueError, match='requires gpu_det'):
        params(gpu_fn=task)
    with pytest.raises(ValueError, match='task detectors'):
        params(gpu_fn=task, gpu_det='other')
    assert params(gpu_fn=task, hybrid_det='jf').gpu_fn is task
    for kwargs in ({'files': ['a.xtc2']}, {'exp': 'x', 'shmem': 'tag'}, {'exp': 'x', 'drp': True}):
        with pytest.raises(NotImplementedError, match='experiment/run'):
            DataSourceBase.__init__(NS(), gpu_fn=task, **kwargs)
    ds = NS(get_filter_timestamps=lambda x: np.array([], dtype=np.uint64))
    DataSourceBase.__init__(ds, exp='x', gpu_det='jf', gpu_fn=task)
    assert ds.dsparms.gpu_fn is task and ds.batch_size == 1


def test_empty_requests_touch_neither_source_nor_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    store = RequestedConstants((), _GpuBudget(0))
    assert not store.refresh(None)
    with pytest.raises(KeyError, match='not declared'):
        store.get('jf', 'pedestals')


@pytest.mark.parametrize('value', [np.array(3, np.uint32), np.empty((0, 3), np.float32),
                                    np.arange(60).reshape(6, 10)[::2, ::2],
                                    np.arange(60, dtype=np.float64).reshape(3, 10, 2)])
def test_gain_only_preserves_scalar_empty_and_segment_layout(value, cpu_uploads):
    budget = _GpuBudget(value.nbytes)
    store = RequestedConstants([('jf', 'pixel_gain')] * 2, budget)
    source = {'jf': {'pixel_gain': (value, {'not_uploaded': object()})}}
    assert store.refresh(source)
    actual = store.get('jf', 'pixel_gain')
    assert actual.shape == value.shape and actual.dtype == value.dtype
    np.testing.assert_array_equal(actual, value)
    assert len(cpu_uploads) == 2 and budget.committed() == value.nbytes
    assert not store.refresh(source) and len(cpu_uploads) == 2
    store.close()
    assert budget.committed() == value.nbytes  # retained alias owns its charge
    del actual
    assert budget.committed() == 0


@pytest.mark.parametrize('bad', [object(), np.array(['text']), np.array([object()]),
                                np.array([1], dtype='>i4')])
def test_invalid_values_rejected_before_any_upload(bad, cpu_uploads):
    store = RequestedConstants([('jf', 'pixel_gain'), ('jf', 'bad')], _GpuBudget(100))
    with pytest.raises(TypeError, match='native numeric'):
        store.refresh({'jf': {'pixel_gain': np.array([1.]), 'bad': bad}})
    assert not cpu_uploads


def test_missing_key_rejected_even_when_other_keys_exist(cpu_uploads):
    store = RequestedConstants([('jf', 'pixel_gain')], _GpuBudget(100))
    with pytest.raises(KeyError, match='pixel_gain'):
        store.refresh({'jf': {'pedestals': np.array([1.])}})
    assert not cpu_uploads


def test_inplace_change_replaces_only_requested_changed_values(cpu_uploads):
    store = RequestedConstants([('jf', 'pixel_gain'), ('jf', 'pixel_status')], _GpuBudget(100))
    gain, status = np.array([1., np.nan]), np.array([2], np.uint8)
    source = {'jf': {'pixel_gain': gain, 'pixel_status': status}}
    store.refresh(source)
    old, same = store.get('jf', 'pixel_gain'), store.get('jf', 'pixel_status')
    assert not store.refresh(source)  # unchanged NaNs do not trigger uploads
    gain[0] = 5
    cpu_uploads.clear()
    assert store.refresh(source)
    assert old[0] == 1 and store.get('jf', 'pixel_gain')[0] == 5
    assert store.get('jf', 'pixel_status') is same
    assert len(cpu_uploads) == 2


def test_replacement_budget_failure_preserves_previous_snapshot(cpu_uploads):
    budget = _GpuBudget(8)
    store = RequestedConstants([('jf', 'gain')], budget)
    store.refresh({'jf': {'gain': np.array(1., np.float64)}})
    cpu_uploads.clear()
    with pytest.raises(GpuMemoryPressureError):
        store.refresh({'jf': {'gain': np.array(2., np.float64)}})
    assert not cpu_uploads and budget.committed() == 8 and budget._held == 0
    assert store.get('jf', 'gain') == 1


def test_refresh_preserves_signed_zero_and_nan_bits(cpu_uploads):
    store = RequestedConstants([('jf', 'gain')], _GpuBudget(100))
    bits = np.array([0, 0x7ff8000000000001], np.uint64)
    source = {'jf': {'gain': bits.view(np.float64)}}
    store.refresh(source)
    bits[:] = [0x8000000000000000, 0x7ff8000000000002]
    assert store.refresh(source)
    np.testing.assert_array_equal(store.get('jf', 'gain').view(np.uint64), bits)


def test_dense_and_descriptor_selectors_preserve_sparse_segment_identity(monkeypatch):
    from test_gpu_raw_layout import raw_binding
    from psana.gpu.gpu_input import GpuDetectorBinding
    configs, raw = raw_binding()
    binding = GpuDetectorBinding('jf', canonical_segment_ids=raw.canonical_segment_ids,
        field_handles_by_segment={}, field_handles_by_name={('raw', 'raw'): raw.field_handles_by_segment})
    monkeypatch.setitem(sys.modules, 'cupy', None)
    args = dict(configs=configs, bindings={'jf': binding}, n_slots=2, budget=_GpuBudget(100))
    assert prepare_task_inputs(GpuTask(noop, [('jf', 'raw', 'raw')]), **args) == {}
    dense = prepare_task_inputs(GpuTask(noop, ['jf.raw']), **args)['jf.raw']
    assert dense.canonical_segment_ids == (9, 4)
    assert dense.det_shape == (2, 512, 1024)
    with pytest.raises(KeyError):
        prepare_task_inputs(GpuTask(noop, [('jf', 'raw', 'missing')]), **args)


def test_step_refresh_follows_drains_and_host_update(monkeypatch, cpu_uploads):
    from psana.gpu import gpu_events as ge
    manager = ge.GpuEventManager.__new__(ge.GpuEventManager)
    manager._task_constants = RequestedConstants([('jf', 'gain')], _GpuBudget(100))
    manager.dsparms = NS(calibconst={'jf': {'gain': np.array(1.)}})
    manager._task_constants.refresh(manager.dsparms.calibconst)
    log = []
    manager.configs = []
    manager._flush_event_pool = lambda: iter(log.append('executions') or ())
    manager._group_inputs = NS(drain_idle=lambda: log.append('inputs'))
    def host(dgrams):
        log.append('host')
        manager.dsparms.calibconst['jf']['gain'][...] = 2
    manager.run = NS(_handle_transition=host)
    manager._trim_gpu_caches = lambda: log.append('trim')
    manager._compute_subbatch_budget = lambda: log.append('admission') or 100
    monkeypatch.setattr(ge, '_iter_step_events', lambda *a: [(ge.TransitionId.BeginStep, [])])
    assert list(manager._handle_steps({0: (b'x', [])})) == []
    assert log == ['executions', 'inputs', 'host', 'trim', 'admission']
    assert manager._task_constants.get('jf', 'gain') == 2
    log.clear()
    list(manager._handle_steps({0: (b'x', [])}))
    assert log == ['executions', 'inputs', 'host']


def test_task_cannot_be_silently_ignored_by_event_processing():
    from psana.gpu.gpu_events import GpuEventManager
    manager = GpuEventManager.__new__(GpuEventManager)
    manager._gpu_task = GpuTask(noop)
    with pytest.raises(NotImplementedError, match='callback execution'):
        next(manager._process_batch({}, {}, {}))
