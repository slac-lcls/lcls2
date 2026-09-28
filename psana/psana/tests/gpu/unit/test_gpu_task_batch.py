"""Batch selection and admission are based on a common event axis."""
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu import GpuTask
from psana.gpu.gpu_admission import AdmissionEvent
from psana.gpu.gpu_events import GpuEventManager
from psana.gpu.gpu_task_batch import BatchInputContext, metadata_bytes, select_task_events
from test_gpu_producer import Event, envelope


def test_selected_order_tail_empty_sources_and_duplicate_rejection():
    events = (Event(8, 102), Event(3, 100), Event(5, 101, False), Event(9, 103))
    selected = select_task_events(events, [envelope(t) for t in (100, 101, 102)])
    assert selected == events[:2]
    assert select_task_events(events, []) == ()
    with pytest.raises(ValueError, match='duplicate selected'):
        select_task_events((events[0], Event(9, 102)), [envelope(102)])


def test_empty_context_has_no_device_work_and_enforces_declarations(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', NS())  # No CUDA/allocator available.
    context = BatchInputContext((), GpuTask(lambda *a: None, ['jf.raw']),
                               {'jf.raw': None}, {'jf': NS(canonical_segment_ids=(9, 4))},
                               None, None, [], batch_id=8, run=51, step_generation=2)
    assert context.size == 0 and context.timestamps == context.batch_event_indices == ()
    assert context.input('jf.raw') is context.present('jf.raw') is None
    assert context.timestamps_gpu is context.batch_event_indices_gpu is None
    assert context.segment_ids('jf') == (9, 4)
    for operation in (lambda: context.input('other.raw'),
                      lambda: context.field('jf', 'raw', 'raw'),
                      lambda: context.calibconst('jf', 'gain'),
                      lambda: context.segment_ids('other')):
        with pytest.raises(KeyError, match='not declared'):
            operation()
    context.close()
    with pytest.raises(RuntimeError, match='closed'):
        context.input('jf.raw')


def test_context_rejects_compacted_input_before_upload(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', NS())
    events = (Event(3, 100), Event(8, 102))
    with pytest.raises(ValueError, match='not aligned'):
        BatchInputContext(events, GpuTask(lambda *a: None, ['jf.raw']),
                          {'jf.raw': NS(events=events[1:])}, {}, None, None, [])


def test_pinned_pool_rounding_does_not_expand_device_upload(monkeypatch):
    from psana.gpu.gpu_budget import _GpuBudget
    class DeviceArray(np.ndarray):
        def set(self, source, stream=None):
            np.copyto(self, source)
    monkeypatch.setitem(sys.modules, 'cupy', NS(
        empty=lambda shape,dtype:np.empty(shape,dtype).view(DeviceArray),
        cuda=NS(alloc_pinned_memory=lambda n:bytearray(512))))
    budget, owners = _GpuBudget(32), []
    events = (Event(3,100), Event(8,102))
    context = BatchInputContext(events, GpuTask(lambda *a:None), {}, {}, None,
                               None, owners, budget=budget)
    assert budget.committed() == 0 and not owners and context.pinned_nbytes == 0
    np.testing.assert_array_equal(context.timestamps_gpu, [100,102])
    assert budget.committed() == 32
    assert owners[0].host.nbytes == owners[0].device.nbytes == 32
    assert context.pinned_nbytes == owners[0].pinned_nbytes == 512
    np.testing.assert_array_equal(context.timestamps_gpu, [100,102])
    np.testing.assert_array_equal(context.batch_event_indices_gpu, [3,8])
    context.seal()
    np.testing.assert_array_equal(context.timestamps_gpu, [100,102])
    assert len(owners) == 1 and budget.committed() == 32


def test_task_admission_includes_absent_dense_rows_and_metadata(monkeypatch):
    from psana.gpu.gpu_batch import GpuBatchView
    from test_core import _make_batch
    monkeypatch.setitem(sys.modules, 'cupy', NS())
    manager = GpuEventManager.__new__(GpuEventManager)
    manager._gpu_task = GpuTask(lambda *a: None, ['fast.raw', 'slow.raw', ('slow', 'raw', 'x')])
    manager.gpu_detector_bindings = {'slow': NS(canonical_segment_ids=(9, 4, 8))}
    counts = []
    manager.input_preparers = {
        name: NS(binding=NS(has_sources=lambda s, stream=stream: stream in s),
                 estimate_subbatch_bytes=lambda n, cost=cost: cost*n,
                 allocation_requirements=lambda n, slot, name=name: counts.append((name,n,slot)) or [])
        for name, stream, cost in [('fast', 0, 10), ('slow', 1, 1000)]}
    view = GpuBatchView(_make_batch(2, descs_per_event=1, stream_ids=[0], bd_size=5))
    costs = manager._event_memory(view)
    extra = metadata_bytes(manager._gpu_task, manager.gpu_detector_bindings, 1)
    assert extra == (2 + 3*8)*8
    assert [e.detector_bytes for e in costs] == [1010+extra]*2
    # A no-source event has no callback row; an absent detector still does.
    requirements = manager._task_input_requirements(costs+[AdmissionEvent((), 0)], 1)
    assert counts == [('fast', 2, 1), ('slow', 2, 1)]
    assert requirements == [(2*extra, 0)]

    counts.clear()
    manager._gpu_task = None
    assert manager._task_input_requirements(costs, 0) == []
    assert counts == [('fast', 2, 0), ('slow', 0, 0)]


def test_dense_only_context_never_touches_cuda_and_cannot_upload_after_seal(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    events = (Event(3,100), Event(8,102))
    data, presence = np.ones((2,1)), np.ones((2,1), dtype=bool)
    owners = []
    context = BatchInputContext(events, GpuTask(lambda *a:None, ['jf.raw']),
        {'jf.raw':NS(events=events, data=data, present=presence)},
        {'jf':NS(canonical_segment_ids=(4,))}, None, None, owners)
    assert context.input('jf.raw') is data and context.present('jf.raw') is presence
    assert context.timestamps == (100,102) and context.segment_ids('jf') == (4,)
    assert not owners and context.pinned_nbytes == 0
    context.seal()
    with pytest.raises(RuntimeError, match='during the callback'):
        context.timestamps_gpu
    assert not owners and context.pinned_nbytes == 0


def test_failed_metadata_initialization_cannot_expose_partial_views_or_retry(monkeypatch):
    calls = []
    class DeviceArray(np.ndarray):
        def set(self, source, stream=None):
            calls.append(1)
            raise ValueError('upload failed')
    monkeypatch.setitem(sys.modules, 'cupy', NS(
        empty=lambda shape,dtype:np.empty(shape,dtype).view(DeviceArray),
        cuda=NS(alloc_pinned_memory=lambda n:bytearray(n))))
    owners = []
    context = BatchInputContext((Event(3,100),), GpuTask(lambda *a:None), {}, {},
                                None, None, owners)
    with pytest.raises(ValueError, match='upload failed'): context.timestamps_gpu
    for access in (lambda:context.timestamps_gpu, lambda:context.batch_event_indices_gpu):
        with pytest.raises(RuntimeError, match='initialization failed'): access()
    assert calls == [1] and len(owners) == 1
