"""Producer identity, declaration boundaries and host-only dispatch selection."""
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu import GpuTask
from psana.gpu.gpu_producer import dispatch_task
from psana.gpu.gpu_events import GpuEventManager


class Event(dict):
    def __init__(self, index, timestamp, present=True):
        super().__init__({0: object()} if present else {})
        self.batch_event_index, self.timestamp = index, timestamp


class Stream:
    current = False
    def __enter__(self):
        self.current = True
    def __exit__(self, *args):
        self.current = False


def envelope(timestamp):
    return NS(dgrams=[NS(timestamp=lambda: timestamp)])


def test_one_batch_invocation_identity_stream_and_context_expiry(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', NS(cuda=NS(Device=lambda: NS(id=0))))
    data = np.array([[7], [0], [13]])
    inputs = NS(size=3, timestamps=(100,101,102), batch_event_indices=(7,10,13),
                batch_id=3, run=51, step_generation=2,
                input=lambda name:data, present=lambda name:np.array([[1],[0],[1]]),
                segment_ids=lambda name:(9,4))
    calls, contexts, owners, pubs, batches = [], [], [], {}, []
    sentinel = object()
    def callback(batch, stream):
        assert stream.current
        contexts.append(batch)
        calls.append((batch.batch_event_indices, batch.timestamps, batch.run,
                      batch.batch_id, batch.step_generation))
        assert batch.segment_ids('jf') == (9,4)
        np.testing.assert_array_equal(batch.input('jf.raw'), data)
        np.testing.assert_array_equal(batch.present('jf.raw'), [[1],[0],[1]])
        batch.keepalive(sentinel)
        return np.array([1])  # Return values do not publish.
    stream = Stream()
    dispatch_task(GpuTask(callback, ['jf.raw']), inputs,
                  {'jf': NS(canonical_segment_ids=(9,4), fields={})},
                  stream, owners, pubs, batches, object())
    assert calls == [((7,10,13),(100,101,102),51,3,2)]
    assert owners == [sentinel] and pubs == {} and batches == [] and not stream.current
    for access in (lambda:contexts[0].keepalive(object()), lambda:contexts[0].timestamps,
                   lambda:contexts[0].input('jf.raw')):
        with pytest.raises(RuntimeError, match='during its callback'): access()
    assert contexts[0]._inputs is None


def test_no_selected_events_invokes_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    def fail(*args): raise AssertionError('unselected callback')
    dispatch_task(GpuTask(fail), NS(size=0), {}, Stream(), [], {}, [], object())


def publication_context(monkeypatch, size=3):
    from psana.gpu.gpu_producer import ProducerContext
    class DeviceArray(np.ndarray):
        @property
        def device(self): return NS(id=0)
    monkeypatch.setitem(sys.modules,'cupy',NS(ndarray=DeviceArray))
    context = ProducerContext(NS(size=size,timestamps=tuple(range(100,100+size))),
                              [],{},[],object(),{'jf','jf.raw'},0)
    return context, lambda shape,dtype=np.float32:np.zeros(shape,dtype).view(DeviceArray)


def test_sparse_publication_groups_are_atomic_and_share_backing(monkeypatch):
    ctx, array = publication_context(monkeypatch)
    first, second = array((2,4)), array((1,),np.uint32)
    ctx.publish('out',first,event_indices=[2,0])
    ctx.publish('out',second,event_indices=np.array([1]))
    assert len(ctx._batches)==2 and len(ctx._owners)==2
    for ts,row in [(102,0),(100,1)]:
        pub=ctx._publications[ts]['out']
        assert pub.batch is ctx._batches[0] and pub.row==row
        assert pub.shape==(4,) and pub.nbytes==16
        assert np.shares_memory(pub.array,first)
    assert ctx._publications[101]['out'].shape==()
    with pytest.raises(ValueError,match='duplicate publication'):
        ctx.publish('out',array((2,)),event_indices=[0,1])
    assert len(ctx._batches)==2 and len(ctx._owners)==2
    ctx.publish('empty',array((3,0,2)))
    assert all(p['empty'].nbytes==0 and p['empty'].shape==(0,2) for p in ctx._publications.values())
    ctx.publish('none_selected',array((0,)),event_indices=[])
    assert not any('none_selected' in p for p in ctx._publications.values())


@pytest.mark.parametrize('indices,error', [([True],TypeError),([0.5],TypeError),
    ([-1],ValueError),([3],ValueError),([0,0],ValueError)])
def test_bad_publication_indices_do_not_register(monkeypatch,indices,error):
    ctx,array=publication_context(monkeypatch)
    with pytest.raises(error):ctx.publish('out',array((len(indices),)),event_indices=indices)
    assert ctx._owners==[] and ctx._batches==[] and ctx._publications=={}


def test_publication_requires_host_indices_and_leading_axis(monkeypatch):
    ctx,array=publication_context(monkeypatch)
    with pytest.raises(TypeError,match='host integer'):ctx.publish('out',array((1,)),array((1,),np.int64))
    with pytest.raises(ValueError,match='leading event-row'):ctx.publish('out',array(()))
    with pytest.raises(ValueError,match='leading axis'):ctx.publish('out',array((2,)))
    with pytest.raises(ValueError,match='reserved'):ctx.publish('jf.raw',array((3,)))
    with pytest.raises(ValueError,match='contiguous'):ctx.publish('out',array((3,4))[:,::2])


def test_manager_supplies_task_and_transition_generation():
    manager = GpuEventManager.__new__(GpuEventManager)
    manager._gpu_task = GpuTask(lambda *args: None)
    manager._task_constants = object()
    manager.gpu_detector_bindings = {'jf': object()}
    manager.run = NS(runnum=51)
    manager._step_generation = 3
    manager._input_batch_id = 5
    manager.input_preparers = {}
    manager.gpu_xtc_parser = object()
    calls = []
    manager.event_pool = NS(submit=lambda *a, **kw: calls.append((a,kw)))
    manager._submit_per_dgram_gpu('subbatch', 'read', ['selected'])
    args, kw = calls[0]
    assert args[:3] == ('subbatch','read',['selected'])
    assert kw['task'] is manager._gpu_task and kw['task_constants'] is manager._task_constants
    assert (kw['batch_id'],kw['run'],kw['step_generation']) == (5,51,3)
