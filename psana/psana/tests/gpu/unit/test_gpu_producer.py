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


def test_selected_identity_missing_dense_rows_and_context_expiry(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', NS(cuda=NS(Device=lambda: NS(id=0))))
    events = (Event(7, 100), Event(10, 101), Event(13, 102), Event(16, 103, False), Event(19, 104))
    batch = NS(events=(events[0], events[2], events[4]),
               data=np.array([[7], [13], [19]]), present=np.ones((3, 1), np.uint8))
    calls, contexts, owners, pubs = [], [], [], {}
    sentinel = object()
    def callback(evt, stream):
        assert stream.current
        contexts.append(evt)
        calls.append((evt.batch_event_index, evt.timestamp, evt.run,
                      evt.batch_id, evt.step_generation))
        assert evt.segment_ids('jf') == (9, 4)
        if evt.timestamp == 101:
            assert evt.input('jf.raw') is evt.present('jf.raw') is None
        else:
            assert evt.input('jf.raw')[0] == evt.batch_event_index
            assert evt.present('jf.raw')[0] == 1
        for access in (lambda: evt.input('other.raw'), lambda: evt.field('jf','raw','raw'),
                       lambda: evt.calibconst('jf','gain'), lambda: evt.segment_ids('other')):
            with pytest.raises(KeyError, match='not declared'):
                access()
        evt.keepalive(sentinel)
        return np.array([1])  # Returning a value does not publish it.
    stream = Stream()
    dispatch_task(GpuTask(callback, ['jf.raw']), events,
                  [envelope(t) for t in (100,101,102,103)], {'jf.raw': batch},
                  {'jf': NS(canonical_segment_ids=(9,4), fields={})}, None,
                  stream, owners, pubs, object(), batch_id=3, run=51, step_generation=2)
    assert calls == [(7,100,51,3,2),(10,101,51,3,2),(13,102,51,3,2)]
    assert owners == [sentinel]*3 and pubs == {} and not stream.current
    with pytest.raises(RuntimeError, match='during its callback'):
        contexts[0].keepalive(object())
    assert contexts[0]._dense is None


def test_no_selected_events_invokes_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', NS(cuda=NS(Device=lambda: NS(id=0))))
    def fail(*args):
        raise AssertionError('unselected callback')
    dispatch_task(GpuTask(fail), (Event(7, 100),), [], {}, {}, None,
                  Stream(), [], {}, object(), batch_id=1, run=51, step_generation=0)


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
