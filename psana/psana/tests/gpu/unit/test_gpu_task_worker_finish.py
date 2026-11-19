"""How a stateful task's run ends (PR #270 review, fixes 1 and 2). Host-only."""
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu import GpuTask, GpuTaskWorker, RunInstance
from psana.gpu.gpu_plan import StepScope
from psana.gpu.gpu_task_worker import TaskWorkers

RI = RunInstance('xpp', 77, 123, 0)


class _Budget:
    def __init__(self):
        self.held = 0

    def limit(self): return 10 ** 6
    def available(self): return 10 ** 6
    def reserve(self, n): self.held += n
    def release(self, n): self.held -= n


class _W(GpuTaskWorker):
    def __init__(self, log, raise_in_end_run_on=None):
        self.log, self.raise_on = log, raise_in_end_run_on

    def process(self, batch, stream): pass

    def end_step(self, ctx, stream):
        ctx.emit('c', key='k', n=np.array([1]))

    def end_run(self, ctx, stream):
        if ctx.stream_index == self.raise_on:
            raise RuntimeError(f'end_run failed on stream {ctx.stream_index}')
        self.log.append(('end_run', ctx.stream_index))

    def close(self):
        self.log.append('close')


def _workers(monkeypatch, n=2, scopes=3, state=1000, **kw):
    monkeypatch.setitem(sys.modules, 'cupy', NS(zeros=np.zeros, empty=np.empty))
    log, out, budget = [], [], _Budget()
    task = GpuTask(factory=lambda: _W(log, **kw), state_bytes_per_stream=state,
                   aggregate_outputs={'c': {'n': 'sum'}}, aggregate_sink=out.append)
    sc = [StepScope(77, k, 0 if k == 0 else 900 + 50 * k) for k in range(scopes)]
    w = TaskWorkers.build(task, streams=[None] * n, budget=budget, run_instance=RI,
                          input_preparers={}, margin=0, scopes=sc)
    return w, sc, log, out, budget


def test_every_instance_closes_when_one_end_run_raises(monkeypatch):
    w, sc, log, out, budget = _workers(monkeypatch, raise_in_end_run_on=0)
    w.enter_scope(0, sc[1], None); w.enter_scope(1, sc[1], None)
    with pytest.raises(RuntimeError, match='stream 0'):
        w.finish_run('complete')
    assert log.count('close') == 2, 'the instance after the failing one still closed'
    assert ('end_run', 1) in log, 'and still ran its end_run'
    assert budget.held == 0, 'the reservation was returned'
    assert all(w.context(i).state.used == 0 for i in range(2))


def test_a_step_the_run_ended_inside_is_not_complete(monkeypatch):
    w, sc, log, out, budget = _workers(monkeypatch, scopes=2)   # scopes 0 and 1 only
    w.enter_scope(0, sc[1], None)                 # the run is cut inside step 1
    w.finish_run('incomplete')
    assert [(r.begin_ts, r.complete) for r in out] == [(sc[1].begin_ts, False)]


def test_only_the_step_the_run_ended_in_is_truncated(monkeypatch):
    w, sc, log, out, budget = _workers(monkeypatch)
    w.enter_scope(0, sc[1], None)
    w.enter_scope(0, sc[2], None)                 # stream 0 moved on: step 1 fully submitted
    w.enter_scope(1, sc[1], None)                 # stream 1 still in step 1 at the cut
    w.finish_run('incomplete')
    by_ts = {r.begin_ts: r.complete for r in out}
    assert by_ts[sc[1].begin_ts] is True and by_ts[sc[2].begin_ts] is False


def test_manager_close_releases_the_pipeline_when_a_worker_raises():
    from psana.gpu import gpu_events
    order = []
    m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
    m._closed = m._closing = False
    m._iter = None
    m.event_pool = NS(_retiring=None)
    m._run_complete = True
    m._flush_event_pool = lambda: iter(())
    m._close_resources = lambda: order.append('close_resources')

    def boom(status):
        raise RuntimeError('worker end_run failed')
    m._task_workers = NS(failed=False, finish_run=boom)
    with pytest.raises(RuntimeError, match='worker end_run'):
        m.close()
    assert order == ['close_resources']
