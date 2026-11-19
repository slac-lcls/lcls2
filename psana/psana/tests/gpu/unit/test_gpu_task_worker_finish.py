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

    # A flush that raises leaves the pipeline for a retry close (the base
    # contract) but has already marked the task failed: the retry closes the
    # instances without end_run, since nothing proved every execution retired.
    def bad_flush():
        raise RuntimeError('join failed')
        yield
    m._closed = m._closing = False
    m._flush_event_pool = bad_flush
    m._task_workers = NS(failed=False, finish_run=lambda status: order.append(status),
                         mark_failed=lambda: order.append('mark_failed'))
    with pytest.raises(RuntimeError, match='join failed'):
        m.close()
    assert order[1:] == ['mark_failed']


def test_a_failure_on_the_way_to_the_pool_marks_the_task_failed():
    """A read or planning error, not just a submission error, fails the task,
    so close() does not export; a consumer that stops iterating does not."""
    from psana.gpu import gpu_events
    m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
    m._done = False
    marks = []
    m._task_workers = NS(mark_failed=lambda: marks.append('failed'))
    m.close = lambda: marks.append('close')

    def bad_read():
        raise RuntimeError('smd read failed')
    m._next_batch = bad_read
    with pytest.raises(RuntimeError, match='smd read'):
        list(m._events())
    assert marks == ['failed', 'close']

    marks.clear()
    m._next_batch = lambda: (None, None, None)
    m._process_batch = lambda *a: iter(['one event'])
    gen = m._events()
    assert next(gen) == 'one event'
    gen.close()                                   # the consumer broke out of the loop
    assert marks == ['close'], 'an early stop is incomplete, not failed'


def test_a_rowless_submission_still_moves_the_cut_point(monkeypatch):
    """Step 2's subbatches selected no task rows, so they placed no hooks,
    but they were submitted: step 1 was fully submitted and stays complete
    when the run is then cut inside step 2."""
    w, sc, log, out, budget = _workers(monkeypatch)
    w.enter_scope(0, sc[1], None); w.enter_scope(1, sc[1], None)
    w.note_submitted(sc[2])                       # what EventPool.submit does for size == 0
    w.finish_run('incomplete')
    by_ts = {r.begin_ts: (r.complete, r.truncated) for r in out}
    assert by_ts[sc[1].begin_ts] == (True, False)
    assert by_ts[sc[2].begin_ts] == (False, True)
