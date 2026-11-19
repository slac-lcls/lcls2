"""Per-stream worker lifecycle for GpuTask(factory=...) (PR #270 phase 1, A2).

Host-only: CuPy is a NumPy stand-in, streams are recording context managers,
the budget is a counter. What is pinned: one instance per stream built inside
a reservation after setup, begin_run once each on its own stream, state drawn
from the reservation, end_run then close exactly once on finish, and the
manager's close() finalizing workers after the pool flush and before its own
resources go.
"""
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu import GpuTask, GpuTaskWorker, RunInstance
from psana.gpu.gpu_budget import GpuMemoryPressureError
from psana.gpu.gpu_task_worker import InputSpec, TaskWorkers, mint_run_instance


class _Stream:
    def __init__(self, log, i):
        self.log, self.i = log, i

    def __enter__(self):
        self.log.append(('enter', self.i))

    def __exit__(self, *exc):
        self.log.append(('exit', self.i))


class _Budget:
    def __init__(self, limit, available):
        self._limit, self._available, self.log = limit, available, []

    def limit(self):
        return self._limit

    def available(self):
        return self._available

    def reserve(self, n):
        self._available -= n
        self.log.append(('reserve', n))

    def release(self, n):
        self._available += n
        self.log.append(('release', n))


class _W(GpuTaskWorker):
    def __init__(self, log, fail_begin_run=False):
        self.log, self.fail = log, fail_begin_run

    def _rec(self, name, ctx):
        self.log.append((name, ctx.stream_index, ctx.run_status))

    def begin_run(self, ctx, stream):
        self._rec('begin_run', ctx)
        if self.fail and ctx.stream_index == 1:
            raise RuntimeError('begin_run failed on stream 1')

    def begin_step(self, ctx, stream):
        self.log.append(('begin_step', ctx.stream_index, ctx.step_scope.index))

    def end_step(self, ctx, stream):
        self.log.append(('end_step', ctx.stream_index, ctx.step_scope.index))

    def process(self, batch, stream):
        pass

    def end_run(self, ctx, stream):
        self._rec('end_run', ctx)

    def close(self):
        self.log.append(('close',))


@pytest.fixture
def numpy_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', NS(zeros=np.zeros, empty=np.empty))


def _build(log, *, n=2, per_stream=1000, limit=10_000, available=5_000, **kw):
    task = GpuTask(factory=lambda: _W(log, **kw), inputs=['jf.raw'],
                   state_bytes_per_stream=per_stream)
    budget = _Budget(limit, available)
    streams = [_Stream(log, i) for i in range(n)]
    preps = {'jf.raw': NS(det_shape=(2, 512, 1024), _dtype=np.dtype(np.uint16))}
    return task, budget, streams, preps


def test_lifecycle_build_state_finish_and_run_instance(numpy_cupy):
    log = []
    task, budget, streams, preps = _build(log)
    ri = mint_run_instance(NS(expt='xpp', runnum=77, timestamp=123))
    assert mint_run_instance(NS(expt='xpp', runnum=77, timestamp=123)).ordinal == ri.ordinal + 1
    assert mint_run_instance(NS(expt='xpp', runnum=78, timestamp=1)).ordinal == 0

    w = TaskWorkers.build(task, streams=streams, budget=budget, run_instance=ri,
                          input_preparers=preps, margin=1000)

    assert budget.log == [('reserve', 2000)], 'reservation charged once, at setup'
    assert log == [('enter', 0), ('begin_run', 0, None), ('exit', 0),
                   ('enter', 1), ('begin_run', 1, None), ('exit', 1)], \
        'begin_run once per instance, each inside its own stream'
    ctx = w.context(1)
    assert ctx.run_instance is ri and ctx.stream_index == 1 and ctx.step_scope is None
    assert ctx.input_spec('jf.raw') == InputSpec((2, 512, 1024), np.dtype(np.uint16))
    assert ctx.state.zeros((100,), np.float64).shape == (100,) and ctx.state.used == 800
    with pytest.raises(GpuMemoryPressureError, match='reservation exceeded'):
        ctx.state.zeros((100,), np.float64)

    log.clear()
    w.finish_run('complete')
    w.finish_run('complete')          # idempotent
    assert log == [('enter', 0), ('end_run', 0, 'complete'), ('close',), ('exit', 0),
                   ('enter', 1), ('end_run', 1, 'complete'), ('close',), ('exit', 1)]
    assert budget.log == [('reserve', 2000), ('release', 2000)] and ctx.state.used == 0

    # A failed run closes without end_run: a partial state is never exported.
    log.clear()
    task, budget, streams, preps = _build(log, n=1)
    w = TaskWorkers.build(task, streams=streams, budget=budget, run_instance=ri,
                          input_preparers=preps, margin=0)
    log.clear()
    w.finish_run('failed')
    assert log == [('enter', 0), ('close',), ('exit', 0)]


def test_setup_failures_leave_nothing_behind(numpy_cupy):
    # Reservation does not fit: diagnosed before any factory call or reserve.
    log = []
    task, budget, streams, preps = _build(log, per_stream=2000, available=5000)
    with pytest.raises(RuntimeError, match=r'4000 bytes does not fit: 5000 .* 1500 must stay free'):
        TaskWorkers.build(task, streams=streams, budget=budget, run_instance=None,
                          input_preparers=preps, margin=1500)
    assert log == [] and budget.log == []

    # begin_run raises on the second instance: both built instances are
    # closed, in order, and the reservation goes back.
    task, budget, streams, preps = _build(log, fail_begin_run=True)
    with pytest.raises(RuntimeError, match='stream 1'):
        TaskWorkers.build(task, streams=streams, budget=budget, run_instance=None,
                          input_preparers=preps, margin=0)
    assert log.count(('close',)) == 2
    assert budget.log == [('reserve', 2000), ('release', 2000)]


def test_manager_close_finalizes_workers_after_flush_before_resources():
    from psana.gpu import gpu_events
    order = []
    m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
    m._closed = m._closing = False
    m._iter = None
    m.event_pool = NS(_retiring=None)

    def flush():
        order.append('flush'); yield from ()
    m._flush_event_pool = flush
    m._close_resources = lambda: order.append('close_resources')
    m._task_workers = NS(failed=False, finish_run=lambda s: order.append(f'finish_run:{s}'))

    m._run_complete = True
    m.close()
    assert order == ['flush', 'finish_run:complete', 'close_resources']

    # Status derivation: early close is incomplete; a process failure wins.
    order.clear(); m._run_complete = False
    m._finish_task_workers(); assert order == ['finish_run:incomplete']
    order.clear(); m._task_workers.failed = True
    m._finish_task_workers(); assert order == ['finish_run:failed']


def _scopes(n):
    from psana.gpu.gpu_plan import StepScope
    return [StepScope(77, k, 0 if k == 0 else 900 + 50 * k) for k in range(n)]


def test_step_hooks_follow_scope_changes_per_stream(numpy_cupy):
    """The review's "Hook placement": per stream, at scope change, with
    begin/end pairs for scopes the stream never saw, and no drain."""
    log = []
    task, budget, streams, preps = _build(log)
    scopes = _scopes(4)
    w = TaskWorkers.build(task, streams=streams, budget=budget, run_instance=None,
                          input_preparers=preps, margin=0, scopes=scopes)
    log.clear()

    w.enter_scope(0, scopes[1], streams[0])     # first submission: catch up 0, open 1
    w.enter_scope(0, scopes[1], streams[0])     # same scope: nothing
    w.enter_scope(0, scopes[2], streams[0])     # change: close 1, open 2
    w.enter_scope(1, scopes[3], streams[1])     # stream 1 never saw 0, 1, 2
    with pytest.raises(RuntimeError, match='must not decrease'):
        w.enter_scope(0, scopes[1], streams[0])
    w.finish_run('complete')

    def hooks(stream):
        return [(r[0], r[2]) for r in log if len(r) == 3 and r[1] == stream and r[0] != 'begin_run']
    assert hooks(0) == [('begin_step', 0), ('end_step', 0), ('begin_step', 1),
                        ('end_step', 1), ('begin_step', 2),
                        # end of stream: close 2, complete 3, end the run
                        ('end_step', 2), ('begin_step', 3), ('end_step', 3), ('end_run', 'complete')]
    assert hooks(1) == [('begin_step', 0), ('end_step', 0), ('begin_step', 1), ('end_step', 1),
                        ('begin_step', 2), ('end_step', 2), ('begin_step', 3),
                        ('end_step', 3), ('end_run', 'complete')]
    # Every hook ran inside its stream: no hook record sits outside an
    # enter/exit pair, and each enter_scope that changed scope entered once.
    depth, outside = 0, 0
    for rec in log:
        if rec[0] == 'enter': depth += 1
        elif rec[0] == 'exit': depth -= 1
        elif rec[0] != 'close' and depth == 0: outside += 1
    assert outside == 0 and depth == 0
    assert sum(1 for r in log if r == ('enter', 0)) == 3 and sum(1 for r in log if r == ('enter', 1)) == 2


def test_manager_records_each_new_scope_once():
    from psana.gpu import gpu_events
    from psana.psexp import TransitionId as T
    m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
    m._step_generation, m._step_begin_ts = 0, 0
    m.run = NS(runnum=77)
    dg = lambda ts: [NS(timestamp=lambda: ts)]
    m._dispatch_transition(T.BeginStep, dg(900))
    m._dispatch_transition(T.BeginStep, dg(900))      # replayed: same timestamp
    m._dispatch_transition(T.BeginStep, dg(950))
    assert [(sc.index, sc.begin_ts) for sc in m._known_scopes()] == [(0, 0), (1, 900), (2, 950)]


def test_emit_in_end_step_reaches_the_sink_merged_across_streams(numpy_cupy):
    """ctx.emit stages on the hook's stream, the reducer merges the two streams,
    completion comes from end_step returning, and the sink is called inline."""
    class _Emitter(GpuTaskWorker):
        def begin_run(self, ctx, stream):
            self.n = ctx.state.zeros((4,), np.uint64)
            with pytest.raises(RuntimeError, match='end_step and end_run only'):
                ctx.emit('cube', key='x', count=self.n)

        def process(self, batch, stream):
            pass

        def end_step(self, ctx, stream):
            k = ctx.step_scope.index
            self.n[k] += 10 * (ctx.stream_index + 1)        # 10 on stream 0, 20 on stream 1
            ctx.emit('cube', key='events', count=self.n[k:k + 1])

        def end_run(self, ctx, stream):
            ctx.emit('cube', key='all', count=self.n)

    out = []
    task = GpuTask(factory=_Emitter, state_bytes_per_stream=1000,
                   aggregate_outputs={'cube': {'count': 'sum'}}, aggregate_sink=out.append)
    scopes = _scopes(3)
    streams = [_Stream([], i) for i in range(2)]
    w = TaskWorkers.build(task, streams=streams, budget=_Budget(10_000, 5_000), run_instance=RI,
                          input_preparers={}, margin=0, scopes=scopes)
    w.enter_scope(0, scopes[1], streams[0])        # stream 0 catches up 0, opens 1
    assert [(r.begin_ts, r.complete) for r in out] == [], 'scope 0 waits for stream 1'
    w.enter_scope(1, scopes[2], streams[1])        # stream 1 catches up 0 and 1, opens 2
    assert [(r.begin_ts, r.streams, r.fields['count'].tolist()) for r in out] == [
        (0, (0, 1), [30])], 'scope 0 complete on both streams, 10 + 20'
    w.finish_run('complete')
    delivered = [(r.begin_ts, r.key, r.fields['count'].tolist(), r.complete) for r in out]
    assert delivered == [(0, 'events', [30], True), (950, 'events', [30], True),
                         (1000, 'events', [30], True), (None, 'all', [30, 30, 30, 0], True)]


RI = RunInstance('xpp', 77, 123, 0)


def test_step_cube_example_sums_and_counts_per_step(numpy_cupy):
    from psana.gpu.examples.step_cube import StepCube
    out = []
    task = GpuTask(factory=lambda: StepCube('cam'), inputs=['cam.raw'],
                   state_bytes_per_stream=StepCube.state_bytes(2, 2, 3),
                   aggregate_outputs=StepCube.OUTPUTS, aggregate_sink=out.append)
    scopes = _scopes(2)
    streams = [_Stream([], i) for i in range(2)]
    preps = {'cam.raw': NS(det_shape=(2, 2, 3), _dtype=np.dtype(np.uint16))}
    w = TaskWorkers.build(task, streams=streams, budget=_Budget(10**6, 10**6), run_instance=RI,
                          input_preparers=preps, margin=0, scopes=scopes)

    def batch(values, present):
        return NS(input=lambda name: np.asarray(values, np.uint16),
                  present=lambda name: np.asarray(present, np.uint8),
                  segment_ids=lambda det: (9, 4))

    w.enter_scope(0, scopes[1], streams[0])
    w.instance(0).process(batch(np.full((2, 2, 2, 3), 5), [[1, 1], [1, 0]]), None)   # 2 events
    w.enter_scope(1, scopes[1], streams[1])
    w.instance(1).process(batch(np.full((1, 2, 2, 3), 7), [[0, 1]]), None)           # 1 event
    w.finish_run('complete')

    from psana.gpu import state_bytes
    assert state_bytes(((2, 2, 3), np.uint64), ((2,), np.uint64)) == 512 + 512, 'pool rounding'
    [rec] = [r for r in out if r.begin_ts == scopes[1].begin_ts]
    assert rec.key == (9, 4) and rec.streams == (0, 1) and rec.complete
    assert rec.fields['sums'].tolist() == np.full((2, 2, 3), 17).tolist(), '2*5 + 7 per pixel'
    assert rec.fields['counts'].tolist() == [2, 2], 'segment presence, not events'
    assert not [r for r in out if r.begin_ts == 0], 'no data in scope 0, so nothing emitted'
