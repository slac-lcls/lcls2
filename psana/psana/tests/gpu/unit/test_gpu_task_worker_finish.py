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


def test_a_finalization_failure_stops_later_science_but_not_cleanup(monkeypatch):
    """R5. The first finalization error latches: no instance after it runs a
    step or run hook, and nothing more reaches the sink. Cleanup still runs
    for every instance, because that is what keeps the budget honest.

    Before this, worker 1 still ended step 1 after worker 0's end_run had
    failed, and that completion could deliver a merged record to the sink --
    a result built partly from a run that had already failed.
    """
    w, sc, log, out, budget = _workers(monkeypatch, raise_in_end_run_on=0)
    w.enter_scope(0, sc[1], None); w.enter_scope(1, sc[1], None)
    with pytest.raises(RuntimeError, match='stream 0'):
        w.finish_run('complete')
    assert log.count('close') == 2, 'every instance still closed'
    assert ('end_run', 1) not in log, 'no end_run after the first failure'
    assert not [r for r in out if r.final], 'and no final record reached the sink'
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
    m._close_resources = lambda collective=True: order.append(
        'close_resources' if collective else 'close_resources_local_only')

    def boom(status):
        raise RuntimeError('worker end_run failed')
    m._task_workers = NS(failed=False, finish_run=boom)
    with pytest.raises(RuntimeError, match='worker end_run'):
        m.close()
    # R4: the finalizer error reaches the fatal guard BEFORE any collective.
    # Shared-constant cleanup barriers over the device group, and peers
    # arriving by another path may never reach it, so entering it here
    # delayed the fatal error until that bounded barrier timed out and could
    # replace it with a cleanup error. The local cleanup still runs: dodging
    # one barrier must not cost the reader, parser and budget their release.
    assert order == ['close_resources_local_only'], \
        'local cleanup ran, the device-group barrier did not'

    # A flush that raises leaves the pipeline for a retry close (the base
    # contract) but has already marked the task failed: the retry closes the
    # instances without end_run, since nothing proved every execution retired.
    def bad_flush():
        raise RuntimeError('join failed')
        yield
    order.clear()
    m._closed = m._closing = False
    m._flush_event_pool = bad_flush
    m._task_workers = NS(failed=False, finish_run=lambda status: order.append(status),
                         mark_failed=lambda: order.append('mark_failed'))
    with pytest.raises(RuntimeError, match='join failed'):
        m.close()
    assert order == ['mark_failed'], 'the retry path still marks the task failed'


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


# --------------------------------------------------------------------------
# PR #344 review, 2026-11-23. One test per required behaviour.
# --------------------------------------------------------------------------

class _Stream:
    """A stream that records when it is entered and when it is waited for."""

    def __init__(self, log, index, fail_sync=False):
        self.log, self.index, self.fail_sync = log, index, fail_sync

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def synchronize(self):
        if self.fail_sync:
            raise RuntimeError(f'join failed on stream {self.index}')
        self.log.append(('join', self.index))


class _HookW(GpuTaskWorker):
    """Records every lifecycle hook with the scope it was given.

    `_W` above logs only end_run and close, so an assertion about step hooks
    against its log passes whatever happens.
    """

    def __init__(self, log):
        self.log = log

    def process(self, batch, stream):
        pass

    def begin_step(self, ctx, stream):
        self.log.append(('begin_step', ctx.stream_index, ctx.step_scope))

    def end_step(self, ctx, stream):
        self.log.append(('end_step', ctx.stream_index, ctx.step_scope))
        # Emit as the real step-cube worker does, so assertions about what
        # reached the sink mean something.
        ctx.emit('c', key='k', n=np.array([1]))

    def end_run(self, ctx, stream):
        self.log.append(('end_run', ctx.stream_index))

    def close(self):
        self.log.append('close')


def _workers_logged(monkeypatch, n=2, scopes=3, state=1000):
    monkeypatch.setitem(sys.modules, 'cupy', NS(zeros=np.zeros, empty=np.empty))
    log, out, budget = [], [], _Budget()
    task = GpuTask(factory=lambda: _HookW(log), state_bytes_per_stream=state,
                   aggregate_outputs={'c': {'n': 'sum'}}, aggregate_sink=out.append)
    sc = [StepScope(77, k, 0 if k == 0 else 900 + 50 * k) for k in range(scopes)]
    w = TaskWorkers.build(task, streams=[None] * n, budget=budget, run_instance=RI,
                          input_preparers={}, margin=0, scopes=sc)
    return w, sc, log, out, budget


def _workers_with(monkeypatch, streams, n=2, scopes=3, state=1000, **kw):
    """_workers, but with stream objects of the caller's choosing."""
    monkeypatch.setitem(sys.modules, 'cupy', NS(zeros=np.zeros, empty=np.empty))
    log, out, budget = [], [], _Budget()
    task = GpuTask(factory=lambda: _W(log, **kw), state_bytes_per_stream=state,
                   aggregate_outputs={'c': {'n': 'sum'}}, aggregate_sink=out.append)
    sc = [StepScope(77, k, 0 if k == 0 else 900 + 50 * k) for k in range(scopes)]
    w = TaskWorkers.build(task, streams=streams, budget=budget, run_instance=RI,
                          input_preparers={}, margin=0, scopes=sc)
    return w, sc, log, out, budget


# -- R1 --------------------------------------------------------------------

def test_read_ahead_past_the_cutoff_does_not_move_the_frontier():
    """R1, at the submission site. `note_submitted` is reached only when the
    subbatch carries events.

    After max_events the manager still submits the rest of the packet with
    nothing selected. Counting those advanced the frontier past the delivery
    cutoff, which reported a fully delivered step as truncated and gave the
    next step hooks and output for events nobody received. Structural, like
    test_every_preparation_step_is_inside_the_mark_failed_handler: the guard
    is what the fix is, and a behavioural test of submit() needs the whole
    pool.
    """
    import ast
    import inspect
    import textwrap
    from psana.gpu.gpu_stream import EventPool
    tree = ast.parse(textwrap.dedent(inspect.getsource(EventPool.submit)))

    def calls_note_submitted(node):
        return any(isinstance(n, ast.Attribute) and n.attr == 'note_submitted'
                   for n in ast.walk(node))

    guarded = [n for n in ast.walk(tree)
               if isinstance(n, ast.If) and calls_note_submitted(n)
               and isinstance(n.test, ast.Name) and n.test.id == 'event_envelopes']
    assert guarded, 'note_submitted must be guarded by `if event_envelopes`'
    assert calls_note_submitted(tree), 'and must still be called'


def test_catch_up_stops_at_the_eligible_frontier(monkeypatch):
    """R1. An incomplete run runs hooks up to the frontier and no further.

    Scopes 2 and 3 were decoded -- the manager reads ahead -- but no eligible
    submission reached them, so they are not this run's steps: no hooks, no
    records. Step 1, the one the run ended inside, is truncated.
    """
    w, sc, log, out, budget = _workers_logged(monkeypatch, scopes=4)
    w.enter_scope(0, sc[1], None); w.enter_scope(1, sc[1], None)
    log.clear()
    w.finish_run('incomplete')
    hooks = [e for e in log if e[0] in ('begin_step', 'end_step')]
    assert {e[2] for e in hooks} == {sc[1]}, \
        f'only the eligible scope gets hooks, got {sorted(e[2].index for e in hooks)}'
    assert {r.begin_ts for r in out} == {sc[1].begin_ts}, 'and only its record'
    assert all(r.truncated for r in out), 'the step the run ended inside is truncated'


def test_a_complete_run_still_covers_a_trailing_empty_step(monkeypatch):
    """R1's limit must not cost a genuinely empty step its hooks. A complete
    run delivered everything that was decoded, so a trailing BeginStep with
    no events is this run's step and is reported complete."""
    w, sc, log, out, budget = _workers(monkeypatch, scopes=3)
    w.enter_scope(0, sc[1], None); w.enter_scope(1, sc[1], None)
    w.finish_run('complete')
    by_ts = {r.begin_ts: (r.complete, r.truncated) for r in out}
    assert by_ts[sc[2].begin_ts] == (True, False), 'the empty step completed'


# -- R2 --------------------------------------------------------------------

def test_state_is_released_only_after_the_streams_work_is_joined(monkeypatch):
    """R2. Hooks enqueue GPU work and emit is optional, so finalization must
    wait for each instance's own stream before close() drops its arrays and
    before the reservation goes back to the budget."""
    log = []
    streams = [_Stream(log, 0)]
    w, sc, wlog, out, budget = _workers_with(monkeypatch, streams, n=1)
    log.clear()
    w.enter_scope(0, sc[1], streams[0])
    log.clear()
    w.finish_run('complete')
    order = [e if isinstance(e, str) else e[0] for e in log + wlog]
    assert 'join' in order, 'the stream was waited for'
    assert order.index('join') < order.index('close'), 'before close()'
    assert budget.held == 0 and w.context(0).state.used == 0


def test_a_failed_join_keeps_the_state_and_the_reservation(monkeypatch):
    """R2. If completion cannot be established the instance is quarantined:
    its arrays stay referenced and its charge stays with the budget, because
    pending work may still be using that memory. The error propagates."""
    log = []
    streams = [_Stream(log, 0, fail_sync=True)]
    w, sc, wlog, out, budget = _workers_with(monkeypatch, streams, n=1)
    w.context(0).state.zeros((10,), np.float64)
    used_before = w.context(0).state.used
    with pytest.raises(RuntimeError, match='join failed'):
        w.finish_run('complete')
    assert budget.held == 1000, 'the reservation was NOT returned'
    assert w.context(0).state.used == used_before, 'and the arrays were not dropped'
    assert w.instance(0) is not None, 'the instance stays referenced'


# -- R3 --------------------------------------------------------------------

def test_an_mpi_processing_failure_marks_the_task_failed():
    """R3. process_batch is the MPI entry point and is driven directly, not
    through _events(), so it must latch failure itself: otherwise a planning
    or read error reached the caller's cleanup with the task unmarked, and a
    flush that happened to succeed let finalization export."""
    from psana.gpu import gpu_events
    m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
    m._done = False
    m.gpu_reader = None
    marks = []
    m._task_workers = NS(mark_failed=lambda: marks.append('failed'))

    def boom(*a):
        raise RuntimeError('planning failed')
        yield
    m._process_batch = boom
    with pytest.raises(RuntimeError, match='planning failed'):
        list(m.process_batch(b'smd'))
    assert marks == ['failed']

    # A consumer that stops early is incomplete, not failed.
    marks.clear()
    m._process_batch = lambda *a: iter(['one event'])
    gen = m.process_batch(b'smd')
    assert next(gen) == 'one event'
    gen.close()
    assert marks == [], 'an early stop does not fail the task'


def test_a_first_flush_failure_marks_failed_before_close_can_export():
    """R3. finish() calls close() in its finally, and close() flushes again.
    A non-sticky error can let that second flush succeed, which used to
    finish the task as `incomplete` and export -- the outer _events() handler
    marks failure only after this finally has returned."""
    from psana.gpu import gpu_events
    m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
    m._closed = False
    order = []
    m._task_workers = NS(mark_failed=lambda: order.append('mark_failed'))
    m.close = lambda: order.append('close')

    def bad_flush():
        raise RuntimeError('flush failed')
        yield
    m._flush_event_pool = bad_flush
    with pytest.raises(RuntimeError, match='flush failed'):
        list(m.finish())
    assert order == ['mark_failed', 'close'], 'latched before close() could export'


# -- R7 --------------------------------------------------------------------

def test_a_starved_worker_reports_its_steps_without_waiting_for_the_run_end(monkeypatch):
    """R7. One worker gets every subbatch, the other none. The idle worker
    still reports the steps it has passed, so the reducer can finish them and
    drop their bookkeeping instead of holding every scope open to the end of
    the run.

    Before this, a 100-step run with one starved worker retained 99
    completion entries and 97 spilled layouts, and none of those steps
    reached the sink until finalization.
    """
    n_steps = 20
    w, sc, log, out, budget = _workers(monkeypatch, scopes=n_steps)
    for k in range(1, n_steps):
        w.enter_scope(0, sc[k], None)       # stream 0 only ever gets rows
        w.service_idle(exclude=0)           # what EventPool.submit now does
    delivered = {r.begin_ts for r in out if r.final}
    assert len(delivered) >= n_steps - 3, \
        f'steps completed during the run, not at its end: {len(delivered)}'
    reducer = w._reducer
    assert len(reducer._done) <= 2, f'bookkeeping retired: {len(reducer._done)}'
    assert len(reducer._seq) <= 2, f'no spilled layouts left over: {len(reducer._seq)}'


def test_servicing_an_idle_worker_twice_does_not_repeat_its_hooks(monkeypatch):
    """R7. service_idle runs on every submission, so it must be idempotent:
    a second call with the frontier unmoved places no further hooks."""
    w, sc, log, out, budget = _workers_logged(monkeypatch, scopes=4)
    w.enter_scope(0, sc[2], None)
    w.service_idle(exclude=0)
    first = list(log)
    assert [e for e in first if e[1] == 1], 'stream 1 was serviced at all'
    w.service_idle(exclude=0)
    w.service_idle(exclude=0)
    assert log == first, 'no hooks were placed twice'


def test_an_idle_worker_caught_up_can_still_receive_later_work(monkeypatch):
    """R7. Being serviced must not close the door: a later nonempty
    submission for a higher scope still places that scope's hooks."""
    w, sc, log, out, budget = _workers_logged(monkeypatch, scopes=4)
    w.enter_scope(0, sc[1], None)
    w.enter_scope(0, sc[2], None)
    w.service_idle(exclude=0)               # stream 1 catches up through scope 1
    log.clear()
    w.enter_scope(1, sc[2], None)           # and now gets rows for scope 2
    assert ('begin_step', 1, sc[2]) in log, 'scope 2 opened on stream 1'


def test_a_hook_error_survives_a_join_that_also_fails(monkeypatch):
    """R5 asks for the FIRST error to be preserved. Cleanup used to live in a
    `finally`, and an exception raised while unwinding replaces the one being
    unwound, so a join that failed after a hook had already failed reported
    the join and lost the hook -- the error the user needs."""
    log = []
    streams = [_Stream(log, 0, fail_sync=True)]
    w, sc, wlog, out, budget = _workers_with(
        monkeypatch, streams, n=1, raise_in_end_run_on=0)
    with pytest.raises(RuntimeError, match='end_run failed'):
        w.finish_run('complete')
    assert budget.held == 1000, 'and the quarantine still holds'


# --------------------------------------------------------------------------
# Follow-ups from the review of these fixes, 2026-11-24.
# --------------------------------------------------------------------------

class _CloseW(GpuTaskWorker):
    """Records the run status its close() sees, and can raise there."""

    def __init__(self, log, raise_in_close_on=None, raise_in_end_run_on=None):
        self.log = log
        self.raise_close_on = raise_in_close_on
        self.raise_end_run_on = raise_in_end_run_on
        self.ctx = None

    def begin_run(self, ctx, stream):
        self.ctx = ctx

    def process(self, batch, stream):
        pass

    def end_run(self, ctx, stream):
        if ctx.stream_index == self.raise_end_run_on:
            raise RuntimeError(f'end_run failed on stream {ctx.stream_index}')

    def close(self):
        self.log.append(('close', self.ctx.stream_index, self.ctx.run_status))
        if self.ctx.stream_index == self.raise_close_on:
            raise RuntimeError(f'close failed on stream {self.ctx.stream_index}')


def _close_workers(monkeypatch, n=2, state=1000, **kw):
    monkeypatch.setitem(sys.modules, 'cupy', NS(zeros=np.zeros, empty=np.empty))
    log, budget = [], _Budget()
    task = GpuTask(factory=lambda: _CloseW(log, **kw), state_bytes_per_stream=state)
    sc = [StepScope(77, k, 0 if k == 0 else 900 + 50 * k) for k in range(3)]
    w = TaskWorkers.build(task, streams=[None] * n, budget=budget, run_instance=RI,
                          input_preparers={}, margin=0, scopes=sc)
    return w, sc, log, budget


def test_a_raising_close_still_returns_the_reservation(monkeypatch):
    """A bug in a user's close() is an error to report, not a reason to hold
    the run's GPU budget for the rest of the process. Both joins succeeded,
    so nothing is pending and the state is safe to drop; only a failed JOIN
    quarantines."""
    w, sc, log, budget = _close_workers(monkeypatch, raise_in_close_on=0)
    with pytest.raises(RuntimeError, match='close failed'):
        w.finish_run('complete')
    assert budget.held == 0, 'the reservation came back'
    assert all(w.context(i).state.used == 0 for i in range(2))
    assert [e[1] for e in log] == [0, 1], 'both instances were closed'


def test_later_instances_see_a_failed_run_status(monkeypatch):
    """After instance 0's end_run raises, instance 1 runs no hooks -- but its
    close() used to be told the run was `complete`. A close() that branches on
    status, writing a summary say, is misled by that."""
    w, sc, log, budget = _close_workers(monkeypatch, raise_in_end_run_on=0)
    with pytest.raises(RuntimeError, match='end_run failed'):
        w.finish_run('complete')
    seen = {slot: status for _, slot, status in log}
    assert seen[1] == 'failed', f'instance 1 saw {seen[1]!r}'
    assert w.failed is True, 'and the task records the finalization failure'


def test_a_single_stream_services_itself(monkeypatch):
    """R7 with one stream: service_idle must be reachable with exclude=None,
    because no other stream exists to service it."""
    w, sc, log, out, budget = _workers_logged(monkeypatch, n=1, scopes=4)
    w.note_submitted(sc[2])                 # eligible, but placed no hooks
    w.service_idle(exclude=None)
    assert [e for e in log if e[0] == 'end_step'], \
        'the only stream closed the steps below the frontier'


def test_submit_services_the_submitting_stream_when_it_placed_no_hooks():
    """The call site for the test above. With `exclude=slot` unconditionally,
    a single-stream run never serviced anyone."""
    import ast
    import inspect
    import textwrap
    from psana.gpu.gpu_stream import EventPool
    tree = ast.parse(textwrap.dedent(inspect.getsource(EventPool.submit)))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'service_idle']
    assert calls, 'submit must service idle workers'
    for call in calls:
        exclude = {k.arg: k.value for k in call.keywords}.get('exclude')
        assert isinstance(exclude, ast.IfExp), \
            'exclude must be conditional, so a lone stream services itself'


def test_a_finalizer_error_aborts_before_any_device_group_barrier():
    """R4 as the review asked it: the fatal guard must Abort before cleanup enters
    a collective.

    The close()-level test above pins the `collective=False` decision. This
    one drives the real _close_resources through the real gpu_error_handler
    with a fake communicator, so the ORDER is what is asserted: the
    device-group barrier never runs, and Abort does.
    """
    from psana.gpu import gpu_events
    from psana.gpu.gpu_mpi import gpu_error_handler

    order = []

    class _Comm:
        def Get_rank(self):
            return 0

        def Abort(self, code):
            order.append(('abort', code))

    class _Constants:
        """Stands in for SharedRequestedConstants with sharing active, whose
        close() barriers over the device group."""
        _sharing = True
        _shared = {'pedestals': object()}

        def close(self):
            order.append('device_group_barrier')

    m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
    m._closed = m._closing = False
    m._iter = None
    m.event_pool = NS(_retiring=None)
    m._run_complete = True
    m._flush_event_pool = lambda: iter(())
    m._drain_pending_gpu_read = lambda: None
    m._group_inputs = None
    m.gpu_xtc_parser = None
    m._gpu_budget = None
    m.gpu_reader = None
    m._output_d2h = None
    m._task_constants = _Constants()

    def boom(status):
        raise RuntimeError('aggregate sink failed')
    m._task_workers = NS(failed=False, finish_run=boom)

    # gpu_error_handler suppresses after Abort, so this does not raise.
    with gpu_error_handler(_Comm()):
        m.close()

    assert order == [('abort', 1)], \
        f'the barrier must not run before the Abort; got {order}'
