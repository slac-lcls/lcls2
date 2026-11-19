"""Per-stream worker instances for ``GpuTask(factory=...)`` (PR #270 phase 1).

The declaration lives in gpu_task; this module owns what the manager does
with it: mint the run instance identity, build one ``GpuTaskWorker`` per CUDA
execution stream at setup inside a budget reservation, hand each ``process``
call to the right instance, and run ``end_run``/``close`` exactly once when
the run's stream is over. CuPy is imported only inside methods that allocate.
"""
from collections import Counter
from contextlib import nullcontext
from typing import NamedTuple

import numpy as np

from .gpu_allocation import POOL_BLOCK, allocation_capacity
from .gpu_budget import GpuMemoryPressureError
from .gpu_task import GpuTaskWorker


class RunInstance(NamedTuple):
    """Identity of one opening of a run, identical on every BD.

    The run number alone can repeat within a job (the same run opened twice,
    two experiments sharing a number), so records that outlive the run carry
    this instead. ``ordinal`` counts earlier openings of the same
    ``(expt, runnum)`` in this process; every rank opens runs in the same
    order, so every BD computes the same value without communicating. Unique
    within a job; cross-job uniqueness is the sink's concern.
    """
    expt: object
    runnum: object
    begin_run_ts: object
    ordinal: int


_openings = Counter()


def mint_run_instance(run):
    """One RunInstance per opened run, from what Run already knows.

    Cached on the run object, so a second events()/steps() call on the same
    Run -- which builds a second manager on the MPI path -- reuses it: the
    ordinal counts openings of the run, not managers.
    """
    cached = getattr(run, '_gpu_run_instance', None)
    if cached is not None:
        return cached
    key = (getattr(run, 'expt', None), getattr(run, 'runnum', None))
    ordinal = _openings[key]
    _openings[key] += 1
    instance = RunInstance(key[0], key[1], getattr(run, 'timestamp', None), ordinal)
    try:
        run._gpu_run_instance = instance
    except (AttributeError, TypeError):
        pass
    return instance


class InputSpec(NamedTuple):
    """Per-event shape and dtype of a declared dense input, from validated
    configuration rather than from any event -- so an instance that never
    receives an event can still size its state."""
    event_shape: tuple
    dtype: object


def state_bytes(*arrays):
    """The reservation ``ctx.state`` needs for these ``(shape, dtype)`` pairs.

    Each allocation is charged at the pool's rounded capacity, not its exact
    byte count, so a reservation summed from ``nbytes`` comes up short by a
    few hundred bytes per array. Use this for ``state_bytes_per_stream``.
    """
    total = 0
    for shape, dtype in arrays:
        dims = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
        nbytes = int(np.prod([int(n) for n in dims], dtype=np.int64)) * np.dtype(dtype).itemsize
        total += ((nbytes + POOL_BLOCK - 1) // POOL_BLOCK) * POOL_BLOCK
    return total


class StateAllocator:
    """Device allocations charged against one instance's setup reservation.

    The reservation was committed to the budget at setup, so allocations here
    are NOT charged to it again; they are counted against the reservation and
    refused beyond it. References are dropped in ``release()``, after which
    the manager returns the reservation to the budget.
    """

    def __init__(self, reservation):
        self.reservation = int(reservation)
        self.used = 0
        self._arrays = []

    def _take(self, shape, dtype, fill):
        import cupy as cp
        dtype = np.dtype(dtype)
        dims = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
        nbytes = int(np.prod([int(n) for n in dims], dtype=np.int64)) * dtype.itemsize
        capacity = allocation_capacity(cp, nbytes)
        if self.used + capacity > self.reservation:
            raise GpuMemoryPressureError(
                f'task state reservation exceeded: {self.used} used + {capacity} '
                f'requested > {self.reservation} reserved per stream '
                f'(GpuTask.state_bytes_per_stream)')
        array = (cp.zeros if fill == 0 else cp.empty)(dims, dtype=dtype)
        self.used += capacity
        self._arrays.append(array)
        return array

    def zeros(self, shape, dtype):
        return self._take(shape, dtype, 0)

    def empty(self, shape, dtype):
        return self._take(shape, dtype, None)

    def release(self):
        self._arrays.clear()
        self.used = 0


class WorkerContext:
    """What a lifecycle hook may see. Deliberately no EnvStore, no scan/step
    payload values, no SlowUpdate values, no communicator (PR #172 scope)."""

    def __init__(self, run_instance, stream_index, state, input_specs, reducer=None):
        self.run_instance = run_instance
        self.stream_index = stream_index
        self.state = state
        self.step_scope = None          # the step a step hook is about
        self.run_status = None          # "complete" | "incomplete" | "failed", in end_run
        self._input_specs = input_specs
        self._reducer = reducer
        self._hook = None               # name of the hook currently running

    def input_spec(self, name):
        try:
            return self._input_specs[name]
        except KeyError:
            raise KeyError(f'{name!r} is not a declared dense input of this task') from None

    def emit(self, name, key, **fields):
        """Deliver a keyed snapshot of device arrays to the aggregate sink.

        Allowed in end_step and end_run only. Each array is copied to the host
        on this instance's stream and the copy is waited for before returning,
        so the worker may clear or reuse the array immediately; this waits on
        one stream's own work, never on the pool. The record is keyed
        (run_instance, step_scope.begin_ts, key) -- begin_ts None in end_run
        -- and merged by the reducer with the other streams' records.
        """
        if self._hook not in ('end_step', 'end_run'):
            raise RuntimeError('ctx.emit is allowed in end_step and end_run only')
        if self._reducer is None:
            raise RuntimeError('ctx.emit requires GpuTask(aggregate_outputs=..., aggregate_sink=...)')
        host = {}
        for fname, array in fields.items():
            get = getattr(array, 'get', None)
            # CuPy: a D2H copy on the current (this instance's) stream that
            # blocks until it lands. NumPy stand-ins pass straight through.
            host[fname] = np.asarray(get() if callable(get) else array)
        begin_ts = None if self.step_scope is None else self.step_scope.begin_ts
        self._reducer.add(name, self.run_instance, begin_ts, key, host, self.stream_index)


def _entered(stream):
    """`with stream:` when it is a CUDA stream; tests pass None."""
    return nullcontext() if stream is None else stream


class TaskWorkers:
    """The per-stream instances of one stateful task, for one run instance."""

    def __init__(self, task, instances, contexts, streams, budget, reserved, scopes,
                 reducer=None):
        self._reducer = reducer
        self._instances = instances
        self._contexts = contexts
        self._streams = streams
        self._budget = budget
        self._reserved = reserved
        # The manager's ordered scope list, shared by reference; scopes[k]
        # has index k. Per slot, the scope whose begin_step is open.
        self._scopes = scopes
        self._current = [None] * len(instances)
        # Scope 0 is the interval before the first BeginStep. Its hook pair
        # is placed only if some stream processed events in it: submission
        # order is non-decreasing, so once a scope >= 1 is submitted no
        # scope-0 subbatch can follow and the decision is final.
        self._scope0_seen = False
        # Highest scope any submission carried, rows or not: the cut point
        # for truncation. A subbatch with no task-selected rows places no
        # hooks, but it still proves every earlier scope was fully submitted.
        self._reached = None
        self.failed = False
        self._finished = False

    @classmethod
    def build(cls, task, *, streams, budget, run_instance, input_preparers, margin,
              scopes=()):
        """Reserve, construct and begin_run every instance, or none of them.

        ``margin`` is the admission headroom the pipeline keeps for itself
        (the manager's 10% rule); the reservation must leave more than that,
        or the first subbatch could never be admitted.
        """
        n = len(streams)
        need = task.state_bytes_per_stream * n
        available = budget.available()
        if need and available - need - margin <= 0:
            raise RuntimeError(
                f'GpuTask.state_bytes_per_stream={task.state_bytes_per_stream} x '
                f'{n} streams = {need} bytes does not fit: {available} bytes are '
                f'available after fixed storage, and {margin} must stay free for '
                f'event admission (limit {budget.limit()})')
        if need:
            budget.reserve(need)
        specs = {name: InputSpec(tuple(p.det_shape), p._dtype)
                 for name, p in (input_preparers or {}).items()
                 if hasattr(p, 'det_shape')}
        reducer = None
        if task.aggregate_sink is not None:
            from .gpu_aggregate import AggregateReducer
            reducer = AggregateReducer(task.aggregate_outputs, n, task.aggregate_sink,
                                       task.aggregate_host_bytes)
        instances, contexts = [], []
        try:
            for index, stream in enumerate(streams):
                instance = task.factory()
                if not isinstance(instance, GpuTaskWorker):
                    raise TypeError('GpuTask.factory must return a GpuTaskWorker, '
                                    f'got {type(instance).__name__}')
                instances.append(instance)
                ctx = WorkerContext(run_instance, index,
                                    StateAllocator(task.state_bytes_per_stream), specs,
                                    reducer)
                contexts.append(ctx)
                with _entered(stream):
                    ctx._hook = 'begin_run'
                    try:
                        instance.begin_run(ctx, stream)
                    finally:
                        ctx._hook = None
        except BaseException:
            # Close only what was constructed, in order, then give the
            # reservation back; no event was admitted, so nothing else is owed.
            # A close() that itself raises must not stop the others or hide
            # the setup error, which is the one the user needs to see.
            # close() runs on the instance's own stream, like every other hook.
            for instance, ctx, stream in zip(instances, contexts, streams):
                try:
                    with _entered(stream):
                        instance.close()
                except BaseException:
                    pass
                finally:
                    ctx.state.release()
            if need:
                budget.release(need)
            raise
        return cls(task, instances, contexts, list(streams), budget, need, scopes, reducer)

    @property
    def n(self):
        return len(self._instances)

    def instance(self, slot):
        return self._instances[slot]

    def context(self, slot):
        return self._contexts[slot]

    def mark_failed(self):
        self.failed = True

    def _step(self, slot, hook, scope, stream, truncated=False):
        ctx = self._contexts[slot]
        ctx.step_scope, ctx._hook = scope, hook
        try:
            getattr(self._instances[slot], hook)(ctx, stream)
        finally:
            ctx._hook = None
        # Completion comes from the framework, not from whether the hook
        # emitted: a stream that finished a step contributes to that step's
        # completion even if it had nothing to say. `truncated` marks a step
        # the run ended inside (max_events, early close), so its merged
        # record is not reported complete.
        if hook == 'end_step' and self._reducer is not None:
            self._reducer.scope_done(slot, ctx.run_instance, scope.begin_ts,
                                     truncated=truncated)

    def _close_scope(self, slot, stream, stop, truncated_from=None):
        """End this stream's open scope, then begin/end every scope below
        `stop` it never saw: steps this stream received no subbatch for still
        complete, with zero-count state, so a reducer can record every
        stream's completion. Scope 0 is skipped unless some stream processed
        events in it. Scopes at or after `truncated_from` are reported
        truncated. Must run inside `with stream:`."""
        def truncated(scope):
            return truncated_from is not None and scope.index >= truncated_from
        current = self._current[slot]
        start = 0
        if current is not None:
            self._step(slot, 'end_step', current, stream, truncated=truncated(current))
            start = current.index + 1
        for scope in self._scopes[start:stop]:
            if scope.index == 0 and not self._scope0_seen:
                continue
            self._step(slot, 'begin_step', scope, stream)
            self._step(slot, 'end_step', scope, stream, truncated=truncated(scope))

    def note_submitted(self, scope):
        """Record that a subbatch of `scope` was submitted, whether or not it
        selected any task rows. Submission order is non-decreasing, so the
        last scope noted is the highest."""
        if self._reached is None or scope.index > self._reached.index:
            self._reached = scope

    def enter_scope(self, slot, scope, stream):
        """Place this stream's step hooks for a submission in `scope`.

        Called by EventPool.submit with `stream` about to run the subbatch.
        Scopes are non-decreasing per stream, so a change means the previous
        scope is finished on this stream: end_step for it, begin/end pairs for
        the scopes in between, begin_step for the new one. Everything is
        enqueued on the stream ahead of process, so "this instance's
        contributions are complete" holds by stream order -- no drain.
        """
        self.note_submitted(scope)
        if scope.index == 0:
            self._scope0_seen = True
        current = self._current[slot]
        if current is not None and scope.index <= current.index:
            if scope.index < current.index:
                raise RuntimeError(
                    f'stream {slot} received scope {scope.index} after scope '
                    f'{current.index}; submission order must not decrease')
            return
        with _entered(stream):
            self._close_scope(slot, stream, scope.index)
            self._step(slot, 'begin_step', scope, stream)
        self._current[slot] = scope

    def finish_run(self, status):
        """end_run (unless failed) then close, per instance on its stream; once.

        Called by the manager after the execution pool has been flushed, so
        every admitted contribution has retired. ``status`` is "complete" only
        when EndRun was seen; a failed run skips end_run -- a partially
        updated sum must not be exported as a result -- and only closes.

        On an incomplete run, the step the run ended inside, and any later
        step, is exported as truncated. Submission order is global and
        non-decreasing, so every scope below the highest one any stream
        reached was fully submitted; only that scope and later ones can be
        short.

        Every instance is finished and closed, and its state released, even
        when another instance's hook raises; the first such error is raised
        after all of them, so one failing worker cannot leave the others
        open or the reservation held.
        """
        if self._finished:
            return
        self._finished = True
        truncated_from = None
        if status != 'complete':
            # The highest scope submitted, with or without task rows. Every
            # scope below it was fully submitted; only it and later ones can
            # be short.
            truncated_from = self._reached.index if self._reached is not None else 0
        first_error = None
        try:
            for slot, (instance, ctx, stream) in enumerate(
                    zip(self._instances, self._contexts, self._streams)):
                ctx.run_status = status
                try:
                    with _entered(stream):
                        try:
                            if status != 'failed':
                                # End of stream: close this stream's open scope,
                                # complete every later scope the run opened that
                                # this stream never saw, then end the run.
                                self._close_scope(slot, stream, len(self._scopes),
                                                  truncated_from)
                                ctx.step_scope, ctx._hook = None, 'end_run'
                                try:
                                    instance.end_run(ctx, stream)
                                finally:
                                    ctx._hook = None
                                if self._reducer is not None:
                                    self._reducer.run_done(
                                        slot, truncated=truncated_from is not None)
                        finally:
                            try:
                                instance.close()
                            finally:
                                ctx.state.release()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        finally:
            # Drop the instances with the reservation: an instance whose
            # close() kept its arrays would otherwise hold device memory the
            # budget now counts as free, for as long as the manager lives.
            self._instances = [None] * len(self._instances)
            if self._reserved:
                self._budget.release(self._reserved)
                self._reserved = 0
        if first_error is not None:
            raise first_error
