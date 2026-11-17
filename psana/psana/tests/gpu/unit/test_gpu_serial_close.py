"""Public serial generator closure must retire every GPU execution."""
from contextlib import closing
from types import SimpleNamespace as NS
import weakref

import numpy as np
import pytest

from psana.event import EventEnvelope
from psana.gpu.gpu_d2h import PublicationD2H
from psana.gpu.gpu_events import GpuEventManager, _GpuOnlyDgram
from psana.gpu.gpu_stream import EventPool, _EventSlot, _failed_execution_pools
from psana.psexp.run import RunSerial
from test_gpu_d2h import cuda, record, Stream


def fixture(cuda, phase='final'):
    log = cuda[0]
    pipeline = PublicationD2H(8192)
    pool = EventPool.__new__(EventPool)
    pool._n, pool._write_idx, pool._retiring = 2, 2, None
    pool._slots = []
    refs = []
    for slot in range(2):
        stamps = [10 + slot * 2, 11 + slot * 2]
        publication = record(cuda, [('count', np.asarray(stamps, np.uint32), stamps)])
        scratch = np.empty(8)
        refs.extend((weakref.ref(scratch), weakref.ref(publication.publication_batches[0].array)))
        item = _EventSlot(
            slot, {}, [EventEnvelope([_GpuOnlyDgram(ts)]) for ts in stamps],
            Stream(log), [publication.lease], {}, producer_owners=[scratch],
            publication_batches=publication.publication_batches,
        )
        pipeline.enqueue(item)
        pool._slots.append(item)
    manager = GpuEventManager.__new__(GpuEventManager)
    manager._iter = None
    manager._done = manager._closed = manager._closing = False
    manager._first_batch_logged = True
    manager.gpu_det_names = []
    manager.event_pool = pool
    manager._output_d2h = pipeline
    manager._drain_pending_gpu_read = lambda: log.append('read-drain')
    manager.gpu_reader = NS(close=lambda: log.append('reader-close'))
    manager._task_constants = NS(close=lambda: log.append('constants-close'))
    batches = iter([(None, None, None)] if phase == 'mid' else [])
    manager._next_batch = lambda: next(batches)
    manager._process_batch = lambda *args: manager._retire_issue_and_yield(None)
    def unexpected_read(*args):
        raise AssertionError('early close issued another read')
    manager._issue_gpu_read = unexpected_read
    run = RunSerial.__new__(RunSerial)
    run.dsparms = NS(gpu_enabled=True)
    run._evt_iter = manager
    run._run_ctx = object()
    return run, manager, refs


def assert_closed(run, manager, refs):
    assert manager._closed and not manager._closing
    assert manager.event_pool.active_count == 0
    assert manager.event_pool._retiring is None
    assert manager._output_d2h.pinned_bytes == 0
    assert all(ref() is None for ref in refs)
    assert list(run.events()) == []  # Closure is terminal, not a pause.


@pytest.mark.parametrize('phase', ['mid', 'final'])
@pytest.mark.parametrize('exit_kind', ['explicit', 'break', 'error'])
def test_public_early_close_drains_all_slots_and_keeps_host_results(cuda, phase, exit_kind):
    run, manager, refs = fixture(cuda, phase)
    events = run.events()
    held = []
    if exit_kind == 'explicit':
        held.append(next(events))
        assert manager.event_pool.active_count == 2
        events.close()
    else:
        def consume():
            with closing(events):
                for event in events:
                    held.append(event)
                    if exit_kind == 'error':
                        raise ValueError('loop-body error')
                    break
        if exit_kind == 'error':
            with pytest.raises(ValueError, match='loop-body error'):
                consume()
        else:
            consume()
    assert_closed(run, manager, refs)
    assert held[0].gpu.get('count').on_cpu == 10
    # Repeated public close is harmless and does not re-close resources.
    before = list(cuda[0])
    events.close()
    assert cuda[0] == before
    assert cuda[0].count('reader-close') == cuda[0].count('constants-close') == 1


def test_public_exhaustion_drains_once(cuda):
    run, manager, refs = fixture(cuda)
    held = list(run.events())
    assert [event.timestamp for event in held] == [10, 11, 12, 13]
    assert_closed(run, manager, refs)
    assert [int(event.gpu.get('count').on_cpu) for event in held] == [10, 11, 12, 13]
    assert cuda[0].count('reader-close') == 1


def test_interrupted_internal_finish_drains_remaining_slots(cuda):
    run, manager, refs = fixture(cuda)
    deliveries = manager.finish()
    envelope = next(deliveries)
    deliveries.close()
    assert_closed(run, manager, refs)
    assert envelope.gpu_state.get('count').on_cpu == 10


def test_public_producer_error_drains_owned_work(cuda):
    run, manager, refs = fixture(cuda)
    def fail():
        raise ValueError('producer error')
    manager._next_batch = fail
    with pytest.raises(ValueError, match='producer error'):
        next(run.events())
    assert_closed(run, manager, refs)


def test_failed_public_close_retains_owners_until_retry(cuda):
    run, manager, refs = fixture(cuda)
    events = run.events()
    held = next(events)
    failed = manager.event_pool._slots[1]
    failed.stream.fail_sync = True
    with pytest.raises(RuntimeError, match='drain failed'):
        events.close()
    assert not manager._closed and not manager._closing
    assert manager.event_pool in _failed_execution_pools
    assert all(ref() is not None for ref in refs[2:])
    assert 'reader-close' not in cuda[0]
    failed.stream.fail_sync = False
    manager.close()  # Retry after the injected device failure, not test teardown.
    assert_closed(run, manager, refs)
    assert manager.event_pool not in _failed_execution_pools
    assert held.gpu.get('count').on_cpu == 10


def test_cpu_only_serial_iteration_keeps_existing_close_behavior():
    log = []
    class Source:
        def __iter__(self):
            return self
        def __next__(self):
            return EventEnvelope([_GpuOnlyDgram(10)])
        def close(self):
            log.append('closed')
    run = RunSerial.__new__(RunSerial)
    run.dsparms = NS(gpu_enabled=False)
    run._evt_iter = Source()
    run._run_ctx = object()
    events = run.events()
    next(events)
    events.close()
    assert not log


def _steps_fixture(cuda):
    """The events() fixture, with a BeginStep ahead of the GPU-only events.

    steps() only yields on a BeginStep envelope, so the stream needs one --
    which the manager emits since #153 PR B step 4. _run_ctx must be real
    enough for _materialize_event and the envstore update.
    """
    run, manager, refs = fixture(cuda)
    from psana.psexp import TransitionId
    begin = NS(timestamp=lambda: 9,
               service=lambda: TransitionId.BeginStep,
               env=lambda: TransitionId.BeginStep << 24)
    original = manager._events

    def with_begin_step():
        yield EventEnvelope(dgrams=[begin], gpu_state=None)
        yield from original()

    manager._events = with_begin_step
    run.esm = None
    run._update_envstore_from_dgrams = lambda dgrams: None
    run._materialize_event = lambda envelope: envelope
    return run, manager, refs


def test_steps_closes_the_gpu_manager_on_early_break(cuda):
    """Abandoning the steps() loop must release the GPU stream.

    RunSerial.events() has always wrapped iteration in try/finally to close
    the manager; steps() inherited Run.steps() and did not. That was harmless
    while steps() yielded nothing on a GPU rank, but #153 PR B step 4 makes
    the manager emit BeginStep envelopes, so without the same wrapper this
    path leaves the reader, budget and pool to garbage collection.
    """
    run, manager, refs = _steps_fixture(cuda)
    steps = run.steps()
    with closing(steps):
        for _ in steps:
            break                      # abandon the outer loop immediately
    assert_closed(run, manager, refs)


def test_steps_closes_the_gpu_manager_on_exhaustion(cuda):
    run, manager, refs = _steps_fixture(cuda)
    collected = list(run.steps())
    assert len(collected) == 1, 'one BeginStep, so one step'
    assert_closed(run, manager, refs)
    assert cuda[0].count('reader-close') == 1


def test_steps_closes_the_gpu_manager_on_a_loop_body_error(cuda):
    run, manager, refs = _steps_fixture(cuda)
    steps = run.steps()
    with pytest.raises(ValueError, match='loop-body error'):
        with closing(steps):
            for _ in steps:
                raise ValueError('loop-body error')
    assert_closed(run, manager, refs)


def test_steps_without_the_finally_would_leak_on_early_break(cuda):
    """Pins that the finally -- not the manager itself -- does the work.

    On normal exhaustion the manager self-closes in finish(), so a test that
    merely runs steps() to completion cannot tell whether RunSerial.steps()
    wraps it. The wrapper only matters on EARLY exit, so compare against the
    inherited Run.steps(), which has no wrapper: it must leave the manager
    open where RunSerial.steps() closes it.
    """
    from psana.psexp.run import Run
    run, manager, _ = _steps_fixture(cuda)

    # The inherited generator, i.e. the pre-fix behaviour.
    bare = Run.steps(run)
    with closing(bare):
        for _ in bare:
            break
    assert not manager._closed, (
        'baseline check: the inherited steps() does not close, which is '
        'exactly the leak this fix addresses')

    # And RunSerial.steps() on the same abandoned stream does close it.
    run2, manager2, refs2 = _steps_fixture(cuda)
    wrapped = run2.steps()
    with closing(wrapped):
        for _ in wrapped:
            break
    assert manager2._closed
    manager.close()
