"""Retirement must close access without losing consumers or backing charges."""
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from psana.gpu.context import GPUResult, GpuEventState, SlotLease
from psana.gpu.gpu_budget import _GpuBudget
from psana.gpu.gpu_allocation import owned_empty
from test_gpu_input_window import Token, window


class Stream:
    def __init__(self):
        self.recorded = []
        self.fail = False
        self.drain_fail = False
    def wait_event(self, event):
        pass
    def record(self, event):
        if self.fail:
            raise RuntimeError('record failed')
        self.recorded.append(event)
    def synchronize(self):
        if self.drain_fail:
            raise RuntimeError('drain failed')


@pytest.fixture
def cuda(monkeypatch):
    stream = Stream()
    monkeypatch.setitem(sys.modules, 'cupy', NS(cuda=NS(
        Stream=NS(null=stream), Event=Token, get_current_stream=lambda: stream)))
    return stream


def test_all_result_consumers_and_copy_stream_are_joined(cuda):
    lease = SlotLease(Token())
    result = GPUResult(np.arange(4), lease)
    first, second = Stream(), Stream()
    with result.on_gpu_view(first):
        pass
    with result.on_gpu_view(second):
        pass
    np.testing.assert_array_equal(result.on_gpu, np.arange(4))
    consumers = first.recorded + second.recorded + cuda.recorded
    assert len(consumers) == 3
    lease.wait_until_safe_to_reuse()
    assert all(event.waits == 1 for event in consumers)
    assert result._arr is None
    with pytest.raises(RuntimeError, match='released'):
        result.on_gpu


def test_open_result_view_pins_slot_and_can_finish_after_retirement_attempt(cuda):
    lease = SlotLease(None)
    result = GPUResult(np.arange(4), lease)
    entered = result.on_gpu_view(cuda)
    delayed = result.on_gpu_view(cuda)
    array = entered.__enter__()
    with pytest.raises(RuntimeError, match='open view'):
        lease.wait_until_safe_to_reuse()
    assert result._arr is array
    with pytest.raises(RuntimeError, match='retiring'):
        delayed.__enter__()
    entered.__exit__(None, None, None)
    lease.wait_until_safe_to_reuse()
    assert result._arr is None


def test_record_and_drain_failure_keeps_view_acquired_for_retry(cuda):
    lease = SlotLease(None)
    result = GPUResult(np.arange(4), lease)
    view = result.on_gpu_view(cuda)
    view.__enter__()
    cuda.fail = cuda.drain_fail = True
    with pytest.raises(RuntimeError, match='drain failed'):
        view.__exit__(None, None, None)
    with pytest.raises(RuntimeError, match='open view'):
        lease.wait_until_safe_to_reuse()
    cuda.fail = cuda.drain_fail = False
    view.__exit__(None, None, None)
    lease.wait_until_safe_to_reuse()
    assert result._arr is None


@pytest.mark.parametrize('record_fails', [False, True])
def test_field_view_exit_can_retry_failed_owner_completion(cuda, record_fails):
    from psana.gpu.gpu_input import InputSlotLease, _GpuFieldViewContext
    owner = window()
    ready = Token()
    parent = InputSlotLease(ready, (owner,))
    context = _GpuFieldViewContext(NS(_lease=parent, _slot_views=lambda stream: None), cuda)
    context.__enter__()
    assert not owner.close()
    parent.wait_until_safe_to_reuse()
    ready.fail = True
    cuda.fail = record_fails
    with pytest.raises(RuntimeError, match='completion failure'):
        context.__exit__(None, None, None)
    assert not owner.released
    registered = len(cuda.recorded)
    ready.fail = False
    context.__exit__(None, None, None)
    assert owner.released and owner.references == 0
    assert len(cuda.recorded) == registered


def test_field_host_copy_record_failure_returns_child_reference(cuda, monkeypatch):
    from psana.gpu.gpu_input import GpuFieldResult, InputSlotLease
    owner = window()
    parent = InputSlotLease(None, (owner,))
    result = GpuFieldResult(NS(segment_ids=(0,)), NS(), parent)
    monkeypatch.setattr(GpuFieldResult, '_slot_views',
                        lambda self, stream: {0: NS(get=lambda: np.arange(4))})
    cuda.fail = True
    with pytest.raises(RuntimeError, match='record failed'):
        result.on_cpu
    assert owner.references == 1  # only the original parent remains
    assert not owner.close()
    parent.wait_until_safe_to_reuse()
    assert owner.released
    np.testing.assert_array_equal(result.on_cpu[0], np.arange(4))


def test_retained_result_state_releases_backing_but_keeps_host_cache(cuda):
    budget = _GpuBudget(100)
    array = owned_empty(np, 40, np.uint8, budget, 'detector')
    lease = SlotLease(None)
    state = GpuEventState({'det.raw': array}, leases={'det.raw': lease})
    result = state.get('det.raw')
    result._cpu_cache = np.arange(3)
    escaped = array[:2]
    del array
    lease.wait_until_safe_to_reuse()
    assert state._gpu_results == {'det.raw': None}
    assert budget.committed() == 40  # escaped raw array still owns its block
    del escaped
    assert budget.committed() == 0
    np.testing.assert_array_equal(result.on_cpu, np.arange(3))


def test_saved_stream_and_locator_reject_access_while_input_window_is_live():
    from psana.gpu.gpu_input import GpuStreamDgramView, InputSlotLease
    from psana.gpu.gpudgram.parser import DeviceFieldLocators
    owner = window()
    planned = owner.acquire()
    lease = InputSlotLease(None, (owner,))
    stream = GpuStreamDgramView(0, 0, owner.batch, owner)
    stream.bind_lease(lease)
    saved_batch = stream.batch
    assert saved_batch.data_gpu is owner.batch.data_gpu
    locator = DeviceFieldLocators(None, np.zeros((1, 2)), Token(), lease)
    lease.wait_until_safe_to_reuse()
    assert not owner.released
    with pytest.raises(RuntimeError, match='released'):
        stream.data_gpu
    with pytest.raises(RuntimeError, match='released'):
        saved_batch.data_gpu
    with pytest.raises(RuntimeError, match='released'):
        locator.rows_gpu
    owner.close()
    planned.wait_until_safe_to_reuse()
    assert owner.batch is None


def test_manager_failed_drain_remains_retryable():
    from psana.gpu.gpu_events import GpuEventManager
    manager = GpuEventManager.__new__(GpuEventManager)
    manager._closed = False
    closed = []
    manager.gpu_reader = NS(close=lambda: closed.append(True))
    manager._drain_pending_gpu_read = lambda: None
    def fail():
        raise RuntimeError('consumer failed')
        yield
    manager._flush_event_pool = fail
    with pytest.raises(RuntimeError):
        manager.close()
    assert not manager._closed and not closed
    manager._flush_event_pool = lambda: iter(())
    manager.close()
    assert manager._closed and closed == [True]


def test_batch_source_advances_outside_exhausted_event_exception():
    from psana.psexp.events import Events
    source_exceptions = []
    def source():
        source_exceptions.append(sys.exc_info()[0])
        yield NS(smd=None, gpu=None)
    events = Events.__new__(Events)
    events._batch_source = source()
    events.shared_state = NS()
    events._evt_man = iter(())
    events._emit_batch_end = lambda: None
    events._gpu_finished = False
    events._batch_event_count = 0
    envelope = NS(dgrams=[object()])
    events.gpu_manager = NS(process_batch=lambda *args: iter((envelope,)))
    assert next(events) is envelope
    assert source_exceptions == [None]


def test_parallel_iterator_close_drains_gpu_manager(monkeypatch):
    from psana.psexp import mpi_ds
    monkeypatch.setattr(mpi_ds, 'nodetype', 'bd')
    closed = []
    manager = NS(close=lambda: closed.append(True))
    envelope = NS(dgrams=[object()])
    run = NS(dsparms=NS(gpu_enabled=True),
             _make_gpu_event_manager=lambda: manager,
             start=lambda **kwargs: iter((envelope,)),
             _handle_transition=lambda dgrams: False,
             _materialize_event=lambda env: env)
    # Setup and teardown are shared with _steps_impl now, so the stand-in
    # needs them bound; they are part of the contract _events_impl relies on.
    run._open_gpu_manager = lambda: mpi_ds.RunParallel._open_gpu_manager(run)
    run._close_gpu_manager = lambda it, mgr: mpi_ds.RunParallel._close_gpu_manager(
        run, it, mgr)
    iterator = mpi_ds.RunParallel._events_impl(run)
    assert next(iterator) is envelope
    iterator.close()
    assert closed == [True]


def _gpu_bd_steps_run(monkeypatch, start, aborts):
    """A RunParallel stand-in on a GPU BD rank whose comm records Abort."""
    from psana.psexp import mpi_ds
    monkeypatch.setattr(mpi_ds, 'nodetype', 'bd')
    comm = NS(Get_rank=lambda: 0, Abort=lambda code: aborts.append(code))
    run = NS(dsparms=NS(gpu_enabled=True), comms=NS(psana_comm=comm),
             _make_gpu_event_manager=lambda: NS(close=lambda: None),
             start=start,
             _update_envstore_from_dgrams=lambda dgrams: None,
             _materialize_event=lambda env: env,
             _run_ctx=object(), esm=None)
    # Bind the real methods lazily, so a method this fix adds is only looked
    # up when the code under test calls it.
    for name in ('_open_gpu_manager', '_close_gpu_manager', '_steps_impl',
                 '_guard_shared_iterator'):
        setattr(run, name,
                lambda *a, _n=name: getattr(mpi_ds.RunParallel, _n)(run, *a))
    return run


def _beginstep_envelope():
    from psana.psexp import TransitionId
    begin = NS(timestamp=lambda: 900,
               service=lambda: TransitionId.BeginStep,
               env=lambda: TransitionId.BeginStep << 24)
    return NS(dgrams=[begin], gpu_state=None)


def test_a_gpu_failure_inside_step_events_aborts_the_mpi_job(monkeypatch):
    """The failure happens on the USER's call stack, not inside run.steps().

    steps() wraps _steps_impl in gpu_error_handler, but step.events() advances
    the shared iterator while that generator is suspended, so an exception
    there bypassed the handler entirely: it reached the user, and closing the
    abandoned steps() generator afterwards showed the handler only
    GeneratorExit. No Abort, and the EB hung. Without _guard_shared_iterator
    this test dies with the RuntimeError and `aborts` stays empty.
    """
    from psana.psexp import mpi_ds

    def start(**kwargs):
        yield _beginstep_envelope()
        raise RuntimeError('KvikIO pread failed')   # the step's first read

    aborts = []
    run = _gpu_bd_steps_run(monkeypatch, start, aborts)
    steps = mpi_ds.RunParallel.steps(run)
    step = next(steps)

    list(step.events())        # the user's loop; must not see the exception

    assert aborts == [1], 'a GPU failure inside step.events() must Abort'


def test_breaking_out_of_a_gpu_step_loop_does_not_abort(monkeypatch):
    """GeneratorExit must still pass through both handler layers: abandoning
    run.steps() is a local decision, not a job failure."""
    from psana.psexp import mpi_ds

    def start(**kwargs):
        yield _beginstep_envelope()
        yield _beginstep_envelope()      # never reached

    aborts = []
    run = _gpu_bd_steps_run(monkeypatch, start, aborts)
    steps = mpi_ds.RunParallel.steps(run)
    next(steps)
    steps.close()

    assert aborts == [], 'closing early must not abort the job'


def test_parallel_steps_iterator_close_drains_gpu_manager(monkeypatch):
    """The same ownership contract for steps() as for events().

    RunParallel.steps() used to return immediately on a GPU BD rank, so there
    was nothing to close. It now shares _events_impl's teardown, and an
    abandoned steps() loop must release the manager (issue #153, GPU-140-03).
    """
    from psana.psexp import mpi_ds
    from psana.psexp import TransitionId
    monkeypatch.setattr(mpi_ds, 'nodetype', 'bd')
    closed = []
    manager = NS(close=lambda: closed.append(True))
    begin = NS(timestamp=lambda: 900,
               service=lambda: TransitionId.BeginStep,
               env=lambda: TransitionId.BeginStep << 24)
    envelope = NS(dgrams=[begin], gpu_state=None)
    run = NS(dsparms=NS(gpu_enabled=True),
             _make_gpu_event_manager=lambda: manager,
             start=lambda **kwargs: iter((envelope,)),
             _update_envstore_from_dgrams=lambda dgrams: None,
             _materialize_event=lambda env: env,
             _run_ctx=object(),
             esm=None)
    run._open_gpu_manager = lambda: mpi_ds.RunParallel._open_gpu_manager(run)
    run._close_gpu_manager = lambda it, mgr: mpi_ds.RunParallel._close_gpu_manager(
        run, it, mgr)
    # No comms on this stand-in, so the guard hands the iterator back as is;
    # the close path below is therefore the unwrapped one.
    run._guard_shared_iterator = (
        lambda it: mpi_ds.RunParallel._guard_shared_iterator(run, it))
    iterator = mpi_ds.RunParallel._steps_impl(run)
    step = next(iterator)
    assert step.evt is envelope, 'one BeginStep, so one Step'
    iterator.close()
    assert closed == [True], 'abandoning steps() must close the manager'
