"""Steps and GPU events together on a real device (issue #153, PR B).

Steps 1-4 of PR B on hardware: the step scope each submission carries, the
ordering of transitions against the events they follow, and the fact that a
BeginStep no longer drains the pool.

What makes this different from the CPU unit tests covering the same logic:
a real GpuTask runs, a real kernel reads the scope and writes per-event
output, and the EventPool holds real CUDA streams -- so a scope that were
wrong, or a transition delivered out of order, would show up as wrong device
data rather than a wrong Python value.

Marked `gpu`; skipped without a CUDA device.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_batched_gather import _gpu_available
from test_gpu_producer_device import bindings, envelopes, fixture
from psana.gpu import GpuTask, gpu_events
from psana.gpu.gpu_plan import StepScope, plan_packet
from psana.gpu.gpu_stream import EventPool
from psana.psexp import TransitionId

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not _gpu_available(), reason='no CUDA device')]


# Writes the submission's step index into every event row, on the device.
_STAMP_KERNEL = r'''
extern "C" __global__
void stamp(unsigned int* out, unsigned long long n, unsigned int step_index) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x
                         + threadIdx.x;
    if (i < n) out[i] = step_index;
}
'''


def _transition(service, timestamp):
    """One (service, dgrams) pair as _iter_step_events yields it.

    `service=lambda s=service: s` binds per call: a bare closure over a loop
    variable reports the last service for every dgram.
    """
    dg = NS(timestamp=lambda ts=timestamp: ts,
            service=lambda s=service: s)
    return (service, [dg])


def _manager(pool, mapping, task, budget, detector):
    m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
    m.event_pool = pool
    m._first_batch_logged = True
    m.gpu_det_names = ['camera']
    m.gpu_detector_bindings = mapping
    m.configs = []
    m._step_generation, m._step_begin_ts = 0, 0
    m._gpu_task = task
    m._gpu_budget = budget
    m._task_constants = None
    m.input_preparers = {'camera.raw': detector}
    m.run = NS(runnum=77, _handle_transition=lambda _: None)
    return m


def test_a_kernel_stamps_the_step_the_events_were_recorded_in():
    """Each submission's device output carries its own scope index.

    The scope travels submission -> BatchInputContext -> callback -> kernel
    -> published rows. Reading it back from the publication is the end of
    that chain, so a break anywhere in it shows up here.
    """
    import cupy as cp
    parser, detector, budget, specs, expected, window = fixture(cp, ((0,),) * 3)
    mapping = bindings(parser, detector)
    kernel = cp.RawKernel(_STAMP_KERNEL, 'stamp')
    kernel.compile()
    seen = []

    def callback(batch, stream):
        scope = batch.step_scope
        assert scope is not None, 'no scope reached the callback'
        assert batch.step_generation == scope.index
        seen.append(scope.index)
        out = cp.zeros((batch.size,), cp.uint32)
        kernel((1,), (max(1, batch.size),),
               (out, np.uint64(batch.size), np.uint32(scope.index)),
               stream=stream)
        batch.publish('step_index', out)

    task = GpuTask(callback, ['camera.raw'], [])
    pool = EventPool(n=1, budget=budget)

    try:
        for index in (0, 2, 5):      # non-contiguous: skipped/empty steps
            scope = StepScope(run=77, index=index, begin_ts=1000 * index)
            record = pool.submit(
                NS(iter_events=lambda: iter(specs)), None, envelopes(specs),
                {'camera.raw': detector}, input_windows=(window,), batch_id=7,
                task=task, detector_bindings=mapping, task_constants=None,
                run=77, step_generation=scope.index, step_scope=scope)
            assert pool.begin_retire_next() is record
            for _ts, outputs in record.publications_by_ts.items():
                stamped = outputs['step_index'].array.get()
                assert int(stamped) == index, (
                    f'event stamped step {int(stamped)}, expected {index}')
            pool.finish_retire_next()
    finally:
        list(pool.flush())
        window.close()

    assert seen == [0, 2, 5]


def test_events_before_a_trailing_beginstep_are_stamped_with_the_old_step():
    """GPU-148-01 on hardware, through the real planner.

    A packet [events..., BeginStep] is planned, the events are submitted with
    the scope the planner resolved, and the kernel stamps that scope. Before
    PR B the manager's generation had already advanced past the trailing
    BeginStep by submission time, so these events would be stamped 1.
    """
    import cupy as cp
    parser, detector, budget, specs, expected, window = fixture(cp, ((0,),) * 3)
    mapping = bindings(parser, detector)
    kernel = cp.RawKernel(_STAMP_KERNEL, 'stamp')
    kernel.compile()
    stamped = []

    def callback(batch, stream):
        out = cp.zeros((batch.size,), cp.uint32)
        kernel((1,), (max(1, batch.size),),
               (out, np.uint64(batch.size),
                np.uint32(batch.step_scope.index)), stream=stream)
        batch.publish('step_index', out)
        stamped.append(batch.step_scope.index)

    task = GpuTask(callback, ['camera.raw'], [])
    pool = EventPool(n=1, budget=budget)
    m = _manager(pool, mapping, task, budget, detector)

    # The packet's events are at timestamps 100..102 (the fixture's) and the
    # BeginStep follows them at 900, which is what makes it "trailing".
    gpu_view = NS(events=[{'timestamp': ts} for ts in (100, 101, 102)],
                  header=NS(n_events=3))
    plan = plan_packet(gpu_view, [_transition(TransitionId.BeginStep, 900)],
                       run=77, generation=0, begin_ts=0)
    assert plan.scope.index == 0, 'the planner must keep the previous scope'
    assert plan.transitions[0][3] == 3, 'the BeginStep follows all 3 events'

    try:
        record = pool.submit(
            NS(iter_events=lambda: iter(specs)), None, envelopes(specs),
            {'camera.raw': detector}, input_windows=(window,), batch_id=7,
            task=task, detector_bindings=mapping, task_constants=None,
            run=77, step_generation=plan.scope.index, step_scope=plan.scope)
        assert pool.begin_retire_next() is record
        for _ts, outputs in record.publications_by_ts.items():
            assert int(outputs['step_index'].array.get()) == 0, (
                'an event before a trailing BeginStep must carry step 0')
        pool.finish_retire_next()
    finally:
        list(pool.flush())
        window.close()

    assert stamped == [0]
    # And the generation still advances, for the next packet's events.
    assert plan.next_generation == 1


def test_a_beginstep_does_not_drain_the_pool(monkeypatch):
    """Step 4's overlap win, observed on a real pool.

    Before PR B, _handle_steps flushed the EventPool at every BeginStep --
    a pipeline bubble at each step boundary. Now only EndRun drains, so work
    submitted before a BeginStep is still in flight after it.
    """
    import cupy as cp
    parser, detector, budget, specs, _, window = fixture(cp, ((0,),) * 3)
    mapping = bindings(parser, detector)

    def callback(batch, stream):
        batch.publish('ones', cp.ones((batch.size,), cp.uint32))

    task = GpuTask(callback, ['camera.raw'], [])
    pool = EventPool(n=2, budget=budget)
    m = _manager(pool, mapping, task, budget, detector)
    monkeypatch.setattr(gpu_events, '_iter_step_events',
                        lambda packet, configs: iter(packet))

    try:
        pool.submit(
            NS(iter_events=lambda: iter(specs)), None, envelopes(specs),
            {'camera.raw': detector}, input_windows=(window,), batch_id=7,
            task=task, detector_bindings=mapping, task_constants=None,
            run=77, step_generation=0,
            step_scope=StepScope(run=77, index=0, begin_ts=0))
        assert pool.active_count == 1

        delivered = list(m._handle_steps(
            {0: ([_transition(TransitionId.BeginStep, 900)], [])}))

        # The BeginStep is queued behind the in-flight submission, so it is
        # NOT yielded yet -- and crucially the submission is still active.
        assert pool.active_count == 1, (
            'BeginStep drained the pool; the step-boundary bubble is back')
        assert delivered == [], 'the BeginStep must wait for its events'

        # EndRun does drain, and releases both transitions in order.
        rest = list(m._handle_steps(
            {0: ([_transition(TransitionId.EndRun, 9000)], [])}))
        assert pool.active_count == 0
        assert [e.dgrams[0].service() for e in rest if e.gpu_state is None] == [
            TransitionId.BeginStep, TransitionId.EndRun]
    finally:
        list(pool.flush())
        window.close()
