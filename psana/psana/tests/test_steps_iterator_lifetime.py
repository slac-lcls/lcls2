"""A CPU Step keeps its events after its parent run.steps() iterator is gone.

`Step` shares the event iterator with the `run.steps()` generator that yielded
it. On the CPU path the pre-#153 code never closed that iterator, so

    step = next(run.steps())
    events = list(step.events())

delivered every event even though the temporary parent was collected right
after `next()`. #153 routed CPU ranks through `_steps_impl`, whose finally
called `_close_gpu_manager`, which closed the iterator unconditionally -- the
Step then held an exhausted iterator and yielded nothing (PR #299 review, C1).

The GPU path is different on purpose: closing the parent releases the GPU
manager, and that is terminal for the stream (see limitations.md and
test_gpu_retirement.py / test_gpu_serial_close.py for the GPU controls). These
tests pin the CPU lifetime only, driving the real RunParallel.steps() on a
CPU BD stand-in with a synthetic two-event step.
"""
import gc
from types import SimpleNamespace as NS

import pytest

from psana.event import EventEnvelope
from psana.psexp import TransitionId as T

BS, L1, ES, ER = T.BeginStep, T.L1Accept, T.EndStep, T.EndRun


class _Dgram:
    """Methods, not instance attributes: Event._complete() walks a dgram's
    __dict__ for detector names and skips only underscore-prefixed keys."""

    def __init__(self, service, ts):
        self._service, self._ts = service, ts

    def env(self):
        return self._service << 24

    def timestamp(self):
        return self._ts

    def service(self):
        return self._service


def _envelope(service, ts):
    return EventEnvelope(dgrams=[_Dgram(service, ts)], gpu_state=None)


def _records():
    # One step holding two events, then the run ends.
    for service, ts in ((BS, 900), (L1, 1000), (L1, 1100), (ES, 1200), (ER, 1300)):
        yield _envelope(service, ts)


@pytest.fixture
def cpu_run(monkeypatch):
    """A RunParallel stand-in on a CPU BD rank: no GPU, no communicator."""
    from psana.psexp import mpi_ds
    monkeypatch.setattr(mpi_ds, 'nodetype', 'bd')
    run = NS(dsparms=NS(gpu_enabled=False),
             start=lambda **kwargs: _records(),
             _update_envstore_from_dgrams=lambda dgrams: None,
             _materialize_event=lambda env: env,
             _run_ctx=object(), esm=None)
    for name in ('_open_gpu_manager', '_close_gpu_manager', '_steps_impl',
                 '_guard_shared_iterator'):
        setattr(run, name,
                lambda *a, _n=name: getattr(mpi_ds.RunParallel, _n)(run, *a))
    return run


def _timestamps(events):
    return [int(e._dgrams[0].timestamp()) for e in events]


def test_a_step_outlives_a_temporary_parent(cpu_run):
    """monarin's reproduction: 2 events at the pre-#153 baseline, 0 at the
    unfixed head."""
    from psana.psexp import mpi_ds

    step = next(mpi_ds.RunParallel.steps(cpu_run))   # parent unreferenced here
    gc.collect()

    assert _timestamps(step.events()) == [1000, 1100], (
        'releasing the parent run.steps() generator must not exhaust the '
        "Step's shared iterator on a CPU rank")


def test_a_step_outlives_an_explicitly_released_parent(cpu_run):
    from psana.psexp import mpi_ds

    steps = mpi_ds.RunParallel.steps(cpu_run)
    step = next(steps)
    del steps
    gc.collect()

    assert _timestamps(step.events()) == [1000, 1100]


def test_control_retained_parent(cpu_run):
    from psana.psexp import mpi_ds

    steps = mpi_ds.RunParallel.steps(cpu_run)
    step = next(steps)
    assert _timestamps(step.events()) == [1000, 1100]
    assert list(steps) == [], 'one BeginStep, so no second step'


def test_control_nested_loop(cpu_run):
    from psana.psexp import mpi_ds

    seen = []
    for step in mpi_ds.RunParallel.steps(cpu_run):
        seen.append(_timestamps(step.events()))
    assert seen == [[1000, 1100]]
