"""Setup collectives abort with a named step instead of hanging.

Eight bugs on this branch came from ranks taking different collective paths,
and every one hung rather than erroring -- holding its nodes until the job
wall clock expired, with no diagnostic. These cover the two mechanisms that
change that: a deadline on setup collectives, and a recorded call order that
unit tests can compare across simulated ranks.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu import gpu_collectives as gc
from psana.gpu.gpu_collectives import (
    CollectiveTimeout, RecordingComm, bounded_allreduce, bounded_barrier,
    checking_enabled, timeout_seconds,
)


@pytest.fixture(autouse=True)
def intercept_abort(monkeypatch):
    """Record aborts instead of terminating the test interpreter.

    _abort calls MPI.COMM_WORLD.Abort, which is correct in a real job --
    the standard only promises best effort on a sub-communicator's group --
    but fatal here. The production hook exists for exactly this.
    """
    monkeypatch.setattr(gc, '_abort_hook', lambda comm: comm.Abort(1))


class Comm:
    """Communicator whose requests complete, or never do."""

    def __init__(self, *, completes=True):
        self.completes = completes
        self.aborted = []
        self.blocking_calls = []

    def Ibarrier(self):
        return NS(Test=lambda: self.completes)

    def Iallreduce(self, sendbuf, recvbuf, op=None):
        if self.completes:
            recvbuf[:] = sendbuf
        return NS(Test=lambda: self.completes)

    def Barrier(self):
        self.blocking_calls.append('Barrier')

    def allreduce(self, value, op=None):
        self.blocking_calls.append('allreduce')
        return value

    def allgather(self, value):
        return [value] * 2

    def Abort(self, code=1):
        self.aborted.append(code)


# ---------------------------------------------------------------------------
# Bounded collectives
# ---------------------------------------------------------------------------

def test_barrier_returns_when_peers_arrive():
    comm = Comm()
    bounded_barrier(comm, 'step', timeout=5)
    assert comm.aborted == []


def test_barrier_aborts_rather_than_hanging():
    """The whole point: a rank whose peers are elsewhere must stop the job
    with a message, not hold the allocation until the wall clock expires."""
    comm = Comm(completes=False)
    with pytest.raises(CollectiveTimeout, match="'late-step'"):
        bounded_barrier(comm, 'late-step', timeout=0.05, warn=0)
    assert comm.aborted == [1]


def test_timeout_message_names_the_step():
    """A hang gives no clue which collective diverged; this must."""
    comm = Comm(completes=False)
    with pytest.raises(CollectiveTimeout) as caught:
        bounded_barrier(comm, 'shared-constants/case-B-drained',
                        timeout=0.05, warn=0)
    assert 'case-B-drained' in str(caught.value)
    assert 'different collectives' in str(caught.value)


def test_slow_but_healthy_collective_warns_and_completes():
    """The failure the fourth review caught: a single short deadline aborts
    healthy runs.

    release-shared-closed is the first collective in teardown, so ranks arrive
    whenever their last batch finished; case-B-drained waits on each rank's
    file I/O. Minutes of legitimate skew must cost a log line, not the job.
    """
    class Slow:
        def __init__(self):
            self.polls = 0
            self.aborted = []

        def Ibarrier(self):
            return NS(Test=self._test)

        def _test(self):
            self.polls += 1
            return self.polls > 3        # completes, just not immediately

        def Abort(self, code=1):
            self.aborted.append(code)

    comm = Slow()
    # With no logger the helper prints, so capture stdout rather than caplog:
    # a GPU worker may have no configured logger and the message must still
    # reach the job's output.
    recorded = []
    logger = NS(warning=lambda m, *a: recorded.append(m % a if a else m),
                error=lambda m, *a: recorded.append(m % a if a else m))
    bounded_barrier(comm, 'release-shared-closed', timeout=30, warn=1e-9,
                    logger=logger)
    assert comm.aborted == []            # healthy: never aborted
    messages = ' '.join(recorded)
    assert 'release-shared-closed' in messages
    assert 'will abort at' in messages   # says what happens next
    assert 'completed after' in messages


def test_abort_deadline_is_far_above_the_warning():
    """A deadline only separates hung from slow if no legitimate skew reaches
    it, so the abort limit must be much larger than the warning."""
    from psana.gpu.gpu_collectives import DEFAULT_TIMEOUT_S, DEFAULT_WARN_S
    assert DEFAULT_WARN_S <= 60
    assert DEFAULT_TIMEOUT_S >= 15 * 60


def test_warning_interval_is_configurable(monkeypatch):
    from psana.gpu.gpu_collectives import warn_seconds
    monkeypatch.setenv('PSANA_GPU_COLLECTIVE_WARN', '5')
    assert warn_seconds() == 5
    monkeypatch.setenv('PSANA_GPU_COLLECTIVE_WARN', 'nonsense')
    assert warn_seconds() == 60.0


def test_buffer_allreduce_is_bounded():
    """The agreement vector was unbounded even though it is one of the
    collectives this branch's hangs occurred in."""
    from psana.gpu.gpu_collectives import bounded_allreduce_buffer
    comm = Comm(completes=False)
    with pytest.raises(CollectiveTimeout, match='settle'):
        bounded_allreduce_buffer(comm, np.array([1]), np.array([0]), 'MAX',
                                 'shared-constants/settle', timeout=0.05,
                                 warn=0)


def test_allreduce_returns_the_reduced_value():
    comm = Comm()
    assert bounded_allreduce(comm, 7, 'SUM', 'step', timeout=5) == 7


def test_allreduce_aborts_rather_than_hanging():
    comm = Comm(completes=False)
    with pytest.raises(CollectiveTimeout):
        bounded_allreduce(comm, 1, 'SUM', 'step', timeout=0.05, warn=0)
    assert comm.aborted == [1]


def test_zero_timeout_uses_the_blocking_form():
    """An escape hatch: a deadline on the event loop would trade a hang for a
    flaky job, so bounding must be switchable off."""
    comm = Comm()
    bounded_barrier(comm, 'step', timeout=0)
    bounded_allreduce(comm, 3, 'SUM', 'step', timeout=0)
    assert comm.blocking_calls == ['Barrier', 'allreduce']


def test_timeout_is_configurable(monkeypatch):
    monkeypatch.setenv('PSANA_GPU_COLLECTIVE_TIMEOUT', '12.5')
    assert timeout_seconds() == 12.5
    from psana.gpu.gpu_collectives import DEFAULT_TIMEOUT_S
    monkeypatch.setenv('PSANA_GPU_COLLECTIVE_TIMEOUT', 'not-a-number')
    assert timeout_seconds() == DEFAULT_TIMEOUT_S   # falls back, never raises
    monkeypatch.delenv('PSANA_GPU_COLLECTIVE_TIMEOUT')
    assert timeout_seconds() == DEFAULT_TIMEOUT_S


def test_checking_is_opt_in(monkeypatch):
    monkeypatch.delenv('PSANA_GPU_CHECK_COLLECTIVES', raising=False)
    assert not checking_enabled()
    monkeypatch.setenv('PSANA_GPU_CHECK_COLLECTIVES', '0')
    assert not checking_enabled()
    monkeypatch.setenv('PSANA_GPU_CHECK_COLLECTIVES', '1')
    assert checking_enabled()


# ---------------------------------------------------------------------------
# Order recording
# ---------------------------------------------------------------------------

def test_recorder_delegates_and_records():
    comm = RecordingComm(Comm())
    comm.Barrier()
    comm.allreduce(1, op='SUM')
    comm.allgather('x')
    assert comm.calls == ['Barrier', 'allreduce', 'allgather']


def test_recorder_passes_non_collective_attributes_through():
    comm = RecordingComm(NS(Get_rank=lambda: 3, size=8,
                            allgather=lambda v: [v]))
    assert comm.Get_rank() == 3 and comm.size == 8
    assert comm.calls == []               # not a collective, not recorded


def test_identical_paths_agree():
    class Agreeing(Comm):
        def allgather(self, value):
            return [value, value]

    comm = RecordingComm(Agreeing())
    comm.Barrier()
    comm.Barrier()
    assert comm.check_agreement('establish') is True


def test_divergent_paths_are_detected():
    """The failure mode the unit tests could not see before: one rank skipping
    a collective the others make."""
    class Diverging(Comm):
        def allgather(self, value):
            # A peer made three calls where this rank made two.
            return [value, ('deadbeefdeadbeef', 3)]

    comm = RecordingComm(Diverging())
    comm.Barrier()
    comm.Barrier()
    assert comm.check_agreement('establish') is False


def test_check_agreement_does_not_record_itself():
    class Agreeing(Comm):
        def allgather(self, value):
            return [value, value]

    comm = RecordingComm(Agreeing())
    comm.Barrier()
    comm.check_agreement('step')
    assert comm.calls == ['Barrier']      # the check used the raw comm


def test_digest_is_order_sensitive():
    """Reordering is as much a divergence as omitting."""
    class Agreeing(Comm):
        def allgather(self, value):
            return [value, value]

    first = RecordingComm(Agreeing())
    first.Barrier()
    first.allreduce(1, op='SUM')

    second = RecordingComm(Agreeing())
    second.allreduce(1, op='SUM')
    second.Barrier()

    assert first.sequence_digest() != second.sequence_digest()
