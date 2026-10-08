"""Bounded collectives for GPU setup, and an optional order checker.

Two problems these address, both learned the hard way on this branch.

**A setup collective that blocks forever costs an allocation.** Eight bugs in
this work came from ranks taking different collective paths; every one hung
rather than erroring, and a hung job holds its nodes until the wall clock
expires with no diagnostic. ``bounded_barrier`` and ``bounded_allreduce`` poll
a non-blocking request to a deadline and then abort, naming the step -- turning
a lost allocation into a sixty-second message.

Only setup and transitions use these. The event loop does not: its collectives
are on the critical path, and a deadline there would trade a hang for a flaky
job.

**Not covered:** three calls carry Python objects, which mpi4py can only send
non-blocking through a two-step size-then-data ``Ibcast``, so they remain
unbounded:

* the manifest ``bcast`` in ``_publish``/``_subscribe``
* ``_intersect``'s ``allgather`` of declared selector sets
* the Case B digest ``bcast`` in ``_refresh_in_place``

Everything expressible as a NumPy buffer -- barriers, the discovery
reductions, the agreement vector and the case code -- is bounded.

**Unit tests cannot see path divergence.** The test fake drives peers
sequentially, so it has no notion of two ranks blocked in different calls --
exactly the failure mode. ``RecordingComm`` wraps a communicator and records
the sequence of collective names; comparing those sequences across ranks
catches a path that skips or reorders a step. Enable with
``PSANA_GPU_CHECK_COLLECTIVES=1``.

The recorder raises confidence rather than proving correctness: a sequential
fake still cannot model true concurrency, so the hardware failure cases remain
the only real proof.
"""
import os
import time


# Warn here; this is long enough that a healthy rendezvous has completed.
DEFAULT_WARN_S = 60.0
# Abort here. A deadline only separates "hung" from "slow" if no legitimate
# skew can reach it. Two of the bounded steps are rendezvous points where
# ranks arrive whenever their own work finishes:
#
#   release-shared-closed  first collective in teardown, so arrival depends on
#                          when each BD rank finished its last batch
#   case-B-drained         gap is each rank's before_upload() drain, including
#                          in-flight file I/O
#
# With uneven event distribution or a loaded filesystem those can be minutes
# apart. Aborting at 60s would kill an otherwise successful run.
DEFAULT_TIMEOUT_S = 1800.0

# Discovery runs before any event work, so its ranks arrive within
# milliseconds of each other -- no legitimate skew to absorb. A hang there is
# a genuine divergence and need not cost half an hour to report.
SETUP_TIMEOUT_S = 120.0

POLL_S = 0.01


def timeout_seconds():
    """Abort deadline for setup collectives. ``0`` disables bounding."""
    raw = os.environ.get('PSANA_GPU_COLLECTIVE_TIMEOUT', '')
    try:
        return float(raw) if raw else DEFAULT_TIMEOUT_S
    except ValueError:
        return DEFAULT_TIMEOUT_S


def warn_seconds():
    """When to log that a collective is slow, without acting on it.

    ``0`` disables the warning while leaving the abort deadline in place.
    """
    raw = os.environ.get('PSANA_GPU_COLLECTIVE_WARN', '')
    try:
        return float(raw) if raw else DEFAULT_WARN_S
    except ValueError:
        return DEFAULT_WARN_S


def checking_enabled():
    return os.environ.get('PSANA_GPU_CHECK_COLLECTIVES', '') not in ('', '0')


class CollectiveTimeout(RuntimeError):
    """A setup collective did not complete within its deadline."""


def _report(message, logger, error=False):
    if logger is not None:
        (logger.error if error else logger.warning)(message)
    else:
        kind = 'ERROR' if error else 'WARNING'
        print(f'[PSANA-GPU-{kind}] {message}', flush=True)


# Set by tests to intercept the abort. An explicit hook rather than
# inspecting ``Comm.Abort.__module__``: that worked on this build (it reports
# 'mpi4py.MPI'), but a compiled build reporting no module would misclassify
# the real sub-communicator as a test double and abort only its own group.
_abort_hook = None


def _abort(comm):
    """Stop the whole job.

    ``MPI.COMM_WORLD`` rather than the communicator passed in: the standard
    only promises a best effort to abort the given communicator's group --
    Open MPI kills everything, other implementations need not -- and the
    communicator here is often a sub-communicator such as ``device_comm``.
    """
    if _abort_hook is not None:
        _abort_hook(comm)
        return
    try:
        from mpi4py import MPI
        MPI.COMM_WORLD.Abort(1)
    except Exception:                                     # noqa: BLE001
        try:
            comm.Abort(1)
        except Exception:                                 # noqa: BLE001
            pass


def _wait(request, step, comm, timeout, logger, warn=None):
    """Poll a non-blocking request: warn when slow, abort when hung.

    Abort rather than raise: a rank reaching the deadline means its peers are
    elsewhere, so raising locally would leave them blocked.

    See ``_abort`` for which communicator is aborted and why.
    """
    warn = warn_seconds() if warn is None else warn
    started = time.monotonic()
    warned = False
    while True:
        if request.Test():
            if warned:
                _report(f'GPU setup collective {step!r} completed after '
                        f'{time.monotonic() - started:.0f}s.', logger)
            return
        waited = time.monotonic() - started
        if not warned and warn and waited >= warn:
            warned = True
            _report(f'GPU setup collective {step!r} has waited {waited:.0f}s. '
                    f'Still waiting; will abort at {timeout:.0f}s. Peers may '
                    'be skewed by uneven work, or in different collectives.',
                    logger)
        if waited >= timeout:
            message = (f'GPU setup collective {step!r} did not complete within '
                       f'{timeout:.0f}s. Ranks are in different collectives; '
                       'this would otherwise hang until the job wall clock '
                       'expired.')
            _report(message, logger, error=True)
            _abort(comm)
            raise CollectiveTimeout(message)
        time.sleep(POLL_S)


def bounded_barrier(comm, step, *, timeout=None, logger=None, warn=None):
    """``comm.Barrier()`` that warns when slow and aborts when hung."""
    timeout = timeout_seconds() if timeout is None else timeout
    if not timeout:
        comm.Barrier()
        return
    _wait(comm.Ibarrier(), step, comm, timeout, logger, warn)


def bounded_allreduce(comm, value, op, step, *, timeout=None, logger=None,
                      warn=None):
    """``comm.allreduce()`` that aborts instead of hanging.

    Uses the buffer form through NumPy so the request can be polled; the
    Python-object form has no non-blocking equivalent in mpi4py.
    """
    timeout = timeout_seconds() if timeout is None else timeout
    if not timeout:
        return comm.allreduce(value, op=op)
    import numpy as np
    send = np.array([value], dtype=np.int64)
    recv = np.zeros_like(send)
    _wait(comm.Iallreduce(send, recv, op=op), step, comm, timeout, logger, warn)
    return int(recv[0])


def bounded_allreduce_buffer(comm, send, recv, op, step, *, timeout=None,
                             logger=None, warn=None):
    """Buffer-form ``Allreduce`` that warns when slow and aborts when hung.

    For callers that already hold NumPy buffers -- the agreement vector in
    shared constants, which was previously unbounded even though it is one of
    the collectives this branch's hangs occurred in.
    """
    timeout = timeout_seconds() if timeout is None else timeout
    if not timeout:
        comm.Allreduce(send, recv, op=op)
        return recv
    _wait(comm.Iallreduce(send, recv, op=op), step, comm, timeout, logger, warn)
    return recv


class RecordingComm:
    """Communicator wrapper that records the order of collective calls.

    Delegates everything to the wrapped communicator and appends a label per
    call. ``sequence_digest`` summarises the calls made so far; comparing that
    across ranks detects a path that skipped or reordered a step -- which is
    what every hang on this branch turned out to be.
    """

    _COLLECTIVES = ('bcast', 'allgather', 'allreduce', 'Allreduce', 'Barrier',
                    'Ibarrier', 'Iallreduce', 'gather', 'Split', 'Free')

    def __init__(self, comm, label=''):
        self._comm = comm
        self._label = label
        self.calls = []

    def __getattr__(self, name):
        attribute = getattr(self._comm, name)
        if name not in self._COLLECTIVES or not callable(attribute):
            return attribute

        def recorded(*args, **kwargs):
            self.calls.append(name)
            return attribute(*args, **kwargs)

        return recorded

    def sequence_digest(self):
        """Stable summary of the calls made so far."""
        from hashlib import blake2b
        hasher = blake2b(digest_size=8)
        hasher.update('|'.join(self.calls).encode())
        return hasher.hexdigest()

    def check_agreement(self, step, logger=None):
        """Confirm every rank made the same calls in the same order.

        Called at a point all ranks reach. Uses the wrapped communicator
        directly so the check does not record itself.
        """
        digest = self.sequence_digest()
        everyone = self._comm.allgather((digest, len(self.calls)))
        if len(set(everyone)) == 1:
            return True
        message = (f'collective order diverged at {step!r}: '
                   f'{sorted(set(everyone))}. This rank made {self.calls}.')
        if logger is not None:
            logger.error(message)
        else:
            print(f'[PSANA-GPU-ERROR] {message}', flush=True)
        return False
