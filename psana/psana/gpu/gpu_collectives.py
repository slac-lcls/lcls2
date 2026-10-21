"""Bounded collectives for GPU setup.

A setup collective that blocks forever costs an allocation. Ranks taking
different collective paths was the dominant failure mode in this work; every
instance hung rather than erroring, and a hung job holds its nodes until the
wall clock expires with no diagnostic. ``bounded_barrier`` and
``bounded_allreduce`` poll a non-blocking request to a deadline and then
abort, naming the step -- turning a lost allocation into a sixty-second
message.

Only setup and transitions use these. The event loop does not: its collectives
are on the critical path, and a deadline there would trade a hang for a flaky
job.

**Not covered:** calls that carry Python objects, which mpi4py can only send
non-blocking through a two-step size-then-data ``Ibcast``. In shared
constants: the manifest ``bcast`` in ``_publish``/``_subscribe`` and
``_agree_on_set``'s ``allgather`` of declared selector sets. In discovery:
the ``allgather`` of hostnames, of visible PCI sets and of device keys, plus
the ``Split`` calls themselves.

What IS bounded is everything expressible as a NumPy buffer: the barriers,
the discovery *reductions* (node and job agreement, the budget-verdict MAX,
the per-device capacity MIN, the sharing-permitted MIN) and the agreement
vector. So a hang in an object-carrying allgather still hangs; the reductions
that follow it are what report a divergence.

Collective-order checking lives in the tests, in
``psana/psana/tests/gpu/collective_recorder.py``: it is a debugging aid rather
than something production needs, and keeping it there means production holds
no communicator wrapper.
"""
import os
import time


# Warn here; this is long enough that a healthy rendezvous has completed.
DEFAULT_WARN_S = 60.0
# Abort here. A deadline only separates "hung" from "slow" if no legitimate
# skew can reach it, and `release-shared-closed` is a rendezvous where ranks
# arrive whenever their own work finishes: it is the first collective in
# teardown, so arrival depends on when each BD rank finished its last batch.
# With uneven event distribution or a loaded filesystem those can be minutes
# apart, and aborting at 60s would kill an otherwise successful run.
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
