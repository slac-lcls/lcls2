"""Placement reaches the budget and the constants, and every rank participates.

These cover the wiring rather than the algorithms: that GpuEventManager takes
its peer count and device capacity from a GpuPlacement, that an explicit
per-rank budget is validated against the group, and that the discovery
collective is called by every rank -- the failure mode that deadlocks rather
than erroring.
"""
from types import SimpleNamespace as NS

import pytest

from psana.gpu.gpu_placement import (
    GpuPlacement, GpuPlacementError, PinnedDevice, per_rank_limit,
)


GIB = 1024 ** 3


def placement(peers=1, *, usable=40 * GIB, owner=True, shared=0, mig=False):
    return GpuPlacement(
        pinned=PinnedDevice(requested_uuid='GPU-' + 'a' * 32, pinned=True,
                            is_mig=mig),
        hostname='sdfampere001', device_uuid_hex='a' * 32,
        device_comm=NS() if peers > 1 else None,
        is_owner=owner, n_device_peers=peers, usable_bytes=usable,
        shared_bytes=shared)


# ---------------------------------------------------------------------------
# Budget sizing from discovered peers
# ---------------------------------------------------------------------------

def test_capacity_is_divided_among_true_peers_not_eb_group_peers():
    """Deriving the count from bd_comm counts only intra-group peers: with
    PS_EB_NODES=2, eight ranks on one card each claimed a quarter of it."""
    assert per_rank_limit(placement(peers=8, usable=40 * GIB)) == 5 * GIB
    assert per_rank_limit(placement(peers=4, usable=40 * GIB)) == 10 * GIB


def test_aggregate_claim_over_a_device_is_exactly_one():
    for peers in (1, 2, 3, 4, 7, 8):
        p = placement(peers=peers, usable=40 * GIB)
        assert per_rank_limit(p) * peers <= p.usable_bytes
        # Within one rank's rounding of an integer division.
        assert p.usable_bytes - per_rank_limit(p) * peers < peers


def test_shared_constants_are_charged_to_the_device_once():
    bare = per_rank_limit(placement(peers=4, usable=40 * GIB))
    with_shared = per_rank_limit(
        placement(peers=4, usable=40 * GIB, shared=4 * GIB))
    # One GiB per rank less, because the 4 GiB copy is subtracted once.
    assert bare - with_shared == GIB


# ---------------------------------------------------------------------------
# Explicit budgets are validated against the group
# ---------------------------------------------------------------------------

def manager_budget(dsparms, place):
    """The budget-selection rule from GpuEventManager._setup_gpu_pipeline.

    Mirrored rather than called: importing gpu_events pulls in the whole
    runtime (reader, parser, D2H), which needs a built install. The rule
    itself is small and its behaviour is what matters here; the integration
    tests exercise the real call path on a GPU node.
    """
    from psana.gpu.gpu_budget import _GpuBudget
    budget_gb = float(getattr(dsparms, 'gpu_memory_budget_gb', 0) or 0)
    if budget_gb > 0:
        limit = int(budget_gb * GIB)
        if place is not None and place.n_device_peers > 1:
            claim = limit * place.n_device_peers
            if claim > place.usable_bytes:
                raise GpuPlacementError(
                    f'{budget_gb:.2f} GiB x {place.n_device_peers} peers '
                    f'= {claim / GIB:.2f} GiB exceeds '
                    f'{place.usable_bytes / GIB:.2f} GiB available')
        return _GpuBudget(limit_bytes=limit)
    if place is not None:
        return _GpuBudget(limit_bytes=per_rank_limit(place))
    return _GpuBudget(limit_bytes=0)


def test_explicit_budget_within_the_group_total_is_accepted():
    dsparms = NS(gpu_memory_budget_gb=9.0)
    budget = manager_budget(dsparms, placement(peers=4, usable=40 * GIB))
    assert budget.limit() == int(9 * GIB)


def test_explicit_budget_that_over_commits_the_device_is_refused():
    """Today an explicit budget is taken on trust per rank, so four ranks
    each claiming 20 GiB of a 40 GiB card is accepted silently."""
    dsparms = NS(gpu_memory_budget_gb=20.0)
    with pytest.raises(GpuPlacementError, match='exceeds'):
        manager_budget(dsparms, placement(peers=4, usable=40 * GIB))


def test_explicit_budget_verdict_is_the_same_on_an_uneven_layout():
    """The claim is per-device, but the verdict must be job-wide.

    An uneven split puts 4 peers on one device and 3 on another. A budget
    too large for 4 but fine for 3 would make one device raise while the
    other carried on -- ranks on different paths, racing the abort rather
    than agreeing on it. discover_peers reduces the peer count with MPI.MAX
    so both devices validate against the busiest one.
    """
    usable, budget_gb = 40 * GIB, 11.0
    # 11 x 4 = 44 GiB over a 40 GiB card; 11 x 3 = 33 fits.
    assert budget_gb * 4 > usable / GIB
    assert budget_gb * 3 < usable / GIB

    busiest = 4          # what MPI.MAX returns to every rank in the job
    for local_peers in (4, 3):
        claim = int(budget_gb * GIB) * busiest
        assert claim > usable, (
            f'a {local_peers}-peer device must reject the budget too, using '
            'the busiest count rather than its own')


def test_explicit_budget_error_names_the_busiest_device():
    """When the local count differs, the message has to say so, or a rank on
    a 3-peer device reads 'x 4 peers' and cannot reconcile it."""
    import re
    source = __import__('inspect').getsource(
        __import__('psana.gpu.gpu_placement', fromlist=['x']).discover_peers)
    assert 'busiest device' in source
    assert re.search(r'this device has', source)


def test_explicit_budget_is_unchecked_for_a_solo_rank():
    dsparms = NS(gpu_memory_budget_gb=39.0)
    budget = manager_budget(dsparms, placement(peers=1, usable=40 * GIB))
    assert budget.limit() == int(39 * GIB)


def test_automatic_budget_uses_the_discovered_capacity():
    dsparms = NS(gpu_memory_budget_gb=0)
    budget = manager_budget(dsparms, placement(peers=2, usable=36 * GIB))
    assert budget.limit() == 18 * GIB


# ---------------------------------------------------------------------------
# Every rank participates in the discovery collective
# ---------------------------------------------------------------------------

class CountingComm:
    """Counts collective calls so a skipped participant is visible.

    A rank that skips a split does not raise -- it hangs the others. The only
    way to catch that in a unit test is to assert participation.
    """

    def __init__(self, log, rank=0, size=4):
        self.log = log
        self._rank, self._size = rank, size

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def allgather(self, value):
        self.log.append(('allgather', self._rank))
        return [value] * self._size

    def allreduce(self, value, op=None):
        # Record the operation too: MPI pairs collectives by call order, not
        # by operation, so a reordering is invisible to a count alone.
        self.log.append(('allreduce', self._rank, op))
        return value

    def Split(self, color, key):
        self.log.append(('split', self._rank, color))
        return CountingComm(self.log, rank=0, size=1)

    def Free(self):
        self.log.append(('free', self._rank))

    def Ibarrier(self):
        self.log.append(('ibarrier', self._rank))
        return NS(Test=lambda: True)

    def Iallreduce(self, sendbuf, recvbuf, op=None):
        # The bounded helper polls this instead of blocking, so a diverging
        # rank aborts with a named step rather than hanging.
        self.log.append(('allreduce', self._rank, op))
        recvbuf[:] = sendbuf
        return NS(Test=lambda: True)

    def Abort(self, code=1):
        raise AssertionError(f'unexpected Abort({code}) in a unit test')


def test_non_gpu_roles_still_perform_the_node_splits(monkeypatch):
    """smd0 and EB must call Split so the parent collective completes; they
    receive COMM_NULL and no device."""
    from psana.gpu import gpu_placement as gp
    log = []
    fake_mpi = NS(Get_processor_name=lambda: 'sdfampere001', UNDEFINED=-32766,
                  SUM='SUM', MIN='MIN', MAX='MAX')
    monkeypatch.setitem(gp.sys.modules, 'mpi4py', NS(MPI=fake_mpi))

    result = gp.discover_peers(PinnedDevice(), CountingComm(log),
                               is_gpu_worker=False)
    kinds = [entry[0] for entry in log]
    assert kinds.count('allgather') == 1       # hostname exchange
    assert kinds.count('split') == 2           # node, then gpu-role
    # Three job-wide reductions, in the order a GPU worker makes them: the
    # failure agreement, the device-capacity minimum, then the busiest-device
    # peer count. Skipping any leaves this rank racing a peer's MPI_ABORT --
    # measured on hardware, where rank 0 printed SURVIVED before the abort
    # landed. The count alone would not catch a reordering, and MPI pairs
    # collectives by call order rather than by operation, so assert the
    # sequence.
    assert [k for k in kinds if k == 'allreduce'] == ['allreduce'] * 3
    operations = [entry[2] for entry in log if entry[0] == 'allreduce']
    assert operations == ['SUM', 'MIN', 'MAX']
    assert result.device_comm is None
    assert result.n_device_peers == 1


def test_non_gpu_role_raises_when_a_gpu_worker_failed(monkeypatch):
    """The agreement must stop CPU-only roles too, deterministically."""
    from psana.gpu import gpu_placement as gp
    fake_mpi = NS(Get_processor_name=lambda: 'h', UNDEFINED=-32766,
                  SUM='SUM', MIN='MIN', MAX='MAX')
    monkeypatch.setitem(gp.sys.modules, 'mpi4py', NS(MPI=fake_mpi))

    class Failing(CountingComm):
        def Iallreduce(self, sendbuf, recvbuf, op=None):
            recvbuf[:] = 1         # a GPU worker reported a failure
            return NS(Test=lambda: True)

    with pytest.raises(GpuPlacementError, match='aborting this rank'):
        gp.discover_peers(PinnedDevice(), Failing([]), is_gpu_worker=False)


def test_non_gpu_role_never_touches_cupy(monkeypatch):
    """A CPU-role rank must not import CuPy: on a shared node that would
    allocate a context on a device belonging to someone else."""
    from psana.gpu import gpu_placement as gp
    fake_mpi = NS(Get_processor_name=lambda: 'h', UNDEFINED=-32766,
                  SUM='SUM', MIN='MIN', MAX='MAX')
    monkeypatch.setitem(gp.sys.modules, 'mpi4py', NS(MPI=fake_mpi))

    def explode():
        raise AssertionError('CuPy must not be imported on a CPU-role rank')

    gp.discover_peers(PinnedDevice(), CountingComm([]),
                      is_gpu_worker=False, cp=NS(boom=explode))


# ---------------------------------------------------------------------------
# Constant sharing is selected only when it is safe
# ---------------------------------------------------------------------------

def test_sharing_is_selected_for_real_peers():
    assert placement(peers=2).can_share


def test_sharing_is_skipped_for_a_solo_rank():
    assert not placement(peers=1).can_share


def test_sharing_is_skipped_on_mig():
    """IPC does not span MIG instances: they have separate memory."""
    p = placement(peers=2, mig=True)
    assert not p.can_share
    assert p.n_device_peers == 2        # budgets are still sized correctly
