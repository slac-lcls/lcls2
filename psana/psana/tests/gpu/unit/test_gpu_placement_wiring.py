"""Placement reaches the budget and the constants, and every rank participates.

These cover the wiring rather than the algorithms: that GpuEventManager takes
its peer count and device capacity from a GpuPlacement, that an explicit
per-rank budget is validated against the group, and that the discovery
collective is called by every rank -- the failure mode that deadlocks rather
than erroring.
"""
import sys
from types import SimpleNamespace as NS

import pytest

from psana.gpu.gpu_placement import (
    GpuPlacement, GpuPlacementError, per_rank_limit,
)


GIB = 1024 ** 3


def placement(peers=1, *, usable=40 * GIB, owner=True, shared=0):
    return GpuPlacement(
        hostname='sdfampere001', device_pci='0:01:00.0',
        device_comm=NS() if peers > 1 else None,
        is_owner=owner, n_device_peers=peers, usable_bytes=usable,
        shared_bytes=shared)


# ---------------------------------------------------------------------------
# Budget sizing from discovered peers
# ---------------------------------------------------------------------------

def test_capacity_is_divided_among_true_peers_not_eb_group_peers():
    """The divisor is the device's real peer count. Deriving it from bd_comm
    would count only intra-group peers, so with PS_EB_NODES=2 eight ranks on
    one card would each claim a quarter of it."""
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

def discover_with_budget(budget_gb, *, local_peers, busiest, usable,
                         sharing_permitted=1):
    """Run the REAL discover_peers on a GPU-worker path with a fake CuPy.

    Calling the production function rather than mirroring its rule: the
    earlier version of these tests reimplemented the budget check here, so
    they passed whatever the shipped code did. The fakes supply only what the
    function reaches out to -- MPI, CuPy and the communicator.
    """
    from psana.gpu import gpu_placement as gp

    # One allreduce per step, answered in the order discover_peers makes them
    # on a GPU worker: node-agreement SUM, job-agreement SUM, capacity MIN,
    # then the busiest-device peak MAX.
    #
    # The sharing-permitted MIN is answered SEPARATELY from the generic 'MIN'
    # slot. Sharing that slot made sharing_permitted come out as
    # bool(usable) -- True only by coincidence, so a test passing usable=0
    # would silently disable sharing and the failure would look like
    # something else.
    answers = {'SUM': 0, 'MIN': usable, 'MAX': busiest}
    # What the OTHER peers contribute to the sharing reduction. This rank's
    # own contribution is read from sendbuf and reduced against it, as a real
    # MIN would, so the test covers discovery actually reading the
    # environment variable -- not just the reduced value arriving. A bug that
    # hardcoded 1 instead of consulting sharing_enabled() would pass if the
    # fake ignored sendbuf.
    peers_permit = sharing_permitted
    mins = {'n': 0, 'sent': None}

    class Comm:
        def __init__(self, rank=0, size=local_peers):
            self._rank, self._size = rank, size

        def Get_rank(self):
            return self._rank

        def Get_size(self):
            return self._size

        def allgather(self, value):
            # Every peer reports the same host and the same single device, so
            # they land in one device group of `local_peers`.
            return [value] * self._size

        def Split(self, color, key):
            return Comm(rank=0, size=local_peers)

        def Free(self):
            pass

        def Iallreduce(self, sendbuf, recvbuf, op=None):
            # Counted, not inferred from the payload: discover_peers makes
            # its MINs in a fixed order, and the FIRST one on a GPU worker is
            # sharing-permitted (on device_comm, right after the Split); the
            # capacity MIN on psana_comm follows. Keying on the step name
            # would be better but Iallreduce is not given it, and keying on
            # the value would break the moment a budget MIN is legitimately
            # 0 or 1 bytes.
            if op == 'MIN':
                # Counted, not inferred from the payload: discover_peers
                # makes its MINs in a fixed order, and the FIRST one on a GPU
                # worker is sharing-permitted (on device_comm, right after
                # the Split); the capacity MIN on psana_comm follows. Keying
                # on the step name would be better but Iallreduce is not
                # given it, and keying on the value would break the moment a
                # budget MIN is legitimately 0 or 1 bytes.
                mins['n'] += 1
                if mins['n'] == 1:
                    mins['sent'] = int(sendbuf[0])
                    recvbuf[:] = min(mins['sent'], peers_permit)
                else:
                    recvbuf[:] = answers['MIN']
            else:
                recvbuf[:] = answers[op]
            return NS(Test=lambda: True)

        def Abort(self, code=1):
            raise AssertionError(f'unexpected Abort({code})')

    cp = NS(cuda=NS(
        runtime=NS(getDeviceCount=lambda: 1,
                   deviceGetPCIBusId=lambda i: '0000:01:00.0',
                   memGetInfo=lambda: (usable, usable)),
        Device=lambda *a: NS(use=lambda: None)))
    cp.get_default_memory_pool = lambda: NS(free_all_blocks=lambda: None)

    fake_mpi = NS(Get_processor_name=lambda: 'sdfampere001', UNDEFINED=-32766,
                  SUM='SUM', MIN='MIN', MAX='MAX')
    import sys as _sys
    saved = _sys.modules.get('mpi4py')
    _sys.modules['mpi4py'] = NS(MPI=fake_mpi)
    try:
        placement = gp.discover_peers(
            Comm(), is_gpu_worker=True, cp=cp, headroom=0.0,
            explicit_limit_bytes=int(budget_gb * GIB))
        # Expose what this rank contributed to the sharing reduction, so a
        # test can assert discovery consulted the environment rather than
        # only that the reduced value arrived.
        placement.contributed_sharing_flag = mins['sent']
        return placement
    finally:
        if saved is None:
            _sys.modules.pop('mpi4py', None)
        else:
            _sys.modules['mpi4py'] = saved


def test_discovery_reduces_the_sharing_opt_out_across_the_group(monkeypatch):
    """One rank with PSANA_GPU_SHARE_CONSTANTS=0 must disable sharing for all.

    The reduction is MIN over device_comm, so the group's answer is 0 if ANY
    peer opted out. Two halves, and both matter:

    * a PEER's opt-out must reach this rank, which is the reduction;
    * THIS rank's environment must reach the reduction, which is discovery
      consulting sharing_enabled() rather than hardcoding a 1.

    The fake reduces this rank's own sendbuf against what the other peers
    contribute, so neither half can pass on the other's behalf.
    """
    monkeypatch.delenv('PSANA_GPU_SHARE_CONSTANTS', raising=False)

    # Every peer opted in: the MIN returns 1 and sharing stays available.
    allowed = discover_with_budget(0, local_peers=4, busiest=4,
                                   usable=40 * GIB, sharing_permitted=1)
    assert allowed.contributed_sharing_flag == 1, \
        'discovery did not contribute this rank\'s opted-in environment'
    assert allowed.sharing_permitted is True
    assert allowed.can_share, 'a 4-peer group with no opt-out must share'

    # A PEER opted out while this rank did not. The group MIN is 0, so nobody
    # shares -- including this rank, which is the point.
    refused = discover_with_budget(0, local_peers=4, busiest=4,
                                   usable=40 * GIB, sharing_permitted=0)
    assert refused.contributed_sharing_flag == 1, \
        'this rank opted in, so it must still have contributed 1'
    assert refused.sharing_permitted is False
    assert not refused.can_share, (
        'one rank opting out must disable sharing for the whole device group; '
        'a rank that still shared would wait in an establishment allgather '
        'the opted-out rank never enters')
    # The peer count is unaffected -- still a 4-rank device group, it just
    # will not share.
    assert refused.n_device_peers == 4

    # And THIS rank opting out while every peer opted in. Catches discovery
    # that ignores the variable and contributes a constant: the reduction
    # would return 1 and sharing would stay on.
    monkeypatch.setenv('PSANA_GPU_SHARE_CONSTANTS', '0')
    mine = discover_with_budget(0, local_peers=4, busiest=4,
                                usable=40 * GIB, sharing_permitted=1)
    assert mine.contributed_sharing_flag == 0, \
        'discovery ignored PSANA_GPU_SHARE_CONSTANTS=0 in this rank\'s env'
    assert mine.sharing_permitted is False
    assert not mine.can_share


def test_explicit_budget_within_the_group_total_is_accepted():
    result = discover_with_budget(9.0, local_peers=4, busiest=4,
                                  usable=40 * GIB)
    # Accepted, and usable_bytes becomes the group's claim on the device.
    assert result.usable_bytes == int(9 * GIB) * 4


def test_explicit_budget_that_over_commits_the_device_is_refused():
    """An explicit budget is per rank, so it has to be checked against the
    group: four ranks each claiming 20 GiB of a 40 GiB card must not be
    accepted."""
    with pytest.raises(GpuPlacementError, match='exceeds'):
        discover_with_budget(20.0, local_peers=4, busiest=4, usable=40 * GIB)


def test_explicit_budget_verdict_is_the_same_on_an_uneven_layout():
    """The claim is per-device, but the verdict must be job-wide.

    An uneven split puts 4 peers on one device and 3 on another. A budget too
    large for 4 but fine for 3 would make one device raise while the other
    carried on -- ranks on different paths, racing the abort rather than
    agreeing on it. discover_peers reduces the peer count with MPI.MAX, so
    the 3-peer device must refuse it too.
    """
    # 11 x 4 = 44 GiB over a 40 GiB card; 11 x 3 = 33 would fit.
    for local_peers in (4, 3):
        with pytest.raises(GpuPlacementError, match='exceeds'):
            discover_with_budget(11.0, local_peers=local_peers, busiest=4,
                                 usable=40 * GIB)


def test_explicit_budget_error_names_the_busiest_device():
    """When the local count differs, the message has to say so, or a rank on
    a 3-peer device reads 'x 4 peers' and cannot reconcile it."""
    with pytest.raises(GpuPlacementError) as caught:
        discover_with_budget(11.0, local_peers=3, busiest=4, usable=40 * GIB)
    message = str(caught.value)
    assert 'busiest device' in message
    assert 'this device has 3' in message, message


def test_explicit_budget_error_omits_the_aside_when_counts_match():
    """A rank whose own count IS the busiest must not be told what its own
    count is -- the parenthetical would read as a contradiction."""
    with pytest.raises(GpuPlacementError) as caught:
        discover_with_budget(11.0, local_peers=4, busiest=4, usable=40 * GIB)
    assert 'this device has' not in str(caught.value)


def test_explicit_budget_is_unchecked_for_a_solo_rank():
    """A lone rank can claim the whole card: there is no one to over-commit
    it against, so discovery accepts the figure as given."""
    result = discover_with_budget(39.0, local_peers=1, busiest=1,
                                  usable=40 * GIB)
    assert result.usable_bytes == int(39 * GIB)


def test_automatic_budget_uses_the_discovered_capacity():
    """With no explicit budget, the limit is the discovered capacity divided
    among true peers -- which is what GpuEventManager reads."""
    assert per_rank_limit(placement(peers=2, usable=36 * GIB)) == 18 * GIB


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
    monkeypatch.setitem(sys.modules, 'mpi4py', NS(MPI=fake_mpi))

    result = gp.discover_peers(CountingComm(log),
                               is_gpu_worker=False)
    kinds = [entry[0] for entry in log]
    assert kinds.count('allgather') == 1       # hostname exchange
    assert kinds.count('split') == 2           # node, then gpu-role
    # Three job-wide reductions, in the order a GPU worker makes them: the
    # failure agreement, the device-capacity minimum, then the busiest-device
    # peer count. Skipping any leaves this rank racing a peer's MPI_ABORT
    # instead of participating in the failure. MPI pairs collectives by call
    # order rather than by operation, so assert the sequence and not just the
    # count.
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
    monkeypatch.setitem(sys.modules, 'mpi4py', NS(MPI=fake_mpi))

    class Failing(CountingComm):
        def Iallreduce(self, sendbuf, recvbuf, op=None):
            recvbuf[:] = 1         # a GPU worker reported a failure
            return NS(Test=lambda: True)

    with pytest.raises(GpuPlacementError, match='aborting this rank'):
        gp.discover_peers(Failing([]), is_gpu_worker=False)


def test_non_gpu_role_never_touches_cupy(monkeypatch):
    """A CPU-role rank must not import CuPy: on a shared node that would
    allocate a context on a device belonging to someone else.

    cp is left None so the code reaches its own `import cupy`, and the module
    is poisoned so that import FAILS. Passing a stand-in instead -- as this
    test used to -- means no import is attempted and the test passes however
    the code behaves.
    """
    from psana.gpu import gpu_placement as gp
    fake_mpi = NS(Get_processor_name=lambda: 'h', UNDEFINED=-32766,
                  SUM='SUM', MIN='MIN', MAX='MAX')
    monkeypatch.setitem(sys.modules, 'mpi4py', NS(MPI=fake_mpi))
    monkeypatch.setitem(sys.modules, 'cupy', None)

    # `import cupy` with sys.modules['cupy'] = None raises ImportError, so a
    # CPU-role rank that reached the import would fail here rather than pass.
    gp.discover_peers(CountingComm([]), is_gpu_worker=False)


# ---------------------------------------------------------------------------
# Constant sharing is selected only when it is safe
# ---------------------------------------------------------------------------

def test_sharing_is_selected_for_real_peers():
    assert placement(peers=2).can_share


def test_sharing_is_skipped_for_a_solo_rank():
    assert not placement(peers=1).can_share


def test_sharing_is_skipped_without_a_communicator():
    """A placement with no device_comm cannot share, whatever its peer count:
    the serial path and non-GPU roles both land here."""
    p = placement(peers=4)
    p.device_comm = None
    assert not p.can_share
    assert p.n_device_peers == 4      # the count is still known; sharing is not
