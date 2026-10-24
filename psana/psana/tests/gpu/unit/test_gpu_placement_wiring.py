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

def discover_with_budget(budget_gb, *, local_peers, usable,
                         other_device_over_commits=0, sharing_permitted=1):
    """Run the REAL discover_peers on a GPU-worker path with a fake CuPy.

    Calling the production function rather than mirroring its rule: the
    earlier version of these tests reimplemented the budget check here, so
    they passed whatever the shipped code did. The fakes supply only what the
    function reaches out to -- MPI, CuPy and the communicator.

    ``usable`` is THIS device's free memory and ``local_peers`` the number of
    ranks on it; together they are the only inputs the budget check may use.
    ``other_device_over_commits`` is what the rest of the job contributes to
    the job-wide verdict, which is how a device that fits still stops when
    another does not.

    The fakes distinguish `psana_comm` from `device_comm`, because which
    communicator a reduction runs on is the thing under test: capacity reduced
    job-wide sizes a roomy card against the busiest card in the job, and
    combining a job-wide capacity MIN with a job-wide peer MAX rejects
    layouts that fit every device.
    """
    from psana.gpu import gpu_placement as gp

    state = {'device_mins': 0, 'psana_maxes': 0, 'sharing_sent': None,
             'capacity_sent': None, 'verdict_sent': None}

    class Comm:
        """One link in the Split chain: psana -> node -> node_gpu -> device."""

        _next = {'psana': 'node', 'node': 'node_gpu', 'node_gpu': 'device'}

        def __init__(self, kind='psana', rank=0, size=local_peers):
            self.kind, self._rank, self._size = kind, rank, size

        def Get_rank(self):
            return self._rank

        def Get_size(self):
            return self._size

        def allgather(self, value):
            # Every peer reports the same host and the same single device, so
            # they land in one device group of `local_peers`.
            return [value] * self._size

        def Split(self, color, key):
            return Comm(kind=self._next[self.kind], rank=0, size=local_peers)

        def Free(self):
            pass

        def Iallreduce(self, sendbuf, recvbuf, op=None):
            mine = int(sendbuf[0])
            if self.kind == 'device':
                # Two MINs on device_comm, in the order discover_peers makes
                # them: sharing-permitted right after the Split, then
                # capacity. Counted rather than keyed on the payload, because
                # a capacity MIN can legitimately be 0 or 1 bytes. Each is
                # reduced against this rank's own sendbuf, as a real MIN
                # would, so a bug that contributed a constant instead of
                # consulting the environment or memGetInfo cannot pass.
                assert op == 'MIN', f'unexpected {op} on device_comm'
                state['device_mins'] += 1
                if state['device_mins'] == 1:
                    state['sharing_sent'] = mine
                    recvbuf[:] = min(mine, sharing_permitted)
                else:
                    state['capacity_sent'] = mine
                    recvbuf[:] = min(mine, usable)
            elif self.kind == 'psana':
                if op == 'MIN':
                    raise AssertionError(
                        'capacity was reduced over psana_comm. Automatic '
                        'sizing must stay local to the device: a job-wide '
                        'minimum sizes an idle 80 GiB card against a loaded '
                        '40 GiB one, and pairing it with a job-wide peer MAX '
                        'rejects layouts that fit every device')
                if op == 'MAX':
                    # The job-wide failure verdict, so every role stops
                    # together even though only this device's ranks evaluated
                    # this device's claim.
                    #
                    # EXACTLY ONE. A second MAX here is a peer count or a
                    # capacity figure being reduced across devices, which is
                    # the reviewer's bug: the claim must be assembled from one
                    # device's own numbers. Keeping capacity on device_comm
                    # while taking the peer count job-wide still mixes two
                    # cards, and the guard above cannot see it.
                    state['psana_maxes'] += 1
                    if state['psana_maxes'] > 1:
                        raise AssertionError(
                            f'{state["psana_maxes"]} MAX reductions on '
                            'psana_comm; only the failure verdict belongs '
                            'there. A second one means peer counts or '
                            'capacity are being combined across devices, so '
                            'a claim can be checked against a capacity no '
                            'device actually has')
                    state['verdict_sent'] = mine
                    recvbuf[:] = max(mine, other_device_over_commits)
                else:
                    recvbuf[:] = mine        # SUM: the failure agreements
            else:
                recvbuf[:] = mine            # node / node_gpu agreement
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
        # Expose what this rank contributed to each reduction, so a test can
        # assert discovery consulted its own inputs rather than only that the
        # reduced value arrived.
        placement.contributed_sharing_flag = state['sharing_sent']
        placement.contributed_capacity = state['capacity_sent']
        placement.contributed_verdict = state['verdict_sent']
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
    allowed = discover_with_budget(0, local_peers=4, usable=40 * GIB,
                                   sharing_permitted=1)
    assert allowed.contributed_sharing_flag == 1, \
        'discovery did not contribute this rank\'s opted-in environment'
    assert allowed.sharing_permitted is True
    assert allowed.can_share, 'a 4-peer group with no opt-out must share'

    # A PEER opted out while this rank did not. The group MIN is 0, so nobody
    # shares -- including this rank, which is the point.
    refused = discover_with_budget(0, local_peers=4, usable=40 * GIB,
                                   sharing_permitted=0)
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
    mine = discover_with_budget(0, local_peers=4, usable=40 * GIB,
                                sharing_permitted=1)
    assert mine.contributed_sharing_flag == 0, \
        'discovery ignored PSANA_GPU_SHARE_CONSTANTS=0 in this rank\'s env'
    assert mine.sharing_permitted is False
    assert not mine.can_share


def test_explicit_budget_within_the_group_total_is_accepted():
    result = discover_with_budget(9.0, local_peers=4, usable=40 * GIB)
    # Accepted, and usable_bytes becomes the group's claim on the device.
    assert result.usable_bytes == int(9 * GIB) * 4


def test_explicit_budget_that_over_commits_the_device_is_refused():
    """An explicit budget is per rank, so it has to be checked against the
    group: four ranks each claiming 20 GiB of a 40 GiB card must not be
    accepted."""
    with pytest.raises(GpuPlacementError, match='exceeds'):
        discover_with_budget(20.0, local_peers=4, usable=40 * GIB)


def test_each_device_is_validated_against_its_own_capacity():
    """A budget every device can honour must be accepted on all of them.

    The reviewer's case, with an explicit 9 GiB per worker:

        device A   40 GiB usable   4 workers   36 GiB claimed   fits
        device B   20 GiB usable   1 worker     9 GiB claimed   fits

    Nothing in the job over-commits. Reducing capacity job-wide with MIN and
    the peer count job-wide with MAX combines figures from different cards --
    9 x 4 > 20 -- and aborts a placement that fits everywhere. The two inputs
    must come from the same device, which means both are reduced over
    `device_comm`.
    """
    a = discover_with_budget(9.0, local_peers=4, usable=40 * GIB)
    assert a.usable_bytes == int(9 * GIB) * 4
    assert a.contributed_verdict == 0, \
        'device A claims 36 of 40 GiB and must raise no objection'

    b = discover_with_budget(9.0, local_peers=1, usable=20 * GIB)
    assert b.usable_bytes == int(9 * GIB)
    assert b.contributed_verdict == 0, \
        'device B claims 9 of 20 GiB and must raise no objection'


def test_automatic_sizing_is_not_shrunk_by_a_busier_device():
    """Capacity is spent by one device's peers, so it is measured there.

    Taking the job-wide minimum charged the roomiest card for the busiest one:
    4 peers on an idle 80 GiB card alongside a loaded 40 GiB card were sized
    at 10 GiB each rather than 20, leaving 40 GiB unused for no reason.
    """
    roomy = discover_with_budget(0, local_peers=4, usable=80 * GIB)
    assert roomy.usable_bytes == 80 * GIB
    assert per_rank_limit(roomy) == 20 * GIB
    # And the figure came from this device's own memGetInfo, not a constant.
    assert roomy.contributed_capacity == 80 * GIB


def test_an_over_commit_on_another_device_stops_this_one_too():
    """The claim is per-device; the verdict is job-wide.

    A budget that over-commits any card is a configuration error for the whole
    job. A device that could honour it and carried on while another aborted
    would leave ranks on different collective paths -- racing the abort rather
    than agreeing on it. So this device evaluates its own claim, which fits,
    and still raises because the job-wide MAX says someone's did not.
    """
    with pytest.raises(GpuPlacementError, match='another GPU in this job'):
        discover_with_budget(9.0, local_peers=1, usable=20 * GIB,
                             other_device_over_commits=1)


def test_the_verdict_message_distinguishes_whose_device_failed():
    """Two different messages, because the actions they imply differ.

    A rank on the offending device reports the figures to fix. A rank on a
    device that fits must not report its own capacity as the cause -- it would
    read as a contradiction, since those numbers add up.
    """
    with pytest.raises(GpuPlacementError) as mine:
        discover_with_budget(20.0, local_peers=4, usable=40 * GIB)
    message = str(mine.value)
    assert 'exceeds' in message
    assert '4 peers' in message and '0:01:00.0' in message, message

    with pytest.raises(GpuPlacementError) as theirs:
        discover_with_budget(9.0, local_peers=1, usable=20 * GIB,
                             other_device_over_commits=1)
    other = str(theirs.value)
    assert 'could have honoured it' in other, other
    assert 'exceeds' not in other, \
        'a device that fits must not be reported as the one that failed'


def test_explicit_budget_is_unchecked_for_a_solo_rank():
    """A lone rank can claim the whole card: there is no one to over-commit
    it against, so discovery accepts the figure as given."""
    result = discover_with_budget(39.0, local_peers=1, usable=40 * GIB)
    assert result.usable_bytes == int(39 * GIB)


def test_automatic_budget_uses_the_discovered_capacity():
    """With no explicit budget, the limit is the discovered capacity divided
    among true peers -- which is what GpuEventManager reads."""
    assert per_rank_limit(placement(peers=2, usable=36 * GIB)) == 18 * GIB


# ---------------------------------------------------------------------------
# The public device accessor (GLINT #256)
# ---------------------------------------------------------------------------
# GLINT's adapter contract is "keep the mask, follow psana's selected device,
# never assume cuda:0, verify physical identity". Following it requires a
# supported way to ask which device psana chose; reaching into
# run._gpu_placement is not one, and that attribute has been reshaped twice.

def test_gpu_device_reports_the_selected_card():
    """What an external framework needs: the physical card, not the ordinal.

    The ordinal is an index into this rank's mask and means nothing outside
    it -- under a reversed mask, ordinal 0 is physical card 1. The bus id
    names the card, and is what GLINT checks too, so the two agree.
    """
    from psana.psexp.run import GpuDevice

    view = GpuDevice(placement(peers=3, usable=40 * GIB))
    assert view.pci_bus_id == '0:01:00.0'
    assert view.peers == 3
    assert view.hostname == 'sdfampere001'
    assert view.is_owner is True
    # Reported, but a caller must not identify a card by it.
    assert isinstance(view.ordinal, int)


def test_gpu_device_is_read_only():
    """Placement is decided once, collectively, inside discovery.

    A settable view would be a second way to influence it, which is a second
    way to get it wrong -- and a framework that wrote to it would diverge
    from what the peer group agreed.
    """
    from psana.psexp.run import GpuDevice

    view = GpuDevice(placement(peers=2, usable=40 * GIB))
    for field, value in (('pci_bus_id', '0:99:00.0'), ('ordinal', 7),
                         ('peers', 99), ('is_owner', False)):
        with pytest.raises(AttributeError, match='read-only'):
            setattr(view, field, value)
    with pytest.raises(AttributeError, match='read-only'):
        del view.peers
    # Unchanged, and the message says why rather than only that.
    assert view.peers == 2 and view.pci_bus_id == '0:01:00.0'


def test_gpu_device_reports_a_shared_card_without_complaint():
    """Several BD workers on one GPU is the normal mode, not an error.

    It is the basis of IPC constant sharing. GLINT treats two ranks on one
    card as a misconfiguration unless --share-gpu is passed, so a psana2-gpu
    reader under GLINT needs that flag as the default. Asserted here so the
    expectation is recorded on this side of the boundary.
    """
    from psana.psexp.run import GpuDevice

    view = GpuDevice(placement(peers=4, usable=40 * GIB))
    assert view.peers == 4
    assert 'peers=4' in repr(view)


def test_gpu_device_is_none_without_a_placement():
    """CPU roles, the serial path and non-GPU jobs have nothing to follow.

    A caller must handle None rather than defaulting to device 0 -- that
    default is the failure the accessor exists to prevent.
    """
    from psana.psexp.run import Run

    bare = Run.__new__(Run)                 # no _gpu_placement attribute
    assert bare.gpu_device is None

    bare._gpu_placement = None              # CPU role: discovery set it None
    assert bare.gpu_device is None

    bare._gpu_placement = placement(peers=2, usable=40 * GIB)
    assert bare.gpu_device is not None
    assert bare.gpu_device.peers == 2


def test_gpu_device_is_importable_from_psana_gpu():
    """GLINT should not have to import from psana.psexp, which is internal."""
    from psana.gpu import GpuDevice as exported
    from psana.psexp.run import GpuDevice as defined
    assert exported is defined


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
    # Two job-wide reductions, in the order a GPU worker makes them: the
    # selection-failure agreement, then the explicit-budget verdict. Skipping
    # either leaves this rank racing a peer's MPI_ABORT instead of
    # participating in the failure. MPI pairs collectives by call order rather
    # than by operation, so assert the sequence and not just the count.
    #
    # Capacity and peer counts are NOT here: both are reduced on device_comm,
    # which a CPU role does not hold, so it cannot evaluate the budget itself
    # and takes the workers' verdict instead.
    assert [k for k in kinds if k == 'allreduce'] == ['allreduce'] * 2
    operations = [entry[2] for entry in log if entry[0] == 'allreduce']
    assert operations == ['SUM', 'MAX']
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
