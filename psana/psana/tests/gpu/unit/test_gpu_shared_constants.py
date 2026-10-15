"""Shared constants are all-or-nothing, and any failure aborts the group.

The protocol is exercised against a fake device_comm that runs every peer in
one process, and a NumPy-backed CuPy stand-in, so no GPU is required. Device
behaviour that cannot be faithfully faked -- real IPC handles, real kernels --
is covered by the integration tests instead.

Two contracts these turn on:

* **one constant set per run** -- ``refresh`` establishes on the first call and
  is a no-op thereafter, which is what makes a returned array safe to hold for
  the whole run;
* **all-or-nothing sharing** -- every peer must declare the same selector set,
  and an IPC failure aborts rather than degrading to private copies. The
  escape hatch for a system where IPC cannot work is
  ``PSANA_GPU_SHARE_CONSTANTS=0``.
"""
import pathlib
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from collective_recorder import RecordingComm                    # noqa: E402
from psana.gpu.gpu_budget import _GpuBudget                      # noqa: E402
from psana.gpu.gpu_placement import GpuPlacement                 # noqa: E402
from psana.gpu.gpu_shared_constants import (                     # noqa: E402
    SharedConstantsError, SharedRequestedConstants, digest_of, sharing_enabled,
)


PEDS = np.arange(24, dtype=np.float32).reshape(3, 2, 2, 2)
GAIN = (PEDS * 0.5 + 1).astype(np.float32)
STATUS = np.ones((3, 2, 2, 2), dtype=np.uint16)


def placement_with_peers(peers):
    """A bare placement with a device communicator, for can_share checks."""
    return GpuPlacement(device_pci='0:01:00.0', device_comm=NS(),
                        n_device_peers=peers)


def host(**overrides):
    values = {'pedestals': PEDS, 'pixel_gain': GAIN, 'pixel_status': STATUS}
    values.update(overrides)
    return {'jf': values}


class FakeComm:
    """Collective over peers that all live in this process.

    Real collectives block until every peer contributes. Peers here are driven
    sequentially, so instead of blocking, each collective records this rank's
    contribution and returns the full set assembled from values the harness
    already knows -- `group['declared']` for allgather, and a position-matched
    broadcast slot. Real MPI ordering is covered by the integration tests.
    """

    def __init__(self, group, rank):
        self.group = group
        self._rank = rank

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return len(self.group['declared'])

    def allgather(self, value):
        """Gather over peers driven sequentially in one process.

        Declaration sets are answered from the harness's own record, so the
        set-agreement verdict does not depend on the order peers are driven
        in. Anything else is answered with this rank's own value repeated,
        which is exact whenever peers agree; disagreement is covered by the
        MPI integration test.
        """
        self.group['contrib'][self._rank] = value
        if isinstance(value, (set, frozenset)):
            return [set(d) for d in self.group['declared']]
        return [value] * self.Get_size()

    def bcast(self, value, root=0):
        """Round-aware broadcast.

        Peers are driven sequentially, so a single last-writer slot would hand
        a follower whichever value the owner wrote most recently. Count each
        rank's calls and match by position instead.
        """
        counts = self.group.setdefault('bcast_n', {})
        index = counts.get(self._rank, 0)
        counts[self._rank] = index + 1
        slots = self.group.setdefault('bcast_slots', {})
        if self._rank == root:
            slots[index] = value
            return value
        return slots.get(index)

    def allreduce(self, value, op=None):
        """Reduction over peers driven sequentially in one process.

        A real allreduce blocks until every peer contributes. Here the first
        caller cannot see the others, so returning a partial reduction would
        make the owner act on an incomplete vote. Treat this rank's own
        contribution as the result instead: exact whenever all peers
        contribute the same value, which is the case under the fake CuPy.
        """
        self.group.setdefault('reduce', []).append((self._rank, value))
        name = getattr(op, 'name', str(op)).upper()
        if 'MIN' in name:
            return value
        return value * self.Get_size() if value else 0

    def Allreduce(self, sendbuf, recvbuf, op=None):
        """Buffer-form reduction, used for the single agreement step.

        Peers run sequentially here, so this rank's own vector is the result.
        That is exact for these tests: the fake CuPy either works for every
        peer or for none, and content disagreement is injected on the rank
        being asserted. Genuine per-rank divergence is covered by the MPI
        integration test.
        """
        self.group.setdefault('allreduce', []).append(
            (self._rank, list(sendbuf)))
        recvbuf[:] = sendbuf
        return recvbuf

    def Barrier(self):
        return None

    def Ibarrier(self):
        """Immediately-complete request, for the bounded barrier helper.

        Peers run sequentially here, so a real barrier would deadlock on the
        first caller. Completing at once is faithful to what the helper needs
        to observe: that every rank reaches the call.
        """
        self.group.setdefault('ibarrier', []).append(self._rank)
        return NS(Test=lambda: True)

    def Iallreduce(self, sendbuf, recvbuf, op=None):
        self.group.setdefault('iallreduce', []).append(self._rank)
        recvbuf[:] = sendbuf
        return NS(Test=lambda: True)

    def Abort(self, code=1):
        raise AssertionError(f'unexpected Abort({code}) in a unit test')

    def Free(self):
        return None


def make_peers(declarations, *, budgets=None, uuid='abcd' * 8, record=False):
    """One SharedRequestedConstants per peer, sharing a fake device_comm.

    ``record=True`` wraps each communicator in the test recorder, which is
    also how the integration drivers enable order checking -- production holds
    no reference to it.
    """
    group = {'declared': [list(d) for d in declarations],
             'contrib': {}, 'bcast': None}
    cp = FakeCupy()
    peers = []
    recorders = []
    for rank, declared in enumerate(declarations):
        comm = FakeComm(group, rank)
        if record:
            comm = RecordingComm(comm, 'shared-constants')
            recorders.append(comm)
        placement = GpuPlacement(
            device_pci=uuid, device_comm=comm,
            is_owner=(rank == 0), n_device_peers=len(declarations),
            usable_bytes=1 << 30)
        budget = (budgets[rank] if budgets else _GpuBudget(1 << 30))
        peers.append(SharedRequestedConstants(declared, budget, placement,
                                              cp=cp))
    if record:
        return peers, cp, recorders
    return peers, cp


class FakeArray(np.ndarray):
    """NumPy array with CuPy's ``set`` so uploads work unchanged."""

    def set(self, source, stream=None):
        np.copyto(self, source)


class FakeCupy:
    """Minimal CuPy stand-in with a byte-addressed fake device heap.

    Allocation returns offsets into one bytearray, so an "IPC handle" is just
    an offset and a follower genuinely reads the owner's bytes -- which is what
    makes the value assertions meaningful.
    """

    def __init__(self):
        self.heap = bytearray(1 << 22)
        # Start above zero: cudaMalloc never returns a null pointer, and the
        # production code treats pointer 0 as "nothing allocated".
        self.next = 512
        self.live = {}                       # ptr -> nbytes
        self.freed = []
        self.opened = {}                     # base -> handle
        self.closed = []
        self.syncs = 0
        # Injection points for the failure-path tests.
        self.fail_malloc = False
        self.fail_open = False
        self.fail_handle = False
        self.cuda = NS(
            runtime=NS(
                malloc=self._malloc, free=self._free,
                ipcGetMemHandle=self._get_handle,
                ipcOpenMemHandle=self._open_handle,
                ipcCloseMemHandle=self._close_handle,
                cudaIpcMemLazyEnablePeerAccess=1,
                memGetInfo=lambda: (1 << 30, 1 << 31),
            ),
            UnownedMemory=lambda ptr, size, owner, **kw: NS(
                ptr=ptr, size=size, owner=owner),
            MemoryPointer=lambda mem, off: NS(mem=mem, ptr=mem.ptr + off),
            Stream=NS(null=NS(synchronize=self._sync)),
            Device=lambda *a: NS(id=0, synchronize=self._sync,
                                 use=lambda: None),
            get_current_stream=lambda: NS(synchronize=self._sync),
        )

    # -- device heap --
    def _malloc(self, nbytes):
        if self.fail_malloc:
            raise RuntimeError('injected cudaMalloc failure')
        ptr = self.next
        self.next += ((int(nbytes) + 511) // 512) * 512
        self.live[ptr] = int(nbytes)
        return ptr

    def _free(self, ptr):
        self.freed.append(ptr)
        self.live.pop(ptr, None)

    def _get_handle(self, ptr):
        if self.fail_handle:
            # Raises AFTER the allocation succeeded, which is the case that
            # leaked the block and its budget charge.
            raise RuntimeError('injected ipcGetMemHandle failure')
        return b'HANDLE' + str(int(ptr)).encode()

    def _open_handle(self, handle, flags):
        if self.fail_open:
            raise RuntimeError('injected ipcOpenMemHandle failure')
        base = int(handle[len(b'HANDLE'):])
        self.opened[base] = handle
        return base

    def _close_handle(self, base):
        self.closed.append(base)
        self.opened.pop(base, None)

    def _sync(self):
        self.syncs += 1

    # -- arrays --
    def ndarray(self, shape, dtype=None, memptr=None):
        count = int(np.prod(shape)) if shape else 1
        offset = int(memptr.ptr)
        view = np.frombuffer(self.heap, dtype=dtype, count=count,
                             offset=offset).reshape(shape)
        return view.view(FakeArray)

    def empty(self, shape, dtype=None):
        return np.empty(shape, dtype=dtype).view(FakeArray)

    def asarray(self, a):
        return np.asarray(a).view(FakeArray)


# ---------------------------------------------------------------------------
# All-or-nothing: identical sets share, any difference shares nothing
# ---------------------------------------------------------------------------

def test_identical_declarations_share_everything():
    decl = [('jf', 'pedestals'), ('jf', 'pixel_gain')]
    peers, cp = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    for peer in peers:
        assert peer.shared_selectors == tuple(decl)
    # One device copy, not two.
    assert len(cp.live) == 2          # pedestals + gain, owned once each


def test_differing_sets_share_nothing():
    """Sharing is all-or-nothing. Peers on one device run the same GpuTask so
    they normally declare the same set; a group that disagrees gets a private
    copy per rank rather than a partial share."""
    owner = [('jf', 'pedestals'), ('jf', 'pixel_gain'), ('jf', 'pixel_status')]
    follower = [('jf', 'pedestals'), ('jf', 'pixel_gain')]
    peers, cp = make_peers([owner, follower])
    for peer in peers:
        peer.refresh(host())

    for peer in peers:
        assert peer.shared_selectors == ()
        assert peer.shared_bytes == 0
        assert peer._placement.sharing == 'off'
    # No un-pooled owner allocation: the private path uses the pool.
    assert cp.live == {}


def test_every_declared_constant_is_readable_when_sets_differ():
    owner = [('jf', 'pedestals'), ('jf', 'pixel_status')]
    follower = [('jf', 'pedestals')]
    peers, _ = make_peers([owner, follower])
    for peer in peers:
        peer.refresh(host())
    np.testing.assert_array_equal(peers[0].get('jf', 'pedestals'), PEDS)
    np.testing.assert_array_equal(peers[0].get('jf', 'pixel_status'), STATUS)
    np.testing.assert_array_equal(peers[1].get('jf', 'pedestals'), PEDS)


def test_disjoint_sets_share_nothing_without_error():
    peers, _ = make_peers([[('jf', 'pedestals')], [('jf', 'pixel_gain')]])
    for peer in peers:
        peer.refresh(host())
    for peer in peers:
        assert peer.shared_selectors == ()
        assert peer.shared_bytes == 0
    np.testing.assert_array_equal(peers[1].get('jf', 'pixel_gain'), GAIN)


def test_order_does_not_count_as_disagreement():
    """The set is what matters, not the order a task happened to list it in:
    two peers running the same task with the selectors written differently
    must still share."""
    first = [('jf', 'pedestals'), ('jf', 'pixel_gain')]
    second = [('jf', 'pixel_gain'), ('jf', 'pedestals')]
    peers, cp = make_peers([first, second])
    for peer in peers:
        peer.refresh(host())
    for peer in peers:
        assert set(peer.shared_selectors) == set(first)
        assert peer._placement.sharing == 'on'
    assert len(cp.live) == 2


def test_undeclared_access_still_fails():
    decl = [('jf', 'pedestals')]
    peers, _ = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    with pytest.raises(KeyError, match='was not declared'):
        peers[1].get('jf', 'pixel_gain')


# ---------------------------------------------------------------------------
# Content disagreement is fatal
# ---------------------------------------------------------------------------

def test_digest_mismatch_aborts_rather_than_substituting():
    """Sharing here would hand the follower the owner's array under the
    follower's own name: wrong results with no error."""
    decl = [('jf', 'pedestals')]
    peers, _ = make_peers([decl, decl])
    peers[0].refresh(host())
    with pytest.raises(SharedConstantsError, match='disagree on content'):
        peers[1].refresh(host(pedestals=PEDS + 1))


def test_layout_mismatch_aborts():
    decl = [('jf', 'pedestals')]
    peers, _ = make_peers([decl, decl])
    peers[0].refresh(host())
    reshaped = PEDS.reshape(3, 2, 4, 1)
    with pytest.raises(SharedConstantsError, match='disagree on layout'):
        peers[1].refresh(host(pedestals=reshaped))


def test_dtype_mismatch_aborts():
    decl = [('jf', 'pedestals')]
    peers, _ = make_peers([decl, decl])
    peers[0].refresh(host())
    with pytest.raises(SharedConstantsError, match='disagree on layout'):
        peers[1].refresh(host(pedestals=PEDS.astype(np.float64)))


def test_digest_detects_content_and_ignores_identity():
    a = np.arange(8, dtype=np.float32)
    assert digest_of(a) == digest_of(a.copy())
    assert digest_of(a) != digest_of(a + 1)
    # Non-contiguous input must still digest its logical content.
    assert digest_of(a.reshape(2, 4)[:, ::2]) == digest_of(
        np.ascontiguousarray(a.reshape(2, 4)[:, ::2]))


# ---------------------------------------------------------------------------
# Accounting: charge once, report imports
# ---------------------------------------------------------------------------

def test_owner_charges_shared_bytes_and_followers_charge_nothing():
    decl = [('jf', 'pedestals'), ('jf', 'pixel_gain')]
    budgets = [_GpuBudget(1 << 30) for _ in range(3)]
    peers, _ = make_peers([decl, decl, decl], budgets=budgets)
    for peer in peers:
        peer.refresh(host())

    expected = PEDS.nbytes + GAIN.nbytes
    assert budgets[0].committed() == expected
    # Charging a follower would shrink its budget by memory it does not own.
    assert budgets[1].committed() == 0
    assert budgets[2].committed() == 0
    assert peers[1].imported_bytes == expected
    assert peers[0].imported_bytes == 0


def test_the_placement_records_a_followers_imported_bytes():
    """The field was declared and documented but never written, so a
    follower's imported bytes always read zero.

    It is the one figure that shows sharing worked from the follower's side:
    memory it reads and is not charged for. Cleared again at teardown, so a
    reading taken after close() does not describe a mapping that is gone.
    """
    decl = [('jf', 'pedestals')]
    peers, _ = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())

    owner, follower = peers
    assert follower._placement.imported_bytes == PEDS.nbytes
    assert owner._placement.imported_bytes == 0, \
        'the owner holds the allocation; it imports nothing'

    for peer in reversed(peers):
        peer.close()
    assert follower._placement.imported_bytes == 0, \
        'the figure outlived the mapping it describes'


def test_shared_bytes_are_reported_by_both_roles():
    decl = [('jf', 'pedestals')]
    peers, _ = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    assert peers[0].shared_bytes == PEDS.nbytes
    assert peers[1].shared_bytes == PEDS.nbytes


# ---------------------------------------------------------------------------
# One constant set per run
# ---------------------------------------------------------------------------

def test_a_second_refresh_does_nothing():
    """Establish once, then no-op.

    BeginStep does not fetch the calibration DB, so a second refresh could
    only find the values already uploaded. It must not upload, synchronise or
    report that anything moved -- reporting True makes the caller recompute
    its subbatch budget every step.
    """
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    syncs, live_before = cp.syncs, sorted(cp.live)

    for peer in peers:
        assert peer.refresh(host()) is False
    assert cp.syncs == syncs, 'a second refresh synchronised the device'
    assert sorted(cp.live) == live_before, 'a second refresh changed allocations'


def test_a_changed_source_is_ignored_after_establishment():
    """Even a DIFFERENT source is ignored on a later call.

    Replacing a shared allocation mid-run would need coordinated retirement
    across peers, because one allocation backs all of them and no rank can
    know when another has stopped reading. Under the one-set-per-run contract
    this cannot arise, so the second call is a no-op rather than a reupload.
    """
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    live_before = sorted(cp.live)

    for peer in peers:
        assert peer.refresh(host(pedestals=PEDS + 7)) is False
    assert sorted(cp.live) == live_before
    # The device still holds the values uploaded at establishment.
    np.testing.assert_array_equal(peers[1].get('jf', 'pedestals'), PEDS)


def test_a_shared_array_stays_valid_for_the_whole_run():
    """The guarantee the one-set-per-run contract buys.

    The allocation is written once and released at teardown, never rewritten
    or replaced while the run is live, so a reference a callback keeps across
    events and steps stays valid and keeps its values. Python cannot prevent
    a caller retaining a reference -- any returned object can be assigned --
    so the contract has to be that retention is safe rather than forbidden.
    """
    decl = [('jf', 'pedestals')]
    peers, _ = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())

    kept = peers[1].get('jf', 'pedestals')          # a follower's view
    before = np.asarray(kept).copy()

    # Several later transitions, including one with a changed source.
    for _ in range(3):
        for peer in peers:
            peer.refresh(host())
    for peer in peers:
        peer.refresh(host(pedestals=PEDS + 7))

    np.testing.assert_array_equal(np.asarray(kept), before)


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

def test_importers_close_before_the_owner_frees():
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    ptr = next(iter(cp.live))

    peers[1].close()                            # importer first
    assert cp.closed, 'importer did not close its mapping'
    assert ptr not in cp.freed, 'owner freed while a mapping could be live'
    peers[0].close()
    assert ptr in cp.freed


def test_close_is_idempotent():
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    for peer in reversed(peers):
        peer.close()
        peer.close()
    assert len(cp.freed) == 1


def test_refresh_after_close_is_refused():
    decl = [('jf', 'pedestals')]
    peers, _ = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    peers[1].close()
    with pytest.raises(SharedConstantsError, match='closed'):
        peers[1].refresh(host())


def test_close_without_sharing_makes_no_collective():
    """A group that shared nothing has no mappings to close in order, so
    close() must not barrier -- a barrier there would have to be matched by
    every peer, and a rank that never established would not reach it."""
    peers, _, recorders = make_peers(
        [[('jf', 'pedestals')], [('jf', 'pixel_gain')]], record=True)
    for peer in peers:
        peer.refresh(host())
    before = [len(r.calls) for r in recorders]
    for peer in peers:
        peer.close()
    assert [len(r.calls) for r in recorders] == before


# ---------------------------------------------------------------------------
# No sharing available
# ---------------------------------------------------------------------------

def test_single_peer_uses_the_inherited_private_path():
    peers, cp = make_peers([[('jf', 'pedestals')]])
    peer = peers[0]
    assert not peer._sharing          # one peer: nothing to share with
    peer.refresh(host())
    assert peer.shared_selectors == ()
    assert cp.live == {}              # no un-pooled owner allocation


def test_no_placement_uses_the_inherited_private_path():
    budget = _GpuBudget(1 << 30)
    peer = SharedRequestedConstants([('jf', 'pedestals')], budget, None)
    assert not peer._sharing


def test_solo_rank_reports_sharing_off():
    peers, _ = make_peers([[('jf', 'pedestals')]])
    peers[0].refresh(host())
    assert peers[0]._placement.sharing == 'off'


def test_successful_sharing_is_reported_as_on():
    decl = [('jf', 'pedestals')]
    peers, _ = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    for peer in peers:
        assert peer._placement.sharing == 'on'
        assert 'sharing=on' in peer._placement.describe()


def test_sharing_can_be_disabled_by_environment(monkeypatch):
    """The escape hatch the IPC abort points users at.

    A system where CUDA IPC cannot work between ranks discovery grouped as
    peers -- MIG, chiefly, whose instances share a PCI bus id -- would
    otherwise abort with no way to proceed.
    """
    monkeypatch.delenv('PSANA_GPU_SHARE_CONSTANTS', raising=False)
    assert sharing_enabled()
    monkeypatch.setenv('PSANA_GPU_SHARE_CONSTANTS', '1')
    assert sharing_enabled()
    monkeypatch.setenv('PSANA_GPU_SHARE_CONSTANTS', '0')
    assert not sharing_enabled()
    monkeypatch.setenv('PSANA_GPU_SHARE_CONSTANTS', 'no')
    assert not sharing_enabled()


def test_an_opted_out_placement_takes_the_private_path():
    """The group's agreed verdict, not this rank's environment.

    can_share folds in sharing_permitted, which discovery reduces with MIN
    over the device group. A rank acting on its own reading of the variable
    would skip the establishment allgather its peers enter -- one of the two
    unbounded collectives, so the job would hang with no diagnostic.
    """
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    for peer in peers:
        peer._placement.sharing_permitted = False
        assert not peer._placement.can_share
    for peer in peers:
        peer.refresh(host())
    for peer in peers:
        assert peer.shared_selectors == ()
        assert peer._placement.sharing == 'off'
    assert cp.live == {}          # no un-pooled owner allocation
    np.testing.assert_array_equal(peers[1].get('jf', 'pedestals'), PEDS)


def test_sharing_permitted_defaults_true_without_discovery():
    """A placement built by hand -- the serial path, the unit tests -- must
    behave as it did before the field existed."""
    assert GpuPlacement().sharing_permitted is True
    assert placement_with_peers(2).can_share


# ---------------------------------------------------------------------------
# IPC failure aborts the group
# ---------------------------------------------------------------------------
# The automatic degrade to private copies was removed: it hid a
# misconfiguration behind an n-times memory cost, and its reset path was the
# source of two separate budget defects. A failure now aborts with a message
# naming PSANA_GPU_SHARE_CONSTANTS=0 as the deliberate way to run without it.

def test_owner_failing_before_bcast_still_publishes_a_marker():
    """If the owner's allocation or upload raises it must still broadcast,
    or followers wait in bcast forever -- then both abort together."""
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    cp.fail_malloc = True                      # owner's _OwnedBlock raises
    with pytest.raises(SharedConstantsError, match='CUDA IPC is unusable'):
        peers[0].refresh(host())
    assert peers[0]._ipc_error is not None
    # The empty manifest reached the follower, which aborts rather than
    # blocking in bcast.
    cp.fail_malloc = False
    with pytest.raises(SharedConstantsError, match='CUDA IPC is unusable'):
        peers[1].refresh(host())


def test_follower_import_failure_aborts_the_group():
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    peers[0].refresh(host())
    cp.fail_open = True
    with pytest.raises(SharedConstantsError, match='CUDA IPC is unusable'):
        peers[1].refresh(host())
    assert peers[1]._ipc_error is not None


def test_the_abort_names_the_way_to_run_without_sharing():
    """An abort with no remedy would strand MIG users, who have no other way
    to run: discovery groups their instances as peers because they share a
    PCI bus id, and the import then necessarily fails."""
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    peers[0].refresh(host())
    cp.fail_open = True
    with pytest.raises(SharedConstantsError) as caught:
        peers[1].refresh(host())
    message = str(caught.value)
    assert 'PSANA_GPU_SHARE_CONSTANTS=0' in message
    assert 'ipcOpenMemHandle' in message, 'the cause is not in the message'


def test_an_ipc_failure_releases_what_was_built_before_aborting():
    """The abort must not leak the owner's allocation or a follower's mapping.

    Release goes through _release_shared, whose barriers every peer reaches,
    rather than happening on whichever rank noticed first.
    """
    decl = [('jf', 'pedestals'), ('jf', 'pixel_gain')]
    peers, cp = make_peers([decl, decl])
    peers[0].refresh(host())
    cp.fail_open = True
    with pytest.raises(SharedConstantsError):
        peers[1].refresh(host())
    assert peers[1]._imported == {}, 'a mapping was left open'
    assert peers[1].imported_bytes == 0


def test_close_after_an_abort_makes_no_collective():
    """The abort already released, so close() must not do it again.

    _settle releases through _release_shared, whose two barriers every peer
    reaches, and then raises. If it left _shared set, the close() that follows
    would take the sharing path and repeat the device sync and both barriers --
    matched only if every peer happens to call close(), which is the caller's
    property and not this class's to assume.
    """
    decl = [('jf', 'pedestals')]
    peers, cp, recorders = make_peers([decl, decl], record=True)
    peers[0].refresh(host())
    cp.fail_open = True
    with pytest.raises(SharedConstantsError):
        peers[1].refresh(host())

    # Scoped to the follower that aborted. Peers run sequentially here and
    # FakeComm's Iallreduce answers each rank with its own contribution, so
    # the owner cannot learn of the follower's failure: it returned from
    # refresh() believing sharing succeeded, and its close() then releases for
    # real. On MPI the reduction carries the flag and both ranks abort, which
    # the follower-import integration case covers.
    follower, recorder = peers[1], recorders[1]
    calls_after_abort = len(recorder.calls)
    syncs_after_abort = cp.syncs
    follower.close()
    assert len(recorder.calls) == calls_after_abort, \
        'close() after an abort made a collective'
    assert cp.syncs == syncs_after_abort, \
        'close() after an abort synchronised the device again'
    follower.close()                    # and still idempotent
    assert len(recorder.calls) == calls_after_abort


def test_an_abort_leaves_no_device_arrays_reachable():
    """get() must not hand back a view of memory the abort released."""
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    peers[0].refresh(host())
    cp.fail_open = True
    with pytest.raises(SharedConstantsError):
        peers[1].refresh(host())
    assert peers[1].shared_selectors == ()
    with pytest.raises(KeyError):
        peers[1].get('jf', 'pedestals')


def test_a_content_abort_releases_the_owners_allocation():
    """The content branch used to raise without releasing.

    At HEAD, `_settle`'s content branch raised immediately: the owner had
    already allocated and uploaded its un-pooled block by then, so every
    content abort leaked a full copy of the constants plus its budget
    reservation. Both branches now release through `_release_shared` before
    raising, which is also what keeps the two barriers matched across peers.

    Asserted on the OWNER, driven into the abort itself. The sequential fake
    cannot propagate the follower's verdict -- ``Allreduce`` answers each rank
    with its own vector -- so a follower-detected mismatch leaves the owner
    believing it succeeded. Injecting the error on the owner's own side is
    what makes it traverse the branch under test. That the whole group aborts
    together is covered on real MPI by the shared-constants driver's guard
    block.
    """
    decl = [('jf', 'pedestals')]
    budgets = [_GpuBudget(1 << 30), _GpuBudget(1 << 30)]
    peers, cp = make_peers([decl, decl], budgets=budgets)

    # Make the owner itself report a content error, so it reaches _settle's
    # content branch with its block already allocated and uploaded.
    owner = peers[0]
    real_publish = owner._publish

    def publish_then_reject(hosts):
        real_publish(hosts)
        return (['injected: a peer rejected the shared constants'], None)

    owner._publish = publish_then_reject

    with pytest.raises(SharedConstantsError, match='injected'):
        owner.refresh(host())

    # The block was allocated before the abort, and releasing is what returns
    # it. At HEAD this branch raised immediately and leaked both.
    assert cp.live == {}, "the owner's un-pooled block was leaked"
    assert budgets[0].committed() == 0, 'the budget charge was never returned'
    assert owner.shared_selectors == ()


def test_failed_publish_releases_its_budget_charge():
    """A block whose upload raises must not keep its charge.

    _OwnedBlock reserves budget and allocates in its constructor, so a later
    failure in view/set/handle left it unreferenced and uncharged-for.
    """
    decl = [('jf', 'pedestals')]
    budgets = [_GpuBudget(1 << 30), _GpuBudget(1 << 30)]
    peers, cp = make_peers([decl, decl], budgets=budgets)
    cp.fail_handle = True                  # ipcGetMemHandle raises after alloc
    with pytest.raises(SharedConstantsError):
        peers[0].refresh(host())
    assert budgets[0].committed() == 0, 'the failed block kept its charge'


def test_failed_import_closes_a_partly_built_block():
    """ipcOpenMemHandle can succeed while view() then raises.

    The block never reaches self._imported, so _release_shared would not
    close it -- and the owner frees its allocation after that barrier, with
    this rank's mapping still open, which CUDA forbids. The owner's half in
    _publish already handled the same window.
    """
    import psana.gpu.gpu_shared_constants as module

    original = module._ImportedBlock
    opened, closed = [], []

    class OpensThenFailsToView(original):
        def __init__(self, cp, handle, nbytes):
            super().__init__(cp, handle, nbytes)
            opened.append(id(self))          # the mapping is now open

        def view(self, *args, **kwargs):
            raise RuntimeError('injected: view fails after the mapping opened')

        def close(self):
            closed.append(id(self))
            return super().close()

    declared = [('jf', 'pedestals')]
    peers, _ = make_peers([declared, declared])
    module._ImportedBlock = OpensThenFailsToView
    try:
        peers[0].refresh(host())
        with pytest.raises(SharedConstantsError):
            peers[1].refresh(host())
    finally:
        module._ImportedBlock = original

    assert opened, 'the test did not reach the import'
    assert len(closed) == len(opened), (
        f'{len(opened) - len(closed)} mapping(s) left open; the owner frees '
        'its allocation underneath them')


# ---------------------------------------------------------------------------
# Collective order: every peer must make the same calls, in the same order
# ---------------------------------------------------------------------------
# Path divergence is what every hang in this work turned out to be, and the
# sequential fake cannot see two ranks blocked in different calls. The
# recorder is the only thing that detects it here.

def test_establishment_records_one_sequence_on_every_peer():
    """MPI pairs collectives by call order rather than by operation, so a rank
    that takes a different route does not raise -- it hangs the others."""
    decl = [('jf', 'pedestals')]
    peers, _, recorders = make_peers([decl, decl], record=True)
    for peer in peers:
        peer.refresh(host())

    sequences = {r.sequence_digest() for r in recorders}
    assert len(sequences) == 1, f'peers diverged at establishment: {sequences}'


def test_later_refreshes_make_no_collectives():
    """A second refresh must not collective at all.

    Under the one-set-per-run contract there is nothing to agree, and a
    collective here would be a liability: every peer would have to reach it,
    so any rank that skipped a transition would hang the rest.
    """
    decl = [('jf', 'pedestals')]
    peers, _, recorders = make_peers([decl, decl], record=True)
    for peer in peers:
        peer.refresh(host())
    after_establish = [len(r.calls) for r in recorders]

    for _ in range(3):
        for peer in peers:
            peer.refresh(host(pedestals=PEDS + 5))

    assert [len(r.calls) for r in recorders] == after_establish, \
        'a later refresh made a collective'


def test_close_records_one_sequence():
    decl = [('jf', 'pedestals')]
    peers, _, recorders = make_peers([decl, decl], record=True)
    for peer in peers:
        peer.refresh(host())
    for peer in peers:
        peer.close()
    sequences = {r.sequence_digest() for r in recorders}
    assert len(sequences) == 1


def test_disagreeing_peers_stay_on_one_path():
    """A group whose sets differ shares nothing, and both ranks must reach the
    same collectives getting there: the verdict comes from one allgather that
    every rank makes."""
    peers, _, recorders = make_peers(
        [[], [('jf', 'pedestals')]], record=True)
    sources = [{}, host()]
    for _ in range(3):
        for peer, source in zip(peers, sources):
            peer.refresh(source)

    sequences = {r.sequence_digest() for r in recorders}
    assert len(sequences) == 1, (
        'a rank declaring nothing diverged from one declaring a selector: '
        f'{[r.calls for r in recorders]}')


# ---------------------------------------------------------------------------
# A task that declares no constants
# ---------------------------------------------------------------------------

def test_empty_declaration_establishes_once():
    """Establishment is tracked by a flag, not by whether the selector tuple
    is non-empty: those two states are indistinguishable for a task that
    declared no constants, and conflating them re-runs establish on every
    transition -- collectives and a cache trim per step for a task that asked
    for nothing."""
    peers, _ = make_peers([[], []])
    trims = [0, 0]

    def trim(index):
        def record():
            trims[index] += 1
        return record

    for _ in range(4):
        for index, peer in enumerate(peers):
            peer.refresh({}, before_upload=trim(index))

    assert trims == [1, 1], f'cache trimmed {trims} times, expected once each'


def test_empty_declaration_reports_that_nothing_moved():
    """refresh() returns True only when something moved. Returning True for an
    empty declaration makes the caller recompute its subbatch budget every
    step."""
    peers, _ = make_peers([[], []])
    first = [peer.refresh({}) for peer in peers]
    later = [peer.refresh({}) for peer in peers]
    assert first == [False, False], 'nothing was declared, so nothing moved'
    assert later == [False, False]


def test_empty_declaration_keeps_peers_on_one_path():
    """Establish still runs once even with nothing declared, because
    _agree_on_set's allgather needs every rank. Skipping it on one rank would
    hang the others."""
    peers, _, recorders = make_peers([[], []], record=True)
    for _ in range(3):
        for peer in peers:
            peer.refresh({})
    sequences = {r.sequence_digest() for r in recorders}
    assert len(sequences) == 1, f'peers diverged: {sequences}'


# ---------------------------------------------------------------------------
# The budget must be sized before the shared copy is allocated (PR 175)
# ---------------------------------------------------------------------------

class WatchingBudget(_GpuBudget):
    """Records the limit in force at each reserve() call.

    The defect was an ordering one, so what matters is not the final limit
    but the limit the owner saw while allocating.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen = []

    def reserve(self, n):
        self.seen.append((int(n), int(self.limit())))
        return super().reserve(n)


def _wire_sizing(peer, budget, usable):
    """Attach the sizing hook exactly as GpuEventManager does."""
    from psana.gpu.gpu_placement import per_rank_limit
    placement = peer._placement
    placement.usable_bytes = usable

    def resize(shared_bytes):
        placement.shared_bytes = int(shared_bytes)
        limit = per_rank_limit(placement)
        if placement.is_owner:
            limit += int(shared_bytes)
        budget.set_limit(limit)

    peer._sizing = resize


def test_budget_is_resized_before_the_shared_copy_is_allocated():
    """The limits must be in place before the shared copy is allocated.

    A 12 GiB set on a 40 GiB four-peer device fits the documented accounting
    -- owner 7 + 12 = 19 GiB -- but charged against usable/peers (10 GiB) the
    shared copy is refused. Scaled down here to the fixture's heap.
    """
    usable, peer_count = 400 * 1024, 4
    shared = np.zeros((16 * 1024,), dtype=np.float32)     # 64 KiB
    declared = [('jf', 'pedestals')]

    budgets = [WatchingBudget(limit_bytes=usable // peer_count)
               for _ in range(peer_count)]
    peers, _ = make_peers([declared] * peer_count, budgets=budgets)
    for peer, budget in zip(peers, budgets):
        _wire_sizing(peer, budget, usable)

    naive_limit = usable // peer_count
    for peer in peers:
        peer.refresh({'jf': {'pedestals': shared}})

    # The owner allocates the shared block; the limit must already include it.
    owner_reserves = budgets[0].seen
    assert owner_reserves, 'the owner never reserved anything'
    first_bytes, limit_then = owner_reserves[0]
    assert first_bytes == shared.nbytes
    assert limit_then > naive_limit, (
        f'owner allocated {first_bytes} bytes against {limit_then}, the '
        f'un-resized {naive_limit}: the limit was raised too late')

    # And the accounting itself: shared counted once, owner gets it back.
    expected_share = (usable - shared.nbytes) // peer_count
    assert budgets[0].limit() == expected_share + shared.nbytes
    assert budgets[1].limit() == expected_share


def test_sizing_is_not_called_when_nothing_is_shared():
    """A group whose sets disagree shares nothing, so the budget must stay at
    usable/peers: calling sizing with this rank's own declaration would
    reserve device memory for a copy that is never made."""
    seen = []
    declared = [('jf', 'pedestals')]
    peers, _ = make_peers([[], declared])
    for peer in peers:
        peer._sizing = seen.append

    sources = [{}, host()]
    for peer, source in zip(peers, sources):
        peer.refresh(source)

    assert seen == [], f'sizing was called with {seen} though nothing is shared'


def test_explicit_budget_is_not_moved_by_sizing():
    """An explicit gpu_memory_budget_gb is the user's ceiling and
    discover_peers has already validated it against the group."""
    declared = [('jf', 'pedestals')]
    peers, _ = make_peers([declared, declared])
    # No sizing hook is wired when an explicit budget is in force, so the
    # limit must be whatever the caller set.
    before = [peer.budget.limit() for peer in peers]
    for peer in peers:
        peer.refresh(host())
    assert [peer.budget.limit() for peer in peers] == before
