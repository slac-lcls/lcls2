"""Shared constants degrade on set difference and abort on content difference.

The protocol is exercised against a fake device_comm that runs every peer in
one process, and a NumPy-backed CuPy stand-in, so no GPU is required. Device
behaviour that cannot be faithfully faked -- real IPC handles, real kernels --
is covered by the integration tests instead.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu.gpu_budget import _GpuBudget
from psana.gpu.gpu_placement import GpuPlacement
from psana.gpu.gpu_shared_constants import (
    SharedConstantsError, SharedRequestedConstants, digest_of,
)


PEDS = np.arange(24, dtype=np.float32).reshape(3, 2, 2, 2)
GAIN = (PEDS * 0.5 + 1).astype(np.float32)
STATUS = np.ones((3, 2, 2, 2), dtype=np.uint16)


def host(**overrides):
    values = {'pedestals': PEDS, 'pixel_gain': GAIN, 'pixel_status': STATUS}
    values.update(overrides)
    return {'jf': values}


class FakeComm:
    """Collective over peers that all live in this process.

    Real collectives block until every peer contributes. Peers here are driven
    sequentially, so instead of blocking, each collective records this rank's
    contribution and returns the full set assembled from values the harness
    already knows -- `group['declared']` for allgather, and a last-writer
    broadcast slot. Sufficient for intersection, manifest validation and case
    selection; real MPI ordering is covered by the integration tests.
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
        intersection does not depend on the order peers are driven in.
        Anything else -- the case letter, device identity -- is answered with
        this rank's own value repeated, which is exact whenever peers agree.
        Disagreement between peers is covered by the MPI integration test.
        """
        self.group['contrib'][self._rank] = value
        if isinstance(value, (set, frozenset)):
            return [set(d) for d in self.group['declared']]
        return [value] * self.Get_size()

    def bcast(self, value, root=0):
        """Round-aware broadcast.

        Peers are driven sequentially and there are now several broadcasts per
        transition (case letter, digests, manifest), so a single last-writer
        slot would hand a follower whichever value the owner wrote most
        recently. Count each rank's calls and match by position instead.
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

        ``force_cases`` makes the MIN/MAX case agreement disagree: the helper
        reduces an ordinal twice, so returning the lower bound for MIN and the
        upper for MAX simulates peers proposing different cases.

        A real allreduce blocks until every peer contributes. Here the first
        caller cannot see the others, so returning a partial reduction would
        make the owner act on an incomplete vote -- it would see "1 of 1
        succeeded" and then diverge from a follower that failed.

        Instead, treat this rank's own contribution as the result. That is
        exact whenever all peers contribute the same value, which is the case
        for both uses: the IPC-capability vote (all succeed or all fail
        together under the fake CuPy) and the error count (the fake never
        produces per-rank IPC errors). Genuine per-rank divergence is covered
        by the MPI integration test, which this fake cannot model.
        """
        self.group.setdefault('reduce', []).append((self._rank, value))
        name = getattr(op, 'name', str(op)).upper()
        forced = self.group.get('force_cases')
        if forced is not None and value in (0, 1, 2):
            codes = ['ABC'.index(c) for c in forced]
            return min(codes) if 'MIN' in name else max(codes)
        if 'MIN' in name:
            return value
        return value * self.Get_size() if value else 0

    def Allreduce(self, sendbuf, recvbuf, op=None):
        """Buffer-form reduction, used for the single agreement step.

        Peers run sequentially here, so this rank's own vector is the result.
        That is exact for the agreement step's purpose in these tests: the
        fake CuPy either works for every peer or for none, and content
        disagreement is injected on the rank being asserted. Genuine per-rank
        divergence is covered by the MPI integration test.
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


class FakeOp:
    def __init__(self, name):
        self.name = name


def make_peers(declarations, *, budgets=None, uuid='abcd' * 8):
    """One SharedRequestedConstants per peer, sharing a fake device_comm."""
    group = {'declared': [list(d) for d in declarations],
             'contrib': {}, 'bcast': None}
    cp = FakeCupy()
    peers = []
    for rank, declared in enumerate(declarations):
        comm = FakeComm(group, rank)
        placement = GpuPlacement(
            device_pci=uuid, device_comm=comm,
            is_owner=(rank == 0), n_device_peers=len(declarations),
            usable_bytes=1 << 30)
        budget = (budgets[rank] if budgets else _GpuBudget(1 << 30))
        peers.append(SharedRequestedConstants(declared, budget, placement,
                                              cp=cp))
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
        itemsize = np.dtype(dtype).itemsize
        offset = int(memptr.ptr)
        view = np.frombuffer(self.heap, dtype=dtype, count=count,
                             offset=offset).reshape(shape)
        return view.view(FakeArray)

    def empty(self, shape, dtype=None):
        return np.empty(shape, dtype=dtype).view(FakeArray)

    def asarray(self, a):
        return np.asarray(a).view(FakeArray)




# ---------------------------------------------------------------------------
# Intersection: differing selector sets degrade
# ---------------------------------------------------------------------------

def test_identical_declarations_share_everything():
    decl = [('jf', 'pedestals'), ('jf', 'pixel_gain')]
    peers, cp = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    for peer in peers:
        assert peer.shared_selectors == tuple(decl)
        assert peer.private_selectors == ()
    # One device copy, not two.
    assert len(cp.live) == 2          # pedestals + gain, owned once each


def test_differing_sets_share_the_intersection():
    """#168 lists differing selectors as a supported case, so a selector a
    peer did not request must not fail the job."""
    owner = [('jf', 'pedestals'), ('jf', 'pixel_gain'), ('jf', 'pixel_status')]
    follower = [('jf', 'pedestals'), ('jf', 'pixel_gain')]
    peers, _ = make_peers([owner, follower])
    for peer in peers:
        peer.refresh(host())

    shared = (('jf', 'pedestals'), ('jf', 'pixel_gain'))
    assert peers[0].shared_selectors == shared
    assert peers[1].shared_selectors == shared
    assert peers[0].private_selectors == (('jf', 'pixel_status'),)
    assert peers[1].private_selectors == ()


def test_every_declared_constant_is_readable_through_either_path():
    owner = [('jf', 'pedestals'), ('jf', 'pixel_status')]
    follower = [('jf', 'pedestals')]
    peers, _ = make_peers([owner, follower])
    for peer in peers:
        peer.refresh(host())
    np.testing.assert_array_equal(peers[0].get('jf', 'pedestals'), PEDS)
    np.testing.assert_array_equal(peers[0].get('jf', 'pixel_status'), STATUS)
    np.testing.assert_array_equal(peers[1].get('jf', 'pedestals'), PEDS)


def test_empty_intersection_shares_nothing_without_error():
    peers, cp = make_peers([[('jf', 'pedestals')], [('jf', 'pixel_gain')]])
    for peer in peers:
        peer.refresh(host())
    for peer in peers:
        assert peer.shared_selectors == ()
        assert peer.shared_bytes == 0
    np.testing.assert_array_equal(peers[1].get('jf', 'pixel_gain'), GAIN)


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


def test_owner_also_charges_its_private_remainder():
    owner = [('jf', 'pedestals'), ('jf', 'pixel_status')]
    follower = [('jf', 'pedestals')]
    budgets = [_GpuBudget(1 << 30), _GpuBudget(1 << 30)]
    peers, _ = make_peers([owner, follower], budgets=budgets)
    for peer in peers:
        peer.refresh(host())
    # Shared bytes plus the private selector nobody else wanted.
    assert budgets[0].committed() >= PEDS.nbytes + STATUS.nbytes
    assert budgets[1].committed() == 0


def test_shared_bytes_are_reported_by_both_roles():
    decl = [('jf', 'pedestals')]
    peers, _ = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    assert peers[0].shared_bytes == PEDS.nbytes
    assert peers[1].shared_bytes == PEDS.nbytes


# ---------------------------------------------------------------------------
# BeginStep: three cases, decided by the owner
# ---------------------------------------------------------------------------

def test_a_second_refresh_does_nothing():
    """One constant set per run: establish once, then no-op.

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
    assert np.asarray(kept) is not None


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


# ---------------------------------------------------------------------------
# Fallbacks: degrade to a private copy, never to a wrong device
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


def test_mig_peers_degrade_rather_than_being_detected():
    """MIG is handled by the capability path, not by detection.

    Grouping is by PCI bus id, which MIG instances of one card share, so two
    ranks on different instances look like peers and sharing is attempted.
    The IPC import then fails -- CUDA IPC cannot span instances, which have
    separate memory -- and the group degrades to private copies together.

    Detecting MIG up front would need an instance UUID, which neither CUDA nor
    plain NVML enumeration provides; the grouping comment in
    gpu_placement.discover_peers records what that would take. The capability
    path has to exist regardless, for IPC failures unrelated to MIG, so this
    test drives MIG through it.
    """
    import psana.gpu.gpu_shared_constants as module

    declared = [('jf', 'pedestals')]
    peers, _ = make_peers([declared, declared])
    original = module._ImportedBlock

    class CannotSpanInstances(original):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('cudaIpcOpenMemHandle: invalid device context')

    module._ImportedBlock = CannotSpanInstances
    try:
        for peer in peers:
            peer.refresh(host())           # must not raise
    finally:
        module._ImportedBlock = original

    follower = peers[1]
    assert follower.shared_selectors == ()
    assert follower.private_selectors == tuple(declared)
    assert 'fallback' in follower._placement.sharing


def test_digest_detects_content_and_ignores_identity():
    a = np.arange(8, dtype=np.float32)
    assert digest_of(a) == digest_of(a.copy())
    assert digest_of(a) != digest_of(a + 1)
    # Non-contiguous input must still digest its logical content.
    assert digest_of(a.reshape(2, 4)[:, ::2]) == digest_of(
        np.ascontiguousarray(a.reshape(2, 4)[:, ::2]))


# ---------------------------------------------------------------------------
# Failure paths: no exception may skip a collective
# ---------------------------------------------------------------------------
# The follow-up review found that an exception could leave peers in different
# collectives: MPI pairs them by call order, not by operation, so a rank that
# reaches a different one gives undefined results and then hangs. These
# assert that each failure still produces the same sequence of calls.

def collective_trace(group):
    """Calls recorded by FakeComm, as (kind, rank) pairs."""
    trace = []
    for rank, _ in group.get('allreduce', []):
        trace.append(('Allreduce', rank))
    return trace


def test_owner_failing_before_bcast_still_publishes_a_marker():
    """If the owner's allocation or upload raises it must still broadcast,
    or followers wait in bcast forever."""
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    cp.fail_malloc = True                      # owner's _OwnedBlock raises
    peers[0].refresh(host())
    # An empty manifest was published rather than nothing at all.
    assert peers[0].shared_selectors == ()     # degraded to private
    assert peers[0]._ipc_error is not None
    cp.fail_malloc = False
    peers[1].refresh(host())
    assert peers[1].shared_selectors == ()
    # Values are still correct through the private path.
    np.testing.assert_array_equal(peers[1].get('jf', 'pedestals'), PEDS)


def test_follower_import_failure_degrades_the_group():
    """A follower's ipcOpenMemHandle failing must not leave it in a different
    collective from its peers."""
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    peers[0].refresh(host())
    cp.fail_open = True
    peers[1].refresh(host())
    assert peers[1]._ipc_error is not None
    assert peers[1].shared_selectors == ()     # fell back, did not hang
    np.testing.assert_array_equal(peers[1].get('jf', 'pedestals'), PEDS)


def test_fallback_closes_importers_before_the_owner_frees():
    """CUDA requires imported mappings closed before the exporter frees, and
    a follower may have opened some handles before the failure.

    Asserted on the follower only: peers are driven sequentially here, so the
    owner has already returned from refresh() and cannot re-enter the
    fallback. The owner's half of the ordering -- Barrier, then free -- is
    covered by the MPI integration test.
    """
    decl = [('jf', 'pedestals'), ('jf', 'pixel_gain')]
    peers, cp = make_peers([decl, decl])
    peers[0].refresh(host())
    cp.fail_open = True
    peers[1].refresh(host())
    # The follower released every mapping it had opened before failing, and
    # holds no shared state afterwards.
    assert peers[1]._imported == {}
    assert peers[1].shared_selectors == ()
    assert peers[1].imported_bytes == 0


def test_failed_publish_releases_its_budget_charge():
    """A block whose upload raises must not keep its charge.

    _OwnedBlock reserves budget and allocates in its constructor, so a later
    failure in view/set/handle left it unreferenced and uncharged-for -- and
    the fallback then uploaded privately, making the owner pay twice.
    """
    decl = [('jf', 'pedestals')]
    budgets = [_GpuBudget(1 << 30), _GpuBudget(1 << 30)]
    peers, cp = make_peers([decl, decl], budgets=budgets)
    cp.fail_handle = True                  # ipcGetMemHandle raises after alloc
    peers[0].refresh(host())
    cp.fail_handle = False
    peers[1].refresh(host())

    # Charged exactly once: for the private copy it fell back to.
    assert budgets[0].committed() == PEDS.nbytes


# ---------------------------------------------------------------------------
# A fallback must be visible
# ---------------------------------------------------------------------------
# Degrading keeps the job running while every peer holds its own copy of the
# constants, so a device that held one copy holds n. Previously the only trace
# was shared_bytes dropping to zero.

def test_successful_sharing_is_reported_as_on():
    decl = [('jf', 'pedestals')]
    peers, _ = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    for peer in peers:
        assert peer._placement.sharing == 'on'
        assert 'sharing=on' in peer._placement.describe()


def test_fallback_is_logged_with_its_reason_and_cost(caplog):
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    peers[0].refresh(host())
    cp.fail_open = True
    with caplog.at_level('WARNING'):
        peers[1].refresh(host())

    messages = [r.getMessage() for r in caplog.records]
    assert any('fell back to private copies' in m for m in messages)
    # The reason, and the memory it now costs, must both be in the line.
    assert any('ipcOpenMemHandle' in m for m in messages)
    assert any('each hold' in m for m in messages)
    # Bytes as well as MiB: a small constant set would otherwise log as
    # "0.0 MiB" and read like a broken calculation.
    assert any('bytes' in m for m in messages)


def test_fallback_is_recorded_on_the_placement():
    decl = [('jf', 'pedestals')]
    peers, cp = make_peers([decl, decl])
    peers[0].refresh(host())
    cp.fail_open = True
    peers[1].refresh(host())
    state = peers[1]._placement.sharing
    assert state.startswith('fallback(')
    assert 'ipcOpenMemHandle' in state
    # And it shows in the placement line, where a regression would be noticed.
    assert 'sharing=fallback(' in peers[1]._placement.describe()


def test_solo_rank_reports_sharing_off():
    peers, _ = make_peers([[('jf', 'pedestals')]])
    peers[0].refresh(host())
    assert peers[0]._placement.sharing == 'off'


# ---------------------------------------------------------------------------
# Collective order: every peer must make the same calls, in the same order
# ---------------------------------------------------------------------------
# Path divergence is what every hang on this branch turned out to be, and the
# sequential fake cannot see two ranks blocked in different calls. The order
# checker is the only thing that detects it, so these assert it is actually
# wired in and that each path records a consistent sequence.

def peers_with_checking(declarations, monkeypatch, **kwargs):
    monkeypatch.setenv('PSANA_GPU_CHECK_COLLECTIVES', '1')
    return make_peers(declarations, **kwargs)


def test_checker_is_wired_in_when_enabled(monkeypatch):
    """Previously the recorder existed but nothing turned it on, so it had
    never run."""
    decl = [('jf', 'pedestals')]
    peers, _ = peers_with_checking([decl, decl], monkeypatch)
    for peer in peers:
        peer.refresh(host())
    for peer in peers:
        assert peer._recorder is not None
        assert peer._recorder.calls, 'no collectives recorded'


def test_checker_is_absent_when_disabled(monkeypatch):
    monkeypatch.delenv('PSANA_GPU_CHECK_COLLECTIVES', raising=False)
    decl = [('jf', 'pedestals')]
    peers, _ = make_peers([decl, decl])
    for peer in peers:
        peer.refresh(host())
    assert all(peer._recorder is None for peer in peers)


def test_establishment_records_one_sequence_on_every_peer(monkeypatch):
    """Peers must make the same collectives in the same order.

    Establishment is now the only path that collectives at all, so this is
    where divergence could hide. MPI pairs collectives by call order rather
    than by operation, so a rank that takes a different route does not raise
    -- it hangs the others.
    """
    decl = [('jf', 'pedestals')]
    peers, _ = peers_with_checking([decl, decl], monkeypatch)
    for peer in peers:
        peer.refresh(host())

    sequences = {peer._recorder.sequence_digest() for peer in peers}
    assert len(sequences) == 1, f'peers diverged at establishment: {sequences}'


def test_later_refreshes_make_no_collectives(monkeypatch):
    """A second refresh must not collective at all.

    Under the one-set-per-run contract there is nothing to agree, and a
    collective here would be a liability: every peer would have to reach it,
    so any rank that skipped a transition would hang the rest.
    """
    decl = [('jf', 'pedestals')]
    peers, _ = peers_with_checking([decl, decl], monkeypatch)
    for peer in peers:
        peer.refresh(host())
    after_establish = [len(peer._recorder.calls) for peer in peers]

    for _ in range(3):
        for peer in peers:
            peer.refresh(host(pedestals=PEDS + 5))

    assert [len(p._recorder.calls) for p in peers] == after_establish, \
        'a later refresh made a collective'


def test_fallback_path_is_recorded(monkeypatch):
    """The degrade path closes, barriers and re-uploads, and each of those is
    recorded.

    Sequence *equality* is deliberately not asserted: peers run sequentially
    in this fake, so the owner has already returned from refresh() and never
    re-enters the degrade path. Agreement on the fallback needs real
    concurrency, which the fail-follower-import integration case provides.
    """
    decl = [('jf', 'pedestals')]
    peers, cp = peers_with_checking([decl, decl], monkeypatch)
    peers[0].refresh(host())
    cp.fail_open = True
    peers[1].refresh(host())
    # The follower recorded the extra collectives the degrade performs.
    assert len(peers[1]._recorder.calls) > len(peers[0]._recorder.calls)
    assert 'Ibarrier' in peers[1]._recorder.calls


def test_close_records_one_sequence(monkeypatch):
    decl = [('jf', 'pedestals')]
    peers, _ = peers_with_checking([decl, decl], monkeypatch)
    for peer in peers:
        peer.refresh(host())
    for peer in peers:
        peer.close()
    sequences = {peer._recorder.sequence_digest() for peer in peers}
    assert len(sequences) == 1


# ---------------------------------------------------------------------------
# A task that declares no constants (PR 175 review)
# ---------------------------------------------------------------------------

def test_empty_declaration_establishes_once():
    """Establishment is tracked by a flag, not by whether the selector tuples
    are non-empty: those two states are indistinguishable for a task that
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


def test_empty_declaration_keeps_peers_on_one_path(monkeypatch):
    """Establish still runs once even with nothing declared, because peers may
    declare different sets by design and _intersect's allgather needs every
    rank. Skipping it on one rank would hang the others."""
    peers = peers_with_checking([[], []], monkeypatch)[0]
    for _ in range(3):
        for peer in peers:
            peer.refresh({})
    sequences = {peer._recorder.sequence_digest() for peer in peers}
    assert len(sequences) == 1, f'peers diverged: {sequences}'


def test_one_peer_declaring_nothing_still_agrees():
    """The degrade design permits unequal declarations. The rank that declared
    nothing must still take part in the intersection, and neither rank may
    end up sharing a selector the other never asked for."""
    declared = [('jf', 'pedestals')]
    peers, _ = make_peers([[], declared])
    sources = [{}, host()]
    for peer, source in zip(peers, sources):
        peer.refresh(source)

    assert peers[0].shared_selectors == ()
    assert peers[1].shared_selectors == (), \
        'a selector only one peer declared must not be shared'
    assert peers[1].private_selectors == tuple(declared)


# ---------------------------------------------------------------------------
# The budget must be sized before the intersection is allocated (PR 175)
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


def test_budget_is_resized_before_the_intersection_is_allocated():
    """The limits must be in place before the intersection is allocated.

    A 12 GiB intersection on a 40 GiB four-peer device fits the documented
    accounting -- owner 7 + 12 = 19 GiB -- but charged against usable/peers
    (10 GiB) the shared copy is refused, the group degrades, and the private
    copy is refused by the same limit. Scaled down here to the fixture's heap.
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


def test_sizing_is_not_called_before_the_intersection_is_known():
    """Sizing must see the negotiated intersection, not this rank's
    declaration: a selector only one peer declared is private, not shared."""
    seen = []
    declared = [('jf', 'pedestals')]
    peers, _ = make_peers([[], declared])
    for peer in peers:
        peer._sizing = seen.append

    sources = [{}, host()]
    for peer, source in zip(peers, sources):
        peer.refresh(source)

    # The intersection is empty, so no shared bytes on either rank.
    assert seen == [0, 0], f'sizing saw {seen}, expected no shared bytes'


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


def test_mixed_declarations_do_not_diverge_across_transitions(monkeypatch):
    """Unequal declarations must not split the collective path.

    A rank that re-enters _establish begins with _intersect's allgather while
    a peer on the case path begins with the case allreduce -- measured as
    ['allgather', 'allgather'] against ['allgather', 'Iallreduce',
    'Iallreduce']. MPI pairs collectives by call order, so that is a hang
    rather than wasted work.
    """
    peers = peers_with_checking([[], [('jf', 'pedestals')]], monkeypatch)[0]
    sources = [{}, host()]
    for _ in range(3):
        for peer, source in zip(peers, sources):
            peer.refresh(source)

    sequences = {peer._recorder.sequence_digest() for peer in peers}
    assert len(sequences) == 1, (
        'a rank declaring nothing diverged from one declaring a selector: '
        f'{[peer._recorder.calls for peer in peers]}')


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
        for peer in peers:
            peer.refresh(host())
    finally:
        module._ImportedBlock = original

    assert opened, 'the test did not reach the import'
    assert len(closed) == len(opened), (
        f'{len(opened) - len(closed)} mapping(s) left open; the owner frees '
        'its allocation underneath them')
    # Scoped to the follower that failed. FakeComm's Iallreduce answers each
    # rank with its own contribution, so a capability flag cannot propagate
    # here and the owner still reports sharing. Whether the whole group
    # degrades together is asserted on real MPI by the follower-import case
    # in mpi_failure_paths.py, which is where it is observable.
    follower = peers[1]
    assert follower.shared_selectors == ()
    assert follower.private_selectors == tuple(declared)


def test_fallback_returns_the_sharing_budget_before_uploading_privately():
    """Sizing the budget before the exchange made the fallback abort.

    The limits are set for ONE device copy -- followers at
    (usable - shared)/peers -- but a capability failure means every rank
    uploads its own full copy. Leaving the sharing limits in place makes that
    upload fail with GpuMemoryPressureError, turning a clean degrade into an
    abort: on a 40 GiB four-peer device a 9-10 GiB intersection fits
    usable/peers (10 GiB) but not (usable - shared)/peers (7 GiB).

    Scaled to the fixture heap: usable 400 KiB, 4 peers, shared 100 KiB, so
    usable/peers == 100 KiB fits and (usable - shared)/peers == 75 KiB does
    not.
    """
    import psana.gpu.gpu_shared_constants as module
    from psana.gpu.gpu_placement import per_rank_limit

    usable, peer_count = 400 * 1024, 4
    shared = np.zeros((25 * 1024,), dtype=np.float32)        # 100 KiB
    assert shared.nbytes <= usable // peer_count
    assert shared.nbytes > (usable - shared.nbytes) // peer_count

    declared = [('jf', 'pedestals')]
    budgets = [_GpuBudget(limit_bytes=usable // peer_count)
               for _ in range(peer_count)]
    peers, _ = make_peers([declared] * peer_count, budgets=budgets)
    for peer, budget in zip(peers, budgets):
        _wire_sizing(peer, budget, usable)

    # Fail the owner's export, so the owner takes the fallback itself rather
    # than learning about it through an allreduce the fake cannot propagate.
    original = module._OwnedBlock

    class FailsToExport(original):
        def handle(self):
            raise RuntimeError('injected: ipcGetMemHandle failed')

    module._OwnedBlock = FailsToExport
    try:
        for peer in peers:
            # Must not raise: the whole point of the fallback is to keep the
            # job running on private copies.
            peer.refresh({'jf': {'pedestals': shared}})
    finally:
        module._OwnedBlock = original

    expected = usable // peer_count
    for index, budget in enumerate(budgets):
        assert budget.limit() == expected, (
            f'peer {index} kept the sharing limit {budget.limit()} instead of '
            f'returning to {expected}; its private copy would be refused')
    assert peers[0]._placement.shared_bytes == 0
    assert all(peer.shared_selectors == () for peer in peers)


