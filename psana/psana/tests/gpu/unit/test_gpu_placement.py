"""Device selection happens inside discovery, and refuses to guess.

Selection needs only a communicator and CuPy, both injected here, so these run
without a GPU. What they cover is the assignment rule and the failure
reporting; that the assignment actually reaches a running job is covered by
``integration/mpi_calib_pixel_exact.py``, which goes through a real
DataSource.
"""
import sys
from types import SimpleNamespace as NS

import pytest

from psana.gpu import gpu_placement as gp
from psana.gpu.gpu_placement import (
    GpuPlacement, GpuPlacementError, device_pci, discover_peers,
    per_rank_limit, release_placement,
)


# What the driver returns, and the normalised form comparisons use:
# _pci_norm strips the domain's leading zeros so nvidia-smi's
# eight-digit domain and CUDA's four-digit one compare equal.
PCI_A = '0000:01:00.0'
PCI_B = '0000:41:00.0'
NORM_A = '0:01:00.0'
NORM_B = '0:41:00.0'
GIB = 1024 ** 3


class Comm:
    """Communicator over peers driven sequentially in one process.

    Each rank's splits return a child whose rank/size the caller supplies, so
    a specific node and device layout can be set up without real MPI.
    """

    def __init__(self, *, rank=0, size=1, node_rank=0, node_size=1,
                 gpu_rank=0, gpu_size=1, log=None):
        self._rank, self._size = rank, size
        self._node = (node_rank, node_size)
        self._gpu = (gpu_rank, gpu_size)
        self.log = log if log is not None else []
        self._splits = 0

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def allgather(self, value):
        self.log.append(('allgather', value))
        return [value] * self._size

    def Split(self, color, key):
        self._splits += 1
        self.log.append(('split', color))
        if self._splits == 1:
            node_rank, node_size = self._node
            child = Comm(rank=node_rank, size=node_size,
                         gpu_rank=self._gpu[0], gpu_size=self._gpu[1],
                         log=self.log)
            child._node = self._gpu          # its first split is the GPU one
            return child
        gpu_rank, gpu_size = self._gpu
        return Comm(rank=gpu_rank, size=gpu_size, log=self.log)

    def Free(self):
        self.log.append(('free', None))

    def Ibarrier(self):
        self.log.append(('barrier', None))
        return NS(Test=lambda: True)

    def Iallreduce(self, sendbuf, recvbuf, op=None):
        self.log.append(('allreduce', op))
        recvbuf[:] = sendbuf
        return NS(Test=lambda: True)

    def Abort(self, code=1):
        raise AssertionError(f'unexpected Abort({code}) in a unit test')


def fake_cupy(*, count=2, pci=None, free=40 * GIB, used=None):
    """Minimal CuPy stand-in. `used` collects ordinals passed to use()."""
    used = [] if used is None else used
    table = pci or {0: PCI_A, 1: PCI_B}

    def get_bus_id(ordinal):
        return table[int(ordinal)]

    # Discovery releases the cached pool before sampling free memory, so the
    # stand-in needs the pool API too.
    pool = NS(free_all_blocks=lambda: None)
    return NS(cuda=NS(
        runtime=NS(getDeviceCount=lambda: count,
                   deviceGetPCIBusId=get_bus_id,
                   memGetInfo=lambda: (free, free)),
        Device=lambda ordinal=0: NS(id=ordinal,
                                    use=lambda: used.append(ordinal))),
              get_default_memory_pool=lambda: pool), used


@pytest.fixture(autouse=True)
def fake_mpi(monkeypatch):
    """discover_peers imports mpi4py for Get_processor_name and the ops."""
    mpi = NS(Get_processor_name=lambda: 'sdfampere001', UNDEFINED=-32766,
             SUM='SUM', MIN='MIN', MAX='MAX')
    monkeypatch.setitem(sys.modules, 'mpi4py', NS(MPI=mpi))
    return mpi


# ---------------------------------------------------------------------------
# Assignment: balanced, and independent of any mask ordering
# ---------------------------------------------------------------------------

def test_rank_index_spreads_workers_over_devices():
    """Index within the node's GPU workers, modulo the visible count.

    CUDA has already applied any launcher mask to the visible set, so an
    index spreads workers evenly however that mask was ordered. Indexing into
    CUDA_VISIBLE_DEVICES entries instead would put masks '2,3' and '3,2' on
    the same device and leave the other idle.
    """
    chosen = []
    for gpu_rank in range(4):
        cp, used = fake_cupy(count=2)
        discover_peers(Comm(rank=gpu_rank, size=4, node_size=4,
                            gpu_rank=gpu_rank, gpu_size=4),
                       is_gpu_worker=True, cp=cp)
        chosen.append(used[-1])
    assert chosen == [0, 1, 0, 1], 'workers must alternate across devices'


def test_opposite_mask_orders_still_use_both_devices():
    """The index is into the SORTED visible PCI ids, not CUDA's ordinals.

    CUDA numbers visible devices in the order the mask lists them, so a rank
    launched with `2,3` and one with `3,2` see the same two cards under
    swapped ordinals. Indexing ordinals directly sends both to the same
    physical GPU and leaves the other idle; indexing the sorted PCI sequence
    gives each rank a different card without any collective.
    """
    # Same two cards, opposite visible order -- what the two masks produce.
    masks = {0: {0: PCI_A, 1: PCI_B},
             1: {0: PCI_B, 1: PCI_A}}
    landed = []
    for gpu_rank, table in masks.items():
        cp, used = fake_cupy(count=2, pci=table)
        placement = discover_peers(
            Comm(rank=gpu_rank, size=2, node_size=2,
                 gpu_rank=gpu_rank, gpu_size=2),
            is_gpu_worker=True, cp=cp)
        # Record the PHYSICAL card, not the ordinal: the ordinals differ by
        # construction, which is the whole point.
        landed.append(placement.device_pci)
        assert used, 'no device was made current'

    assert len(set(landed)) == 2, (
        f'both ranks landed on {landed[0]}; one card is doing double duty '
        'while the other sits idle')


def test_identical_masks_still_round_robin():
    """The ordinary case must not regress: with the same visible order,
    workers alternate as before."""
    landed = []
    for gpu_rank in range(4):
        cp, _ = fake_cupy(count=2)
        placement = discover_peers(
            Comm(rank=gpu_rank, size=4, node_size=4,
                 gpu_rank=gpu_rank, gpu_size=4),
            is_gpu_worker=True, cp=cp)
        landed.append(placement.device_pci)
    assert landed == [NORM_A, NORM_B, NORM_A, NORM_B]


def test_single_visible_device_is_used_by_every_worker():
    """A launcher mask of one device is just a visible count of one."""
    for gpu_rank in range(3):
        cp, used = fake_cupy(count=1, pci={0: PCI_A})
        discover_peers(Comm(rank=gpu_rank, size=3, node_size=3,
                            gpu_rank=gpu_rank, gpu_size=3),
                       is_gpu_worker=True, cp=cp)
        assert used == [0]


def test_selection_records_the_ordinal_for_other_threads():
    """use() is per-thread, so the ordinal is kept for any worker thread that
    has to make the same device current later."""
    cp, _ = fake_cupy(count=2)
    placement = discover_peers(Comm(gpu_rank=1, gpu_size=2, size=2,
                                    node_size=2),
                               is_gpu_worker=True, cp=cp)
    assert placement.device_ordinal == 1
    assert placement.visible_count == 2


# ---------------------------------------------------------------------------
# Per-run rediscovery
# ---------------------------------------------------------------------------

def test_discovery_releases_the_cached_pool_before_sizing():
    """Discovery runs per run, so run N+1 must not be sized against run N.

    CuPy's pool keeps what the previous run allocated until it is asked to
    release it, and `usable` is a MIN over the device group -- so one rank
    holding a stale pool drags every peer on its card down, not just itself.
    The release has to happen before memGetInfo is sampled, which is what this
    pins.
    """
    order = []
    GIB = 1024 ** 3

    class Pool:
        def free_all_blocks(self):
            order.append('free_all_blocks')

    def mem_info():
        order.append('memGetInfo')
        return (40 * GIB, 40 * GIB)

    cp = NS(cuda=NS(
        runtime=NS(getDeviceCount=lambda: 1,
                   deviceGetPCIBusId=lambda i: PCI_A,
                   memGetInfo=mem_info),
        Device=lambda ordinal=0: NS(id=ordinal, use=lambda: None)),
        get_default_memory_pool=lambda: Pool())

    discover_peers(Comm(), is_gpu_worker=True, cp=cp, headroom=0.0)

    assert 'free_all_blocks' in order, 'the cached pool was never released'
    assert order.index('free_all_blocks') < order.index('memGetInfo'), (
        f'free memory was sampled before the pool was released: {order}; '
        'a second run would size itself against the first run\'s allocations')


def test_discovery_survives_a_pool_that_cannot_be_released():
    """Releasing the pool is an optimisation for the NEXT run, not a
    precondition for this one: a CuPy build without the pool API, or a release
    that raises, must not fail a run whose device is already selected."""
    def explode():
        raise RuntimeError('no default memory pool')

    cp = NS(cuda=NS(
        runtime=NS(getDeviceCount=lambda: 1,
                   deviceGetPCIBusId=lambda i: PCI_A,
                   memGetInfo=lambda: (40 * GIB, 40 * GIB)),
        Device=lambda ordinal=0: NS(id=ordinal, use=lambda: None)),
        get_default_memory_pool=explode)

    placement = discover_peers(Comm(), is_gpu_worker=True, cp=cp,
                               headroom=0.0)
    assert placement.device_pci == NORM_A
    assert placement.usable_bytes == 40 * GIB


def test_a_masked_device_is_made_current_by_its_local_ordinal():
    """A non-default physical GPU must become current by its LOCAL ordinal.

    CUDA renumbers a mask to 0..n-1, so with `CUDA_VISIBLE_DEVICES=3,4` the
    rank that should run on physical GPU 4 has to call `use(1)`, not `use(4)`.
    Anything that consumes the device afterwards -- a user kernel, or an
    external framework such as GLINT -- inherits the current device, so
    selecting the right physical card but the wrong local ordinal would put
    application state on a device this rank does not own.
    """
    # mask '3,4' as CUDA presents it: local ordinal -> physical PCI id.
    masked = {0: '0:03:00.0', 1: '0:04:00.0'}
    landed = {}
    for gpu_rank in (0, 1):
        cp, used = fake_cupy(count=2, pci=masked)
        placement = discover_peers(
            Comm(rank=gpu_rank, size=2, node_size=2,
                 gpu_rank=gpu_rank, gpu_size=2),
            is_gpu_worker=True, cp=cp)
        # use() must have been called with the LOCAL ordinal...
        assert used, 'no device was made current'
        assert used[-1] == placement.device_ordinal
        # ...and the ordinal must name the card the placement reports.
        assert masked[placement.device_ordinal] == placement.device_pci
        landed[gpu_rank] = (placement.device_ordinal, placement.device_pci)

    # Both cards used, and the second physical GPU is reached as ordinal 1.
    assert {v[1] for v in landed.values()} == {'0:03:00.0', '0:04:00.0'}
    by_pci = {pci: ordinal for ordinal, pci in landed.values()}
    assert by_pci['0:04:00.0'] == 1, (
        'the rank on the second masked GPU must make local device 1 current, '
        f'got {by_pci["0:04:00.0"]}')


# ---------------------------------------------------------------------------
# Identity: PCI bus id from the driver, never CuPy's truncated UUID
# ---------------------------------------------------------------------------

def test_grouping_key_is_the_driver_pci_id():
    cp, _ = fake_cupy(count=2)
    placement = discover_peers(Comm(gpu_rank=1, gpu_size=2, size=2,
                                    node_size=2),
                               is_gpu_worker=True, cp=cp)
    assert placement.device_pci == NORM_B


def test_pci_ids_are_normalised_for_comparison():
    """nvidia-smi prints an eight-digit domain, CUDA four."""
    cp, _ = fake_cupy(pci={0: '00000000:01:00.0'})
    assert device_pci(cp, 0) == NORM_A


def test_uuid_is_never_read_from_cuda():
    """getDeviceProperties()['uuid'] is 16 bytes truncated at the first NUL,
    so a UUID containing a zero byte yields a short, wrong key -- measured on
    sdfampere014, where c9ef90546100fd2e... arrived as five bytes.

    Asserted behaviourally: a CuPy whose getDeviceProperties raises must not
    disturb discovery, which it would if the grouping key came from there.
    """
    def explode(ordinal):
        raise AssertionError('getDeviceProperties must not be consulted')

    cp = NS(cuda=NS(
        runtime=NS(getDeviceCount=lambda: 2,
                   deviceGetPCIBusId=lambda i: {0: PCI_A, 1: PCI_B}[int(i)],
                   getDeviceProperties=explode,
                   memGetInfo=lambda: (40 * GIB, 40 * GIB)),
        Device=lambda ordinal=0: NS(id=ordinal, use=lambda: None)),
        get_default_memory_pool=lambda: NS(free_all_blocks=lambda: None))
    placement = discover_peers(Comm(gpu_rank=1, gpu_size=2, size=2,
                                    node_size=2),
                               is_gpu_worker=True, cp=cp)
    assert placement.device_pci == NORM_B


# ---------------------------------------------------------------------------
# Failures are reported, never guessed around
# ---------------------------------------------------------------------------

def test_no_visible_device_is_a_configuration_error():
    """An empty CUDA_VISIBLE_DEVICES reaches CUDA as zero devices. For a GPU
    role that is a configuration error, and the message has to say so rather
    than letting Device(0) raise a bare CUDA error."""
    cp, _ = fake_cupy(count=0, pci={})
    with pytest.raises(GpuPlacementError, match='exposed no GPU'):
        discover_peers(Comm(), is_gpu_worker=True, cp=cp)


def test_a_failed_selection_is_reported_not_guessed_around():
    """A device this rank cannot take must stop the job with the reason,
    rather than leaving it running on whatever device was current."""
    def explode():
        raise RuntimeError('cudaSetDevice failed')

    cp = NS(cuda=NS(
        runtime=NS(getDeviceCount=lambda: 2,
                   deviceGetPCIBusId=lambda i: PCI_A,
                   memGetInfo=lambda: (40 * GIB, 40 * GIB)),
        Device=lambda ordinal=0: NS(id=ordinal, use=explode)))
    with pytest.raises(GpuPlacementError) as caught:
        discover_peers(Comm(), is_gpu_worker=True, cp=cp)
    message = str(caught.value)
    assert 'cannot select a GPU' in message
    assert 'cudaSetDevice failed' in message, \
        'the underlying CUDA error must survive into the message'


def test_a_peer_failure_stops_this_rank_too():
    """The group agrees before grouping, so one rank's failure does not leave
    the others waiting in the next collective."""
    class PeerFailed(Comm):
        def Iallreduce(self, sendbuf, recvbuf, op=None):
            recvbuf[:] = 1        # a peer reported a failure
            return NS(Test=lambda: True)

    cp, _ = fake_cupy()
    with pytest.raises(GpuPlacementError, match='failed device selection'):
        discover_peers(PeerFailed(), is_gpu_worker=True, cp=cp)


# ---------------------------------------------------------------------------
# Non-GPU roles participate without touching CUDA
# ---------------------------------------------------------------------------

def test_non_gpu_role_joins_every_collective_in_order():
    """smd0 and EB must make the same calls a GPU worker makes. Skipping one
    does not raise -- it hangs the others, and a CPU-only rank that returned
    early was measured racing a peer's MPI_ABORT."""
    log = []
    placement = discover_peers(Comm(size=4, log=log), is_gpu_worker=False)
    kinds = [entry[0] for entry in log]
    assert kinds.count('allgather') == 1
    assert kinds.count('split') == 2
    operations = [entry[1] for entry in log if entry[0] == 'allreduce']
    assert operations == ['SUM', 'MAX'], (
        'a non-GPU role must make the same psana_comm reductions, in the same '
        'order: MPI pairs collectives by call order, not by operation. The '
        'capacity MIN is not among them because it runs on device_comm, '
        'which this rank does not hold')
    assert placement.device_comm is None
    assert placement.n_device_peers == 1


def test_non_gpu_role_never_touches_cupy(monkeypatch):
    """On a shared node importing CuPy would allocate a context on a device
    that belongs to someone else.

    cp is left None so the code reaches its own `import cupy`, and the module
    is poisoned so that import FAILS. Passing a stand-in instead -- as this
    test used to -- means no import is attempted and the test passes however
    the code behaves.
    """
    monkeypatch.setitem(sys.modules, 'cupy', None)
    discover_peers(Comm(), is_gpu_worker=False)


def test_non_gpu_role_raises_when_a_gpu_worker_failed():
    class Failing(Comm):
        def Iallreduce(self, sendbuf, recvbuf, op=None):
            recvbuf[:] = 1
            return NS(Test=lambda: True)

    with pytest.raises(GpuPlacementError, match='aborting this rank'):
        discover_peers(Failing(), is_gpu_worker=False)


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

def test_capacity_is_divided_among_discovered_peers():
    cp, _ = fake_cupy(count=1, pci={0: PCI_A}, free=40 * GIB)
    placement = discover_peers(Comm(size=4, node_size=4, gpu_size=4),
                               is_gpu_worker=True, cp=cp, headroom=0.0)
    assert placement.n_device_peers == 4
    assert per_rank_limit(placement) == 10 * GIB


def test_headroom_is_withheld_from_the_budget():
    cp, _ = fake_cupy(count=1, pci={0: PCI_A}, free=40 * GIB)
    placement = discover_peers(Comm(), is_gpu_worker=True, cp=cp,
                               headroom=0.10)
    assert placement.usable_bytes == int(40 * GIB * 0.9)


def test_shared_bytes_are_counted_once():
    placement = GpuPlacement(usable_bytes=40 * GIB, n_device_peers=4,
                             shared_bytes=4 * GIB)
    assert per_rank_limit(placement) == 9 * GIB


def test_limit_never_goes_negative():
    placement = GpuPlacement(usable_bytes=GIB, n_device_peers=2,
                             shared_bytes=4 * GIB)
    assert per_rank_limit(placement) == 0


def test_explicit_budget_within_the_device_is_accepted():
    cp, _ = fake_cupy(count=1, pci={0: PCI_A}, free=40 * GIB)
    placement = discover_peers(Comm(size=4, node_size=4, gpu_size=4),
                               is_gpu_worker=True, cp=cp, headroom=0.0,
                               explicit_limit_bytes=9 * GIB)
    assert placement.usable_bytes == 36 * GIB


def test_explicit_budget_that_over_commits_the_device_is_refused():
    """N ranks each claiming the whole device was accepted silently."""
    cp, _ = fake_cupy(count=1, pci={0: PCI_A}, free=40 * GIB)
    with pytest.raises(GpuPlacementError, match='exceeds'):
        discover_peers(Comm(size=4, node_size=4, gpu_size=4),
                       is_gpu_worker=True, cp=cp, headroom=0.0,
                       explicit_limit_bytes=20 * GIB)


# ---------------------------------------------------------------------------
# Sharing eligibility and teardown
# ---------------------------------------------------------------------------

def test_sharing_needs_more_than_one_peer():
    assert not GpuPlacement(device_comm=NS(), n_device_peers=1).can_share
    assert GpuPlacement(device_comm=NS(), n_device_peers=2).can_share
    assert not GpuPlacement(device_comm=None, n_device_peers=4).can_share


def test_placement_describes_itself_for_the_log():
    """Peer counts and budgets are otherwise invisible until something runs
    out of memory, so the line has to carry both."""
    placement = GpuPlacement(hostname='sdfampere001', device_pci=NORM_A,
                             device_ordinal=1, visible_count=2,
                             n_device_peers=4, is_owner=True,
                             usable_bytes=18 * GIB, sharing='on')
    line = placement.describe()
    for token in ('sdfampere001', NORM_A, 'ordinal=1', 'peers=4', 'owner=True',
                  '18.00GiB', 'sharing=on'):
        assert token in line


def test_release_frees_the_device_communicator():
    """Discovery runs per run, so the previous run's comm must be released."""
    freed = []
    placement = GpuPlacement(device_comm=NS(Free=lambda: freed.append(1)))
    release_placement(placement)
    assert freed == [1]
    assert placement.device_comm is None


def test_release_tolerates_an_already_freed_communicator():
    def explode():
        raise RuntimeError('MPI_Comm_free on a freed comm')

    placement = GpuPlacement(device_comm=NS(Free=explode))
    release_placement(placement)          # must not raise during teardown
    assert placement.device_comm is None
