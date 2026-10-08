"""Device selection and peer discovery for GPU BD ranks.

Two phases, in this order:

    phase 1  pin_device()      choose a device from the launcher environment
    phase 2  discover_peers()  make it current, then group ranks by identity

Phase 1 deliberately does NOT narrow ``CUDA_VISIBLE_DEVICES``. It cannot:
``import psana`` loads mpi4py, whose CUDA-aware MPI calls ``cuInit`` at library
load, and once CUDA is initialised a mask write is silently ignored. Measured
on sdfampere002 -- after ``import psana`` the driver reports CUDA already
initialised, so the branch that narrowed the mask was unreachable in any real
job.

Selecting rather than masking is sufficient. Measured on sdfampere014 with
4 ranks over 2 GPUs (``validation/ipc-integration-20261007/probe_late_leak.py``,
job 40151108): with both devices visible throughout, every rank's allocations
landed on the device it selected and the driver reported each PID resident on
exactly one GPU. No context leaked onto device 0.

The visible-device count is therefore recorded for the placement log but is not
a correctness signal. Identity is verified one way: the PCI bus id of the
current device must be the one chosen in phase 1.

A launcher may still narrow the mask -- ``srun --gpus-per-task=1``, or a
wrapper setting ``CUDA_VISIBLE_DEVICES`` from ``OMPI_COMM_WORLD_LOCAL_RANK``
before Python starts. That is defence in depth, not a requirement: a mask
containing one device is simply a permitted set of size one, and the same
single path handles it.

Replaces the ``init_gpu_rank`` + ``bd_ranks_sharing_gpu`` pair, which derived
both the device and the peer count from EB-group-local rank arithmetic. That
pair also wrote the mask too late to take effect, leaving every rank on
device 0 with only a warning.
"""
import os
import subprocess
import sys
from dataclasses import dataclass, field


# Node-local rank, in preference order. All are set by the launcher in each
# process's environment before MPI_Init, so none requires a communicator.
LOCAL_RANK_VARS = (
    'OMPI_COMM_WORLD_LOCAL_RANK',   # Open MPI / PRRTE
    'PMIX_LOCAL_RANK',              # PMIx (newer Open MPI, Slurm PMIx)
    'SLURM_LOCALID',                # srun
    'MV2_COMM_WORLD_LOCAL_RANK',    # MVAPICH2
    'PMI_LOCAL_RANK',               # MPICH / Intel MPI
)

# Modules whose import initialises CUDA. Reported in the placement log so a
# surprising device set can be explained; no longer a gate on anything, since
# the mask is not narrowed and selection works regardless.
CUDA_INITIALISERS = ('cupy', 'mpi4py.MPI', 'torch', 'numba.cuda', 'pycuda')


class GpuPlacementError(RuntimeError):
    """Device selection or peer discovery could not be completed safely."""


def uuid_hex(value):
    """Hex digits of a device UUID, ignoring any ``GPU-``/``MIG-`` prefix.

    ``cudaGetDeviceProperties`` returns 16 raw bytes with no prefix and no
    GPU/MIG discriminator, so comparisons must be prefix-insensitive and
    MIG-ness must come from NVML or nvidia-smi instead.
    """
    return ''.join(c for c in str(value).lower() if c in '0123456789abcdef')


def _pci_norm(value):
    """Comparable PCI bus id.

    nvidia-smi prints an eight-digit domain (``00000000:01:00.0``) while
    ``cudaDeviceGetPCIBusId`` prints four (``0000:01:00.0``).
    """
    text = str(value).strip().upper()
    parts = text.split(':')
    if len(parts) == 3:
        parts[0] = parts[0].lstrip('0') or '0'
        return ':'.join(parts)
    return text


@dataclass(frozen=True)
class PinnedDevice:
    """Outcome of phase 1. Host-only; contains no CUDA or MPI state."""

    requested_uuid: str = ''      # what we asked CUDA_VISIBLE_DEVICES for
    requested_pci: str = ''
    is_mig: bool = False
    local_rank: int = -1
    local_rank_source: str = ''
    permitted: tuple = ()         # launcher's device set, as seen by us
    n_node_devices: int = 0
    identity_source: str = ''     # 'nvml' | 'nvidia-smi' | ''
    pinned: bool = False
    warnings: tuple = ()

    @property
    def usable(self):
        """True when a device was selected and may be verified after import."""
        return self.pinned and bool(self.requested_uuid)


@dataclass
class GpuPlacement:
    """Phase 1 plus phase 2: who this rank is, and who shares its device."""

    pinned: PinnedDevice
    hostname: str = ''
    device_uuid_hex: str = ''     # what the driver actually reports
    device_ordinal: int = -1
    visible_count: int = 0
    device_comm: object = None    # MPI.Comm over ranks sharing this device
    is_owner: bool = False
    n_device_peers: int = 1
    usable_bytes: int = 0
    shared_bytes: int = 0         # intersection bytes, filled in by sharing
    imported_bytes: int = 0       # non-owning bytes this rank reads
    # 'on'   sharing established
    # 'off'  not attempted (solo rank, MIG, no communicator)
    # 'fallback(<reason>)'  an IPC failure degraded the group to private
    #        copies, which keeps the job running while using n times the
    #        memory for constants -- otherwise invisible.
    sharing: str = 'off'
    warnings: list = field(default_factory=list)

    @property
    def is_mig(self):
        return self.pinned.is_mig

    @property
    def can_share(self):
        """IPC sharing is possible only between real peers on one full GPU."""
        return (self.device_comm is not None
                and self.n_device_peers > 1
                and not self.is_mig)

    def describe(self):
        """One-line placement record. The only practical way to confirm
        placement on a real allocation, and its absence is why the previous
        arithmetic defects went unnoticed."""
        return (
            f'gpu placement host={self.hostname} '
            f'uuid={self.device_uuid_hex[:16]}... '
            f'ordinal={self.device_ordinal} mig={self.is_mig} '
            f'local_rank={self.pinned.local_rank}'
            f'({self.pinned.local_rank_source}) '
            f'visible={self.visible_count} peers={self.n_device_peers} '
            f'owner={self.is_owner} '
            f'limit={self.usable_bytes / 1024 ** 3:.2f}GiB '
            f'sharing={self.sharing} '
            f'permitted={",".join(self.pinned.permitted) or "<unset>"}'
        )


# ---------------------------------------------------------------------------
# Phase 1: pin before MPI
# ---------------------------------------------------------------------------

def local_rank():
    """(rank, variable) from the launcher environment, or (None, None)."""
    for name in LOCAL_RANK_VARS:
        value = os.environ.get(name)
        if value is not None and value.strip().lstrip('-').isdigit():
            return int(value), name
    return None, None


def _nvml_identity():
    """index -> {uuid, pci, mig} via NVML. Creates no CUDA context."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:                                     # noqa: BLE001
        return None
    try:
        table = {}
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            pci = pynvml.nvmlDeviceGetPciInfo(handle).busId
            pci = pci.decode() if isinstance(pci, bytes) else pci
            try:
                current, _pending = pynvml.nvmlDeviceGetMigMode(handle)
                mig = current == pynvml.NVML_DEVICE_MIG_ENABLE
            except Exception:                             # noqa: BLE001
                # Not MIG-capable, or the query is unsupported on this driver.
                mig = uuid.startswith('MIG-')
            table[index] = {'uuid': uuid, 'pci': _pci_norm(pci), 'mig': bool(mig)}
        return table
    except Exception:                                     # noqa: BLE001
        return None
    finally:
        # Always shut down, including on the exception path.
        try:
            pynvml.nvmlShutdown()
        except Exception:                                 # noqa: BLE001
            pass


def _smi_identity():
    """index -> {uuid, pci, mig} via nvidia-smi. Creates no CUDA context."""
    try:
        output = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:                                     # noqa: BLE001
        return None
    table = {}
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
        table[int(parts[0])] = {'uuid': parts[1], 'pci': _pci_norm(parts[2]),
                                'mig': parts[1].startswith('MIG-')}
    return table or None


def node_devices():
    """(table, source). NVML is preferred; nvidia-smi is the fallback.

    Both report identity without initialising CUDA, which is what allows this
    to run before the device is chosen.
    """
    table = _nvml_identity()
    if table:
        return table, 'nvml'
    table = _smi_identity()
    if table:
        return table, 'nvidia-smi'
    return {}, ''


def pin_device(*, local_rank_override=None):
    """Choose this rank's GPU from the launcher environment.

    Records the choice; ``discover_peers`` makes it current. The mask is not
    narrowed, because by the time any caller inside psana can reach this the
    package import has already initialised CUDA and a mask write would have no
    effect.

    The device is identified by **PCI bus id**, not by index: NVML and
    nvidia-smi number devices by PCI bus while CUDA honours
    ``CUDA_DEVICE_ORDER``, and Slurm ``ConstrainDevices`` can renumber the
    mask relative to the allocation. The bus id is stable under both.

    The UUID is recorded alongside it and used as the peer-grouping key, where
    it distinguishes MIG instances that share a bus id. It is taken from
    NVML/nvidia-smi, never read back from CUDA: ``getDeviceProperties()`` hands
    CuPy a 16-byte array that is truncated at the first NUL, so a UUID
    containing a zero byte yields a short, wrong value.
    """
    warnings = []

    # Deterministic index order, for the diagnostics that still report indices.
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

    table, source = node_devices()
    if not table:
        warnings.append('no GPU identity available; not pinning')
        return PinnedDevice(identity_source=source, warnings=tuple(warnings))

    # An UNSET mask and an EMPTY mask mean opposite things, so they must not
    # share a branch. Unset is "the launcher did not restrict anything", so
    # every device on the node is permitted. Empty -- `CUDA_VISIBLE_DEVICES=""`
    # -- is how a launcher says "this rank gets NO GPU", and CUDA itself
    # reports zero devices for it. Expanding that to the whole node would let
    # a rank select a device outside its allocation and quietly share another
    # job's card.
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    if mask is None:
        permitted = tuple(str(i) for i in sorted(table))
    else:
        permitted = tuple(p.strip() for p in mask.split(',') if p.strip())
    if not permitted:
        warnings.append(
            'CUDA_VISIBLE_DEVICES is set but empty, so the launcher has '
            'exposed no GPU to this rank; not pinning. A GPU role here is a '
            'configuration error -- discovery will report it rather than '
            'selecting a device outside the allocation.')
        return PinnedDevice(permitted=permitted, identity_source=source,
                            n_node_devices=len(table), warnings=tuple(warnings))

    if local_rank_override is not None:
        rank, rank_source = int(local_rank_override), 'override'
    else:
        rank, rank_source = local_rank()
    if rank is None:
        rank, rank_source = 0, ''
        warnings.append(
            'no launcher local-rank variable found (looked for '
            f'{", ".join(LOCAL_RANK_VARS)}); every rank would select the same '
            'device. Check the launcher.')

    entry = permitted[rank % len(permitted)]
    if entry.startswith(('GPU-', 'MIG-')):
        record = next((v for v in table.values()
                       if uuid_hex(v['uuid']) == uuid_hex(entry)), None)
        if record is None:
            # NVML and `nvidia-smi --query-gpu` enumerate parent GPUs only, so
            # a MIG instance UUID is never in the table and this rank ends up
            # unpinned. Grouping then falls back to the PCI bus id, which MIG
            # instances of one card share, so ranks on *different* instances
            # look like peers: sharing is attempted and fails over to private
            # copies, and the budget is divided by too many peers. Safe but
            # wasteful, and silent without this.
            warnings.append(
                f'permitted entry {entry!r} is not in the node device map; '
                'MIG instance UUIDs are not enumerated by NVML, so this rank '
                'is unpinned and peer grouping falls back to the PCI bus id. '
                'MIG is not a supported configuration.')
    else:
        record = table.get(int(entry))
        if record is None:
            # Slurm ConstrainDevices=yes can renumber the mask relative to the
            # allocation while NVML reports physical indices.
            warnings.append(
                f'permitted entry {entry!r} is absent from the node device map '
                f'{sorted(table)}; possible cgroup renumbering')
    if record is None:
        return PinnedDevice(local_rank=rank, local_rank_source=rank_source,
                            permitted=permitted, n_node_devices=len(table),
                            identity_source=source, warnings=tuple(warnings))

    return PinnedDevice(
        requested_uuid=record['uuid'], requested_pci=record['pci'],
        is_mig=record['mig'], local_rank=rank, local_rank_source=rank_source,
        permitted=permitted, n_node_devices=len(table),
        identity_source=source, pinned=True, warnings=tuple(warnings))


# ---------------------------------------------------------------------------
# Phase 2: verify and discover, after MPI exists
# ---------------------------------------------------------------------------

def device_pci(cp, ordinal):
    """PCI bus id of a visible device, normalised for comparison.

    Used in preference to ``getDeviceProperties()['uuid']``: that is a 16-byte
    array which CuPy surfaces as Python bytes truncated at the first NUL, so
    any GPU whose UUID contains a zero byte produces a short, wrong value.
    Measured on sdfampere014: ``c9ef90546100fd2e...`` arrived as
    ``b'\xc9\xef\x90Ta'``. deviceGetPCIBusId is a NUL-terminated string.
    """
    value = cp.cuda.runtime.deviceGetPCIBusId(int(ordinal))
    return _pci_norm(value.decode() if isinstance(value, bytes) else value)


def select_device(pinned, cp):
    """Make the requested device current when the mask could not be narrowed.

    Every device in the launcher's mask is visible, so the right one is
    chosen by matching PCI bus ids and calling ``Device.use``. A mask the
    launcher already narrowed to one device needs no special case: it is a
    permitted set of size one. Returns True when a device was selected.
    """
    if not pinned.usable:
        return False
    wanted = _pci_norm(pinned.requested_pci)
    matches = [ordinal for ordinal in range(int(cp.cuda.runtime.getDeviceCount()))
               if device_pci(cp, ordinal) == wanted]
    if not matches:
        return False
    if len(matches) > 1:
        # MIG instances of one card share a PCI bus id, so the bus id no
        # longer identifies a device. CUDA's own UUID cannot break the tie
        # either: getDeviceProperties() truncates it at the first NUL byte.
        # Taking matches[0] would silently select the wrong instance and give
        # this rank another rank's memory, so refuse.
        #
        # Reachable only on a MIG-partitioned node. S3DF advertises no MIG
        # today, but the h200 and b200 partitions are MIG-capable hardware.
        # Narrowing the mask before Python starts (srun --gpus-per-task=1, or
        # a wrapper setting CUDA_VISIBLE_DEVICES to the MIG UUID) leaves one
        # visible device and makes this unambiguous.
        raise GpuPlacementError(
            f'{len(matches)} visible devices share pci {wanted} (ordinals '
            f'{matches}), so the pci id does not identify one device. This is '
            'a MIG-partitioned card; CUDA reports a truncated UUID and cannot '
            'disambiguate it. Narrow CUDA_VISIBLE_DEVICES to one device '
            'before Python starts.')
    ordinal = matches[0]
    # NOTE: use() sets the current device for the CALLING THREAD only.
    # Nothing in the pipeline does CuPy work on a worker thread today
    # (KvikIO's threads are host-side I/O), but any thread added later on
    # this path would silently default to device 0. Such a thread must call
    # Device(ordinal).use() itself; placement records the ordinal for that
    # purpose.
    cp.cuda.Device(ordinal).use()
    return True


def verify_pin(pinned, cp):
    """Confirm the current device is the one chosen. One rule: PCI bus id.

    Returns ``(uuid, ordinal, visible, errors)``. The UUID is the full value
    recorded at pin time, never read back from
    ``getDeviceProperties()['uuid']`` -- that is a 16-byte array which CuPy
    surfaces truncated at the first NUL, so a UUID containing a zero byte
    yields a short, wrong key. Measured on sdfampere014, where
    ``c9ef90546100fd2e...`` arrived as five bytes.

    ``visible`` is returned for the placement log. It is deliberately not
    checked: the mask is never narrowed, and a wide visible set was measured
    not to cause misplaced allocations.
    """
    errors = []
    visible = int(cp.cuda.runtime.getDeviceCount())
    ordinal = int(cp.cuda.Device().id)
    actual_pci = device_pci(cp, ordinal)
    actual_uuid = pinned.requested_uuid

    if pinned.usable:
        if actual_pci != _pci_norm(pinned.requested_pci):
            errors.append(
                f'device at pci {actual_pci} is not the requested '
                f'{pinned.requested_pci}: this rank holds a device it did '
                'not select')
    else:
        # Nothing was chosen, so identity comes from the driver alone and the
        # PCI id is the only reliable grouping key available.
        actual_uuid = actual_pci
    return actual_uuid, ordinal, visible, errors


def release_placement(placement):
    """Free a placement's communicator. Discovery runs once per run, so the
    previous run's device_comm must be released or they accumulate."""
    comm = getattr(placement, 'device_comm', None)
    if comm is not None:
        try:
            comm.Free()
        except Exception:                                 # noqa: BLE001
            pass
        placement.device_comm = None


def discover_peers(pinned, comm, *, is_gpu_worker, cp=None,
                   headroom=0.10, explicit_limit_bytes=0, logger=None):
    """Group ranks by physical device and size per-rank budgets.

    ``comm`` must be a communicator every participating rank holds -- psana's
    ``psana_comm`` -- because the node split is collective over all of it.
    Non-GPU roles pass ``is_gpu_worker=False``; they must still call this so
    the splits complete, and they receive a placement with no device.

    Peer counts come from device identity reported by the driver, never from
    rank arithmetic, so they are independent of EB-group topology.
    """
    from mpi4py import MPI

    hostname = MPI.Get_processor_name()
    placement = GpuPlacement(pinned=pinned, hostname=hostname)

    # Node split: collective over every member of comm.
    hosts = comm.allgather(hostname)
    colors = {h: i for i, h in enumerate(dict.fromkeys(hosts))}
    node_comm = comm.Split(colors[hostname], comm.Get_rank())
    try:
        # GPU split: every node member participates; non-GPU roles get
        # COMM_NULL and must not touch it.
        node_gpu_comm = node_comm.Split(
            0 if is_gpu_worker else MPI.UNDEFINED, node_comm.Get_rank())
    finally:
        node_comm.Free()

    if not is_gpu_worker:
        # Join every collective a GPU worker reaches, in the same order, then
        # stop. Returning early leaves this rank racing a peer's MPI_ABORT --
        # measured: with a bogus device on one worker, the CPU-only rank
        # printed SURVIVED before the abort landed.
        from .gpu_collectives import SETUP_TIMEOUT_S, bounded_allreduce
        if bounded_allreduce(comm, 0, MPI.SUM, 'discovery/job-agreement',
                             timeout=SETUP_TIMEOUT_S, logger=logger):
            raise GpuPlacementError(
                'a GPU worker failed device verification; aborting this rank '
                'so the job stops deterministically rather than waiting for '
                'a peer abort to arrive')
        # Neutral contribution to the device-capacity minimum.
        bounded_allreduce(comm, 2 ** 62, MPI.MIN, 'discovery/budget',
                          timeout=SETUP_TIMEOUT_S, logger=logger)
        # Neutral for MAX: a real peer count is always >= 1. Same position in
        # the sequence as the GPU-worker path, which is what MPI pairs on.
        bounded_allreduce(comm, 0, MPI.MAX, 'discovery/max-peers',
                          timeout=SETUP_TIMEOUT_S, logger=logger)
        return placement

    if cp is None:
        import cupy as cp                                 # noqa: PLC0415

    # Make the chosen device current, then verify. One path: a mask narrowed
    # by the launcher is simply a permitted set of size one, and select_device
    # finds that one device by PCI bus id like any other.
    if pinned.permitted == () and os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        # A GPU role with an empty mask is a configuration error, and saying so
        # here is what makes pin_device's warning true. Otherwise this rank
        # reaches verify_pin, cp.cuda.Device().id raises on a context with no
        # devices, and the user gets a raw CUDA error turned into an abort by
        # gpu_error_handler rather than the reason.
        errors = ['the launcher exposed no GPU to this BD rank '
                  '(CUDA_VISIBLE_DEVICES is set but empty), so it cannot do '
                  'GPU work; give it a device or run it as a CPU role']
        actual, ordinal, visible = '', -1, 0
    elif pinned.usable and not select_device(pinned, cp):
        errors = [f'no visible device has pci {pinned.requested_pci}; '
                  f'permitted={",".join(pinned.permitted) or "<unset>"}']
        actual, ordinal, visible = pinned.requested_uuid, -1, 0
    else:
        actual, ordinal, visible, errors = verify_pin(pinned, cp)
    placement.device_uuid_hex = actual
    placement.device_ordinal = ordinal
    placement.visible_count = visible

    # A failure on one rank must not hang the others, and must reach ranks
    # outside this node's GPU group. Two steps, in a fixed order every rank
    # follows: the node's GPU workers agree first, then the whole comm --
    # including EB, smd0 and workers on other nodes, which contribute zero.
    # A shorter deadline than the default: discovery ranks arrive together,
    # so there is no legitimate skew to absorb and a hang is real divergence.
    from .gpu_collectives import SETUP_TIMEOUT_S, bounded_allreduce
    n_failed = bounded_allreduce(node_gpu_comm, 1 if errors else 0, MPI.SUM,
                                 'discovery/node-agreement',
                                 timeout=SETUP_TIMEOUT_S, logger=logger)
    job_failed = bounded_allreduce(comm, n_failed, MPI.SUM,
                                   'discovery/job-agreement',
                                   timeout=SETUP_TIMEOUT_S, logger=logger)
    if job_failed:
        node_gpu_comm.Free()
        if errors:
            raise GpuPlacementError('; '.join(errors))
        raise GpuPlacementError(
            f'{job_failed} peer rank(s) failed device verification; '
            'aborting this rank so the job does not hang')

    # Group by (hostname, device identity). Using the UUID rather than the PCI
    # bus id keeps MIG instances apart: they share a bus id but have separate
    # memory, so they are not peers and cannot share an allocation.
    keys = node_gpu_comm.allgather((hostname, actual))
    distinct = sorted(set(keys))
    placement.device_comm = node_gpu_comm.Split(
        distinct.index((hostname, actual)), node_gpu_comm.Get_rank())
    node_gpu_comm.Free()

    placement.n_device_peers = placement.device_comm.Get_size()
    placement.is_owner = placement.device_comm.Get_rank() == 0

    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    # Base the budget on what is actually free, not on the device total: the
    # CUDA context and the Jungfrau shared caches are allocated before
    # discovery runs, and counting them as available over-commits the device.
    #
    # memGetInfo is device-wide but not time-invariant: peers sample it at
    # slightly different moments while each other's contexts are still
    # settling, so the raw values differ by a few MiB and per-rank limits
    # derived from them sum to marginally more than the device. Take the
    # minimum across the group so every peer uses one figure.
    # Reduced over every rank in `comm`, not per device and not per node: the
    # explicit budget is validated against this figure, so a per-group minimum
    # would let one group raise while another continues. Non-GPU roles
    # contribute the neutral value below, in the branch they return from.
    observed = int(free_bytes * (1.0 - headroom))
    placement.usable_bytes = bounded_allreduce(
        comm, observed, MPI.MIN, 'discovery/budget',
        timeout=SETUP_TIMEOUT_S, logger=logger)

    # Busiest device in the job. Needed so the explicit-budget check below
    # reaches the same verdict on every rank: `usable_bytes` is already
    # job-wide, but the claim is per-device, and peer counts differ between
    # devices in an uneven layout. Validating against the local count would
    # make a budget that is too large for a 4-peer device raise there while
    # a 3-peer device carried on -- ranks on different paths, racing the
    # abort rather than agreeing on it.
    #
    # Rejecting job-wide is also the correct answer, not merely the uniform
    # one: if any device cannot honour the per-rank budget, the job cannot
    # run with it.
    #
    # Unconditional, because it is a collective; the check that uses it is
    # not. Non-GPU roles contribute 0 in the branch they return from, which
    # is neutral for MAX since a real count is always >= 1.
    max_peers = bounded_allreduce(
        comm, placement.n_device_peers, MPI.MAX, 'discovery/max-peers',
        timeout=SETUP_TIMEOUT_S, logger=logger)

    if explicit_limit_bytes:
        # An explicit per-rank budget must be honourable by the whole group:
        # N ranks each claiming the whole device was previously accepted
        # silently.
        total_claim = int(explicit_limit_bytes) * max_peers
        if total_claim > placement.usable_bytes:
            placement.device_comm.Free()
            placement.device_comm = None
            crowded = ('' if max_peers == placement.n_device_peers else
                       f' (this device has {placement.n_device_peers})')
            raise GpuPlacementError(
                f'gpu_memory_budget_gb={explicit_limit_bytes / 1024 ** 3:.2f} GiB '
                f'x {max_peers} peers on the busiest device{crowded} = '
                f'{total_claim / 1024 ** 3:.2f} GiB exceeds the '
                f'{placement.usable_bytes / 1024 ** 3:.2f} GiB usable per '
                'device')
        placement.usable_bytes = int(explicit_limit_bytes) * placement.n_device_peers

    placement.warnings.extend(pinned.warnings)
    if logger is not None:
        logger.info(placement.describe())
    return placement


def per_rank_limit(placement):
    """Device capacity divided among true peers, shared bytes counted once."""
    usable = max(0, placement.usable_bytes - placement.shared_bytes)
    return usable // max(1, placement.n_device_peers)
