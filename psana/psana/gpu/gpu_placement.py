"""Device selection and peer discovery for GPU BD ranks.

``discover_peers()`` makes this rank's device current and groups ranks that
share a physical GPU, collective over psana's ``psana_comm``.

Selection happens here rather than in the caller, because this is where the
node's GPU-worker communicator exists -- a device chosen elsewhere has to
survive the trip, and a choice that fails to arrive leaves every BD worker on
CUDA's default device. The device is made current with ``Device.use()`` under
whatever mask the launcher set; narrowing ``CUDA_VISIBLE_DEVICES`` from Python
has no effect, because ``import psana`` loads mpi4py whose CUDA-aware MPI
calls ``cuInit`` at library load (measured on sdfampere002).

Assignment is a deterministic least-loaded pass over every node worker's
visible PCI ids, gathered once on the node's GPU-worker communicator, keyed on
PCI bus id rather than CUDA ordinal. Devices are identified by
``deviceGetPCIBusId`` and never by ``getDeviceProperties()['uuid']``, which
CuPy truncates at the first NUL.

docs/device_placement_and_shared_constants.md has the reasoning, the worked
rank layouts and the measurements.
"""
import os
from dataclasses import dataclass


class GpuPlacementError(RuntimeError):
    """Device selection or peer discovery could not be completed safely."""


def sharing_enabled():
    """Whether constant sharing may be attempted, per this rank's environment.

    The escape hatch for a system where CUDA IPC does not work between peers
    that discovery grouped together -- MIG instances of one card share a PCI
    bus id, so they are grouped as peers and the import then fails, which
    aborts. Opting out costs n times the device memory for constants but keeps
    the job running, which is what the removed automatic fallback used to do
    silently.

    Read once, in ``discover_peers``, and reduced over the device group into
    ``GpuPlacement.sharing_permitted``. Callers should use that rather than
    this: a rank acting on its own reading would take a collective path its
    peers do not.

    Off for ``0``, ``no``, ``false`` and ``off``, case-insensitively, because
    a user who writes ``false`` and silently keeps sharing has been misled.
    Anything else -- including an empty value -- leaves sharing on.
    """
    value = os.environ.get('PSANA_GPU_SHARE_CONSTANTS', '').strip().lower()
    return value not in ('0', 'no', 'false', 'off')


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


@dataclass
class GpuPlacement:
    """Which physical device this rank holds, and who shares it."""

    hostname: str = ''
    device_pci: str = ''          # grouping key, from the driver
    device_ordinal: int = -1
    visible_count: int = 0
    device_comm: object = None    # MPI.Comm over ranks sharing this device
    is_owner: bool = False
    n_device_peers: int = 1
    usable_bytes: int = 0
    shared_bytes: int = 0         # shared bytes, filled in by sharing
    imported_bytes: int = 0       # non-owning bytes this rank reads
    # Whether the GROUP permits IPC sharing: PSANA_GPU_SHARE_CONSTANTS reduced
    # with MIN over device_comm, so one rank's opt-out disables it for all of
    # them and no peer can take a different collective path. True by default
    # so a placement built without discovery -- the unit tests, the serial
    # path -- behaves as before.
    sharing_permitted: bool = True
    # 'on'   sharing established
    # 'off'  not attempted: a solo rank, no communicator, peers that declared
    #        different selector sets, or PSANA_GPU_SHARE_CONSTANTS=0
    sharing: str = 'off'

    @property
    def can_share(self):
        """IPC sharing is possible only between real peers on one device.

        Necessary, not sufficient: on a MIG-partitioned card this is True for
        ranks that are not really peers, because instances share a PCI bus id.
        The IPC import then fails and the job aborts, naming
        PSANA_GPU_SHARE_CONSTANTS=0. See the grouping comment in
        ``discover_peers``.

        ``sharing_permitted`` is the group's agreed answer to that variable,
        not this rank's reading of it, so every peer decides alike.
        """
        return (self.device_comm is not None and self.n_device_peers > 1
                and self.sharing_permitted)

    def describe(self):
        """One-line placement record. The only practical way to confirm
        placement on a real allocation: peer counts and budgets are
        otherwise invisible until something runs out of memory."""
        return (
            f'gpu placement host={self.hostname} pci={self.device_pci} '
            f'ordinal={self.device_ordinal} '
            f'visible={self.visible_count} peers={self.n_device_peers} '
            f'owner={self.is_owner} '
            f'limit={self.usable_bytes / 1024 ** 3:.2f}GiB '
            f'sharing={self.sharing}'
        )
        # imported_bytes is deliberately absent. The only production caller
        # logs this at the end of discover_peers, before sharing has been
        # established, so the figure would always be zero here -- measured as
        # zero occurrences across a full multi-node run. It is written onto
        # the placement by SharedRequestedConstants for callers that read it
        # after setup, which the validation harnesses do.


def device_pci(cp, ordinal):
    """PCI bus id of a visible device, normalised for comparison.

    Used in preference to ``getDeviceProperties()['uuid']``, which CuPy
    surfaces truncated at the first NUL: measured on sdfampere014, a 16-byte
    UUID arrived as 5 bytes, which can collide with another card's.
    deviceGetPCIBusId is a NUL-terminated string.
    """
    value = cp.cuda.runtime.deviceGetPCIBusId(int(ordinal))
    return _pci_norm(value.decode() if isinstance(value, bytes) else value)


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


def per_rank_limit(placement):
    """Device capacity divided among true peers, shared bytes counted once."""
    usable = max(0, placement.usable_bytes - placement.shared_bytes)
    return usable // max(1, placement.n_device_peers)


def discover_peers(comm, *, is_gpu_worker, cp=None, headroom=0.10,
                   explicit_limit_bytes=0, logger=None):
    """Select this rank's device, then group ranks by physical device.

    ``comm`` must be a communicator every participating rank holds -- psana's
    ``psana_comm`` -- because the node split is collective over all of it.
    Non-GPU roles pass ``is_gpu_worker=False``; they must still call this so
    the splits complete, and they receive a placement with no device.

    Peer counts come from device identity reported by the driver, never from
    rank arithmetic, so they are independent of EB-group topology.
    """
    from mpi4py import MPI

    hostname = MPI.Get_processor_name()
    placement = GpuPlacement(hostname=hostname)

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

    from .gpu_collectives import SETUP_TIMEOUT_S, bounded_allreduce

    if not is_gpu_worker:
        # Join every collective a GPU worker reaches, in the same order, then
        # stop. Returning early leaves this rank racing a peer's MPI_ABORT --
        # measured: with a bogus device on one worker, the CPU-only rank
        # printed SURVIVED before the abort landed.
        if bounded_allreduce(comm, 0, MPI.SUM, 'discovery/job-agreement',
                             timeout=SETUP_TIMEOUT_S, logger=logger):
            raise GpuPlacementError(
                'a GPU worker failed device selection; aborting this rank '
                'so the job stops deterministically rather than waiting for '
                'a peer abort to arrive')
        # Neutral contribution to the device-capacity minimum.
        usable = bounded_allreduce(comm, 2 ** 62, MPI.MIN, 'discovery/budget',
                                   timeout=SETUP_TIMEOUT_S, logger=logger)
        # Neutral for MAX: a real peer count is always >= 1. Same position in
        # the sequence as the GPU-worker path, which is what MPI pairs on.
        max_peers = bounded_allreduce(comm, 0, MPI.MAX, 'discovery/max-peers',
                                      timeout=SETUP_TIMEOUT_S, logger=logger)
        # Both reductions are job-wide, so a CPU-only rank can apply the same
        # explicit-budget check its GPU peers apply and raise with them. It
        # would otherwise carry on while they abort -- the same racing-the-
        # abort pattern avoided above for selection failures.
        if explicit_limit_bytes and max_peers:
            if int(explicit_limit_bytes) * max_peers > usable:
                raise GpuPlacementError(
                    'gpu_memory_budget_gb exceeds what the busiest device in '
                    'the job can honour; the GPU ranks report the figures')
        return placement

    if cp is None:
        import cupy as cp                                 # noqa: PLC0415

    # Assignment works on sorted PCI ids rather than CUDA ordinals, because
    # ordinals follow the mask's order -- see "Why not a simple index" in
    # docs/device_placement_and_shared_constants.md.
    errors = []
    ordinal, actual_pci, visible = -1, '', 0
    try:
        visible = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:                              # noqa: BLE001
        errors.append(f'cannot count visible GPUs: {type(exc).__name__}: {exc}')
    if not errors and visible <= 0:
        # An empty CUDA_VISIBLE_DEVICES reaches here: CUDA reports zero
        # devices for it, which for a GPU role is a configuration error.
        errors.append(
            'the launcher exposed no GPU to this BD rank '
            f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}), '
            'so it cannot do GPU work; give it a device or run it as a CPU '
            'role')
    # Every node worker's visible set is needed, not just this rank's: masks
    # exposing different sets collide otherwise. The allgather runs whether or
    # not selection succeeded, so the collective sequence stays identical
    # across ranks.
    my_pcis = []
    if not errors:
        try:
            my_pcis = [device_pci(cp, i) for i in range(visible)]
        except Exception as exc:                          # noqa: BLE001
            errors.append(f'cannot read PCI ids for {visible} visible GPUs: '
                          f'{type(exc).__name__}: {exc}')
    node_sets = node_gpu_comm.allgather(sorted(my_pcis))

    if not errors:
        try:
            # Deterministic least-loaded assignment: every rank runs this
            # identical loop over the identical gathered data, visiting ranks
            # in communicator order with ties broken on the PCI id, so they
            # agree without further communication.
            load = {}
            target = None
            for peer, peer_pcis in enumerate(node_sets):
                if not peer_pcis:
                    continue
                pick = min(peer_pcis, key=lambda p: (load.get(p, 0), p))
                load[pick] = load.get(pick, 0) + 1
                if peer == node_gpu_comm.Get_rank():
                    target = pick
            ordinal = my_pcis.index(target)
            # use() binds the device for THIS thread only. No Python thread
            # in the pipeline issues CUDA work of its own -- KvikIO's pool
            # runs on the calling thread's context -- so one call suffices.
            # The ordinal is recorded for the log, and is what a future worker
            # thread would need to call use() itself.
            cp.cuda.Device(ordinal).use()
            actual_pci = device_pci(cp, ordinal)
        except Exception as exc:                          # noqa: BLE001
            errors.append(f'cannot select a GPU among {visible} visible: '
                          f'{type(exc).__name__}: {exc}')

    if not errors:
        # Return the previous run's cached blocks before anyone samples free
        # memory: discovery runs per run, and a stale pool makes the card look
        # full. Because `usable` is a job-wide MIN, one rank's stale pool
        # lowers every rank's limit. Placed here because the node agreement
        # and the device-group Split both come later, so every peer has freed
        # before any of them samples memGetInfo.
        #
        # Its own try: this only improves the NEXT run's sizing, so a failure
        # logs rather than joining `errors` and failing a run whose device is
        # already selected.
        try:
            import gc
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as exc:                          # noqa: BLE001
            if logger is not None:
                logger.warning(
                    'could not release the cached CuPy pool before sizing '
                    '(%s: %s); a previous run\'s blocks may still count '
                    'against this run\'s budget', type(exc).__name__, exc)

    placement.device_pci = actual_pci
    placement.device_ordinal = ordinal
    placement.visible_count = visible

    # A failure on one rank must not hang the others, and must reach ranks
    # outside this node's GPU group. Two steps, in a fixed order every rank
    # follows: the node's GPU workers agree first, then the whole comm --
    # including EB, smd0 and workers on other nodes, which contribute zero.
    # A shorter deadline than the default: discovery ranks arrive together,
    # so there is no legitimate skew to absorb and a hang is real divergence.
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
        # job_failed counts contributions, not failing ranks -- every worker
        # on a node contributes its node's n_failed -- so it is reported as a
        # boolean rather than a misleading number.
        raise GpuPlacementError(
            'a peer rank failed device selection; aborting this rank so the '
            'job does not hang. The failing rank reports the reason.')

    # Group by (hostname, PCI bus id) as the driver reports it, so peer counts
    # never depend on rank arithmetic. The bus id rather than a UUID because
    # CuPy truncates getDeviceProperties()['uuid'] at the first NUL -- see
    # the device_pci() docstring.
    #
    # MIG LIMITATION: instances of one card share a bus id, so ranks on
    # different instances group here as peers though IPC cannot span them.
    # Separating them needs nvmlDeviceGetMigDeviceHandleByIndex, which is not
    # used because no MIG node is advertised on S3DF and the code would be
    # untestable. docs/limitations.md records the consequences;
    # PSANA_GPU_SHARE_CONSTANTS=0 is the supported way to run on MIG.
    keys = node_gpu_comm.allgather((hostname, actual_pci))
    distinct = sorted(set(keys))
    placement.device_comm = node_gpu_comm.Split(
        distinct.index((hostname, actual_pci)), node_gpu_comm.Get_rank())
    node_gpu_comm.Free()

    placement.n_device_peers = placement.device_comm.Get_size()
    placement.is_owner = placement.device_comm.Get_rank() == 0

    # Whether constant sharing may be attempted, agreed across the group
    # rather than read per rank. PSANA_GPU_SHARE_CONSTANTS comes from each
    # rank's own environment, and a rank that opted out while its peers did
    # not would skip the establishment allgather the others enter -- one of
    # the two unbounded collectives, so the job hangs with no diagnostic.
    # Measured precedent: the multinode harness set CUDA_VISIBLE_DEVICES
    # without `mpirun -x`, and the two nodes' ranks ended up with different
    # masks. MIN, so one opt-out disables sharing for the whole group.
    placement.sharing_permitted = bool(bounded_allreduce(
        placement.device_comm, 1 if sharing_enabled() else 0, MPI.MIN,
        'discovery/sharing-permitted',
        timeout=SETUP_TIMEOUT_S, logger=logger))

    free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
    # What is actually free, not the device total: the CUDA context is already
    # allocated, and counting it as available over-commits the device.
    #
    # memGetInfo is not time-invariant -- peers sample it while each other's
    # contexts settle, so raw values differ by a few MiB -- hence the MIN, so
    # every peer uses one figure. Reduced over all of `comm` rather than per
    # device: the explicit budget is validated against it, and a per-group
    # minimum would let one group raise while another continued.
    observed = int(free_bytes * (1.0 - headroom))
    placement.usable_bytes = bounded_allreduce(
        comm, observed, MPI.MIN, 'discovery/budget',
        timeout=SETUP_TIMEOUT_S, logger=logger)

    # Busiest device in the job, so the explicit-budget check below reaches
    # the same verdict everywhere: the claim is per-device and peer counts
    # differ in an uneven layout, so validating against the local count would
    # make one device raise while another carried on. Rejecting job-wide is
    # also correct on its own terms -- if any device cannot honour the budget,
    # the job cannot run with it.
    #
    # Unconditional because it is a collective; the check that uses it is not.
    max_peers = bounded_allreduce(
        comm, placement.n_device_peers, MPI.MAX, 'discovery/max-peers',
        timeout=SETUP_TIMEOUT_S, logger=logger)

    if explicit_limit_bytes:
        # N ranks each claiming the whole device must not pass silently.
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

    if logger is not None:
        logger.info(placement.describe())
    return placement
