"""Device selection and peer discovery for GPU BD ranks.

``discover_peers()`` makes this rank's device current and groups ranks that
share a physical GPU, collective over psana's ``psana_comm``.

Selection happens here, inside discovery, where the node's GPU-worker
communicator already exists. The device is made current with ``Device.use()``
under whatever mask the launcher set; narrowing
``CUDA_VISIBLE_DEVICES`` from Python would have no effect, because ``import
psana`` loads mpi4py and its CUDA-aware MPI calls ``cuInit`` at library load,
after which the variable is ignored (measured on sdfampere002).

``use()`` under a wide mask is sufficient: measured on sdfampere014 with
4 ranks over 2 GPUs (job 40151108), every rank's allocations landed on the
device it selected and the driver reported each PID resident on exactly one
GPU.

Assignment is a deterministic least-loaded pass over every node worker's
visible PCI ids, gathered once on the node's GPU-worker communicator. Each rank
runs the identical loop over the identical data, so they agree without further
communication.

Two reasons it is not a simple index. CUDA numbers visible devices in the order
the mask lists them, so indexing ordinals puts masks ``2,3`` and ``3,2`` on the
same physical GPU. And indexing only this rank's own sorted set is not enough
either: with ``{A,B}`` on two ranks and ``{C,D}`` on two others, each indexes
its own list and they land A, D, A, D with B and C idle. Only a view of what
every worker can see produces a balanced assignment.

Using the communicator rank rather than a launcher local rank also removes the
gaps smd0 and EB ranks leave in a node-local numbering.

Devices are identified by PCI bus id from ``deviceGetPCIBusId``. Never by
``getDeviceProperties()['uuid']``: that is a 16-byte array which CuPy surfaces
truncated at the first NUL, so a UUID containing a zero byte yields a short,
wrong key -- measured on sdfampere014, where ``c9ef90546100fd2e...`` arrived as
five bytes.
"""
import os
from dataclasses import dataclass


class GpuPlacementError(RuntimeError):
    """Device selection or peer discovery could not be completed safely."""


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
    shared_bytes: int = 0         # intersection bytes, filled in by sharing
    imported_bytes: int = 0       # non-owning bytes this rank reads
    # 'on'   sharing established
    # 'off'  not attempted (solo rank, no communicator)
    # 'fallback(<reason>)'  an IPC failure degraded the group to private
    #        copies, which keeps the job running while using n times the
    #        memory for constants -- otherwise invisible.
    sharing: str = 'off'

    @property
    def can_share(self):
        """IPC sharing is possible only between real peers on one device.

        A necessary condition, not a sufficient one: on a MIG-partitioned
        card this returns True for ranks that are not really peers, because
        instances share a PCI bus id. The IPC import then fails and the group
        degrades to private copies. See the grouping comment in
        ``discover_peers``.
        """
        return self.device_comm is not None and self.n_device_peers > 1

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

    # Assign by index within this node's GPU workers, which removes the gaps
    # a launcher local rank leaves where smd0 and EB ranks sit.
    #
    # The index is into the visible PCI ids SORTED, not into CUDA's own
    # ordinals. CUDA numbers visible devices in the order the mask lists them,
    # so indexing ordinals directly inherits the mask's order: with `2,3` on
    # one rank and `3,2` on another, ordinal 0 and ordinal 1 are both physical
    # GPU 2, and the other card sits idle. Sorting first gives every rank that
    # can see the same set the same sequence to index, whatever order its own
    # mask used, and needs no collective to agree on.
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
    # Assignment needs every node worker's visible set, because indexing only
    # this rank's own set is not enough: with masks that expose DIFFERENT sets
    # -- {A,B} on two ranks and {C,D} on two others -- each rank indexes its
    # own list and they land A, D, A, D, leaving B and C idle. The allgather
    # is the one extra collective, and it runs on every GPU worker whether or
    # not selection succeeded, so the sequence stays identical across ranks.
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
            # Deterministic least-loaded assignment. Every rank runs this
            # identical loop over the identical gathered data, so all of them
            # reach the same answer without further communication. Ranks are
            # visited in communicator order and ties break on the PCI id, so
            # there is nothing order-dependent left.
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
            # use() binds the device for THIS thread only. Nothing in the
            # pipeline currently needs that elsewhere -- the reader's thread
            # pool is inside KvikIO and runs on the calling thread's context,
            # and no Python thread here issues CUDA work of its own. The
            # ordinal is recorded for the placement log, and is what a future
            # worker thread would need to call use() itself.
            cp.cuda.Device(ordinal).use()
            actual_pci = device_pci(cp, ordinal)
        except Exception as exc:                          # noqa: BLE001
            errors.append(f'cannot select a GPU among {visible} visible: '
                          f'{type(exc).__name__}: {exc}')

    if not errors:
        # Return the previous run's cached blocks before anyone samples free
        # memory. Discovery runs per run, and CuPy's pool holds what run N
        # allocated until asked to release it -- so run N+1 would size itself
        # against a card that still looks full. Because `usable` is a job-wide
        # MIN, one rank's stale pool lowers every rank's limit, not just its
        # own.
        #
        # Here rather than in the caller: the node agreement and the
        # device-group Split both come after this point, so every peer on the
        # node has freed before any of them samples memGetInfo.
        #
        # Its own try, because this only improves the NEXT run's sizing. A
        # CuPy build without the pool API, or a release that raises, is no
        # reason to fail a run whose device is already selected -- so it logs
        # and carries on rather than joining `errors`.
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
        # job_failed is a count of contributions, not of failing ranks: every
        # GPU worker on a node contributes its node's n_failed, so the figure
        # is inflated by the node's worker count. Reported as a boolean rather
        # than a misleading number.
        raise GpuPlacementError(
            'a peer rank failed device selection; aborting this rank so the '
            'job does not hang. The failing rank reports the reason.')

    # Group by (hostname, PCI bus id) as the driver reports it, so peer counts
    # never depend on rank arithmetic.
    #
    # The bus id, not a UUID, because the UUID is not obtainable here:
    # getDeviceProperties()['uuid'] is 16 raw bytes that CuPy surfaces
    # truncated at the first NUL, so a UUID containing a zero byte yields a
    # short key that can collide with another card's. deviceGetPCIBusId is
    # NUL-terminated and describes the device this rank actually holds, having
    # been read back after use().
    #
    # MIG LIMITATION. Instances of one card share a bus id, so ranks on
    # *different* instances group together here. They are not really peers:
    # instances have separate memory and CUDA IPC cannot span them, so
    # sharing is attempted and the import fails.
    #
    # The group then degrades to private copies with a peer count inflated by
    # however many instances were merged, and each rank's limit becomes
    # usable/inflated_peers. For constants larger than that share the private
    # upload is refused and the job aborts, so MIG is not merely wasteful --
    # it can fail outright. This is why MIG is listed as unsupported rather
    # than as a degraded mode.
    #
    # Separating them needs an identity CUDA will not give us. NVML would,
    # via nvmlDeviceGetMigDeviceHandleByIndex: plain nvmlDeviceGetUUID and
    # `nvidia-smi --query-gpu=uuid` enumerate parent GPUs only, so an
    # instance UUID is absent from either. Since CUDA is already initialised
    # by the time we run, querying NVML here would be cheap -- the reason it
    # is not done is that no MIG node is advertised on S3DF, so the code
    # would be untestable. Revisit if that changes; until then MIG is
    # documented as unsupported in docs/limitations.md.
    keys = node_gpu_comm.allgather((hostname, actual_pci))
    distinct = sorted(set(keys))
    placement.device_comm = node_gpu_comm.Split(
        distinct.index((hostname, actual_pci)), node_gpu_comm.Get_rank())
    node_gpu_comm.Free()

    placement.n_device_peers = placement.device_comm.Get_size()
    placement.is_owner = placement.device_comm.Get_rank() == 0

    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    # Base the budget on what is actually free, not on the device total: the
    # CUDA context is allocated before discovery runs, and counting it as
    # available over-commits the device. (The Jungfrau shared caches are host
    # MPISharedMemory, not device memory, so they do not appear here.)
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
        # Validated against the busiest device in the job, so every rank
        # reaches the same verdict; N ranks each claiming the whole device
        # must not be accepted silently.
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
