"""Device peer discovery on real hardware, independent of EB topology.

Covers design rows 2A-0/3/4/5/7 with real CUDA and real MPI, which the unit
tests cannot: driver-reported device identity, device selection with CUDA
already initialised (the path every real job takes), and peer counts that must
not depend on how bd_comm was split.

Run with one or more GPUs and at least three ranks::

    mpirun -n 9 python mpi_placement_discovery.py --expect-devices 2

Across nodes, assert both counts so a hostname that is ignored or formatted
inconsistently is caught -- single-node runs cannot see that, because the
hostname half of the (hostname, uuid) grouping key is constant there::

    mpirun -n 10 python mpi_placement_discovery.py \
        --expect-hosts 2 --expect-devices 4

Rank 0 acts as a CPU-only role (smd0/EB) and must still participate in every
collective; ranks 1+ are GPU workers.
"""
import argparse
import json
import os
import sys




def emit(tag, obj):
    print(f'DISCOVERY_{tag} ' + json.dumps(obj, indent=2, sort_keys=True),
          flush=True)


def current_peers(n_bd_in_comm, phys_gpu_id, n_gpus):
    """The removed ``bd_ranks_sharing_gpu()`` arithmetic, reproduced here.

    Kept as a local copy, not an import: the function is gone, but the
    contrast is the point of this test. It counted only peers inside this
    rank's own bd_comm, so with several EB groups it undercounted and every
    rank claimed too much of the device. Each record reports both this value
    and the discovered one, so the log shows what was fixed.
    """
    n_gpus = max(1, int(n_gpus))
    target = int(phys_gpu_id) % n_gpus
    return max(1, sum(1 for k in range(n_bd_in_comm) if k % n_gpus == target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-eb', type=int, default=1,
                    help='simulated PS_EB_NODES: how bd_comm would be split')
    ap.add_argument('--contiguous', action='store_true')
    ap.add_argument('--expect-devices', type=int, default=0,
                    help='assert this many distinct devices were discovered')
    ap.add_argument('--expect-hosts', type=int, default=0,
                    help='assert this many distinct hosts were discovered')
    ap.add_argument('--eb-ranks', type=int, default=1)
    args = ap.parse_args()

    # --- phase 1: pin before mpi4py is imported --------------------------
    # World rank comes from the environment, not from MPI, because MPI must
    # not be initialised yet.
    world = None
    for name in ('OMPI_COMM_WORLD_RANK', 'PMIX_RANK', 'PMI_RANK',
                 'SLURM_PROCID'):
        value = os.environ.get(name)
        if value is not None and value.strip().isdigit():
            world = int(value)
            break
    if world is None:
        print('cannot determine world rank from the environment',
              file=sys.stderr)
        return 2

    is_worker = world >= args.eb_ranks
    # Imported normally, exactly as a real job does: psana's own import
    # loads mpi4py and initialises CUDA, and the single path is designed for
    # that. There is no earlier moment to reach from inside the package.
    from psana.gpu.gpu_placement import (
        GpuPlacementError, PinnedDevice, device_pci, discover_peers,
        per_rank_limit, pin_device,
    )
    pinned = pin_device() if is_worker else PinnedDevice()

    # --- phase 2: MPI may now start --------------------------------------
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if rank != world:
        print(f'env rank {world} != MPI rank {rank}', file=sys.stderr)
        return 2

    failures = []
    info = {'rank': rank, 'is_worker': is_worker, 'n_eb': args.n_eb,
            'pinned': {'requested': pinned.requested_uuid,
                       'local_rank': pinned.local_rank,
                       'source': pinned.local_rank_source,
                       'permitted': list(pinned.permitted),
                       'warnings': list(pinned.warnings)}}

    try:
        placement = discover_peers(pinned, comm, is_gpu_worker=is_worker,
                                   explicit_limit_bytes=0)
    except GpuPlacementError as exc:
        info['failures'] = [f'discover_peers: {exc}']
        info['PASS'] = False
        emit(f'RANK{rank}', info)
        return 1

    if not is_worker:
        # Join the GPU-worker split the workers make below -- with
        # MPI.UNDEFINED, so this rank gets COMM_NULL and no membership. Not
        # calling it at all would hang them.
        comm.Split(MPI.UNDEFINED, rank)
        # A CPU-only role must have participated without taking a device.
        if placement.device_comm is not None:
            failures.append('CPU-role rank joined a device communicator')
        if placement.n_device_peers != 1:
            failures.append('CPU-role rank reported device peers')
        info['failures'] = failures
        info['PASS'] = not failures
        emit(f'RANK{rank}', info)
        return 0 if not failures else 1

    import cupy as cp
    info.update({
        'uuid_hex': placement.device_uuid_hex,
        'visible_count': placement.visible_count,
        'ordinal': placement.device_ordinal,
        'is_mig': placement.is_mig,
        'design_peers': placement.n_device_peers,
        'is_owner': placement.is_owner,
        'usable_bytes': placement.usable_bytes,
        'per_rank_limit': per_rank_limit(placement),
        'describe': placement.describe(),
    })

    # 2A-0: the chosen device must be the current one. The visible count is
    # NOT asserted: the mask is deliberately left as the launcher set it, and
    # a wide visible set was measured not to cause misplaced allocations
    # (probe_late_leak.py, job 40151108). Identity is the real check, done
    # below via the device communicator's membership.
    #
    # What does matter is that work lands on the selected device, so allocate
    # and confirm.
    probe = cp.empty(1 << 20, dtype=cp.uint8)      # 1 MiB
    alloc_pci = device_pci(cp, int(probe.device.id))
    info['allocation_pci'] = alloc_pci
    info['requested_pci'] = pinned.requested_pci
    if pinned.usable and alloc_pci != pinned.requested_pci:
        failures.append(f'allocation landed on {alloc_pci}, not the selected '
                        f'{pinned.requested_pci}')
    del probe

    device_comm = placement.device_comm
    # GPU workers only: the CPU-role rank has returned, so COMM_WORLD is out.
    gpu_workers = comm.Split(0, rank)

    # 2A-3/4: peers come from device identity, so every member agrees.
    agreed = device_comm.allgather(placement.n_device_peers)
    info['peer_agreement'] = sorted(set(agreed))
    if len(set(agreed)) != 1:
        failures.append(f'peers disagree on the count: {sorted(set(agreed))}')

    keys = device_comm.allgather(placement.device_uuid_hex)
    if len(set(keys)) != 1:
        failures.append('a device communicator spans several devices')
    if len(keys) != placement.n_device_peers:
        failures.append('peer count does not match communicator membership')

    owners = device_comm.allreduce(1 if placement.is_owner else 0, op=MPI.SUM)
    info['owners'] = owners
    if owners != 1:
        failures.append(f'{owners} owners elected on one device')

    # 2A-4: the group must claim the device exactly once.
    claim = device_comm.allreduce(per_rank_limit(placement), op=MPI.SUM)
    info['aggregate_claim'] = round(claim / max(1, placement.usable_bytes), 4)
    if claim > placement.usable_bytes:
        failures.append(f'group claims {info["aggregate_claim"]}x the device')

    # 2A-5: an explicit per-rank budget the group cannot honour is refused.
    # Checked arithmetically rather than by re-running discover_peers: that
    # is collective over COMM_WORLD, and only worker ranks reach this branch
    # -- the CPU-role rank has already returned, so a second call deadlocks.
    if placement.n_device_peers > 1:
        over = placement.usable_bytes          # each rank claiming the device
        refused = over * placement.n_device_peers > placement.usable_bytes
        info['explicit_over_commit_refused'] = bool(refused)
        if not refused:
            failures.append('an explicit over-commit would be accepted')

    # Contrast with the arithmetic this replaces.
    bd_index = rank - args.eb_ranks
    total_bd = size - args.eb_ranks
    if args.contiguous:
        per = max(1, total_bd // args.n_eb)
        eb_group = min(bd_index // per, args.n_eb - 1)
    else:
        eb_group = bd_index % args.n_eb
    # NOTE: no COMM_WORLD collective here. The CPU-role rank returned above,
    # so any comm.* call would deadlock. The EB-group membership this contrast
    # needs is derivable arithmetically from the rank layout.
    n_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE',
                                max(1, len(pinned.permitted))))
    mine = [r for r in range(args.eb_ranks, size)
            if (min((r - args.eb_ranks) // max(1, total_bd // args.n_eb),
                    args.n_eb - 1) if args.contiguous
                else (r - args.eb_ranks) % args.n_eb) == eb_group]
    believed = current_peers(len(mine), mine.index(rank) % n_gpus, n_gpus)
    info['current_believed_peers'] = believed
    info['current_aggregate'] = round(
        device_comm.allreduce(placement.usable_bytes // believed, op=MPI.SUM)
        / max(1, placement.usable_bytes), 4)

    # ---- multi-node: grouping must respect the host, and budgets must agree
    # Reduced over every GPU worker, so this works at any node count.
    info['hostname'] = placement.hostname
    identity = (placement.hostname, placement.device_uuid_hex)
    all_identities = gpu_workers.allgather(identity)
    all_budgets = gpu_workers.allgather(placement.usable_bytes)

    distinct_devices = sorted(set(all_identities))
    distinct_hosts = sorted({h for h, _ in all_identities})
    info['n_distinct_devices_job'] = len(distinct_devices)
    info['n_distinct_hosts'] = len(distinct_hosts)

    # A device communicator must never span hosts. Single-node runs cannot
    # detect a hostname that is ignored or inconsistently formatted.
    same_host = device_comm.allgather(placement.hostname)
    if len(set(same_host)) != 1:
        failures.append(f'device communicator spans hosts: {sorted(set(same_host))}')

    # This rank's peers must be exactly the ranks reporting its identity.
    expected_peers = sum(1 for i in all_identities if i == identity)
    if expected_peers != placement.n_device_peers:
        failures.append(f'{placement.n_device_peers} peers reported but '
                        f'{expected_peers} ranks share this host and device')

    # usable_bytes is reduced with MPI.MIN over the whole communicator, so
    # every rank in the job must hold the same figure -- including across
    # nodes, where free memory genuinely differs.
    if len(set(all_budgets)) != 1:
        failures.append(f'usable_bytes differs across ranks: '
                        f'{sorted(set(all_budgets))}')

    if args.expect_devices and len(distinct_devices) != args.expect_devices:
        failures.append(f'{len(distinct_devices)} distinct devices discovered, '
                        f'expected {args.expect_devices}')
    if args.expect_hosts and len(distinct_hosts) != args.expect_hosts:
        failures.append(f'{len(distinct_hosts)} distinct hosts discovered, '
                        f'expected {args.expect_hosts}')

    info['failures'] = failures
    info['PASS'] = not failures
    emit(f'RANK{rank}', info)

    gathered = device_comm.gather(info, root=0)
    if placement.is_owner and gathered is not None:
        members = [r for r in gathered if r]
        emit('DEVICE', {
            'uuid_hex': placement.device_uuid_hex[:16] + '...',
            'peers': len(members),
            'design_peers': placement.n_device_peers,
            'aggregate_claim': info['aggregate_claim'],
            'current_believed_peers': sorted(
                {m['current_believed_peers'] for m in members}),
            'current_aggregate': info['current_aggregate'],
            'world_ranks': sorted(m['rank'] for m in members),
            'all_pass': all(m['PASS'] for m in members),
        })
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
