"""Placement through the real DataSource -> Run -> GpuEventManager path.

The other placement drivers call ``discover_peers`` directly, so they prove the
algorithm but not the wiring: a device chosen correctly and then not applied,
or applied but never reaching the manager, looks identical to them. This driver
touches none of the placement helpers. It builds a real ``DataSource`` with
``gpu_det``, takes a real ``Run``, asks it for a real ``GpuEventManager``, and
asserts on what came out the far end:

* a placement exists on the Run, and reached the manager;
* the device it names is the device this rank's allocations land on;
* the peer count matches the device communicator's real membership;
* the budget the manager is enforcing came from that placement;
* ranks on one device agree, and the group claims the card exactly once;
* the GPU workers between them land on every visible device;
* non-GPU roles hold no placement and never import CuPy.

The last cross-rank check is the one that cannot be made per rank. A rank that
is wrongly placed is still self-consistent -- it reports a device, allocates on
that device, and its current device matches -- so every per-rank assertion
above passes while the job quietly runs N workers on one of several GPUs. Only
comparing the devices actually used against the number CUDA offers catches
that, and it is derived rather than passed in.

Run with PS_PARALLEL=mpi and one or more GPUs::

    PS_EB_NODES=1 mpirun -n 6 python mpi_real_path_placement.py

``--expect-devices N`` and ``--expect-peers N`` additionally pin the layout
for a case whose rank count and mask are fixed by the launcher.
"""
import argparse
import json
import os
import sys

import numpy as np


def emit(tag, obj):
    print(f'REALPATH_{tag} ' + json.dumps(obj, indent=2, sort_keys=True),
          flush=True)


class _NoPlacement(Exception):
    """Skip the per-rank checks while still reaching the shared collectives."""


def unused(evt, stream):
    raise AssertionError('this driver exercises setup, not callbacks')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expect-devices', type=int, default=0,
                    help='assert this many distinct devices across the job')
    ap.add_argument('--expect-peers', type=int, default=0,
                    help='assert this many peers on this rank\'s device')
    args = ap.parse_args()

    from mpi4py import MPI
    from psana import DataSource
    from psana.gpu import GpuTask

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    task = GpuTask(unused, ['jungfrau.raw'], [('jungfrau', 'pixel_gain')])
    ds = DataSource(exp='mfx100848724', run=51,
                    dir='/sdf/data/lcls/ds/prj/public01/xtc',
                    detectors=['jungfrau'], max_events=1,
                    gpu_det='jungfrau', gpu_fn=task)
    run = next(ds.runs())
    role = run.comms.node_type()

    # The mask is recorded so a reversed-mask run is distinguishable from a
    # natural-order one: with CUDA_VISIBLE_DEVICES reversed or narrowed, local
    # ordinals do not match physical device numbers.
    failures, checks, info = [], {}, {
        'rank': rank, 'role': role,
        'mask': os.environ.get('CUDA_VISIBLE_DEVICES')}

    if role != 'bd':
        # A CPU-role rank must come out of setup with no placement and no
        # CuPy: on a shared node, importing it would create a context on a
        # device belonging to someone else.
        checks['no_placement'] = getattr(run, '_gpu_placement', None) is None
        checks['no_cupy'] = 'cupy' not in sys.modules
        if not checks['no_placement']:
            failures.append('a CPU-role rank holds a GPU placement')
        if not checks['no_cupy']:
            failures.append('a CPU-role rank imported CuPy')
        info.update({'checks': checks, 'failures': failures,
                     'PASS': not failures})
        emit(f'RANK{rank}', info)
        gathered = comm.gather(info, root=0)
        comm.Barrier()
        run.close_shared_memory()
        if rank == 0:
            # World rank 0 is always smd0, so this is the branch that runs in
            # any real MPI job -- expect_devices has to be passed here or it
            # is never asserted at all.
            summarise(gathered, expect_devices=args.expect_devices)
        return 0 if not failures else 1

    import cupy as cp
    from psana.gpu.gpu_placement import device_pci, per_rank_limit

    # Setup must have left a placement on the Run. Note that a populated
    # placement is necessary but not sufficient: discovery can run and still
    # describe a device this rank did not take, which the identity checks
    # below are for.
    placement = getattr(run, '_gpu_placement', None)
    checks['run_has_placement'] = placement is not None
    if placement is None:
        # Report and fall through to the shared gather/Barrier below. An early
        # return here would leave every peer blocked in gather, so the harness
        # would report a 180s timeout instead of the missing placement.
        failures.append('setup left no GPU placement on the Run; '
                        '_discover_gpu_placement did not reach this rank')

    manager = run._make_gpu_event_manager() if placement is not None else None
    try:
        if manager is None:
            raise _NoPlacement
        # 1. The manager must be using the placement setup produced, not a
        #    default. Identity, not equality: a copy would hide a handoff bug.
        checks['manager_got_the_placement'] = manager._placement is placement
        if not checks['manager_got_the_placement']:
            failures.append('GpuEventManager did not receive the Run\'s '
                            'placement object')

        # 2. The device the placement names must be the device this rank's
        #    allocations actually land on. A placement that describes a device
        #    nothing was selected on is the failure this catches: the record
        #    looks right while the work goes elsewhere.
        probe = cp.empty(1 << 20, dtype=cp.uint8)          # 1 MiB
        allocation_pci = device_pci(cp, int(probe.device.id))
        del probe
        info['placement_pci'] = placement.device_pci
        info['allocation_pci'] = allocation_pci
        checks['allocations_land_on_the_named_device'] = (
            allocation_pci == placement.device_pci)
        if not checks['allocations_land_on_the_named_device']:
            failures.append(
                f'allocations land on {allocation_pci} but the placement '
                f'names {placement.device_pci}')

        # 3. Device.use() must have been called by discovery, so the current
        #    device is the selected ordinal rather than CUDA's default.
        info['current_ordinal'] = int(cp.cuda.Device().id)
        info['placement_ordinal'] = placement.device_ordinal
        checks['current_device_is_the_selected_one'] = (
            int(cp.cuda.Device().id) == placement.device_ordinal)
        if not checks['current_device_is_the_selected_one']:
            failures.append(
                f'current device is ordinal {cp.cuda.Device().id}, the '
                f'placement selected {placement.device_ordinal}; '
                'Device.use() did not run')

        # 4. The peer count must match the communicator's real membership,
        #    not arithmetic. A count derived from EB-group rank arithmetic
        #    over-committed the device by 2.0x at PS_EB_NODES=2.
        peers = placement.device_comm
        info['n_device_peers'] = placement.n_device_peers
        if peers is not None:
            checks['peers_match_communicator'] = (
                peers.Get_size() == placement.n_device_peers)
            if not checks['peers_match_communicator']:
                failures.append(
                    f'placement reports {placement.n_device_peers} peers but '
                    f'the device communicator has {peers.Get_size()}')

            # Every member must see the same device and the same count.
            keys = peers.allgather(placement.device_pci)
            counts = peers.allgather(placement.n_device_peers)
            checks['one_device_per_group'] = len(set(keys)) == 1
            checks['peers_agree_on_the_count'] = len(set(counts)) == 1
            if not checks['one_device_per_group']:
                failures.append(f'a device group spans several devices: '
                                f'{sorted(set(keys))}')
            if not checks['peers_agree_on_the_count']:
                failures.append(f'peers disagree on the count: '
                                f'{sorted(set(counts))}')

            # 5. The group must claim its card exactly once.
            claim = peers.allreduce(per_rank_limit(placement), op=MPI.SUM)
            ratio = round(claim / max(1, placement.usable_bytes), 4)
            info['aggregate_claim'] = ratio
            checks['group_claims_the_device_once'] = (
                claim <= placement.usable_bytes)
            if not checks['group_claims_the_device_once']:
                failures.append(f'the group claims {ratio}x its device')

            if args.expect_peers and placement.n_device_peers != args.expect_peers:
                failures.append(f'expected {args.expect_peers} peers, '
                                f'discovered {placement.n_device_peers}')

        # 6. The budget the manager enforces must come from the placement,
        #    not from _GpuBudget.auto(). Without an explicit
        #    gpu_memory_budget_gb the limit is the per-rank share; shared
        #    constants may have raised the owner's, so allow that.
        limit = manager._gpu_budget.limit()
        share = per_rank_limit(placement)
        info['budget_limit'] = int(limit)
        info['per_rank_limit'] = int(share)
        info['shared_bytes'] = int(placement.shared_bytes)
        checks['budget_came_from_the_placement'] = limit in (
            share, share + placement.shared_bytes)
        if not checks['budget_came_from_the_placement']:
            failures.append(
                f'budget limit {limit} is neither the per-rank share {share} '
                f'nor the owner share {share + placement.shared_bytes}; '
                'the manager sized itself independently of the placement')

        # 7. The constants the task declared must have been uploaded, and
        #    read back correctly. This is the whole point of the pipeline, so
        #    a placement that is right on paper but unusable still fails.
        value = manager._task_constants.get('jungfrau', 'pixel_gain')
        source = run.dsparms.calibconst['jungfrau']['pixel_gain'][0]
        checks['constants_readable'] = (
            value.shape == source.shape and value.dtype == source.dtype)
        if not checks['constants_readable']:
            failures.append(f'constant shape/dtype {value.shape}/{value.dtype} '
                            f'!= source {source.shape}/{source.dtype}')
        else:
            try:
                np.testing.assert_array_equal(cp.asnumpy(value), source)
                checks['constants_correct'] = True
            except AssertionError:
                checks['constants_correct'] = False
                failures.append('constant values differ from the source')

        # 8. The cross-rank check, in summarise(). A wrongly placed rank is
        #    still self-consistent -- it reports a device, allocates on it,
        #    and its current device matches -- so checks 1-7 cannot see N
        #    workers crowded onto one of several visible GPUs. Report the
        #    visible count so rank 0 can compare it with the devices used.
        info['visible_count'] = placement.visible_count
        info['describe'] = placement.describe()
    except _NoPlacement:
        pass
    finally:
        if manager is not None:
            manager.close()

    info.update({'checks': checks, 'failures': failures, 'PASS': not failures})
    emit(f'RANK{rank}', info)

    gathered = comm.gather(info, root=0)
    comm.Barrier()
    run.close_shared_memory()
    if rank == 0:
        summarise(gathered, expect_devices=args.expect_devices)
    return 0 if not failures else 1


def summarise(gathered, expect_devices=0):
    """Job-wide record, emitted by rank 0 only."""
    members = [r for r in gathered if r]
    workers = [r for r in members if r.get('role') == 'bd']
    devices = sorted({r['placement_pci'] for r in workers
                      if 'placement_pci' in r})
    failures = [f for r in members for f in r.get('failures', [])]

    # Unconditional, not gated on --expect-devices: with V devices visible
    # and W >= V GPU workers on a node, the workers must land on all V.
    # Derived from what CUDA reported, so every case gets the check without
    # having to state an expectation.
    visible = {r['visible_count'] for r in workers if 'visible_count' in r}
    if len(visible) == 1 and workers:
        offered = visible.pop()
        reachable = min(offered, len(workers))
        if offered > 1 and len(devices) < reachable:
            failures.append(
                f'{len(workers)} GPU workers landed on {len(devices)} '
                f'device(s) of {offered} visible; expected {reachable}. '
                'Each rank can still report itself correctly placed, so '
                'check how the device is selected, not how it is recorded.')

    if expect_devices and len(devices) != expect_devices:
        failures.append(f'expected {expect_devices} distinct devices, '
                        f'discovered {len(devices)}: {devices}')

    emit('SUMMARY', {
        'ranks': len(members),
        'gpu_workers': len(workers),
        'distinct_devices': devices,
        'peers_by_device': {
            pci: sum(1 for r in workers if r.get('placement_pci') == pci)
            for pci in devices},
        'aggregate_claims': sorted({r['aggregate_claim'] for r in workers
                                    if 'aggregate_claim' in r}),
        'masks': sorted({r['mask'] for r in members if r.get('mask')}),
        'ordinal_by_rank': {r['rank']: r.get('placement_ordinal')
                            for r in workers},
        'pci_by_rank': {r['rank']: r.get('placement_pci') for r in workers},
        'all_pass': not failures,
        'failures': failures[:10],
    })


if __name__ == '__main__':
    # A failure on one rank must stop the job: its peers would otherwise block
    # in their next collective. SystemExit is deliberately not caught -- it is
    # how a clean exit is signalled.
    try:
        status = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        from mpi4py import MPI
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)
        raise
    if status:
        from mpi4py import MPI
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)
    sys.exit(status)
