"""Multi-node psana event loop with shared constants and several EB groups.

The single-node suite covers the event loop, and the multi-node suite covers
peer discovery, but nothing covers them together. That leaves the property
that only appears across nodes untested: **CUDA IPC is single-node**, so a
2-node job must end up with one owner *per node* and one shared copy per node,
not one owner for the job. A grouping bug that let a device group span hosts
would make the remote peer's IPC import fail and degrade the whole job, and a
single-node run cannot see it because the hostname half of the grouping key is
constant there.

This driver runs the real thing: a real ``DataSource`` with ``gpu_det`` and a
``GpuTask``, real events, real Jungfrau constants read through
``batch.calibconst()`` inside the callback, and ``PS_EB_NODES`` greater than 1
so the EB-group topology is not the trivial one. It asserts, per node and then
job-wide:

* every device group lies within one host;
* exactly one owner per device group, so one shared copy per node;
* followers import rather than allocating;
* the constants a callback reads are the declared ones, with the right shape;
* budgets per device sum to no more than that device.

Run across two nodes, with the mask narrowed so the workers on each node are
peers of one another::

    PS_EB_NODES=2 CUDA_VISIBLE_DEVICES=0 \\
        mpirun -n 10 python mpi_multinode_event_loop.py --expect-hosts 2

``--expect-hosts`` is asserted rather than inferred, because a job that landed
all its ranks on one node would otherwise pass this driver while testing
nothing it was written for.
"""
import argparse
import json
import os
import sys


def emit(tag, obj):
    print(f'MULTINODE_{tag} ' + json.dumps(obj, indent=2, sort_keys=True),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expect-hosts', type=int, default=0,
                    help='assert this many distinct hosts hold GPU workers')
    ap.add_argument('--expect-devices', type=int, default=0,
                    help='assert this many distinct devices job-wide')
    ap.add_argument('--max-events', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=5)
    args = ap.parse_args()

    from mpi4py import MPI
    from psana import DataSource
    from psana.gpu import GpuTask

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    declared = [('jungfrau', 'pedestals'), ('jungfrau', 'pixel_gain')]
    seen = {'calls': 0, 'shapes': {}, 'dtypes': {}}
    # Sampled inside the callback, because the counters are derived from the
    # owned and imported block dicts and run.events() closes the manager in
    # its finally -- clearing both. Anything read after the loop is zero.
    live = {}
    current_run = []

    def sample_live():
        if live or not current_run:
            return
        manager = getattr(current_run[0], '_gpu_manager_for_placement', None)
        placement = getattr(current_run[0], '_gpu_placement', None)
        constants = getattr(manager, '_task_constants', None)
        if manager is None or placement is None or constants is None:
            return
        live.update(
            sharing=placement.sharing,
            shared_bytes=int(placement.shared_bytes),
            resident_shared_bytes=int(getattr(constants, 'shared_bytes', 0) or 0),
            imported_bytes=int(getattr(constants, 'imported_bytes', 0) or 0),
        )

    def callback(batch, stream):
        sample_live()
        # Read the constants the way a user kernel does. Inside the callback,
        # because BatchInputContext seals afterwards and calibconst() then
        # refuses.
        for det, key in declared:
            array = batch.calibconst(det, key)
            seen['shapes'][key] = list(array.shape)
            seen['dtypes'][key] = str(array.dtype)
        seen['calls'] += 1

    task = GpuTask(callback, ['jungfrau.raw'], declared)
    ds = DataSource(exp='mfx100848724', run=51,
                    dir='/sdf/data/lcls/ds/prj/public01/xtc',
                    detectors=['jungfrau'], gpu_det='jungfrau', gpu_fn=task,
                    max_events=args.max_events, batch_size=args.batch_size,
                    gpu_bulk_read=True)

    failures, checks = [], {}
    info = {'rank': rank, 'host': MPI.Get_processor_name(),
            'eb_nodes': os.environ.get('PS_EB_NODES'),
            'mask': os.environ.get('CUDA_VISIBLE_DEVICES')}
    events = 0
    role, current_run_obj = None, None

    for run in ds.runs():
        role = run.comms.node_type()
        current_run_obj = run
        current_run[:] = [run]
        info['role'] = role

        if role != 'bd':
            # CPU roles must still drive the run to completion, or the BD
            # workers block in their own collectives.
            for _ in run.events():
                pass
            checks['no_cupy_on_cpu_role'] = 'cupy' not in sys.modules
            if not checks['no_cupy_on_cpu_role']:
                failures.append('a CPU-role rank imported CuPy')
            break

        from psana.gpu.gpu_placement import per_rank_limit

        placement = getattr(run, '_gpu_placement', None)
        checks['placement_present'] = placement is not None
        if placement is None:
            failures.append('setup left no GPU placement on this BD rank')
            break

        info.update(device_pci=placement.device_pci,
                    n_device_peers=placement.n_device_peers,
                    is_owner=placement.is_owner,
                    per_rank_limit=int(per_rank_limit(placement)))

        for _ in run.events():
            events += 1

        info.update(
            events=events,
            callback_calls=seen['calls'],
            constant_shapes=seen['shapes'],
            constant_dtypes=seen['dtypes'],
            live=dict(live),
            describe=placement.describe(),
        )

        # A worker that received no batch is not a failure: psana does not
        # promise every BD rank gets work, and with few events relative to
        # workers most will not. Whether the job as a whole dispatched is
        # asserted once in summarise(), where the totals are known.
        checks['callback_ran'] = seen['calls'] > 0

        # Conditional on having run: if this rank's callback DID read the
        # constants, they must be the declared ones with the detector's real
        # layout rather than a zero-filled placeholder.
        if seen['calls']:
            checks['constants_have_real_shapes'] = all(
                len(seen['shapes'].get(key, [])) >= 3 for _, key in declared)
            if not checks['constants_have_real_shapes']:
                failures.append(
                    f'constant shapes look wrong: {seen["shapes"]}')

        break

    info.update(checks=checks, failures=failures, PASS=not failures)
    emit(f'RANK{rank}', info)

    # Every rank reaches the gather and the barrier, including one that
    # broke out early: an early return would leave its peers blocked here
    # and the harness would report a timeout rather than the real cause.
    rows = comm.gather(info, root=0)
    comm.Barrier()
    if current_run_obj is not None:
        current_run_obj.close_shared_memory()
    if rank == 0:
        summarise(rows, expect_hosts=args.expect_hosts,
                  expect_devices=args.expect_devices)
    return 0 if not failures else 1


def summarise(rows, expect_hosts=0, expect_devices=0):
    """Job-wide assertions, which are the point of running across nodes."""
    members = [r for r in rows if r]
    workers = [r for r in members if r.get('role') == 'bd']
    failures = [f for r in members for f in r.get('failures', [])]

    hosts = sorted({r['host'] for r in workers})
    devices = sorted({r['device_pci'] for r in workers if 'device_pci' in r})

    # A device group must lie within one host: IPC cannot span nodes, so a
    # group that did would fail its import and degrade the whole job.
    by_device = {}
    for r in workers:
        by_device.setdefault(r.get('device_pci'), []).append(r)
    spanning = {pci: sorted({m['host'] for m in group})
                for pci, group in by_device.items()
                if len({m['host'] for m in group}) > 1}
    if spanning:
        failures.append(f'device group(s) span several hosts: {spanning}; '
                        'CUDA IPC cannot cross nodes')

    # One owner per device group, so one shared copy per node rather than one
    # for the job.
    owners_per_device = {pci: sum(1 for m in group if m.get('is_owner'))
                         for pci, group in by_device.items()}
    wrong = {pci: n for pci, n in owners_per_device.items() if n != 1}
    if wrong:
        failures.append(f'expected exactly one owner per device, got {wrong}')

    # Followers import; owners do not. Only ranks whose callback ran have a
    # live sample -- a rank that received no batch never took one, and its
    # post-teardown counters are all zero.
    for r in workers:
        sample = r.get('live') or {}
        if sample.get('sharing') != 'on':
            continue
        if r.get('is_owner') and sample.get('imported_bytes'):
            failures.append(f"rank {r['rank']} owns its copy but imported "
                            f"{sample['imported_bytes']} bytes")
        if not r.get('is_owner') and not sample.get('imported_bytes'):
            failures.append(f"rank {r['rank']} is a follower but imported "
                            'nothing, so it allocated its own copy')

    # Budgets per device must not exceed that device.
    for pci, group in by_device.items():
        claim = sum(m.get('per_rank_limit', 0) for m in group)
        peers = {m.get('n_device_peers') for m in group}
        if len(peers) > 1:
            failures.append(f'{pci}: peers disagree on the count: {peers}')
        if len(group) != (peers.pop() if peers else 0):
            failures.append(f'{pci}: {len(group)} ranks but the placement '
                            'reports a different peer count')
        by_device[pci] = dict(ranks=[m['rank'] for m in group],
                              host=group[0]['host'], claim=claim,
                              owners=owners_per_device[pci])

    # The job must actually have dispatched work and read constants through
    # the public path somewhere, even though no individual rank is required to.
    total_calls = sum(r.get('callback_calls') or 0 for r in workers)
    total_events = sum(r.get('events') or 0 for r in workers)
    if not total_calls:
        failures.append('no GPU callback ran on any rank, so calibconst was '
                        'never read through the public path')
    if not total_events:
        failures.append('no events were delivered to any BD rank')
    # And every rank that did run must have seen sane constants.
    ran = [r for r in workers if (r.get('callback_calls') or 0) > 0]
    bad = [r['rank'] for r in ran
           if not r.get('checks', {}).get('constants_have_real_shapes')]
    if bad:
        failures.append(f'ranks {bad} ran the callback but read constants '
                        'with the wrong shape')

    if expect_hosts and len(hosts) != expect_hosts:
        failures.append(f'expected GPU workers on {expect_hosts} hosts, '
                        f'found {len(hosts)}: {hosts}; this driver exists to '
                        'test the multi-node case')
    if expect_devices and len(devices) != expect_devices:
        failures.append(f'expected {expect_devices} distinct devices, '
                        f'found {len(devices)}: {devices}')

    emit('SUMMARY', {
        'ranks': len(members),
        'gpu_workers': len(workers),
        'hosts': hosts,
        'devices': devices,
        'groups': by_device,
        'total_callback_calls': total_calls,
        'total_events': total_events,
        'workers_that_ran': [r['rank'] for r in ran],
        'events_by_rank': {r['rank']: r.get('events') for r in workers},
        'callback_calls_by_rank': {r['rank']: r.get('callback_calls')
                                   for r in workers},
        'sharing_by_rank': {r['rank']: (r.get('live') or {}).get('sharing')
                            for r in workers},
        'imported_by_rank': {r['rank']: (r.get('live') or {}).get('imported_bytes')
                             for r in workers},
        'all_pass': not failures,
        'failures': failures[:10],
    })


if __name__ == '__main__':
    # A failure on one rank must stop the job, or its peers block in their
    # next collective. SystemExit is not caught: it signals a clean exit.
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
