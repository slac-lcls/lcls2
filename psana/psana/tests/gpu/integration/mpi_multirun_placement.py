"""Per-run rediscovery: run N+1 must not be sized against run N's memory.

Discovery runs once per run, so everything it decides -- the device, the peer
group, the budget -- is decided again for each run. Nothing exercised that
twice: every other driver takes a single run, which is why two reviews in a row
raised the same concern from reading the code rather than from a measurement.

Two things go wrong if a run does not clean up after itself.

**Budget collapse.** ``usable_bytes`` comes from ``memGetInfo``, reduced with
``MPI.MIN`` across the job. CuPy's pool keeps whatever run N allocated until
asked to release it, so run N+1 measures a card that still looks full -- and
because the reduction is a job-wide minimum, one rank's stale pool lowers
*every* rank's limit on *every* device. A four-peer 40 GiB card holding 9 GiB
per rank from run 1 leaves ~4 GiB free, so run 2's per-rank limit is ~0.9 GiB
instead of ~9, and setup either aborts or runs with tiny subbatches.

**Communicator leak.** Discovery builds a ``device_comm`` per run. A job over
many runs that never frees them reaches the MPI communicator limit.

Both are invisible on one run, and neither shows up as a wrong answer -- the
first degrades throughput or aborts at setup, the second fails only after
hundreds of runs. Hence the budget comparison below rather than a correctness
check on the data.

Run with a multi-run experiment, which is what makes this possible::

    PS_EB_NODES=1 mpirun -n 6 python mpi_multirun_placement.py \\
        --exp mfx101629726 --runs 15,16,17

Dataset choice is not free. ``mfx100848724``, which every other driver uses,
has only r0051. ``mfx100852324`` has three runs but cannot be used: its
stream 7 describes jungfrau segments 11-31 twice under the ``config``
algorithm, and ``detector_field_handles`` refuses a segment owned by two
Configure ``Names`` records -- so setup aborts with ``field
jungfrau.config.user.gainMode:gainModeEnum segment 19 is owned by more than
one routed Configure Names record`` before any event is read. That is a
pre-existing issue unrelated to placement (the caller discards every
``config`` key immediately after the call), and it reproduces on one run just
as readily. ``mfx101555026`` (r0008-r0012) and ``mfx101629726`` (r0015-r0032)
both build handles cleanly and either will do.

**This driver cannot pass yet.** psana2 hangs at the second run of any
multi-run MPI job: the second run's BD workers each receive zero events and
the EB rank never returns from it. That is pre-existing on this branch's
parent (``features/psana2-gpu``) and reproduces with no GPU code involved --
measurements, controls and the localisation are in
``psana/psana/gpu/docs/multirun_hang.md``. The driver is kept because it is
the reproducer, and because it becomes the rediscovery test the moment the
multi-run path works.

``max_events`` applies per run, not to the DataSource as a whole: ``_setup_run``
builds a fresh ``SmdReaderManager`` for each entry of ``runnum_list``, which
resets ``n_processed_events`` (measured -- ``run=[77,78,79]`` with
``max_events=10`` yields 10 events from each of the three runs, not 10 in
total). So a cap here bounds each run rather than ending the job inside run 1,
and every run's ``events()`` still terminates on its own and runs the teardown
in ``_events_impl``'s ``finally`` -- which is half of what this driver is
checking.
"""
import argparse
import json
import os
import sys


def emit(tag, obj):
    print(f'MULTIRUN_{tag} ' + json.dumps(obj, indent=2, sort_keys=True),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='mfx101629726')
    ap.add_argument('--runs', default='15,16,17',
                    help='comma-separated run numbers; at least two')
    ap.add_argument('--dir', default='/sdf/data/lcls/ds/prj/public01/xtc')
    ap.add_argument('--max-events', type=int, default=20,
                    help='per-run cap; psana resets the count for each run '
                         'in the list')
    ap.add_argument('--batch-size', type=int, default=5)
    ap.add_argument('--tolerance-mib', type=float, default=512.0,
                    help='how far a later run may be sized below the first '
                         'before this is called a collapse')
    args = ap.parse_args()

    runs = [int(r) for r in args.runs.split(',') if r.strip()]
    if len(runs) < 2:
        print('need at least two runs to exercise rediscovery',
              file=sys.stderr)
        return 2

    from mpi4py import MPI
    from psana import DataSource
    from psana.gpu import GpuTask

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    declared = [('jungfrau', 'pedestals')]
    calls = {'n': 0}

    def callback(batch, stream):
        # Read a constant the way a user kernel does, so each run really goes
        # through establishment and teardown of the shared block rather than
        # just allocating a budget.
        batch.calibconst('jungfrau', 'pedestals')
        calls['n'] += 1

    task = GpuTask(callback, ['jungfrau.raw'], declared)
    # A LIST of runs. One DataSource then yields several, and discovery runs
    # once per run -- the path this driver exists to cover.
    ds = DataSource(exp=args.exp, run=runs, dir=args.dir,
                    detectors=['jungfrau'], gpu_det='jungfrau', gpu_fn=task,
                    max_events=args.max_events,
                    batch_size=args.batch_size, gpu_bulk_read=True)

    failures, checks = [], {}
    info = {'rank': rank, 'host': MPI.Get_processor_name(),
            'requested_runs': runs,
            'mask': os.environ.get('CUDA_VISIBLE_DEVICES')}
    seen, role, last_run = [], None, None

    for run in ds.runs():
        role = run.comms.node_type()
        last_run = run
        info['role'] = role
        placement = getattr(run, '_gpu_placement', None)
        before = calls['n']

        # Iterated to exhaustion rather than broken out of, so _events_impl's
        # finally runs: that is where the manager closes and the placement's
        # device_comm is freed, and whether it ran is what the next run's
        # budget reflects.
        events = 0
        for _ in run.events():
            events += 1

        record = dict(runnum=run.runnum, events=events,
                      callback_calls=calls['n'] - before)
        if role == 'bd' and placement is not None:
            record.update(usable=int(placement.usable_bytes),
                          peers=placement.n_device_peers,
                          pci=placement.device_pci,
                          owner=bool(placement.is_owner),
                          sharing=placement.sharing)
        seen.append(record)
        emit(f'RANK{rank}_RUN{run.runnum}', record)

    info.update(runs=seen, n_runs=len(seen), callback_calls=calls['n'])

    # Every requested run must have been yielded. If the job ends early the
    # later runs' discovery never happens, and every remaining check below
    # would pass while testing a single run.
    checks['saw_every_run'] = len(seen) == len(runs)
    if not checks['saw_every_run']:
        failures.append(f'requested {len(runs)} runs, saw {len(seen)}: '
                        f'{[r["runnum"] for r in seen]}')

    if role == 'bd':
        sized = [r for r in seen if 'usable' in r]
        checks['every_run_had_a_placement'] = len(sized) == len(seen)
        if len(sized) != len(seen):
            missing = [r['runnum'] for r in seen if 'usable' not in r]
            failures.append(f'runs {missing} left no GPU placement, so '
                            'discovery did not run for them')

        if len(sized) >= 2:
            first, tol = sized[0], int(args.tolerance_mib * 1024 ** 2)
            # THE assertion this driver exists for. A later run's budget must
            # not collapse because an earlier run's allocations are still
            # held. Free memory is not time-invariant -- other tenants, the
            # driver's own overhead -- so compare against a tolerance rather
            # than demanding equality; the failure being guarded against is a
            # multi-GiB cliff, not drift.
            drops = []
            for r in sized[1:]:
                drop = first['usable'] - r['usable']
                drops.append(dict(runnum=r['runnum'], drop_mib=drop // 1024 ** 2))
                if drop > tol:
                    failures.append(
                        f"run {r['runnum']} was sized at "
                        f"{r['usable'] / 1024 ** 3:.2f} GiB against run "
                        f"{first['runnum']}'s {first['usable'] / 1024 ** 3:.2f}"
                        f" GiB, a drop of {drop / 1024 ** 2:.0f} MiB; the "
                        'earlier run\'s device memory was still held when '
                        'this run was sized')
            info['drops'] = drops
            checks['budget_held_across_runs'] = not failures

            # Identity comes from the driver, so one rank cannot move between
            # cards from run to run. If it does, selection is not
            # deterministic and the peer groups differ per run.
            cards = sorted({r['pci'] for r in sized})
            checks['same_device_every_run'] = len(cards) == 1
            if len(cards) > 1:
                failures.append(f'this rank moved between devices across '
                                f'runs: {cards}')

            peers = sorted({r['peers'] for r in sized})
            checks['same_peer_count_every_run'] = len(peers) == 1
            if len(peers) > 1:
                failures.append(f'peer count changed across runs: {peers}')

            # Ownership must be stable too: a group that re-elected a
            # different owner each run would allocate a new shared copy
            # without the previous one being accounted for.
            owners = sorted({r['owner'] for r in sized})
            checks['same_owner_role_every_run'] = len(owners) == 1
            if len(owners) > 1:
                failures.append('this rank was owner in some runs and not '
                                'others, so the group re-elected mid-job')
    elif role is not None:
        # A CPU role must still drive every run to completion, and must not
        # have touched CuPy.
        checks['no_cupy_on_cpu_role'] = 'cupy' not in sys.modules
        if not checks['no_cupy_on_cpu_role']:
            failures.append('a CPU-role rank imported CuPy')

    info.update(checks=checks, failures=failures, PASS=not failures)
    emit(f'RANK{rank}', info)

    # Every rank reaches the gather, including one that recorded a failure:
    # an early return would leave its peers blocked here and the harness
    # would report a timeout instead of the real cause.
    rows = comm.gather(info, root=0)
    comm.Barrier()
    if last_run is not None:
        last_run.close_shared_memory()
    if rank == 0:
        summarise(rows, runs)
    # Hold every rank until rank 0 has finished writing the summary. A rank
    # with failures calls MPI_Abort as soon as main() returns, which would
    # otherwise kill rank 0 mid-emit and leave the launcher parsing a
    # truncated record -- reported as "no rank records emitted" rather than
    # as the real failure.
    comm.Barrier()
    return 0 if not failures else 1


def summarise(rows, requested):
    """Job-wide view. Per-rank checks cannot see whether the budget collapse
    happened on a rank other than this one -- and because the reduction is a
    job-wide MIN, one rank's stale pool is enough to lower everybody."""
    members = [r for r in rows if r]
    workers = [r for r in members if r.get('role') == 'bd']
    failures = [f for r in members for f in r.get('failures', [])]

    per_run = {}
    for r in workers:
        for x in r.get('runs', []):
            if 'usable' in x:
                per_run.setdefault(x['runnum'], []).append(
                    dict(rank=r['rank'], usable=x['usable'],
                         peers=x['peers'], pci=x['pci'],
                         events=x['events'], calls=x['callback_calls']))

    if len(per_run) < 2:
        failures.append(f'only {len(per_run)} run(s) were sized across the '
                        'whole job; rediscovery was never exercised, so this '
                        'case proves nothing even though it may report passes')

    # Work must actually have been dispatched in every run. A run that yielded
    # no events would get discovery and teardown but allocate nothing, so its
    # budget would look healthy regardless.
    for runnum, ms in sorted(per_run.items()):
        if not sum(m['events'] for m in ms):
            failures.append(f'run {runnum} delivered no events to any BD '
                            'rank, so nothing was allocated and its budget '
                            'says nothing about cleanup')
        if not sum(m['calls'] for m in ms):
            failures.append(f'run {runnum} ran no GPU callback, so constants '
                            'were never established for it')

    emit('SUMMARY', {
        'requested_runs': requested,
        'runs_sized': sorted(per_run),
        'gpu_workers': len(workers),
        # Per rank across runs, which is how the collapse would show: one
        # rank's list falling off a cliff after its first entry.
        'usable_gib_by_rank': {
            r['rank']: [round(x['usable'] / 1024 ** 3, 2)
                        for x in r.get('runs', []) if 'usable' in x]
            for r in workers},
        'peers_by_run': {run: sorted({m['peers'] for m in ms})
                         for run, ms in sorted(per_run.items())},
        'devices_by_run': {run: sorted({m['pci'] for m in ms})
                           for run, ms in sorted(per_run.items())},
        'events_by_run': {run: sum(m['events'] for m in ms)
                          for run, ms in sorted(per_run.items())},
        'calls_by_run': {run: sum(m['calls'] for m in ms)
                         for run, ms in sorted(per_run.items())},
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
