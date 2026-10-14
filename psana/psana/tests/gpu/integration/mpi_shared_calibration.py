"""Real Jungfrau calibration while the constants are shared over CUDA IPC.

Two halves are covered separately elsewhere and never together:

``user_calibration_driver.py`` runs the real calibration kernel through the
public API and compares every panel against an independent CPU reference, but
leaves ``CUDA_VISIBLE_DEVICES`` as the launcher set it, so on a multi-GPU node
its BD workers land on different devices and sharing never engages.

``mpi_shared_constants.py`` exercises sharing at 2 and 4 peers, but builds
``SharedRequestedConstants`` directly with a hand-made budget and ~768 KiB
synthetic arrays. It never goes through ``GpuEventManager``, so the sizing
callback, the ``placement.can_share`` branch and teardown are not covered, and
the arrays are nothing like real pedestals.

This driver is the combination: every BD worker pinned to one device so they
are peers, the real detector's pedestals and pixel_gain shared from a single
device allocation, and the calibration still required to be pixel-exact
against a CPU reference. A follower here reads its constants out of the
owner's memory through an imported IPC mapping, so a shared view that is
subtly wrong -- a wrong offset, a truncated mapping, values that never
arrived -- shows up as mismatched pixels rather than as nothing at all.

It also reports the device memory the sharing actually saved, on the real
constants rather than a fixture.

Run with every rank on one GPU, so the BD workers are peers::

    CUDA_VISIBLE_DEVICES=0 PS_EB_NODES=1 mpirun -n 4 \\
        python mpi_shared_calibration.py --user-dir <copied-kernel-dir>

``--expect-peers N`` asserts the layout, and ``--require-sharing`` fails if
sharing did not engage, which is what makes this case different from running
the calibration driver and hoping.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np


def emit(tag, obj):
    print(f'SHAREDCALIB_{tag} ' + json.dumps(obj, indent=2, sort_keys=True),
          flush=True)


def batch_constant(run, selector):
    """This rank's device array for a declared selector, or None.

    Read from the live manager rather than from a callback argument, because
    the point is to hold it OUTSIDE the callback that produced it.
    """
    manager = getattr(run, '_gpu_manager_for_placement', None)
    constants = getattr(manager, '_task_constants', None)
    if constants is None:
        return None
    try:
        return constants.get(*selector)
    except Exception:                                     # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user-dir', type=Path, required=True,
                    help='directory holding the copied jungfrau_calibration.py')
    ap.add_argument('--expect-peers', type=int, default=0)
    ap.add_argument('--require-sharing', action='store_true',
                    help='fail unless constants were shared over IPC')
    ap.add_argument('--offset', action='store_true')
    ap.add_argument('--max-events', type=int, default=13)
    ap.add_argument('--batch-size', type=int, default=5)
    args = ap.parse_args()

    sys.path.insert(0, str(args.user_dir.resolve()))
    user_module = importlib.import_module('jungfrau_calibration')
    assert Path(user_module.__file__).resolve().parent == args.user_dir.resolve()

    from mpi4py import MPI
    from psana import DataSource
    from psana.gpu import GpuTask
    from psana.tests.gpu.user_calibration_reference import calibrate

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    user = user_module.JungfrauCalibration(use_offset=args.offset)
    # Not an assert: python -O strips those, and this invariant -- declaring a
    # task must not import CuPy -- is the kind that fails silently.
    if 'cupy' in sys.modules:
        emit('RANK' + str(comm.Get_rank()),
             {'PASS': False,
              'failures': ['CuPy was imported while declaring the task']})
        return 1

    batches = []
    # Counters must be read while the constants are LIVE. Reading them after
    # the event loop finishes samples a torn-down state: close() clears the
    # owned and imported dicts, so shared_bytes and imported_bytes both go to
    # zero and the only charge left is each rank's own pipeline memory.
    live = {}
    # Mutable cell so the callback can see the Run once it exists; the
    # callback is constructed before DataSource is.
    current_run = []

    def sample_live_sharing():
        if live or not current_run:
            return
        run = current_run[0]
        manager = getattr(run, '_gpu_manager_for_placement', None)
        placement = getattr(run, '_gpu_placement', None)
        constants = getattr(manager, '_task_constants', None)
        if manager is None or placement is None or constants is None:
            return
        live.update(
            sharing=placement.sharing,
            shared_selectors=[list(x) for x in
                              (getattr(constants, 'shared_selectors', ()) or ())],
            private_selectors=[list(x) for x in
                               (getattr(constants, 'private_selectors', ()) or ())],
            # Two different figures, both wanted. placement.shared_bytes is
            # the group's agreed intersection size; constants.shared_bytes is
            # what THIS rank has resident for it -- owned blocks on the owner,
            # imported mappings on a follower. They must agree while sharing
            # is live, and both drop to zero after close().
            shared_bytes=int(placement.shared_bytes),
            resident_shared_bytes=int(getattr(constants, 'shared_bytes', 0) or 0),
            imported_bytes=int(getattr(constants, 'imported_bytes', 0) or 0),
            charged_bytes=int(manager._gpu_budget.committed()),
            budget_limit=int(manager._gpu_budget.limit()),
        )

    def callback(batch, stream):
        sample_live_sharing()
        user(batch, stream)
        batches.append(batch.size)

    task = GpuTask(callback, inputs=user.inputs, calibconst=user.calibconst)
    ds = DataSource(exp='mfx100848724', run=51,
                    dir='/sdf/data/lcls/ds/prj/public01/xtc',
                    detectors=['jungfrau'], gpu_det='jungfrau', gpu_fn=task,
                    max_events=args.max_events, batch_size=args.batch_size,
                    n_gpu_streams=2, gpu_bulk_read=True)

    failures, checks = [], {}
    info = {'rank': rank, 'declared': [list(s) for s in user.calibconst]}
    events_checked = 0

    for run in ds.runs():
        role = run.comms.node_type()
        info['role'] = role
        if role != 'bd':
            # Must still reach the end of the run so the BD workers' own
            # collectives complete.
            for _ in run.events():
                pass
            checks['no_cupy_on_cpu_role'] = 'cupy' not in sys.modules
            if not checks['no_cupy_on_cpu_role']:
                failures.append('a CPU-role rank imported CuPy')
            break

        import cupy as cp
        det = run.Detector('jungfrau')
        current_run[:] = [run]
        placement = getattr(run, '_gpu_placement', None)

        checks['placement_present'] = placement is not None
        if placement is None:
            failures.append('no GPU placement on the Run')
            break

        info['device_pci'] = placement.device_pci
        info['n_device_peers'] = placement.n_device_peers
        info['is_owner'] = placement.is_owner

        # Every BD worker must be on the SAME device, or there is nothing to
        # share and this case is not testing what it claims to.
        checks['peers_on_one_device'] = placement.n_device_peers > 1
        if args.require_sharing and not checks['peers_on_one_device']:
            failures.append(
                f'only {placement.n_device_peers} peer(s) on this device; '
                'pin every rank to one GPU (CUDA_VISIBLE_DEVICES=0) or this '
                'case cannot exercise sharing')
        if args.expect_peers and placement.n_device_peers != args.expect_peers:
            failures.append(f'expected {args.expect_peers} peers, discovered '
                            f'{placement.n_device_peers}')

        # Run the real calibration and compare every panel to the CPU
        # reference computed from this rank's own host constants. The host
        # copies are this rank's; the DEVICE copies a follower calibrates
        # from are the owner's, imported over IPC. Equality therefore proves
        # the imported view carries the right values at the right offsets.
        host_constants = {key: det.calibconst[key][0]
                          for _, key in user.calibconst}
        # Hold a device constant across the whole event loop. One upload per
        # run means the allocation is never replaced, so this must stay valid
        # and keep its values -- on a follower it is the owner's memory, read
        # through an imported mapping.
        held = None
        held_before = None
        held_checks = 0
        held_mismatches = 0

        for event in run.events():
            fields = event.gpu.detector('jungfrau').field('raw', 'raw').on_cpu
            segments = tuple(fields.segment_ids)
            panels = []
            for segment in segments:
                panel = fields[segment]
                if panel.ndim == 3 and panel.shape[0] == 1:
                    panel = panel[0]
                panels.append(panel)
            raw = np.stack(panels)[None]
            expected = calibrate(raw, np.ones(raw.shape[:2], np.uint8),
                                 segments, host_constants,
                                 use_offset=args.offset)[0]
            if held is None:
                held = batch_constant(run, user.calibconst[0])
                if held is not None:
                    held_before = cp.asnumpy(held).copy()
            elif held_before is not None:
                # Still the same array object, obtained on an earlier event:
                # one upload per run means it must still hold its values.
                held_checks += 1
                if not np.array_equal(cp.asnumpy(held), held_before):
                    held_mismatches += 1
            actual = event.gpu.get(user.output).on_cpu
            try:
                np.testing.assert_array_equal(actual, expected)
            except AssertionError:
                bad = int(np.count_nonzero(actual != expected))
                failures.append(
                    f'event {int(event.timestamp)}: {bad} pixel(s) differ '
                    'from the CPU reference while calibrating from shared '
                    'constants')
                break
            events_checked += 1

        # Retention was checked inside the loop, on every event after the
        # first. It cannot be checked here: run.events() closes the GPU
        # manager in its finally block when the generator exhausts, which
        # frees the owner's allocation and closes the importers' mappings.
        # That is the documented boundary -- valid for the run, not past it.
        checks['held_constant_stayed_valid'] = held_mismatches == 0
        if held_mismatches:
            failures.append(
                f'a constant held across events changed value on '
                f'{held_mismatches} of {events_checked} events; the shared '
                'allocation was rewritten mid-run')
        info['held_checks'] = held_checks

        # The manager created for this run holds the constants object. By
        # here the event loop has finished, so these are TEARDOWN values: a
        # clean close() clears the owned and imported dicts, which is why the
        # sharing assertions below read `live`, captured in the callback.
        manager = getattr(run, '_gpu_manager_for_placement', None)
        checks['manager_available'] = manager is not None
        if manager is None:
            failures.append('the run exposed no GPU event manager')
            break

        info.update({
            'live': dict(live),
            'after_loop': {
                'shared_bytes': int(placement.shared_bytes),
                'charged_bytes': int(manager._gpu_budget.committed()),
            },
            'events_checked': events_checked,
            'batches': batches,
            'describe': placement.describe(),
        })

        checks['live_sample_captured'] = bool(live)
        if not live:
            failures.append('no live sharing sample was taken; the callback '
                            'never ran, so nothing about sharing was observed')
        elif args.require_sharing:
            checks['sharing_engaged'] = live['sharing'] == 'on'
            if not checks['sharing_engaged']:
                failures.append(f"sharing did not engage: "
                                f"sharing={live['sharing']!r}")

            checks['constants_were_shared'] = bool(live['shared_selectors'])
            if not live['shared_selectors']:
                failures.append('no selector was shared; every rank uploaded '
                                'its own copy')

            # The group agreed a size, and this rank must actually have that
            # much resident for it -- owned on the owner, imported on a
            # follower.
            checks['resident_matches_the_agreed_size'] = (
                live['resident_shared_bytes'] == live['shared_bytes'])
            if not checks['resident_matches_the_agreed_size']:
                failures.append(
                    f"this rank has {live['resident_shared_bytes']} bytes "
                    f"resident for a {live['shared_bytes']}-byte intersection")

            # One device copy: the owner is charged for it, a follower imports
            # a view and is charged nothing for the shared set.
            if placement.is_owner:
                checks['owner_charged_for_the_copy'] = (
                    live['charged_bytes'] >= live['shared_bytes'])
                if not checks['owner_charged_for_the_copy']:
                    failures.append(
                        f"owner charged {live['charged_bytes']} bytes while "
                        f"holding {live['shared_bytes']} for the group")
                checks['owner_imports_nothing'] = live['imported_bytes'] == 0
                if live['imported_bytes']:
                    failures.append(f"the owner imported "
                                    f"{live['imported_bytes']} bytes")
            else:
                checks['follower_imported_a_view'] = live['imported_bytes'] > 0
                if not checks['follower_imported_a_view']:
                    failures.append('a follower imported nothing, so it is '
                                    'reading its own copy')

        # The shared copy must be gone from the constants object itself. A
        # budget total is the wrong thing to compare: the pipeline allocates
        # per-batch buffers as the loop runs, so the owner's committed bytes
        # at the first callback do not predict what remains at the end.
        #
        # placement.shared_bytes is the group's agreed size and is not reset,
        # so read the per-rank resident figure, which sums the owned and
        # imported blocks and therefore goes to zero once they are released.
        constants = getattr(manager, '_task_constants', None)
        resident_after = int(getattr(constants, 'shared_bytes', 0) or 0)
        imported_after = int(getattr(constants, 'imported_bytes', 0) or 0)
        info['after_loop'].update(resident_shared_bytes=resident_after,
                                  imported_bytes=imported_after)
        checks['shared_copy_released_at_teardown'] = (
            resident_after == 0 and imported_after == 0)
        if not checks['shared_copy_released_at_teardown']:
            failures.append(
                f'after the loop this rank still has {resident_after} shared '
                f'and {imported_after} imported bytes resident; teardown did '
                'not release them, so the next run on this device inherits it')

        checks['events_calibrated'] = events_checked > 0
        if not events_checked:
            failures.append('no event was calibrated, so nothing was proven')
        break

    info.update({'checks': checks, 'failures': failures,
                 'PASS': not failures})
    emit(f'RANK{rank}', info)

    rows = comm.gather(info, root=0)
    comm.Barrier()
    if rank == 0:
        summarise(rows, require_sharing=args.require_sharing)
    return 0 if not failures else 1


def summarise(rows, require_sharing=False):
    members = [r for r in rows if r]
    workers = [r for r in members if r.get('role') == 'bd']
    failures = [f for r in members for f in r.get('failures', [])]

    devices = sorted({r['device_pci'] for r in workers if 'device_pci' in r})
    if require_sharing and len(devices) > 1:
        failures.append(f'GPU workers spread over {len(devices)} devices '
                        f'{devices}; pin them to one to exercise sharing')

    owners = [r for r in workers if r.get('is_owner')]
    followers = [r for r in workers if not r.get('is_owner')]

    # From the live sample, not the post-loop state: close() clears the owned
    # and imported dicts, so anything read after the event loop reports zero.
    def live_of(record, key, default=0):
        return (record.get('live') or {}).get(key, default)

    shared_bytes = max((live_of(r, 'shared_bytes') for r in workers), default=0)

    # What sharing saved on the real constants: every follower would otherwise
    # hold its own copy of the shared set.
    saved = shared_bytes * len(followers)

    emit('SUMMARY', {
        'ranks': len(members),
        'gpu_workers': len(workers),
        'devices': devices,
        'owners': len(owners),
        'followers': len(followers),
        'shared_bytes': shared_bytes,
        'shared_mib': round(shared_bytes / 1024 ** 2, 1),
        'device_memory_saved_mib': round(saved / 1024 ** 2, 1),
        'imported_by_follower': {r['rank']: live_of(r, 'imported_bytes')
                                 for r in followers},
        'charged_live_by_rank': {r['rank']: live_of(r, 'charged_bytes')
                                 for r in workers},
        'charged_after_loop_by_rank': {
            r['rank']: (r.get('after_loop') or {}).get('charged_bytes')
            for r in workers},
        'events_checked': sum(r.get('events_checked', 0) for r in workers),
        'all_pass': not failures,
        'failures': failures[:10],
    })


if __name__ == '__main__':
    # A real failure on one rank must abort the job, or its peers block in
    # their next collective. SystemExit is excluded deliberately: sys.exit()
    # raises it, so catching BaseException here would turn this driver's own
    # clean exit into an MPI_Abort and a nonzero launcher status.
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
            # Checks failed on this rank; stop the peers too rather than let
            # them wait, and give the launcher a nonzero status.
            MPI.COMM_WORLD.Abort(1)
    sys.exit(status)
