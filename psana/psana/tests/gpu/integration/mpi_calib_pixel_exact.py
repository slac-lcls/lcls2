"""Multi-rank psana event loop: GPU calibration, pixel-exact against the CPU.

One test for the whole branch. A real MPI job -- smd0, event builders and BD
workers, assigned by psana -- runs GPU calibration over real Jungfrau data and
compares every event against ``det.raw.calib(evt)`` computed on the CPU in the
same loop.

The point is that this covers the branch's machinery WITHOUT asserting on any
of it. If device discovery put two workers on one GPU and mis-sized their
budgets, if a constant-sharing follower read the owner's memory through a
mapping with the wrong offset, if a BD worker got a batch meant for another
rank -- the pixels come out wrong. Values being right is the end-to-end
statement that the internals agreed.

Run with several ranks. With one GPU visible the BD workers are peers of one
another, so constant sharing engages; with the mask left wide they spread over
the node's cards and each shares with nobody::

    CUDA_VISIBLE_DEVICES=0 PS_EB_NODES=1 mpirun -n 4 \\
        python mpi_calib_pixel_exact.py
    PS_EB_NODES=2 mpirun -n 9 python mpi_calib_pixel_exact.py

``hybrid_det`` is what makes the comparison self-contained: the detector is
routed to the GPU *and* the CPU interface stays available, so each rank checks
its own events against psana's own calibration with no reference file, no
second pass and no fixture to keep in step.
"""
import argparse
import json
import os
import sys

import numpy as np


def emit(tag, obj):
    print(f'CALIB_{tag} ' + json.dumps(obj, indent=2, sort_keys=True),
          flush=True)


def calibration_planes(exp, run, directory):
    """Which optional calibration planes this detector publishes.

    The kernel's policy flags decide which constants it REQUESTS, and asking
    for one the detector does not publish raises KeyError during setup -- so
    they cannot be hardcoded. They also have to match what det.raw.calib()
    applies, or the two sides compute different things and a mismatch would
    mean nothing. Measured on mfx100848724 r51: pixel_offset and pixel_status
    are present, status_extra is not.
    """
    from psana import DataSource
    ds = DataSource(exp=exp, run=run, dir=directory, max_events=1)
    det = next(ds.runs()).Detector('jungfrau')
    return {key for key in ('pixel_offset', 'pixel_status', 'status_extra')
            if (det.calibconst.get(key) or (None,))[0] is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exp', default='mfx100848724')
    ap.add_argument('--run', type=int, default=51)
    ap.add_argument('--dir', default='/sdf/data/lcls/ds/prj/public01/xtc')
    ap.add_argument('--max-events', type=int, default=13)
    ap.add_argument('--batch-size', type=int, default=5)
    ap.add_argument('--expect-events', type=int, default=0,
                    help='assert the job calibrated exactly this many events')
    args = ap.parse_args()

    from mpi4py import MPI
    from psana import DataSource
    from psana.gpu import GpuTask
    from psana.gpu.examples.jungfrau_calibration import JungfrauCalibration

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    full = (1 << 64) - 1
    planes = calibration_planes(args.exp, args.run, args.dir)
    user = JungfrauCalibration(
        use_offset='pixel_offset' in planes,
        status_bits=full if 'pixel_status' in planes else 0,
        stextra_bits=full if 'status_extra' in planes else 0)

    ds = DataSource(exp=args.exp, run=args.run, dir=args.dir,
                    detectors=['jungfrau'],
                    # hybrid, not gpu_det: keeps det.raw.calib() usable so the
                    # CPU reference comes from the same event.
                    hybrid_det='jungfrau',
                    gpu_fn=GpuTask(user, inputs=user.inputs,
                                   calibconst=user.calibconst),
                    max_events=args.max_events, batch_size=args.batch_size,
                    n_gpu_streams=2, gpu_bulk_read=True)

    failures, stamps = [], []
    info = {'rank': rank, 'host': MPI.Get_processor_name(),
            'planes': sorted(planes),
            'eb_nodes': os.environ.get('PS_EB_NODES'),
            'mask': os.environ.get('CUDA_VISIBLE_DEVICES')}

    for run in ds.runs():
        info['role'] = run.comms.node_type()
        det = run.Detector('jungfrau')
        for evt in run.events():
            stamp = int(evt.timestamp)

            # The GPU result, as a user kernel published it.
            actual = np.asarray(evt.gpu.get(user.output).on_cpu)
            # psana's own calibration of the same event, on the CPU.
            cpu = np.asarray(det.raw.calib(evt, cmpars=None))
            # Index the CPU panels by the GPU's routing order, which need not
            # be the physical 0..31.
            segments = list(evt.gpu.detector('jungfrau')
                            .field('raw', 'raw').on_cpu.segment_ids)
            expected = cpu[segments]

            if actual.shape != expected.shape:
                actual = actual.reshape(expected.shape)
            if not np.array_equal(actual, expected):
                bad = int(np.count_nonzero(actual != expected))
                worst = float(np.max(np.abs(actual.astype(np.float64)
                                            - expected.astype(np.float64))))
                failures.append(
                    f'event {stamp}: {bad} of {expected.size} pixels differ '
                    f'from det.raw.calib(), worst |delta| {worst:g}')
            stamps.append(stamp)
        break                       # one run is enough for a value check

    info.update(events=len(stamps), stamps=stamps,
                callback_calls=user.calls,
                failures=failures, PASS=not failures)
    emit(f'RANK{rank}', info)

    # Every rank reaches the gather, including one that found a mismatch: an
    # early exit would leave its peers blocked here and the launcher would
    # report a timeout instead of the wrong pixels.
    rows = comm.gather(info, root=0)
    comm.Barrier()
    if rank == 0:
        summarise(rows, args)
    # Hold every rank until the summary is written, or a failing rank's
    # MPI_Abort truncates it.
    comm.Barrier()
    return 0 if not failures else 1


def summarise(rows, args):
    """Job-wide checks: the ones a single rank cannot make."""
    members = [r for r in rows if r]
    workers = [r for r in members if r.get('role') == 'bd']
    failures = [f for r in members for f in r.get('failures', [])]

    everything = [s for r in workers for s in r.get('stamps', [])]
    total = sum(r.get('events', 0) for r in workers)

    # Work must have been distributed, not all landed on one worker or none.
    if not total:
        failures.append('no BD rank calibrated anything, so the comparison '
                        'never ran')
    if args.expect_events and total != args.expect_events:
        failures.append(f'the job calibrated {total} events, expected '
                        f'{args.expect_events}; psana dropped or duplicated '
                        'work across the BD ranks')
    if everything and len(everything) != len(set(everything)):
        failures.append('an event was delivered to more than one BD rank')

    emit('SUMMARY', {
        'ranks': len(members),
        'gpu_workers': len(workers),
        'hosts': sorted({r['host'] for r in workers if r.get('host')}),
        'events_total': total,
        'events_by_rank': {r['rank']: r.get('events') for r in workers},
        'callback_calls_by_rank': {r['rank']: r.get('callback_calls')
                                   for r in workers},
        'planes': sorted({p for r in members for p in r.get('planes', [])}),
        'all_pass': not failures,
        'failures': failures[:10],
    })


if __name__ == '__main__':
    # A mismatch on one rank must stop the job, or its peers block in their
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
