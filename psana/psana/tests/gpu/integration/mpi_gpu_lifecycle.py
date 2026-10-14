"""Four-rank close/error acceptance. Callback failure must abort, never time out."""
from contextlib import closing
import gc
import json
import sys
import weakref

import numpy as np


def main():
    from mpi4py import MPI
    from psana import DataSource
    from psana.gpu import GpuTask
    from psana.psexp.mpi_ds import RunParallel
    from psana.psexp.run import Run

    # Isolate lifecycle acceptance from calibration-service latency. Real
    # requested constants are exercised separately by mpi_gpu_publication.py.
    gain_fixture = np.array([3], np.uint32)
    def setup_constants(run):
        run._calib_const = {"jungfrau": {"pixel_gain": (gain_fixture, {})}}
        run.dsparms.calibconst = run._calib_const
    Run._setup_run_calibconst = setup_constants

    mode, bulk, exit_kind = sys.argv[1:]
    comm = MPI.COMM_WORLD
    assert comm.size == 4
    refs, managers, held, calls = [], [], [], []
    original = RunParallel._make_gpu_event_manager

    def capture(run):
        manager = original(run)
        managers.append(manager)
        close = manager.close
        def checked_close():
            close()
            assert manager._closed and manager.event_pool.active_count == 0
            assert manager._output_d2h.pinned_bytes == 0
            print('MPI_LIFECYCLE_DRAINED', comm.rank, flush=True)
        manager.close = checked_close
        return manager
    RunParallel._make_gpu_event_manager = capture

    def callback(batch, stream):
        import cupy as cp
        calls.append(batch.size)
        scratch = cp.empty((batch.size,), cp.uint32)
        batch.keepalive(scratch)
        output = cp.empty_like(scratch)
        batch.publish('count', output)
        batch.publish('timestamp', batch.timestamps_gpu)
        gain = batch.calibconst('jungfrau', 'pixel_gain')
        refs.extend(weakref.ref(a) for a in (scratch, output, gain))
        scratch.fill(3)
        cp.add(scratch, np.uint32(4), out=output)
        if exit_kind == 'callback':
            raise ValueError('stage6 callback failure after launch')

    opts = dict(exp='mfx100848724', run=51, dir='/sdf/data/lcls/ds/prj/public01/xtc',
                detectors=['jungfrau'], max_events=40, batch_size=5, n_gpu_streams=2,
                gpu_fn=GpuTask(callback, ['jungfrau.raw'], [('jungfrau', 'pixel_gain')]),
                gpu_d2h_pinned_bytes=8192, gpu_bulk_read=bool(int(bulk)))
    opts['hybrid_det' if mode == 'hybrid' else 'gpu_det'] = 'jungfrau'
    role = None
    for run in DataSource(**opts).runs():
        role = run.comms.node_type()
        try:
            with closing(run.events()) as events:
                for event in events:
                    held.append(event)
                    # Leave published D2H unmaterialized until after closing.
                    if exit_kind == 'error':
                        raise ValueError('stage6 loop-body error')
                    if exit_kind == 'explicit':
                        events.close()
                    break
        except ValueError as exc:
            assert exit_kind == 'error' and str(exc) == 'stage6 loop-body error'
    gc.collect()
    if role == 'bd':
        assert managers and all(m._closed and m.event_pool.active_count == 0 for m in managers)
        assert all(ref() is None for ref in refs)
        assert len(held) <= 1
    else:
        assert 'cupy' not in sys.modules
    for event in held:
        assert event.gpu.get('count').on_cpu == 7
        assert event.gpu.get('timestamp').on_cpu == event.timestamp
    # Remove instrumentation roots before Run destruction: retaining a manager
    # also retains its Run and shared MPI windows on BD ranks alone.
    RunParallel._make_gpu_event_manager = original
    for captured in managers:
        del captured.close
    managers.clear()
    if role == 'bd':
        del captured
    rows = comm.gather(dict(role=role, seen=len(held), calls=calls), root=0)
    if comm.rank == 0:
        assert sum(r['seen'] for r in rows) == 2
        print('MPI_LIFECYCLE_OK ' + json.dumps(dict(mode=mode, bulk=bulk, exit=exit_kind, ranks=rows)), flush=True)


if __name__ == '__main__':
    main()
