"""Four-rank Stage 2 setup acceptance: host-only services, per-BD uploads.

Run with PS_PARALLEL=mpi, PS_EB_NODES=1, PS_SRV_NODES=0 and one shared GPU.
Callback dispatch is not implemented, so this driver exercises run setup only.
"""
import sys
import numpy as np


def unused(evt, stream):
    raise AssertionError('Stage 2 must not invoke callbacks')


def main():
    from mpi4py import MPI
    from psana import DataSource
    from psana.gpu import GpuTask
    comm = MPI.COMM_WORLD
    assert comm.size == 4
    task = GpuTask(unused, ['jungfrau.raw'], [('jungfrau', 'pixel_gain')])
    ds = DataSource(exp='mfx100848724', run=51, dir='/sdf/data/lcls/ds/prj/public01/xtc',
                    detectors=['jungfrau'], max_events=1, gpu_det='jungfrau', gpu_fn=task)
    run = next(ds.runs())
    role = run.comms.node_type()
    if role == 'bd':
        import cupy as cp
        manager = run._make_gpu_event_manager()
        try:
            assert manager._gpu_task is task
            assert set(manager.input_preparers) == {'jungfrau.raw'}
            assert set(manager._task_constants._device) == {('jungfrau', 'pixel_gain')}
            source = run.dsparms.calibconst['jungfrau']['pixel_gain'][0]
            value = manager._task_constants.get('jungfrau', 'pixel_gain')
            assert value.shape == source.shape and value.dtype == source.dtype
            np.testing.assert_array_equal(cp.asnumpy(value), source)
            assert all(a['category'] in ('fixed', 'task-constants')
                       for a in manager._gpu_budget.allocation_snapshot())
        finally:
            manager.close()
    else:
        assert 'cupy' not in sys.modules, role
    roles = comm.gather(role, root=0)
    comm.Barrier()
    run.close_shared_memory()
    if comm.rank == 0:
        assert sorted(roles) == ['bd', 'bd', 'eb', 'smd0'], roles
        print('MPI_GPU_TASK_OK roles=' + repr(roles), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        from mpi4py import MPI
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
