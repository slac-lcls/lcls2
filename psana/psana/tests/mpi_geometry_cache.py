"""Collective startup followed by a geometry request on only one MPI rank."""
from types import SimpleNamespace as NS

from mpi4py import MPI
import numpy as np

from psana.psexp.mpi_ds import RunParallel
from psana.psexp.mpi_shmem import MPISharedMemory
from psana.utils import Logger
from geometry_cache_fixture import GeometryDetector, assert_expected


def main():
    comm = MPI.COMM_WORLD
    assert comm.size >= 2
    memory = MPISharedMemory(shm_comm=comm)
    run = NS(
        comms=NS(get_shmem_comm=lambda: comm, psana_comm=comm),
        _geo_shared_mem=memory, logger=Logger(name='geometry-cache-test'),
        _iter_jungfrau_raw=lambda **kwargs: [
            ('camera', 'raw', GeometryDetector, None, {})],
    )
    # Exercise the actual startup call sites, including explicit collective opt-in.
    RunParallel._setup_jungfrau_shared_caches(run)
    detector = GeometryDetector()
    detector._shared_geo_cache = run._shared_geo_cache
    assert len(memory._handles) == 5
    for method in ('_pixel_coords', '_pixel_coord_indexes'):
        result = getattr(detector, method)()
        assert_expected(method, result)
        assert all(any(np.shares_memory(a, h.array) for h in memory._handles.values())
                   for a in result)
    assert detector.geometry.calls == [0, 0]  # Every startup hit remains shared.

    # Rotate the sole requester through leader and follower. Other ranks wait
    # in a different collective; a hidden accessor collective would hang.
    for requester in (0, 1):
        if comm.rank == requester:
            for method in ('_pixel_coords', '_pixel_coord_indexes'):
                result = getattr(detector, method)(all_segs=True, cframe=1)
                assert_expected(method, result, all_segs=True, cframe=1)
                calls = detector.geometry.calls.copy()
                again = getattr(detector, method)(all_segs=True, cframe=1)
                assert detector.geometry.calls == calls
                assert all(a is b for a, b in zip(result, again))
            assert len(memory._handles) == 5
        comm.Barrier()
    assert comm.allreduce(int(bool(run._shared_geo_cache._local_arrays))) == 2
    memory.close()
    if comm.rank == 0:
        print('MPI_GEOMETRY_CACHE_OK shared startup; isolated leader/follower misses; exact arrays', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
