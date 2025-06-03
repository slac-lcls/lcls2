import pytest
import random
import vals
import numpy as np

# Only import MPI if available
try:
    from mpi4py import MPI
    mpi_available = True
except ImportError:
    mpi_available = False

from psana import DataSource
from setup_input_files import setup_input_files

@pytest.mark.skipif(not mpi_available or MPI.COMM_WORLD.Get_size() < 3,
                    reason="Requires MPI with at least 3 ranks")
def test_runparallel_build_table(tmp_path):
    """
    Test for building the timestamp-to-offset table in RunParallel with MPI.
    This ensures that psana2 can selectively fetch events by timestamp.

    Requires running with at least 3 MPI ranks (smd0, eb, and at least one bd).
    """
    xtc_dir = prepare_xtc_dir(tmp_path)

    ds = DataSource(exp='xpptut15', run=14, dir=str(xtc_dir))
    run = next(ds.runs())

    # Only BigDataNode ranks will populate _ts_table
    with run.build_table() as success:
        if success:
            # Extract first 10 valid timestamps (those with non-empty offset info)
            valid_ts = sorted(k for k in run._ts_table if run._ts_table[k])[:10]
            assert len(valid_ts) == 10, "Expected at least 10 L1Accept events with offsets"

            # Randomly pick 3 timestamps to test
            sample_ts = random.sample(valid_ts, 3)
            for ts in sample_ts:
                evt = run.event(ts)
                assert evt is not None
                assert len(evt._dgrams) > 0

                det = run.Detector('xppcspad')
                padarray = vals.padarray
                assert(np.array_equal(det.raw.calib(evt),np.stack((padarray,padarray,padarray,padarray))))


def prepare_xtc_dir(tmp_path):
    comm = MPI.COMM_WORLD if mpi_available else None
    rank = comm.Get_rank() if comm else 0

    if rank == 0:
        xtc_dir = setup_input_files(tmp_path)
    else:
        xtc_dir = tmp_path  # Other ranks will use the same path without setup

    # Barrier to ensure rank 0 completes setup before others proceed
    if comm:
        comm.Barrier()

    return xtc_dir

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    if not mpi_available or MPI.COMM_WORLD.Get_size() < 3:
        print("This script requires MPI with at least 3 ranks.")
        exit(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_runparallel_build_table(tmp_path)
