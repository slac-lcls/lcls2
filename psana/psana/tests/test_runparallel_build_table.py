import pytest

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
    xtc_dir = setup_input_files(tmp_path)

    ds = DataSource(exp='xpptut15', run=14, dir=str(xtc_dir))
    run = next(ds.runs())

    # Only BigDataNode ranks will populate _ts_table
    with run.build_table() as success:
        if success:
            # Ensure that _ts_table is populated with some valid timestamps
            assert hasattr(run, '_ts_table')
            assert isinstance(run._ts_table, dict)
            assert len(run._ts_table) > 0

            # Optionally verify a few entries have non-empty offsets
            non_empty_entries = [ts for ts, v in run._ts_table.items() if v]
            assert len(non_empty_entries) > 0

            # Sample the first few entries to verify we can access the events
            #for i, ts in enumerate(sorted(non_empty_entries[:5])):
            #    evt = run.event(ts)
            #    assert evt is not None
            #assert len(evt._dgrams) > 0


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    if not mpi_available or MPI.COMM_WORLD.Get_size() < 3:
        print("This script requires MPI with at least 3 ranks.")
        exit(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_runparallel_build_table(tmp_path)
