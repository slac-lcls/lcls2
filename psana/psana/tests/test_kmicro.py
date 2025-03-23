import os
import numpy as np
import pytest
from psana import DataSource

@pytest.mark.kmicro
def test_kmicro_data():
    xtc_dir = '/cds/home/opr/tstopr/data/drp/tst/tstx00117/xtc'

    if not os.path.exists(xtc_dir):
        pytest.skip(f"xtc_dir does not exist: {xtc_dir}")

    ds = DataSource(exp='tstx00117', run=271, dir=xtc_dir, max_events=10)

    for run in ds.runs():
        det = run.Detector('kmicro')

        for evt in run.events():
            xpos = det.raw.xpos(evt)
            ypos = det.raw.ypos(evt)
            time = det.raw.time(evt)

            # Assert all are NumPy arrays
            assert xpos is None or isinstance(xpos, np.ndarray), "xpos must be a numpy array or None"
            assert ypos is None or isinstance(ypos, np.ndarray), "ypos must be a numpy array or None"
            assert time is None or isinstance(time, np.ndarray), "time must be a numpy array or None"

            # Assert none are all zeros
            assert np.any(xpos != 0), "xpos is all zeros"
            assert np.any(ypos != 0), "ypos is all zeros"
            assert np.any(time != 0), "time is all zeros"

            # Metadata checks
            assert run.expt == 'tstx00117'
            assert run.runnum == 271
            assert run.timestamp != 0  # timestamp is non-zero
