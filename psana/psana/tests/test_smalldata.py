

import os
from setup_input_files import setup_input_files
import run_smalldata
from psana.pscalib.calib import MDBWebUtils

def test_smalldata(tmp_path, monkeypatch):
    def unexpected_calibration(*args, **kwargs):
        raise AssertionError('The synthetic smalldata test must not fetch calibration')

    monkeypatch.setattr(MDBWebUtils, 'calib_constants_all_types', unexpected_calibration)
    setup_input_files(tmp_path) # tmp_path is from pytest
    os.environ['TEST_XTC_DIR'] = str(tmp_path)
    run_smalldata.main(tmp_path)
    return
