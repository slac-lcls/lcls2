

import os
from setup_input_files import setup_input_files
import run_smalldata

def test_smalldata(tmp_path):
    setup_input_files(tmp_path) # tmp_path is from pytest
    # breakpoint()
    os.environ['TEST_XTC_DIR'] = str(tmp_path)
    run_smalldata.main(tmp_path)
    return


