

import os
import run_smalldata
from setup_input_files import setup_input_files

def test_smalldata(tmp_path):
    setup_input_files(tmp_path) # tmp_path is from pytest
    os.environ['TEST_XTC_DIR'] = str(tmp_path)
    run_smalldata.main()
    return

