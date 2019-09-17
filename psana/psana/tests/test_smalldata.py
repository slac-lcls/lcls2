

import os
from setup_input_files import setup_input_files
import run_smalldata

def test_smalldata(tmp_path):

    # remove this line when import of SmallData is fixed (due
    # to shmem test crash)
    if run_smalldata.DOTEST is False: return
    setup_input_files(tmp_path) # tmp_path is from pytest
    os.environ['TEST_XTC_DIR'] = str(tmp_path)
    run_smalldata.main()
    return
