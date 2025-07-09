# NOTE:
# To test on 'real' bigdata:
# xtc_dir = "/reg/d/psdm/xpp/xpptut15/scratch/mona/test"
# >bsub -n 64 -q psfehq -o log.txt mpirun python user_loops.py

#import logging
#logger = logging.getLogger('psana.psexp.node')
#logger.setLevel(logging.DEBUG)
#ch = logging.StreamHandler()
#ch.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#ch.setFormatter(formatter)
#logger.addHandler(ch)

import os
import vals
import numpy as np
from psana2 import DataSource

def filter_fn(evt):
    # Create a detector within filter is possible
    run = evt.run()
    det = run.Detector('xppcspad')

    # Access step event
    step_evt = run.step(evt)
    return True

xtc_dir = os.path.join(os.environ.get('TEST_XTC_DIR', os.getcwd()),'.tmp')

# Usecase 1a : two iterators with filter function
ds = DataSource(exp='xpptut15', run=14, dir=xtc_dir, filter=filter_fn)
#beginJobCode
for run in ds.runs():
    #beginRunCode
    # Detector interface identified by detector name
    det = run.Detector('xppcspad')

    # Environment values are accessed also through detector interface
    #edet = run.Detector('HX2:DVD:GCC:01:PMON')
    edet = run.Detector('HX2:DVD:GCC:01:PMON,hello1')
    sdet = run.Detector('motor2')

    for evt in run.events():
        padarray = vals.padarray
        # 4 segments, two per file
        assert(np.array_equal(det.raw.calib(evt),np.stack((padarray,padarray,padarray,padarray))))
        assert edet.dtype == float
        assert sdet.dtype == float
        assert edet(evt) is None or edet(evt) == 41.0
        assert sdet(evt) == 42.0
        assert run.expt == 'xpptut15' # this is from xtc file
        assert run.runnum == 14
        assert run.timestamp == 4294967297
    #endRunCode
#endJobCode

# Usecase#2 looping through steps
ds = DataSource(exp='xpptut15', run=14, dir=xtc_dir, batch_size=10)
for run in ds.runs():
    det = run.Detector('xppcspad')
    for step in run.steps():
        for evt in step.events():
            padarray = vals.padarray
            assert(np.array_equal(det.raw.calib(evt),np.stack((padarray,padarray,padarray,padarray))))

# Usecase#3: singlefile ds
ds = DataSource(files=os.path.join(xtc_dir,'xpptut15-r0014-s000-c000.xtc2'))
for run in ds.runs():
    det = run.Detector('xppcspad')
    edet = run.Detector('HX2:DVD:GCC:01:PMON')
    sdet = run.Detector('motor2')
    for step in run.steps():
        for evt in step.events():
            calib = det.raw.calib(evt)
            assert calib.shape == (2,3,6)
    assert run.expt == 'xpptut15'
    assert run.runnum == 14
