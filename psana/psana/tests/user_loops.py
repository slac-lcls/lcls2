# NOTE:
# To test on 'real' bigdata:
# xtc_dir = "/reg/d/psdm/xpp/xpptut15/scratch/mona/test"
# >bsub -n 64 -q psfehq -o log.txt mpirun python user_loops.py

# cpo found this on the web as a way to get mpirun to exit when
# one of the ranks has an exception
import sys
# Global error handler
def global_except_hook(exctype, value, traceback):
    sys.stderr.write("except_hook. Calling MPI_Abort().\n")
    # NOTE: mpi4py must be imported inside exception handler, not globally.
    # In chainermn, mpi4py import is carefully delayed, because
    # mpi4py automatically call MPI_Init() and cause a crash on Infiniband environment.
    import mpi4py.MPI
    mpi4py.MPI.COMM_WORLD.Abort(1)
    sys.__excepthook__(exctype, value, traceback)
sys.excepthook = global_except_hook

# for debugging...
#import logging
#logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-10s) %(message)s',)

import os
import vals
import numpy as np
from psana import DataSource

def filter_fn(evt):
    return True

xtc_dir = os.path.join(os.environ.get('TEST_XTC_DIR', os.getcwd()),'.tmp')

# Usecase 1a : two iterators with filter function
ds = DataSource(exp='xpptut13', run=1, dir=xtc_dir, filter=filter_fn, monitor=True)
#beginJobCode
for run in ds.runs():
    #beginRunCode
    # Detector interface identified by detector name
    det = run.Detector('xppcspad')

    # Environment values are accessed also through detector interface
    edet = run.Detector('HX2:DVD:GCC:01:PMON')
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

# Usecase 1b : two iterators without filter function
ds = DataSource(exp='xpptut13', run=1, dir=xtc_dir)
for run in ds.runs():
    det = run.Detector('xppcspad')
    for evt in run.events():
        padarray = vals.padarray
        assert(np.array_equal(det.raw.calib(evt),np.stack((padarray,padarray,padarray,padarray))))


# Usecase#2 looping through steps
ds = DataSource(exp='xpptut13', run=1, dir=xtc_dir, filter=filter_fn)
for run in ds.runs():
    det = run.Detector('xppcspad')
    for step in run.steps():
        for evt in step.events():
            padarray = vals.padarray
            assert(np.array_equal(det.raw.calib(evt),np.stack((padarray,padarray,padarray,padarray))))

# Usecase#3: singlefile ds
ds = DataSource(files=os.path.join(xtc_dir,'data-r0001-s00.xtc2'))
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
