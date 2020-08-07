import os
from psana import DataSource
import vals
import numpy as np

# cpo found this on the web as a way to get mpirun to exit when
# one of the ranks has an exception
from psana.psexp.tools import mode
if mode == 'mpi':
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

def filter_fn(evt):
    return True

xtc_dir = os.path.join(os.environ.get('TEST_XTC_DIR', os.getcwd()),'.tmp')

ds = DataSource(exp='xpptut13', run=1, dir=xtc_dir, filter=filter_fn)
n_events = 0
def event_fn(event, det):
    global n_events
    padarray = vals.padarray
    assert(np.array_equal(det.raw.calib(event),np.stack((padarray,padarray,padarray,padarray))))
    n_events += 1

for run in ds.runs():
    det = run.Detector('xppcspad')
    edet = run.Detector('HX2:DVD:GCC:01:PMON')
    assert run.expt == 'xpptut15'
    assert run.runnum == 14
    run.analyze(event_fn=event_fn, det=det)

if mode == 'legion':
    import pygion
    pygion.execution_fence(block=True)

print('n_events', n_events)
# FIXME: Elliott: This fails right now in Legion
# assert n_events == 10
