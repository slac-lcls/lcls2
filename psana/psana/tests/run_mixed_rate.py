from psana import DataSource
import os
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def filter_fn(evt):
    return True

# Test mixed rate detectors
xtc_mixedrate_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mixed_rate')
ds = DataSource(exp='xpptut15', run=1, dir=xtc_mixedrate_dir)
sendbuf = np.zeros(1, dtype='i')
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, 1], dtype='i')

for run in ds.runs():
    det = run.Detector('xppcspad')
    edet = run.Detector('HX2:DVD:GCC:01:PMON')
    for evt in run.events():
        sendbuf += 1
        assert evt._size == 2 # both test files have xppcspad
        if evt._nanoseconds < 30: # first SlowUpdate is ts 30
            assert edet(evt) is None
        else:
            assert edet(evt) == 41.0

comm.Gather(sendbuf, recvbuf, root=0)
if rank == 0:
    assert np.sum(recvbuf) == 100 # need this to make sure that events loop is active

