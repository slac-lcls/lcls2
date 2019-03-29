# Test datasource class
# More exhaustive than user_loops.py or user_callback.py

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

import os
from psana import DataSource
import numpy as np
import vals
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def filter_fn(evt):
    return True

xtc_dir = os.path.join(os.getcwd(),'.tmp')

# Usecase 1a : two iterators with filter function
ds = DataSource('exp=xpptut13:run=1:dir=%s'%(xtc_dir), filter=filter_fn)

sendbuf = np.zeros(1, dtype='i')
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, 1], dtype='i')

for run in ds.runs():
    det = run.Detector('xppcspad')
    edet = run.Detector('XPP:VARS:STRING:01')
    for evt in run.events():
        sendbuf += 1
        padarray = vals.padarray
        assert(np.array_equal(det.raw.calib(evt),np.stack((padarray,padarray))))
        assert evt._size == 2 # check that two dgrams are in there
        assert edet(evt) == "Test String"

comm.Gather(sendbuf, recvbuf, root=0)
if rank == 0:
    assert np.sum(recvbuf) == 2 # need this to make sure that events loop is active

# Usecase 1b : two iterators without filter function
ds = DataSource('exp=xpptut13:run=1:dir=%s'%(xtc_dir))

sendbuf = np.zeros(1, dtype='i')
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, 1], dtype='i')

for run in ds.runs():
    det = run.Detector('xppcspad')
    for evt in run.events():
        sendbuf += 1
        padarray = vals.padarray
        assert(np.array_equal(det.raw.calib(evt),np.stack((padarray,padarray))))
        assert evt._size == 2 # check that two dgrams are in there

comm.Gather(sendbuf, recvbuf, root=0)
if rank == 0:
    assert np.sum(recvbuf) == 2 # need this to make sure that events loop is active

# Usecase 2: one iterator 
sendbuf = np.zeros(1, dtype='i')
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, 1], dtype='i')

for evt in ds.events():
    sendbuf += 1
    padarray = vals.padarray
    assert(np.array_equal(det.raw.calib(evt),np.stack((padarray,padarray))))
    assert evt._size == 2 # check that two dgrams are in there

comm.Gather(sendbuf, recvbuf, root=0)
if rank == 0:
    assert np.sum(recvbuf) == 2 # need this to make sure that events loop is active

# Usecase 3: reading smalldata w/o bigdata
ds = DataSource("exp=xpptut13:run=2:dir=%s"%(xtc_dir))

sendbuf = np.zeros(1, dtype='i')
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, 1], dtype='i')

for run in ds.runs():
    # FIXME: mona how to handle epics data for smalldata-only exp?
    for evt in run.events():
        sendbuf += 1
        assert evt._size == 2 # check that two dgrams are in there

comm.Gather(sendbuf, recvbuf, root=0)
if rank == 0:
    assert np.sum(recvbuf) == 2 # need this to make sure that events loop is active
