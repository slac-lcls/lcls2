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

#import logging
#logging.basicConfig(level=logging.INFO, format='(%(threadName)-10s) %(message)s',)# filename="log.log", filemode="w")

import os
from psana import DataSource
import numpy as np
import vals
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

xtc_dir = os.path.join(os.environ.get('TEST_XTC_DIR', os.getcwd()),'.tmp')

def smd_callback(run):
    n_bd_nodes = size - 2
    for i_evt, evt in enumerate(run.events()):
        dest = (evt.timestamp % n_bd_nodes) + 1
        evt._proxy_evt.set_destination(dest)
        yield evt

def test_standard():
    # Usecase 1a : two iterators with filter function
    ds = DataSource(exp='xpptut13', run=1, dir=xtc_dir, batch_size=1)

    sendbuf = np.zeros(1, dtype='i')
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty([size, 1], dtype='i')

    for run in ds.runs():
        det = run.Detector('xppcspad')
        edet = run.Detector('HX2:DVD:GCC:01:PMON')
        infodet = run.Detector('epicsinfo')
        for evt in run.events():
            sendbuf += 1
            padarray = vals.padarray
            assert(np.array_equal(det.raw.calib(evt),np.stack((padarray,padarray,padarray,padarray))))
            assert evt._size == 2 # check that two dgrams are in there
            assert edet(evt) is None or edet(evt) == 41.0

    comm.Gather(sendbuf, recvbuf, root=0)
    if rank == 0:
        assert np.sum(recvbuf) == 10 # need this to make sure that events loop is active

def test_no_filter():
    # Usecase 1b : two iterators without filter function
    ds = DataSource(exp='xpptut13', run=1, dir=xtc_dir)

    sendbuf = np.zeros(1, dtype='i')
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty([size, 1], dtype='i')

    for run in ds.runs():
        det = run.Detector('xppcspad')
        for evt in run.events():
            sendbuf += 1
            padarray = vals.padarray
            assert(np.array_equal(det.raw.calib(evt),np.stack((padarray,padarray,padarray,padarray))))
            assert evt._size == 2 # check that two dgrams are in there

    comm.Gather(sendbuf, recvbuf, root=0)
    if rank == 0:
        assert np.sum(recvbuf) == 10 # need this to make sure that events loop is active

def test_step():
    # Usecase 3: test looping over steps
    ds = DataSource(exp='xpptut13', run=1, dir=xtc_dir)

    sendbuf = np.zeros(1, dtype='i')
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty([size, 1], dtype='i')

    for run in ds.runs():
        det = run.Detector('xppcspad')
        edet = run.Detector('HX2:DVD:GCC:01:PMON')
        for step in run.steps():
            for evt in step.events():
                sendbuf += 1
                padarray = vals.padarray
                assert(np.array_equal(det.raw.calib(evt),np.stack((padarray,padarray,padarray,padarray))))
                assert evt._size == 2 # check that two dgrams are in there
                assert edet(evt) is None or edet(evt) == 41.0

    comm.Gather(sendbuf, recvbuf, root=0)
    if rank == 0:
        assert np.sum(recvbuf) == 10 # need this to make sure that events loop is active

def test_select_detectors():
    # Usecase 4 : selecting only xppcspad
    ds = DataSource(exp='xpptut13', run=1, dir=xtc_dir, detectors=['xppcspad'])

    sendbuf = np.zeros(1, dtype='i')
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty([size, 1], dtype='i')

    for run in ds.runs():
        det = run.Detector('xppcspad')
        for evt in run.events():
            sendbuf += 1
            assert evt._size == 2 # both test files have xppcspad

    comm.Gather(sendbuf, recvbuf, root=0)
    if rank == 0:
        assert np.sum(recvbuf) == 10 # need this to make sure that events loop is active

def test_callback(batch_size):
    ds = DataSource(exp='xpptut13', run=1, dir=xtc_dir, smd_callback=smd_callback , batch_size=batch_size)

    sendbuf = np.zeros(1, dtype='i')
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty([size, 1], dtype='i')

    for run in ds.runs():
        det = run.Detector('xppcspad')
        edet = run.Detector('HX2:DVD:GCC:01:PMON')
        for step in run.steps():
            for evt in step.events():
                sendbuf += 1
                padarray = vals.padarray
                assert(np.array_equal(det.raw.calib(evt),np.stack((padarray,padarray,padarray,padarray))))
                assert evt._size == 2 # check that two dgrams are in there

    comm.Gather(sendbuf, recvbuf, root=0)
    if rank == 0:
        assert np.sum(recvbuf) == 10 # need this to make sure that events loop is active

if __name__ == "__main__":
    test_standard()
    test_no_filter()
    test_step()
    test_select_detectors()
    if size >= 3:
        test_callback(1)
        test_callback(5)
