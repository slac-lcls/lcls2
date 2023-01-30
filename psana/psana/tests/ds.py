# Test datasource class
# More exhaustive than user_loops.py or user_callback.py

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
    ds = DataSource(exp='xpptut13', run=1, dir=xtc_dir, 
            detectors=['xppcspad_2'], 
            xdetectors=['epicsinfo'])

    sendbuf = np.zeros(1, dtype='i')
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty([size, 1], dtype='i')

    for run in ds.runs():
        det = run.Detector('xppcspad')
        for evt in run.events():
            sendbuf += 1
            assert evt._size == 1 # only s02 has xppcspad and no epicsinfo

    comm.Gather(sendbuf, recvbuf, root=0)
    if rank == 0:
        assert np.sum(recvbuf) == 10 # need this to make sure that events loop is active

def test_replace_with_smd():
    # Usecase 4 : selecting only xppcspad
    ds = DataSource(exp='xpptut13', run=1, dir=xtc_dir, detectors=['epicsinfo'], small_xtc=['epicsinfo'])

    sendbuf = np.zeros(1, dtype='i')
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty([size, 1], dtype='i')

    for run in ds.runs():
        det = run.Detector('xppcspad')
        for evt in run.events():
            sendbuf += 1

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

def test_multi_seg_epics():
    xtc_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data', 'multi-seg-epics')
    ds = DataSource(exp="xpptut15", run=1, dir=xtc_dir)
    run = next(ds.runs())

    hsd = run.Detector('hsd')
    andor = run.Detector('andor')
    # Epics variables from segment 0 (stream 0)
    epicsvar1 = run.Detector('epicsvar1')
    epicsvar2 = run.Detector('epicsvar2')
    # Epics variable from segment 1 (stream 1)
    epics2 = run.Detector('background')

    for i_evt, evt in enumerate(run.events()):
        if i_evt > 5:
            epicsvar1_val = epicsvar1(evt)
            epicsvar2_val = epicsvar2(evt)
            epics2_val = epics2(evt)
            assert epicsvar1_val == "hello"
            assert epicsvar2_val == "world"
            assert epics2_val.shape == (10,10)


if __name__ == "__main__":
    test_standard()
    test_no_filter()
    test_step()
    test_select_detectors()
    test_replace_with_smd()
    test_multi_seg_epics()
    if size >= 3:
        test_callback(1)
        test_callback(5)
