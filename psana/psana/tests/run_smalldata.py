

import os
# cpo found this on the web as a way to get mpirun to exit when
# one of the ranks has an exception
import sys
from glob import glob

import h5py
import numpy as np
from mpi4py import MPI
from test_shmem import Test as ShmemTest

from psana import DataSource


# Global error handler
def global_except_hook(exctype, value, traceback):
    sys.stderr.write("except_hook. Calling MPI_Abort().\n")
    # NOTE: mpi4py must be imported inside exception handler, not globally.
    # In chainermn, mpi4py import is carefully delayed, because
    # mpi4py automatically call MPI_Init() and cause a crash on Infiniband environment.
    import mpi4py.MPI
    mpi4py.MPI.COMM_WORLD.Abort(1)
    sys.__excepthook__(exctype, value, traceback)

#sys.excepthook = global_except_hook

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


global CALLBACK_OUTPUT
CALLBACK_OUTPUT = []


def gen_h5(source='xtc', pid=None):

    def test_callback(data_dict):
        CALLBACK_OUTPUT.append(data_dict['oneint'])


    if source == 'xtc':
        xtc_dir = os.path.join(os.environ.get('TEST_XTC_DIR', os.getcwd()),'.tmp')
        ds = DataSource(exp='xpptut15', run=14, dir=xtc_dir, filter=lambda x : True, batch_size=2)
    elif source == 'shmem':
        # Add a timeout to break infinite reads in reader thread
        # Needed since introducing the threaded dgrammanager for shared memory
        os.environ["PSANA_TESTS_SHMEM_TMO"] = "10" # In seconds

        ds = DataSource(shmem='shmem_test_' + pid)

    smd = ds.smalldata(filename='smalldata_test.h5', batch_size=5,
                       callbacks=[test_callback])

    for run in ds.runs():
        # test that we can make a Detector, which is somewhat subtle
        # because SRV cores make dummy detectors using NullDataSource/NullRun
        run.Detector('xppcspad')
        for i,evt in enumerate(run.events()):

            print('event:', i)

            smd.event(evt,
                      timestamp=evt.timestamp,
                      oneint=1,
                      twofloat=2.0,
                      arrint=np.ones(2, dtype=int),
                      arrfloat=np.ones(2, dtype=float)
                      # ragged_
                      )


            if evt.timestamp % 2 == 0:
                smd.event(evt.timestamp, # make sure passing int works
                          every_other_missing=2)
                smd.event(evt,
                          unaligned_int=3,
                          align_group="mygroup")

            if (rank % 2 == 0) and (smd._type == 'client'):
                smd.event(evt, missing_vds=1)
        # Todo: MONA put this in because with the new change in DgramManager,
        # it continues to look for new run in shmem mode. In this data file,
        # we have only one run so we needed to find a way to stop the process.
        # Future: We might need to test in case where where actually have multple
        # runs. For this, we may need an eof flag coming from the datasource.
        break
    # Similar to Mona's comment above... We hang here because a change in the
    # DgramManager for threaded shared memory has a thread that doesn't exit
    # After end run the reading thread tries to reconnect and enters a loop
    # at the ShmemClient.cc level
    if hasattr(ds, "dm"):
        # Guard for NullDataSource
        ds.dm.close_reader()

    if smd.summary:
        smd.save_summary({'summary_array' : np.arange(3)}, summary_int=1)
    smd.done()

    return


class SmallDataTest:

    def __init__(self):
        self.fn = 'smalldata_test.h5'
        print('TESTING --> %s' % self.fn)
        f = h5py.File(self.fn, 'r')
        self.f = f
        return

    def test_int(self):
        assert np.all(np.array(self.f['/oneint']) == 1)
        return

    def test_float(self):
        assert np.all(np.array(self.f['/twofloat']) == 2.0)
        return

    def test_arrint(self):
        a = np.array(self.f['/arrint'])
        assert np.all(a == 1), a
        assert a.shape[1] == 2, a
        assert a.dtype == int, a
        return

    def test_arrfloat(self):
        a = np.array(self.f['/arrfloat'])
        assert np.all(a == 1.0), a
        assert a.shape[1] == 2, a
        assert a.dtype == float, a
        return

    def test_unaligned(self, mode):
        d = np.array(self.f['mygroup/unaligned_int'])
        assert np.all(d == 3), d
        if mode=='xtc':
            assert d.shape == (5,), d
        return

    def test_every_other_missing(self, mode):
        d = np.array(self.f['/every_other_missing'])
        if mode=='xtc':
            assert np.sum((d == 2)) == 5, d
            assert np.sum((d == -99999)) == 5, d
        return

    def test_summary(self):
        d = np.array(self.f['/summary_array'])
        assert np.all(d == np.arange(3))
        d2 = np.array(self.f['/summary_int'])
        assert d2 == 1

    # test that if one of the srv h5 files is
    # missing a dataset completely that it is filled
    # in correctly by the h5 "missing data" feature
    # ("fill_value")
    # def test_missing_vds(self): return



# -----------------------

def run_test(mode, tmp_path):

    if rank == 0:
        for fn in glob(".?_smalldata_test.h5"):
            os.remove(fn)
    comm.barrier()

    if mode == 'xtc':
        gen_h5('xtc')
    elif mode == 'shmem':
        pid = None
        if rank == 0:
            pid = str(os.getpid())
            tmp_file = tmp_path / '.tmp/shmem/data_shmem.xtc2'
            ShmemTest.setup_input_files(tmp_path  / '.tmp')
            ShmemTest.launch_server(tmp_file, pid)

        pid = comm.bcast(pid, root=0)
        gen_h5('shmem', pid=pid)

    # make sure everyone is finished writing test file
    # then test with a single rank
    comm.barrier()
    if rank == 0:
        testobj = SmallDataTest()
        testobj.test_int()
        testobj.test_float()
        testobj.test_arrint()
        testobj.test_arrfloat()

        # currently these tests count the number of events,
        # however, that number is not deterministic for shmem (depends on speed)
        testobj.test_every_other_missing(mode)
        testobj.test_unaligned(mode)

    assert CALLBACK_OUTPUT == [1,] * len(CALLBACK_OUTPUT), CALLBACK_OUTPUT

    return


# pytest test_smalldata.py will call ONLY .main()
# NOTE : could merge test_smalldata.py into this file
def main(tmp_path):

    import platform
    # breakpoint()

    run_test('xtc', tmp_path)
    # don't test shmem on macOS, and centos7 in TRAVIS is failing for not-understood reasons
    if platform.system()!='Darwin' and os.getenv('LCLS_TRAVIS') is None:
        # cpo: this started hanging on aug 12 2025 presumably because of
        # changes to shmem to allow queuing of transitions
        run_test('shmem', tmp_path)
        pass
    return

# pytest byhand_mpi.py will call this section of the code (w/mpirun)
if __name__ == '__main__':

    import pathlib

    # COMMENT IN TO RUN python ...
    #from setup_input_files import setup_input_files
    #tmp_path = pathlib.Path('.')
    #setup_input_files(tmp_path) # tmp_path is from pytest :: makes .tmp
    #main(tmp_path)
    #os.system('rm -r .tmp')
    # COMMENT IN TO RUN pytest ...
    tmp_path = pathlib.Path(os.environ.get('TEST_XTC_DIR', os.getcwd()))
    main(tmp_path)
