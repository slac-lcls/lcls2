""" This shows and tests psana2 early termination for MPI processes.
"""
# Gets MPI for debugging/print out
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from psana import DataSource
import os



def run_terminate(ds):
    for run in ds.runs():
        det = run.Detector('xppcspad')
        cn_events = 0
        for i_evt, evt in enumerate(run.events()):
            if i_evt == 4:
                ds.terminate()
            cn_events +=1

        if rank == 2 or size == 1:
            print(f'rank: {rank} got {cn_events} events')
            assert cn_events == 5

def run_test_early_termination():
    xtc_dir = os.path.join(os.environ.get('TEST_XTC_DIR', os.getcwd()),'.tmp')
    os.environ['PS_SMD_N_EVENTS'] = '2'

    # Tests RunParallel and RunSerial """
    ds = DataSource(exp='xpptut15', run=14, dir=xtc_dir)
    run_terminate(ds)

    # Tests RunSingleFile
    if size == 1:
        xtc_file = os.path.join(xtc_dir, 'xpptut15-r0014-s000-c000.xtc2')
        ds = DataSource(files=xtc_file)
        run_terminate(ds)

if __name__ == "__main__":
    run_test_early_termination()
