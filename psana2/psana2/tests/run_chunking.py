from psana2 import DataSource
import os
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_test_chunking():
    xtc_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data', 'chunking')
    ds = DataSource(exp='xpptut15', run=14, dir=xtc_dir, batch_size=1)
    sendbuf = np.zeros(1, dtype='i')
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty([size, 1], dtype='i')

    run = next(ds.runs())
    for i, evt in enumerate(run.events()):
        sendbuf += 1

    comm.Gather(sendbuf, recvbuf, root=0)
    if rank == 0:
        assert np.sum(recvbuf) == 15 # need this to make sure that events loop is active

if __name__ == "__main__":
    run_test_chunking()

