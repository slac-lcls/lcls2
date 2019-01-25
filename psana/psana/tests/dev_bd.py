import os
from psana import DataSource
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def filter_fn(evt):
    return True

comm.Barrier()
st = MPI.Wtime()
max_events = 1000000
#xtc_dir = "/ffb01/monarin/hsd"
xtc_dir = "/reg/d/psdm/xpp/xpptut15/scratch/mona/hsd"
ds = DataSource('exp=xpptut13:run=1:dir=%s'%(xtc_dir), filter=filter_fn, max_events=max_events)

sendbuf = np.zeros(1, dtype='i') 
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, 1], dtype='i')

for run in ds.runs():
    #det = run.Detector('xppcspad')
    for evt in run.events():
        sendbuf += 1 

comm.Gather(sendbuf, recvbuf, root=0)

comm.Barrier()
en = MPI.Wtime()
if rank == 0:
    n_events = np.sum(recvbuf)
    print('#events: %d total elapsed (s): %6.2f rate (kHz): %6.2f'%(n_events, en-st, n_events/((en-st)*1000)))

