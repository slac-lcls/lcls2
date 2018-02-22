import sys
from psana import DataSource
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'Parallel read requires at least 2 ranks.'

BUFSIZE=0x4000

def master():
    ds = DataSource(['smd.xtc','smd_1.xtc'])
    for evt in ds:
        rankreq = comm.recv(source=MPI.ANY_SOURCE)
        offsets = [pydgram.info.offsetAlg.intOffset for pydgram in evt]
        comm.send(offsets, dest=rankreq)
    for rankreq in range(size-1):
        rankreq = comm.recv(source=MPI.ANY_SOURCE)
        comm.send('endrun', dest=rankreq)

def client():
    ds = DataSource(bigdata_files, config_bytes=config_bytes)
    while True:
        comm.send(rank, dest=0)
        offsets = comm.recv(source=0)
        if offsets == 'endrun': break

        evt = ds.jump(offsets=offsets)
        print('rank: ', rank, 'reads', offsets)


if __name__ == "__main__":
    if rank == 0:
        bigdata_files = ['data.xtc', 'data_1.xtc']
    else:
        bigdata_files = None
    bigdata_files = comm.bcast(bigdata_files, root=0)
   
    if rank == 0:
        ds = DataSource(bigdata_files)
        config_bytes = np.array([np.resize(config.buf, BUFSIZE).tobytes() for config in ds.configs], dtype='%sb'%BUFSIZE)
    else:
        ds = None
        config_bytes = np.empty(len(bigdata_files), dtype='%sb'%BUFSIZE)
    comm.Bcast(config_bytes, root=0)
    
    if rank == 0:
        master()
    else:
        client()
