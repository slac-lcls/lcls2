from psana import dgramchunk, dgram
import os, time
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size > 1

n_events = 10000
def smd_server(fd, config):
    dchunk = dgramchunk.DgramChunk(fd)
    displacement = memoryview(config).shape[0]
    view = dchunk.get(displacement, n_events)
    rankreq = np.empty(1, dtype='i')
    while view != 0:
        comm.Recv(rankreq, source=MPI.ANY_SOURCE)
        comm.Send(view, dest=rankreq[0])
        displacement += view.nbytes
        view = dchunk.get(displacement, n_events)
    
    for i in range(size-1):
        comm.Recv(rankreq, source=MPI.ANY_SOURCE)
        comm.Send(bytearray(b'end'), dest=rankreq[0])

def smd_client(config):
    while True:
        comm.Send(np.array([rank], dtype='i'), dest=0)
        
        info = MPI.Status()
        comm.Probe(MPI.ANY_SOURCE, MPI.ANY_TAG, info)
        count = info.Get_elements(MPI.BYTE)
        view = bytearray(count)
        comm.Recv(view, source=0)
        
        if view.startswith(b'end'):
            break
        
        offset = 0
        while offset < count:
            d = dgram.Dgram(config=config, view=view, offset=offset)
            offset += memoryview(d).shape[0]


if __name__ == "__main__":
    comm.Barrier()
    ts0 = MPI.Wtime()
    if rank == 0:
        fd = os.open('/reg/d/psdm/xpp/xpptut15/scratch/mona/smd.xtc', os.O_RDONLY)
        config = dgram.Dgram(file_descriptor=fd)
        config_size = np.array([memoryview(config).shape[0]], dtype='i')
    else:
        config = dgram.Dgram()
        config_size = np.empty(1, dtype='i')
    
    comm.Bcast(config_size, root=0)
    comm.Bcast([config, config_size[0], MPI.BYTE], root=0)
    comm.Barrier()
    ts1 = MPI.Wtime()

    if rank == 0:
        smd_server(fd, config)
    else:
        smd_client(config)
    
    comm.Barrier()
    ts2 = MPI.Wtime()
    if rank == 0:
        print("Total: %6.2f s Bcast: %6.2f s Rate: %6.2f MHz"%(ts2-ts0, ts1-ts0, 1/(ts2-ts0)))
