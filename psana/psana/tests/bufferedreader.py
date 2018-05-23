from psana import bufferedreader, dgram
import os, time
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size > 1

n_events = 1000
FN_L = 100

def smd_server(fds):
    """ Sends byte data to client
    Finds matching events across all smd files
    then sends byte data to all clients
    FIXME: now first smd is assumed to be the
    fastest dectector.
    """
    assert len(fds) > 0
    
    eof = False
    limit_ts = 0
    bufrs = [bufferedreader.BufferedReader(fd) for fd in fds]
    while not eof:
        views = bytearray()
        for i, bufr in enumerate(bufrs):
            if i == 0:
                view = bufr.get(n_events)
                limit_ts = bufr.last_ts()
                if view == 0:
                    eof = True
                    break
            else:
                view = bufr.get(n_events, limit_ts = limit_ts)
            
            views.extend(view)
            if i < len(fds) - 1:
                views.extend(b'endofstream')

        if views:
            rankreq = np.empty(1, dtype='i')
            comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            comm.Send(views, dest=rankreq[0])
    
    for i in range(size-1):
        comm.Recv(rankreq, source=MPI.ANY_SOURCE)
        comm.Send(bytearray(b'eof'), dest=rankreq[0])

def smd_client(configs):
    while True:
        comm.Send(np.array([rank], dtype='i'), dest=0)
        
        info = MPI.Status()
        comm.Probe(MPI.ANY_SOURCE, MPI.ANY_TAG, info)
        count = info.Get_elements(MPI.BYTE)
        view = bytearray(count)
        comm.Recv(view, source=0)
        if view.startswith(b'eof'):
            break
        
        views = view.split(b'endofstream')
        for i in range(len(views)):
            offset = 0
            while offset < memoryview(views[i]).nbytes:
                d = dgram.Dgram(config=configs[i], view=views[i], offset=offset)
                offset += memoryview(d).shape[0]


if __name__ == "__main__":
    comm.Barrier()
    ts0 = MPI.Wtime()
    nfiles = 2 
    
    # broadcast smd files
    if rank == 0:
        smd_files = np.array(['/reg/d/psdm/xpp/xpptut15/scratch/mona/smd-00.xtc', '/reg/d/psdm/xpp/xpptut15/scratch/mona/smd-01.xtc'], dtype='U%s'%FN_L)
    else:
        smd_files = np.empty(nfiles, dtype='U%s'%FN_L)
    comm.Bcast([smd_files, MPI.CHAR], root=0)

    # broadcast configs
    if rank == 0:
        fds = [os.open(smd_file, os.O_RDONLY) for smd_file in smd_files]
        configs = [dgram.Dgram(file_descriptor=fd) for fd in fds]
        nbytes = np.array([memoryview(config).shape[0] for config in configs], dtype='i')
    else:
        configs = [dgram.Dgram()] * nfiles
        nbytes = np.empty(nfiles, dtype='i')
    comm.Bcast(nbytes, root=0) 
    for i in range(nfiles):
        comm.Bcast([configs[i], nbytes[i], MPI.BYTE], root=0)
    
    comm.Barrier()
    ts1 = MPI.Wtime()
    
    # start sending smd
    if rank == 0:
        smd_server(fds)
    else:
        smd_client(configs)
    
    comm.Barrier()
    ts2 = MPI.Wtime()
    if rank == 0:
        print("Total: %6.2f s Bcast: %6.2f s Rate: %6.2f MHz"%(ts2-ts0, ts1-ts0, 14.7/(ts2-ts0)))
