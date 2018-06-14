from psana import dgram
from psana.smdreader import SmdReader
from psana.event import Event
from psana.eventbuilder import EventBuilder
import os, time
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size > 1

n_events = 10000
FN_L = 100

def filter(evt):
    return True

def smd_0(fds, n_smd_nodes):
    """ Sends blocks of smds to smd_node
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp to an smd_node.
    """
    assert len(fds) > 0
    
    smdr = SmdReader(fds)
    got_events = -1
    rankreq = np.empty(1, dtype='i')
    while got_events != 0:
        smdr.get(n_events)
        got_events = smdr.got_events
        views = bytearray()
        for i in range(len(fds)):
            view = smdr.view(i)
            if view != 0:
                views.extend(view)
            if i < len(fds) - 1:
                views.extend(b'endofstream')

        if views:
            comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            comm.Send(views, dest=rankreq[0], tag=12)
    
    for i in range(n_smd_nodes):
        comm.Recv(rankreq, source=MPI.ANY_SOURCE)
        comm.Send(bytearray(b'eof'), dest=rankreq[0], tag=12)

def smd_node(configs, n_bd_nodes, batch_size=1):
    """Handles both smd_0 and bd_nodes
    Receives blocks of smds from smd_0 then assembles
    offsets and dgramsizes into a numpy array. Sends
    this np array to bd_nodes that are registered to it."""
    rankreq = np.empty(1, dtype='i')
    view = 0
    while True:
        if not view:
            # handles requests from smd_0
            comm.Send(np.array([rank], dtype='i'), dest=0)
            info = MPI.Status()
            comm.Probe(source=0, tag=12, status=info)
            count = info.Get_elements(MPI.BYTE)
            view = bytearray(count)
            comm.Recv(view, source=0, tag=12)
            if view.startswith(b'eof'):
                break
        
        else:
            # send offset/size(s) to bd_node
            views = view.split(b'endofstream')
            n_views = len(views)
            assert n_views == len(configs)
            
            # build batch of events
            eb = EventBuilder(views, configs)
            batch = eb.build(batch_size=batch_size, filter=filter)
            while eb.nevents:
                comm.Recv(rankreq, source=MPI.ANY_SOURCE, tag=13)
                comm.Send(batch, dest=rankreq[0])
                batch = eb.build(batch_size=batch_size, filter=filter)

            view = 0

    for i in range(n_bd_nodes):
        comm.Recv(rankreq, source=MPI.ANY_SOURCE, tag=13)
        comm.Send(bytearray(b'eof'), dest=rankreq[0])


def bd_node(configs, smd_node_id):
    cn_events = 0
    while True:
        comm.Send(np.array([rank], dtype='i'), dest=smd_node_id, tag=13)
        info = MPI.Status()
        comm.Probe(source=smd_node_id, tag=MPI.ANY_TAG, status=info)
        count = info.Get_elements(MPI.BYTE)
        view = bytearray(count)
        comm.Recv(view, source=smd_node_id)
        if view.startswith(b'eof'):
            break
        
        
        views = view.split(b'endofevt')
        #for i in range(len(views)-1):
        #    evt = Event.from_bytes(configs, views[i])
        cn_events += len(views) - 1

    print('bd_node', rank, 'received', cn_events, 'events')
        


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
    
    # start smd-bd nodes
    n_smd_nodes = int(round((size - 1) * .25))
    n_bd_nodes = size - 1 - n_smd_nodes
    if rank == 0:
        smd_0(fds, n_smd_nodes)
    elif rank < n_smd_nodes + 1:
        bd_nodes = (np.arange(size)[n_smd_nodes+1:] % n_smd_nodes) + 1
        smd_node(configs, len(bd_nodes[bd_nodes==rank]), batch_size=100)
    else:
        smd_node_id = (rank % n_smd_nodes) + 1
        bd_node(configs, smd_node_id)
    
    comm.Barrier()
    ts2 = MPI.Wtime()
    if rank == 0:
        print("Total: %6.2f s Bcast: %6.2f s Rate: %6.2f MHz"%(ts2-ts0, ts1-ts0, 14.7/(ts2-ts0)))
