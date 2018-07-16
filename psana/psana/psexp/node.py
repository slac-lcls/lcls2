import os
import numpy as np
from psana.smdreader import SmdReader
from psana.eventbuilder import EventBuilder
from psana.event import Event
from mpi4py import MPI

class Node(object):
    def __init__(self, mpi):
        self.mpi = mpi

class Smd0(Node):
    """ Sends blocks of smds to smd_node
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp to an smd_node.
    """
    def __init__(self, mpi, fds, n_smd_nodes, max_events=0):
        Node.__init__(self, mpi)
        rank = self.mpi.rank
        comm = self.mpi.comm
        assert len(fds) > 0
        smdr = SmdReader(fds)
        got_events = -1
        processed_events = 0
        rankreq = np.empty(1, dtype='i')

        n_events = int(os.environ.get('PS_SMD_N_EVENTS', 100))
        if max_events:
            if max_events < n_events:
                n_events = max_events

        while got_events != 0:
            smdr.get(n_events)
            got_events = smdr.got_events
            processed_events += got_events
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

            if max_events:
                if processed_events == max_events:
                    break

        for i in range(n_smd_nodes):
            comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            comm.Send(bytearray(b'eof'), dest=rankreq[0], tag=12)

class SmdNode(Node):
    """Handles both smd_0 and bd_nodes
    Receives blocks of smds from smd_0 then assembles
    offsets and dgramsizes into a numpy array. Sends
    this np array to bd_nodes that are registered to it."""
    def __init__(self, mpi, configs, n_bd_nodes, batch_size=1, filter=0):
        Node.__init__(self, mpi)
        rank = self.mpi.rank
        comm = self.mpi.comm 
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

class BigDataNode(Node):
    def __init__(self, mpi, smd_configs, dm, smd_node_id):
        Node.__init__(self, mpi)
        self.smd_configs = smd_configs
        self.dm = dm
        self.smd_node_id = smd_node_id

    def events(self):
        rank = self.mpi.rank
        comm = self.mpi.comm
        while True:
            comm.Send(np.array([rank], dtype='i'), dest=self.smd_node_id, tag=13)
            info = MPI.Status()
            comm.Probe(source=self.smd_node_id, tag=MPI.ANY_TAG, status=info)
            count = info.Get_elements(MPI.BYTE)
            view = bytearray(count)
            comm.Recv(view, source=self.smd_node_id)
            if view.startswith(b'eof'):
                break

            views = view.split(b'endofevt')
            for event_bytes in views:
                if event_bytes:
                    evt = Event().from_bytes(self.smd_configs, event_bytes)
                    # get big data
                    ofsz = np.asarray([[d.info.offsetAlg.intOffset, d.info.offsetAlg.intDgramSize] \
                            for d in evt])
                    bd_evt = self.dm.next(offsets=ofsz[:,0], sizes=ofsz[:,1], read_chunk=False)
                    yield bd_evt



