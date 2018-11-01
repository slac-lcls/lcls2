import os
from sys import byteorder
import numpy as np
from psana.smdreader import SmdReader
from psana.eventbuilder import EventBuilder
from psana.event import Event
from psana.psexp.packet_footer import PacketFooter

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class Smd0(object):
    """ Sends blocks of smds to smd_node
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp to an smd_node.
    """
    def __init__(self, fds, n_smd_nodes, max_events=0):
        self.fds = fds
        self.n_files = len(fds)
        assert len(self.fds) > 0
        self.smdr = SmdReader(fds)

        self.n_smd_nodes = n_smd_nodes

        self.n_events = int(os.environ.get('PS_SMD_N_EVENTS', 1000))
        self.max_events = max_events
        self.processed_events = 0
        if self.max_events:
            if self.max_events < self.n_events:
                self.n_events = self.max_events

        self.run_mpi()

    def run_mpi(self):
        rankreq = np.empty(1, dtype='i')

        for chunk in self.chunks():
            comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            comm.Send(chunk, dest=rankreq[0], tag=12)

        for i in range(self.n_smd_nodes):
            comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            comm.Send(bytearray(b'eof'), dest=rankreq[0], tag=12)

    def chunks(self):
        got_events = -1
        while got_events != 0:
            self.smdr.get(self.n_events)
            got_events = self.smdr.got_events
            self.processed_events += got_events
            view = bytearray()
            pf = PacketFooter(n_packets=self.n_files)
            for i in range(self.n_files):
                if self.smdr.view(i) != 0:
                    view.extend(self.smdr.view(i))
                    pf.set_size(i, memoryview(self.smdr.view(i)).shape[0])

            if view:
                view.extend(pf.footer) # attach footer 
                yield view

            if self.max_events:
                if self.processed_events >= self.max_events:
                    break

class SmdNode(object):
    """Handles both smd_0 and bd_nodes
    Receives blocks of smds from smd_0 then assembles
    offsets and dgramsizes into a numpy array. Sends
    this np array to bd_nodes that are registered to it."""
    def __init__(self, configs, n_bd_nodes=1, batch_size=1, filter=0):
        self.configs = configs
        self.n_bd_nodes = n_bd_nodes
        self.batch_size = batch_size
        self.filter = filter

    def run_mpi(self):
        rankreq = np.empty(1, dtype='i')
        view = 0
        while True:
            # handles requests from smd_0
            comm.Send(np.array([rank], dtype='i'), dest=0)
            info = MPI.Status()
            comm.Probe(source=0, tag=12, status=info)
            count = info.Get_elements(MPI.BYTE)
            view = bytearray(count)
            comm.Recv(view, source=0, tag=12)
            if view.startswith(b'eof'):
                break

            pf = PacketFooter(view=view)
            views = pf.split_packets()

            # build batch of events
            for batch in self.batches(views):
                comm.Recv(rankreq, source=MPI.ANY_SOURCE, tag=13)
                comm.Send(batch, dest=rankreq[0])

            view = 0

        for i in range(self.n_bd_nodes):
            comm.Recv(rankreq, source=MPI.ANY_SOURCE, tag=13)
            comm.Send(bytearray(b'eof'), dest=rankreq[0])

    def batches(self, chunk):
        eb = EventBuilder(chunk, self.configs)
        batch = eb.build(batch_size=self.batch_size, filter=self.filter)
        while eb.nevents:
            yield batch
            batch = eb.build(batch_size=self.batch_size, filter=self.filter)

class BigDataNode(object):
    def __init__(self, smd_configs, dm, smd_node_id=None):
        self.smd_configs = smd_configs
        self.dm = dm
        self.smd_node_id = smd_node_id

    def run_mpi(self):
        while True:
            comm.Send(np.array([rank], dtype='i'), dest=self.smd_node_id, tag=13)
            info = MPI.Status()
            comm.Probe(source=self.smd_node_id, tag=MPI.ANY_TAG, status=info)
            count = info.Get_elements(MPI.BYTE)
            view = bytearray(count)
            comm.Recv(view, source=self.smd_node_id)
            if view.startswith(b'eof'):
                break

            for event in self.events(view):
                yield event

    def events(self, view):
        views = view.split(b'endofevt')
        for event_bytes in views:
            if event_bytes:
                evt = Event().from_bytes(self.smd_configs, event_bytes)
                # get big data
                ofsz = np.asarray([[d.info.offsetAlg.intOffset, d.info.offsetAlg.intDgramSize] \
                        for d in evt])
                bd_evt = self.dm.jump(ofsz[:,0], ofsz[:,1])
                yield bd_evt

def run_node(run, nodetype, nsmds, max_events, batch_size, filter_callback):
    if nodetype == 'smd0':
        Smd0(run.smd_dm.fds, nsmds, max_events=max_events)
    elif nodetype == 'smd':
        bd_node_ids = (np.arange(size)[nsmds+1:] % nsmds) + 1
        smd_node = SmdNode(run.smd_configs, len(bd_node_ids[bd_node_ids==rank]), \
                           batch_size=batch_size, filter=filter_callback)
        smd_node.run_mpi()
    elif nodetype == 'bd':
        smd_node_id = (rank % nsmds) + 1
        bd_node = BigDataNode(run.smd_configs, run.dm, smd_node_id)
        for evt in bd_node.run_mpi():
            yield evt
