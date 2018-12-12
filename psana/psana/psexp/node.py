import os
from sys import byteorder
import numpy as np
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.eventbuilder_manager import EventBuilderManager
from psana.psexp.event_manager import EventManager

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
        self.smdr_man = SmdReaderManager(fds, max_events)
        self.n_smd_nodes = n_smd_nodes

        self.run_mpi()

    def run_mpi(self):
        rankreq = np.empty(1, dtype='i')

        for chunk in self.smdr_man.chunks():
            comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            comm.Send(chunk, dest=rankreq[0], tag=12)

        for i in range(self.n_smd_nodes):
            comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            comm.Send(bytearray(), dest=rankreq[0], tag=12)

class SmdNode(object):
    """Handles both smd_0 and bd_nodes
    Receives blocks of smds from smd_0 then assembles
    offsets and dgramsizes into a numpy array. Sends
    this np array to bd_nodes that are registered to it."""
    def __init__(self, configs, n_bd_nodes=1, batch_size=1, filter=0):
        self.eb_man = EventBuilderManager(configs, batch_size, filter)
        self.n_bd_nodes = n_bd_nodes

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
            if count == 0:
                break

            # build batch of events
            for batch in self.eb_man.batches(view):
                comm.Recv(rankreq, source=MPI.ANY_SOURCE, tag=13)
                comm.Send(batch, dest=rankreq[0])

            view = 0

        for i in range(self.n_bd_nodes):
            comm.Recv(rankreq, source=MPI.ANY_SOURCE, tag=13)
            comm.Send(bytearray(), dest=rankreq[0])

class BigDataNode(object):
    def __init__(self, smd_configs, dm, smd_node_id=None):
        self.evt_man = EventManager(smd_configs, dm)
        self.smd_node_id = smd_node_id

    def run_mpi(self):
        while True:
            comm.Send(np.array([rank], dtype='i'), dest=self.smd_node_id, tag=13)
            info = MPI.Status()
            comm.Probe(source=self.smd_node_id, tag=MPI.ANY_TAG, status=info)
            count = info.Get_elements(MPI.BYTE)
            view = bytearray(count)
            comm.Recv(view, source=self.smd_node_id)
            if count == 0:
                break

            for event in self.evt_man.events(view):
                yield event

def run_node(run, nodetype, nsmds, smd0_threads, max_events, batch_size, filter_callback):
    if nodetype == 'smd0':
        Smd0(run.smd_dm.fds, nsmds, max_events=max_events)
    elif nodetype == 'smd':
        bd_node_ids = (np.arange(size)[nsmds+smd0_threads:] % nsmds) + smd0_threads
        smd_node = SmdNode(run.smd_configs, len(bd_node_ids[bd_node_ids==rank]), \
                           batch_size=batch_size, filter=filter_callback)
        smd_node.run_mpi()
    elif nodetype == 'bd':
        smd_node_id = (rank % nsmds) + smd0_threads
        bd_node = BigDataNode(run.smd_configs, run.dm, smd_node_id)
        for evt in bd_node.run_mpi():
            yield evt
