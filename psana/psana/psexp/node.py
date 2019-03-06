import os
from sys import byteorder
import numpy as np
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.eventbuilder_manager import EventBuilderManager
from psana.psexp.event_manager import EventManager
from psana.psexp.epicsstore import EpicsStore
from psana.psexp.packet_footer import PacketFooter
from psana.event import Event

from mpi4py import MPI
comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()
group = comm.Get_group() # This this the world group

# Setting up group communications
# Ex. PS_SMD_NODES=3 mpirun -n 13
#       1   4   7   10
#   0   2   5   8   11
#       3   6   9   12
#-smd_group-
#       -bd_main_group-
#       color
#       0   0   0   0
#       1   1   1   1
#       2   2   2   2
#       bd_main_rank        bd_rank
#       0   3   6   9       0   1   2   3
#       1   4   7   10      0   1   2   3
#       2   5   8   11      0   1   2   3

PS_SMD_NODES = int(os.environ.get('PS_SMD_NODES', 1))
smd_group = group.Incl(range(PS_SMD_NODES + 1))
bd_main_group = group.Excl([0])

smd_comm = comm.Create(smd_group)
smd_rank = 0
smd_size = 0
if smd_comm != MPI.COMM_NULL:
    smd_rank = smd_comm.Get_rank()
    smd_size = smd_comm.Get_size()

bd_main_comm = comm.Create(bd_main_group)
bd_main_rank = 0
bd_main_size = 0
bd_rank = 0
bd_size = 0
color = 0
nodetype = None
if bd_main_comm != MPI.COMM_NULL:
    bd_main_rank = bd_main_comm.Get_rank()
    bd_main_size = bd_main_comm.Get_size()

    # Split bigdata main comm to PS_SMD_NODES groups
    color = bd_main_rank % PS_SMD_NODES
    bd_comm = bd_main_comm.Split(color, bd_main_rank)
    bd_rank = bd_comm.Get_rank()
    bd_size = bd_comm.Get_size()

    if bd_rank == 0:
        nodetype = 'smd'
    else:
        nodetype = 'bd'

if nodetype is None:
    nodetype = 'smd0' # if no nodetype assigned, I must be smd0

class Smd0(object):
    """ Sends blocks of smds to smd_node
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp to an smd_node.
    """
    def __init__(self, run):
        self.smdr_man = SmdReaderManager(run.smd_dm.fds, run.max_events)
        self.run = run
        self.run_mpi()

    def run_mpi(self):
        rankreq = np.empty(1, dtype='i')

        for smd_chunk in self.smdr_man.chunks():
            # Creates a chunk from smd and epics data to send to SmdNode
            # Anatomy of a chunk (pf=packet_footer):
            # [ [smd0][smd1][smd2][pf] ][ [epics_evt0][epics_evt1][pf] ][ pf ]
            #   ----- smd_chunk ------     --------- epics_chunk -----
            # -------------------------- chunk ------------------------------
            
            # Update EpicsStore and checkout Epics events
            # FIXME: mona do bookeeping to avoid sending duplicates.
            self.run.epics_store.update(self.run.epics_reader.read(), \
                    self.run.epics_reader._config, min_ts=self.smdr_man.min_ts)
            epics_chunk = self.run.epics_store.checkout_by_ts_range(\
                    self.smdr_man.min_ts, self.smdr_man.max_ts, to_bytes=True)

            pf = PacketFooter(2)
            pf.set_size(0, memoryview(smd_chunk).nbytes)
            pf.set_size(1, memoryview(epics_chunk).nbytes)
            chunk = smd_chunk + epics_chunk + pf.footer

            smd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            smd_comm.Send(chunk, dest=rankreq[0])

        for i in range(PS_SMD_NODES):
            smd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            smd_comm.Send(bytearray(), dest=rankreq[0])

class SmdNode(object):
    """Handles both smd_0 and bd_nodes
    Receives blocks of smds from smd_0 then assembles
    offsets and dgramsizes into a numpy array. Sends
    this np array to bd_nodes that are registered to it."""
    def __init__(self, run):
        self.eb_man = EventBuilderManager(run.smd_configs, run.batch_size, run.filter_callback)
        self.n_bd_nodes = bd_comm.Get_size() - 1
        self.run = run

    def run_mpi(self):
        rankreq = np.empty(1, dtype='i')
        while True:
            # handles requests from smd_0
            smd_comm.Send(np.array([smd_rank], dtype='i'), dest=0)
            info = MPI.Status()
            smd_comm.Probe(source=0, status=info)
            count = info.Get_elements(MPI.BYTE)
            chunk = bytearray(count)
            smd_comm.Recv(chunk, source=0)
            if count == 0:
                break
           
            # Unpack the chunk received from Smd0
            pf = PacketFooter(view=chunk)
            smd_chunk, epics_chunk = pf.split_packets()
            
            # Rebuild epics_store (FIXME: with bookkeeping, this will
            # just be an update operation (not rebuild)
            epics_events = self.run.epics_store._from_bytes(self.run.epics_config, epics_chunk)
            self.run.epics_store.force_update(epics_events)

            # build batch of events
            for smd_batch in self.eb_man.batches(smd_chunk):
                
                epics_batch = self.run.epics_store.checkout_by_ts_range(\
                        self.eb_man.min_ts, self.eb_man.max_ts, to_bytes=True)

                pf = PacketFooter(2)
                pf.set_size(0, memoryview(smd_batch).nbytes)
                pf.set_size(1, memoryview(epics_batch).nbytes)
                batch = smd_batch + epics_batch + pf.footer
                
                bd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
                bd_comm.Send(batch, dest=rankreq[0])
                
        for i in range(self.n_bd_nodes):
            bd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            bd_comm.Send(bytearray(), dest=rankreq[0])

class BigDataNode(object):
    def __init__(self, run):
        self.evt_man = EventManager(run.smd_configs, run.dm, \
                filter_fn=run.filter_callback)
        self.run = run

    def run_mpi(self):
        while True:
            bd_comm.Send(np.array([bd_rank], dtype='i'), dest=0)
            info = MPI.Status()
            bd_comm.Probe(source=0, tag=MPI.ANY_TAG, status=info)
            count = info.Get_elements(MPI.BYTE)
            chunk = bytearray(count)
            bd_comm.Recv(chunk, source=0)
            if count == 0:
                break
            
            pf = PacketFooter(view=chunk)
            smd_chunk, epics_chunk = pf.split_packets()
            
            epics_events = self.run.epics_store._from_bytes(self.run.epics_config, epics_chunk)
            self.run.epics_store.force_update(epics_events)

            for event in self.evt_man.events(smd_chunk):
                yield event

def run_node(run):
    if nodetype == 'smd0':
        Smd0(run)
    elif nodetype == 'smd':
        smd_node = SmdNode(run)
        smd_node.run_mpi()
    elif nodetype == 'bd':
        bd_node = BigDataNode(run)
        for evt in bd_node.run_mpi():
            yield evt
