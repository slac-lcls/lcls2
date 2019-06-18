import os
from sys import byteorder
import numpy as np
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.eventbuilder_manager import EventBuilderManager
from psana.psexp.event_manager import EventManager
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

class EpicsManager(object):
    """ Keeps epics data and their send history. """
    def __init__(self, client_size, n_epics):
        self.n_epics = n_epics
        self.epics_bufs = [bytearray() for i in range(self.n_epics)]
        self.epics_send_history = []
        # Initialize no. of sent bytes to 0 for evtbuilder
        # [[offset_epics0, offset_epics1, ], [offset_epics0, offset_epics1, ], ...]
        # [ --------evtbuilder0------------, --------evtbuilder1------------ ,
        for i in range(1, client_size):
            self.epics_send_history.append([0]*self.n_epics)

    def extend_buffers(self, views):
        for i, view in enumerate(views):
            self.epics_bufs[i].extend(view)

    def get_buffer(self, client_id):
        """ Returns new epics data (if any) for this client
        then updates the sent record."""
        epics_chunk = bytearray()

        if self.n_epics: # do nothing if no epics data found
            indexed_id = client_id - 1 # rank 0 has no send history.
            pf = PacketFooter(self.n_epics)
            for i, epics_buf in enumerate(self.epics_bufs):
                current_buf = self.epics_bufs[i]
                current_offset = self.epics_send_history[indexed_id][i]
                current_buf_size = memoryview(current_buf).shape[0]
                pf.set_size(i, current_buf_size - current_offset)
                epics_chunk.extend(current_buf[current_offset:])
                self.epics_send_history[indexed_id][i] = current_buf_size
            epics_chunk.extend(pf.footer)
        
        return epics_chunk
        


class Smd0(object):
    """ Sends blocks of smds to smd_node
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp to an smd_node.
    """
    def __init__(self, run):
        self.smdr_man = SmdReaderManager(run.smd_dm.fds, run.max_events)
        self.run = run
        self.epics_man = EpicsManager(smd_size, self.run.epics_store.n_files)
        self.run_mpi()

    def run_mpi(self):
        rankreq = np.empty(1, dtype='i')

        for smd_chunk in self.smdr_man.chunks():
            # Creates a chunk from smd and epics data to send to SmdNode
            # Anatomy of a chunk (pf=packet_footer):
            # [ [smd0][smd1][smd2][pf] ][ [epics0][epics1][epics2][pf] ][ pf ]
            #   ----- smd_chunk ------    ---------epics_chunk------- 
            # -------------------------- chunk ------------------------------
            
            # Read new epics data as available in the queue
            # then send only unseen portion of data to the evtbuilder rank.
            self.epics_man.extend_buffers(self.run.epics_reader.read())
            smd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            epics_chunk = self.epics_man.get_buffer(rankreq[0])

            pf = PacketFooter(2)
            pf.set_size(0, memoryview(smd_chunk).shape[0])
            pf.set_size(1, memoryview(epics_chunk).shape[0])
            chunk = smd_chunk + epics_chunk + pf.footer

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
        self.eb_man = EventBuilderManager(run.smd_configs, batch_size=run.batch_size, filter_fn=run.filter_callback, destination=run.destination)
        self.n_bd_nodes = bd_comm.Get_size() - 1
        self.run = run
        self.epics_man = EpicsManager(bd_size, self.run.epics_store.n_files)

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
            
            # Unpack epics chunk and updates run's epics_store and epics_manager  
            pfe = PacketFooter(view=epics_chunk)
            epics_views = pfe.split_packets()
            self.run.epics_store.update(epics_views)
            self.epics_man.extend_buffers(epics_views)
            
            # Build batch of events
            for smd_batch_dict in self.eb_man.batches(smd_chunk):

                # If single item and dest_rank=0, send to any bigdata nodes.
                if 0 in smd_batch_dict.keys():
                    smd_batch, _ = smd_batch_dict[0]

                    bd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)

                    epics_batch = self.epics_man.get_buffer(rankreq[0])

                    _pf = PacketFooter(2)
                    _pf.set_size(0, memoryview(smd_batch).shape[0])
                    _pf.set_size(1, memoryview(epics_batch).shape[0])
                    batch = smd_batch + epics_batch + _pf.footer
                    
                    bd_comm.Send(batch, dest=rankreq[0])

                # With > 1 dest_rank, start looping until all dest_rank batches
                # have been sent.
                else:
                    while smd_batch_dict:
                        bd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)

                        if rankreq[0] in smd_batch_dict:
                            smd_batch, _ = smd_batch_dict[rankreq[0]]

                            epics_batch = self.epics_man.get_buffer(rankreq[0])

                            _pf = PacketFooter(2)
                            _pf.set_size(0, memoryview(smd_batch).shape[0])
                            _pf.set_size(1, memoryview(epics_batch).shape[0])
                            batch = smd_batch + epics_batch + _pf.footer
                            
                            bd_comm.Send(batch, dest=rankreq[0])

                            del smd_batch_dict[rankreq[0]] # done sending 
                        else:
                            bd_comm.Send(bytearray(b'wait'), dest=rankreq[0])

        # Done - kill all alive bigdata nodes
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

            if chunk == bytearray(b'wait'):
                continue
            
            pf = PacketFooter(view=chunk)
            smd_chunk, epics_chunk = pf.split_packets()
            
            pfe = PacketFooter(view=epics_chunk)
            epics_views = pfe.split_packets()
            self.run.epics_store.update(epics_views)

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
