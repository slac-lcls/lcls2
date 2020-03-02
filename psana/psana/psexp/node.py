from sys import byteorder
import numpy as np
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.eventbuilder_manager import EventBuilderManager
from psana.psexp.packet_footer import PacketFooter
from psana.psexp.step import Step
from psana.psexp.events import Events
from psana.psexp.event_manager import TransitionId
from psana.dgram import Dgram
import os
from mpi4py import MPI

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

class Communicators(object):
    # Reserved nodes are for external applications (e.g. smalldata
    # servers).  These nodes will do nothing for event/step iterators
    # (see run_node method below).  The "psana_group" consists of
    # all non-reserved ranks and is used for smd0/smd/bd cores.
    comm = None
    world_rank  = 0
    world_size  = 1
    world_group = None
    smd_comm = None
    smd_rank = 0
    smd_size = 0
    bd_main_comm = None
    bd_main_rank = 0
    bd_main_size = 0
    bd_rank = 0
    bd_size = 0
    color = 0
    _nodetype = None
    bd_comm = None

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.world_rank  = self.comm.Get_rank()
        self.world_size  = self.comm.Get_size()
        self.world_group = self.comm.Get_group()

        PS_SRV_NODES = int(os.environ.get('PS_SRV_NODES', 0))
        PS_SMD_NODES = int(os.environ.get('PS_SMD_NODES', 1))
        self.n_smd_nodes = PS_SMD_NODES

        if (self.world_size - PS_SRV_NODES) < 3:
            raise Exception('Too few MPI cores to run parallel psana.'
                            '\nYou need 3 + #PS_SRV_NODES (currently: %d)'
                            '\n\tCurrent cores:  %d'
                            '\n\tRequired:       %d' 
                            ''% (PS_SRV_NODES,self.world_size, PS_SRV_NODES+3))

        self.psana_group    = self.world_group.Excl(range(self.world_size-PS_SRV_NODES, self.world_size))
        self.psana_comm     = self.comm.Create(self.psana_group)

        self.smd_group      = self.psana_group.Incl(range(PS_SMD_NODES + 1))
        self.bd_main_group  = self.psana_group.Excl([0])
        self._bd_only_group = MPI.Group.Difference(self.bd_main_group,self.smd_group)
        self._srv_group     = MPI.Group.Difference(self.world_group,self.psana_group)

        self.smd_comm = self.comm.Create(self.smd_group)
        self.bd_main_comm = self.comm.Create(self.bd_main_group)
        
        if self.smd_comm != MPI.COMM_NULL:
            self.smd_rank = self.smd_comm.Get_rank()
            self.smd_size = self.smd_comm.Get_size()

        if self.bd_main_comm != MPI.COMM_NULL:
            self.bd_main_rank = self.bd_main_comm.Get_rank()
            self.bd_main_size = self.bd_main_comm.Get_size()

            # Split bigdata main comm to PS_SMD_NODES groups
            color = self.bd_main_rank % PS_SMD_NODES
            self.bd_comm = self.bd_main_comm.Split(color, self.bd_main_rank)
            self.bd_rank = self.bd_comm.Get_rank()
            self.bd_size = self.bd_comm.Get_size()

            if self.bd_rank == 0:
                self._nodetype = 'smd'
            else:
                self._nodetype = 'bd'

        if self.world_rank==0:
            self._nodetype = 'smd0'
        elif self.world_rank>=self.psana_group.Get_size():
            self._nodetype = 'srv'
    
    def bd_group(self):
        return self._bd_only_group

    def srv_group(self):
        return self._srv_group

    def node_type(self):
        return self._nodetype


class StepHistory(object):
    """ Keeps step data and their send history. """
    def __init__(self, client_size, n_smds):
        self.n_smds = n_smds
        self.bufs = []
        for i in range(self.n_smds):
            self.bufs.append(bytearray())
        self.send_history = []
        # Initialize no. of sent bytes to 0 for clients
        # [[offset_update0, offset_update1, ], [offset_update0, offset_update1, ], ...]
        # [ -----------client 0------------- ,  ----------- client 1------------ ,
        for i in range(1, client_size):
            self.send_history.append(np.zeros(self.n_smds, dtype=np.int))

    def extend_buffers(self, views, client_id, as_event=False):
        idx = client_id - 1 # rank 0 has no send history.
        # Views is either list of smdchunks or events
        if not as_event:
            # For Smd0
            for i_smd, view in enumerate(views):
                self.bufs[i_smd].extend(view)
                self.send_history[idx][i_smd] += view.nbytes
        else:
            # For EventBuilder
            for i_evt, evt_bytes in enumerate(views):
                pf = PacketFooter(view=evt_bytes)
                assert pf.n_packets == self.n_smds
                for i_smd, dg_bytes in enumerate(pf.split_packets()):
                    self.bufs[i_smd].extend(dg_bytes)
                    self.send_history[idx][i_smd] += dg_bytes.nbytes
    
    def update_history(self, views, client_id):
        indexed_id = client_id - 1 # rank 0 has no send history.
        for i, view in enumerate(views):
            self.send_history[indexed_id][i] += view.nbytes

    def get_buffer(self, client_id):
        """ Returns new step data (if any) for this client
        then updates the sent record."""
        views = []
        
        if self.n_smds: # do nothing if no step data found
            indexed_id = client_id - 1 # rank 0 has no send history.
            views = [bytearray() for i in range(self.n_smds)]
            for i, buf in enumerate(self.bufs):
                current_buf = self.bufs[i]
                current_offset = self.send_history[indexed_id][i]
                current_buf_size = memoryview(current_buf).nbytes
                if current_offset < current_buf_size:
                    views[i].extend(current_buf[current_offset:])
                    self.send_history[indexed_id][i] = current_buf_size
        
        return views

def repack_for_eb(smd_chunk, step_views, configs):
    """ Smd0 uses this to prepend missing step views
    to the smd_chunk (just data with the same limit timestamp from all
    smd files - not event-built yet). 
    """
    if step_views:
        smd_chunk_pf = PacketFooter(view=smd_chunk)
        new_chunk_pf = PacketFooter(n_packets=smd_chunk_pf.n_packets)
        new_chunk = bytearray()
        for i, (smd_view, step_view) in enumerate(zip(smd_chunk_pf.split_packets(), step_views)):
            new_chunk.extend(step_view+bytearray(smd_view))
            new_chunk_pf.set_size(i, memoryview(step_view).nbytes + smd_view.nbytes)
        new_chunk.extend(new_chunk_pf.footer)
        return new_chunk
    else:
        return smd_chunk 


def repack_for_bd(smd_batch, step_views, configs, client=-1):
    """ EventBuilder Node uses this to prepend missing step views 
    to the smd_batch. Unlike repack_for_eb (used by Smd0), this output 
    chunk contains list of pre-built events."""
    if step_views:
        batch_pf = PacketFooter(view=smd_batch)
        
        # Create bytearray containing a list of events from step_views 
        steps = bytearray()
        n_smds = len(step_views)
        offsets = np.zeros(n_smds, dtype=np.int) 
        n_steps = 0
        step_sizes = []
        while offsets[0] < memoryview(step_views[0]).nbytes:
            step_pf = PacketFooter(n_packets=n_smds)
            step_size = 0
            for i, (config, view) in enumerate(zip(configs, step_views)):
                d = Dgram(config=config, view=view, offset=offsets[i])
                steps.extend(d)
                offsets[i] += d._size
                step_size += d._size
                step_pf.set_size(i, d._size)
            
            steps.extend(step_pf.footer)
            step_sizes.append(step_size + memoryview(step_pf.footer).nbytes)
            n_steps += 1
        
        # Create new batch with total_events = smd_batch_events + step_events 
        new_batch_pf = PacketFooter(n_packets = batch_pf.n_packets + n_steps)
        for i in range(n_steps):
            new_batch_pf.set_size(i, step_sizes[i])
        
        for i in range(n_steps, new_batch_pf.n_packets):
            new_batch_pf.set_size(i, batch_pf.get_size(i-n_steps))

        new_batch = bytearray()
        new_batch.extend(steps)
        new_batch.extend(smd_batch[:memoryview(smd_batch).nbytes-memoryview(batch_pf.footer).nbytes])
        new_batch.extend(new_batch_pf.footer)
        return new_batch
    else:
        return smd_batch

class Smd0(object):
    """ Sends blocks of smds to smd_node
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp to an smd_node.
    """
    def __init__(self, run):
        self.smdr_man = SmdReaderManager(run)
        self.run = run
        self.step_hist = StepHistory(self.run.comms.smd_size, len(self.run.configs))
        self.run_mpi()

    def run_mpi(self):
        rankreq = np.empty(1, dtype='i')

        for (smd_chunk, step_chunk) in self.smdr_man.chunks():
            # Creates a chunk from smd and epics data to send to SmdNode
            # Anatomy of a chunk (pf=packet_footer):
            # [ [smd0][smd1][smd2][pf] ][ [epics0][epics1][epics2][pf] ][ pf ]
            #   ----- smd_chunk ------    ---------epics_chunk------- 
            # -------------------------- chunk ------------------------------
            
            # Read new epics data as available in the queue
            # then send only unseen portion of data to the evtbuilder rank.
            if not smd_chunk: break

            self.run.comms.smd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            
            # Check missing steps for the current client
            missing_step_views = self.step_hist.get_buffer(rankreq[0])

            # Update step buffers (after getting the missing steps
            step_pf = PacketFooter(view=step_chunk)
            step_views = step_pf.split_packets()
            self.step_hist.extend_buffers(step_views, rankreq[0])

            smd_extended = repack_for_eb(smd_chunk, missing_step_views, self.run.configs)
            
            self.run.comms.smd_comm.Send(smd_extended, dest=rankreq[0])
        
        for i in range(self.run.comms.n_smd_nodes):
            self.run.comms.smd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            self.run.comms.smd_comm.Send(bytearray(), dest=rankreq[0])

class SmdNode(object):
    """Handles both smd_0 and bd_nodes
    Receives blocks of smds from smd_0 then assembles
    offsets and dgramsizes into a numpy array. Sends
    this np array to bd_nodes that are registered to it."""
    def __init__(self, run):
        self.run = run
        self.step_hist = StepHistory(self.run.comms.bd_size, len(self.run.configs))
        self.waiting_bds = []

    def pack(self, *args):
        pf = PacketFooter(len(args))
        batch = bytearray()
        for i, arg in enumerate(args):
            pf.set_size(i, memoryview(arg).shape[0])
            batch += arg
        batch += pf.footer
        return batch

    def _send_to_dest(self, dest_rank, smd_batch_dict, step_batch_dict, eb_man):
        bd_comm = self.run.comms.bd_comm
        smd_batch, _ = smd_batch_dict[dest_rank]
        missing_step_views = self.step_hist.get_buffer(dest_rank)
        batch = repack_for_bd(smd_batch, missing_step_views, self.run.configs, client=dest_rank)
        bd_comm.Send(batch, dest=dest_rank)
        del smd_batch_dict[dest_rank] # done sending
        
        step_batch, _ = step_batch_dict[dest_rank]
        if eb_man.eb.nsteps > 0 and memoryview(step_batch).nbytes > 0:  
            step_pf = PacketFooter(view=step_batch)
            self.step_hist.extend_buffers(step_pf.split_packets(), dest_rank, as_event=True)
        del step_batch_dict[dest_rank] # done adding

    def run_mpi(self):
        rankreq = np.empty(1, dtype='i')
        smd_comm   = self.run.comms.smd_comm
        n_bd_nodes = self.run.comms.bd_comm.Get_size() - 1
        bd_comm    = self.run.comms.bd_comm
        smd_rank   = self.run.comms.smd_rank
        
        while True:
            smd_comm.Send(np.array([smd_rank], dtype='i'), dest=0)
            info = MPI.Status()
            smd_comm.Probe(source=0, status=info)
            count = info.Get_elements(MPI.BYTE)
            smd_chunk = bytearray(count)
            smd_comm.Recv(smd_chunk, source=0)
            if not smd_chunk:
                break
           
            eb_man = EventBuilderManager(smd_chunk, self.run) 
        
            # Build batch of events
            for smd_batch_dict, step_batch_dict  in eb_man.batches():
                
                # If single item and dest_rank=0, send to any bigdata nodes.
                if 0 in smd_batch_dict.keys():
                    smd_batch, _ = smd_batch_dict[0]
                    step_batch, _ = step_batch_dict[0]
                    bd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
                    
                    missing_step_views = self.step_hist.get_buffer(rankreq[0])
                    batch = repack_for_bd(smd_batch, missing_step_views, self.run.configs, client=rankreq[0])
                    bd_comm.Send(batch, dest=rankreq[0])
                    
                    if eb_man.eb.nsteps > 0 and memoryview(step_batch).nbytes > 0:  
                        step_pf = PacketFooter(view=step_batch)
                        self.step_hist.extend_buffers(step_pf.split_packets(), rankreq[0], as_event=True)
                    
                          
                # With > 1 dest_rank, start looping until all dest_rank batches
                # have been sent.
                else:
                    while smd_batch_dict:
                        sent = False
                        if self.waiting_bds: # Check first if there are bd nodes waiting
                            copied_waiting_bds = self.waiting_bds[:]
                            for dest_rank in copied_waiting_bds:
                                if dest_rank in smd_batch_dict:
                                    self._send_to_dest(dest_rank, smd_batch_dict, step_batch_dict, eb_man)
                                    self.waiting_bds.remove(dest_rank)
                                    sent = True
                        
                        if not sent:
                            bd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
                            dest_rank = rankreq[0]
                            if dest_rank in smd_batch_dict:
                                self._send_to_dest(dest_rank, smd_batch_dict, step_batch_dict, eb_man)
                            else:
                                self.waiting_bds.append(dest_rank)
                        

        # Done 
        # - kill idling nodes
        for dest_rank in self.waiting_bds:
            bd_comm.Send(bytearray(), dest=dest_rank)
        
        # - kill all other nodes
        for i in range(n_bd_nodes-len(self.waiting_bds)):
            bd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            bd_comm.Send(bytearray(), dest=rankreq[0])

class BigDataNode(object):
    def __init__(self, run):
        self.run = run
        self.step_max_ts = 0

    def run_mpi(self):

        def get_smd():
            bd_comm = self.run.comms.bd_comm
            bd_rank = self.run.comms.bd_rank
            bd_comm.Send(np.array([bd_rank], dtype='i'), dest=0)
            info = MPI.Status()
            bd_comm.Probe(source=0, tag=MPI.ANY_TAG, status=info)
            count = info.Get_elements(MPI.BYTE)
            chunk = bytearray(count)
            bd_comm.Recv(chunk, source=0)
            return chunk
        
        events = Events(self.run, get_smd=get_smd)
        if self.run.scan:
            for evt in events:
                if evt.service() == TransitionId.BeginStep:
                    if evt.timestamp > self.step_max_ts:
                        self.step_max_ts = evt.timestamp
                        yield Step(evt, events)

        else:
            for evt in events:
                if evt.service() == TransitionId.L1Accept:
                    yield evt
