from sys import byteorder
import numpy as np
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.eventbuilder_manager import EventBuilderManager
from psana.psexp.event_manager import EventManager
from psana.psexp.packet_footer import PacketFooter
from psana.event import Event
from psana.psexp.step import Step
from psana import dgram
from psana.psexp.psana_mpi import PsanaMPI
from mpi4py import MPI


class UpdateManager(object):
    """ Keeps epics data and their send history. """
    def __init__(self, client_size, n_smds):
        self.n_smds = n_smds
        self.bufs = [bytearray() for i in range(self.n_smds)]
        self.send_history = []
        # Initialize no. of sent bytes to 0 for evtbuilder
        # [[offset_update0, offset_update1, ], [offset_update0, offset_update1, ], ...]
        # [ ---------evtbuilder0------------ ,  ---------evtbuilder1------------ ,
        for i in range(1, client_size):
            self.send_history.append([0]*self.n_smds)

    def extend_buffers(self, views):
        for i, view in enumerate(views):
            self.bufs[i].extend(view)

    def get_buffer(self, client_id):
        """ Returns new epics data (if any) for this client
        then updates the sent record."""
        update_chunk = bytearray()

        if self.n_smds: # do nothing if no epics data found
            indexed_id = client_id - 1 # rank 0 has no send history.
            pf = PacketFooter(self.n_smds)
            for i, buf in enumerate(self.bufs):
                current_buf = self.bufs[i]
                current_offset = self.send_history[indexed_id][i]
                current_buf_size = memoryview(current_buf).shape[0]
                pf.set_size(i, current_buf_size - current_offset)
                update_chunk.extend(current_buf[current_offset:])
                self.send_history[indexed_id][i] = current_buf_size
            update_chunk.extend(pf.footer)
        
        return update_chunk
        


class Smd0(object):
    """ Sends blocks of smds to smd_node
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp to an smd_node.
    """
    def __init__(self, run, psmpi):
        self.smdr_man = SmdReaderManager(run.smd_dm.fds, run.max_events)
        self.run = run
        self.epics_man = UpdateManager(psmpi.smd_size, self.run.ssm.stores['epics'].n_files)
        self.run_mpi(psmpi)

    def run_mpi(self, psmpi):
        rankreq = np.empty(1, dtype='i')

        for (smd_chunk, update_chunk) in self.smdr_man.chunks():
            # Creates a chunk from smd and epics data to send to SmdNode
            # Anatomy of a chunk (pf=packet_footer):
            # [ [smd0][smd1][smd2][pf] ][ [epics0][epics1][epics2][pf] ][ pf ]
            #   ----- smd_chunk ------    ---------epics_chunk------- 
            # -------------------------- chunk ------------------------------
            
            # Read new epics data as available in the queue
            # then send only unseen portion of data to the evtbuilder rank.
            update_pf = PacketFooter(view=update_chunk)
            self.epics_man.extend_buffers(update_pf.split_packets())
            psmpi.smd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            epics_chunk = self.epics_man.get_buffer(rankreq[0])

            pf = PacketFooter(2)
            pf.set_size(0, memoryview(smd_chunk).shape[0])
            pf.set_size(1, memoryview(epics_chunk).shape[0])
            chunk = smd_chunk + epics_chunk + pf.footer

            psmpi.smd_comm.Send(chunk, dest=rankreq[0])

        for i in range(psmpi.n_smd_nodes):
            psmpi.smd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            psmpi.smd_comm.Send(bytearray(), dest=rankreq[0])

class SmdNode(object):
    """Handles both smd_0 and bd_nodes
    Receives blocks of smds from smd_0 then assembles
    offsets and dgramsizes into a numpy array. Sends
    this np array to bd_nodes that are registered to it."""
    def __init__(self, run):
        self.run = run
        self._update_dgram_pos = 0 # bookkeeping for running in scan mode

    def pack(self, *args):
        pf = PacketFooter(len(args))
        batch = bytearray()
        for i, arg in enumerate(args):
            pf.set_size(i, memoryview(arg).shape[0])
            batch += arg
        batch += pf.footer
        return batch

    def run_mpi(self, psmpi):
        rankreq = np.empty(1, dtype='i')
        current_step_pos = 0
        smd_comm = psmpi.smd_comm
        n_bd_nodes = psmpi.bd_comm.Get_size() - 1
        bd_comm = psmpi.bd_comm
        smd_rank = psmpi.smd_rank
        epics_man = UpdateManager(psmpi.bd_size, self.run.ssm.stores['epics'].n_files)

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
            smd_chunk, step_chunk = pf.split_packets()
        
            eb_man = EventBuilderManager(smd_chunk, self.run.configs, \
                    batch_size=self.run.batch_size, filter_fn=self.run.filter_callback, \
                    destination=self.run.destination)
            
            # Unpack step chunk and updates run's epics store and epics_manager  
            step_pf = PacketFooter(view=step_chunk)
            step_views = step_pf.split_packets()
            self.run.ssm.update(step_views)
            epics_man.extend_buffers(step_views)

            # Build batch of events
            step_dgrams = [sd for sd in self.run.ssm.stores['scan'].dgrams()][current_step_pos:]
            n_step_dgrams = len(step_dgrams)
            for i,step_dgram in enumerate(step_dgrams):
                if i < n_step_dgrams - 1:
                    limit_ts = step_dgrams[i + 1].seq.timestamp()
                    current_step_pos += 1
                else:
                    limit_ts = -1
                
                for smd_batch_dict in eb_man.batches(limit_ts=limit_ts):
                    # If single item and dest_rank=0, send to any bigdata nodes.
                    if 0 in smd_batch_dict.keys():
                        smd_batch, _ = smd_batch_dict[0]
                        bd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
                        epics_batch = epics_man.get_buffer(rankreq[0])
                        batch = self.pack(smd_batch, epics_batch) 
                        bd_comm.Send(batch, dest=rankreq[0])

                    # With > 1 dest_rank, start looping until all dest_rank batches
                    # have been sent.
                    else:
                        while smd_batch_dict:
                            bd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)

                            if rankreq[0] in smd_batch_dict:
                                smd_batch, _ = smd_batch_dict[rankreq[0]]
                                epics_batch = epics_man.get_buffer(rankreq[0])
                                batch = self.pack(smd_batch, epics_batch) 
                                bd_comm.Send(batch, dest=rankreq[0])
                                del smd_batch_dict[rankreq[0]] # done sending 
                            else:
                                bd_comm.Send(bytearray(b'wait'), dest=rankreq[0])

        # Done - kill all alive bigdata nodes
        for i in range(n_bd_nodes):
            bd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            bd_comm.Send(bytearray(), dest=rankreq[0])

class BigDataNode(object):
    def __init__(self, run):
        self.evt_man = EventManager(run.configs, run.dm, \
                filter_fn=run.filter_callback)
        self.run = run

    def run_mpi(self, psmpi):
        bd_comm = psmpi.bd_comm
        bd_rank = psmpi.bd_rank
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
            smd_chunk, step_chunk = pf.split_packets()
            
            pfe = PacketFooter(view=step_chunk)
            step_views = pfe.split_packets()
            self.run.ssm.update(step_views)

            if self.run.scan: 
                yield Step(self.run, smd_batch=smd_chunk)
            else:
                for event in self.evt_man.events(smd_chunk):
                    yield event

def run_node(run):
    psmpi = PsanaMPI()
    if psmpi._nodetype == 'smd0':
        Smd0(run, psmpi)
    elif psmpi._nodetype == 'smd':
        smd_node = SmdNode(run)
        smd_node.run_mpi(psmpi)
    elif psmpi._nodetype == 'bd':
        bd_node = BigDataNode(run)
        for evt in bd_node.run_mpi(psmpi):
            yield evt
    elif psmpi._nodetype == 'srv':
        # tell the iterator to do nothing
        return


