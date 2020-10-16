from sys import byteorder
import numpy as np
from psana.psexp import *
from psana.dgram import Dgram
import os
from mpi4py import MPI
import logging
import time, pickle

s_eb_wait_smd0 = PrometheusManager.get_metric('psana_eb_wait_smd0')
s_bd_wait_eb = PrometheusManager.get_metric('psana_bd_wait_eb')

# Setting up group communications
# Ex. PS_EB_NODES=3 mpirun -n 13
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
        PS_EB_NODES = int(os.environ.get('PS_EB_NODES', 1))
        self.n_eb_nodes = PS_EB_NODES

        if (self.world_size - PS_SRV_NODES) < 3:
            raise Exception('Too few MPI cores to run parallel psana.'
                            '\nYou need 3 + #PS_SRV_NODES (currently: %d)'
                            '\n\tCurrent cores:  %d'
                            '\n\tRequired:       %d' 
                            ''% (PS_SRV_NODES,self.world_size, PS_SRV_NODES+3))

        self.psana_group    = self.world_group.Excl(range(self.world_size-PS_SRV_NODES, self.world_size))
        self.psana_comm     = self.comm.Create(self.psana_group)

        self.smd_group      = self.psana_group.Incl(range(PS_EB_NODES + 1))
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

            # Split bigdata main comm to PS_EB_NODES groups
            color = self.bd_main_rank % PS_EB_NODES
            self.bd_comm = self.bd_main_comm.Split(color, self.bd_main_rank)
            self.bd_rank = self.bd_comm.Get_rank()
            self.bd_size = self.bd_comm.Get_size()

            if self.bd_rank == 0:
                self._nodetype = 'eb'
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
    def __init__(self, client_size, n_bufs):
        self.n_bufs = n_bufs
        self.bufs = []
        for i in range(self.n_bufs):
            self.bufs.append(bytearray())
        self.send_history = []
        # Initialize no. of sent bytes to 0 for clients
        # [[offset_update0, offset_update1, ], [offset_update0, offset_update1, ], ...]
        # [ -----------client 0------------- ,  ----------- client 1------------ ,
        for i in range(1, client_size):
            self.send_history.append(np.zeros(self.n_bufs, dtype=np.int))


    def extend_buffers(self, views, client_id, as_event=False, as_calib=False):
        idx = client_id - 1 # rank 0 has no send history.
        # Views is either list of smdchunks or events
        if not as_event:
            # For Smd0
            for i_smd, view in enumerate(views):
                self.bufs[i_smd].extend(view)
                if as_calib:
                    logging.debug(f"node.py: extend_buffers[{i_smd}] client={client_id} self.bufs[{i_smd}]={memoryview(self.bufs[i_smd]).nbytes}")
                if not as_calib:
                    self.send_history[idx][i_smd] += memoryview(view).nbytes
        else:
            # For EventBuilder
            for i_evt, evt_bytes in enumerate(views):
                pf = PacketFooter(view=evt_bytes)
                assert pf.n_packets == self.n_bufs
                for i_smd, dg_bytes in enumerate(pf.split_packets()):
                    self.bufs[i_smd].extend(dg_bytes)
                    if not as_calib:
                        self.send_history[idx][i_smd] += dg_bytes.nbytes
    
    def get_buffer(self, client_id):
        """ Returns new step data (if any) for this client
        then updates the sent record."""
        views = []
        
        if self.n_bufs: # do nothing if no step data found
            indexed_id = client_id - 1 # rank 0 has no send history.
            views = [bytearray() for i in range(self.n_bufs)]
            for i, buf in enumerate(self.bufs):
                current_buf = self.bufs[i]
                current_offset = self.send_history[indexed_id][i]
                current_buf_size = memoryview(current_buf).nbytes
                if current_offset < current_buf_size:
                    views[i].extend(current_buf[current_offset:])
                    self.send_history[indexed_id][i] = current_buf_size
        
        return views


def repack_for_eb(smd_chunk, step_views, configs, calibconst_pkt):
    """ Smd0 uses this to prepend missing step views
    to the smd_chunk (just data with the same limit timestamp from all
    smd files - not event-built yet). 
    """
    smd_extended = smd_chunk
    if step_views:
        smd_chunk_pf = PacketFooter(view=smd_chunk)
        smd_extended_pf = PacketFooter(n_packets=smd_chunk_pf.n_packets)
        smd_extended = bytearray()
        for i, (smd_view, step_view) in enumerate(zip(smd_chunk_pf.split_packets(), step_views)):
            smd_extended.extend(step_view+bytearray(smd_view))
            smd_extended_pf.set_size(i, memoryview(step_view).nbytes + smd_view.nbytes)
        smd_extended.extend(smd_extended_pf.footer)
    
    # add calibconst packet
    new_chunk_pf = PacketFooter(2)
    new_chunk_pf.set_size(0, memoryview(smd_extended).nbytes)
    smd_extended.extend(calibconst_pkt)
    new_chunk_pf.set_size(1, memoryview(calibconst_pkt).nbytes)
    logging.debug(f"node.py: repack_for_eb smd_chunk={memoryview(smd_chunk).nbytes} n_step_views={len(step_views)} calibconst_pkt={memoryview(calibconst_pkt).nbytes}")
    smd_extended.extend(new_chunk_pf.footer)
    
    return smd_extended



def repack_for_bd(smd_batch, step_views, configs, calibconst_pkt, client):
    """ EventBuilder Node uses this to prepend missing step views 
    to the smd_batch. Unlike repack_for_eb (used by Smd0), this output 
    chunk contains list of pre-built events."""
    extended_batch = smd_batch
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
        
        # Create extended batch with total_events = smd_batch_events + step_events 
        extended_batch_pf = PacketFooter(n_packets = batch_pf.n_packets + n_steps)
        for i in range(n_steps):
            extended_batch_pf.set_size(i, step_sizes[i])
        
        for i in range(n_steps, extended_batch_pf.n_packets):
            extended_batch_pf.set_size(i, batch_pf.get_size(i-n_steps))

        extended_batch = bytearray()
        extended_batch.extend(steps)
        extended_batch.extend(smd_batch[:memoryview(smd_batch).nbytes-memoryview(batch_pf.footer).nbytes])
        extended_batch.extend(extended_batch_pf.footer)

    # add calibconst packet
    new_chunk_pf = PacketFooter(2)
    new_chunk_pf.set_size(0, memoryview(extended_batch).nbytes)
    logging.debug(f"extended_batch={memoryview(extended_batch).nbytes}") 
    logging.debug(f"calibconst_pkt={memoryview(calibconst_pkt).nbytes}")
    extended_batch.extend(calibconst_pkt)
    new_chunk_pf.set_size(1, memoryview(calibconst_pkt).nbytes)
    extended_batch.extend(new_chunk_pf.footer)
    
    logging.debug(f"node.py: repack_for_bd {client} smd_batch={memoryview(smd_batch).nbytes} n_step_views={len(step_views)} calibconst_pkt={memoryview(calibconst_pkt).nbytes} new_chunk={memoryview(extended_batch).nbytes}")
    return extended_batch


class Smd0(object):
    """ Sends blocks of smds to smd_node
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp to an smd_node.
    """
    def __init__(self, comms, configs, smdr_man):
        self.comms     = comms  
        self.configs   = configs
        self.smdr_man  = smdr_man
        self.step_hist = StepHistory(self.comms.smd_size, self.smdr_man.n_files)
        self.calib_hist= StepHistory(self.comms.smd_size, 1)
        self.c_sent    = self.smdr_man.dsparms.prom_man.get_metric('psana_smd0_sent')
        self.run_mpi()


    def run_mpi(self):
        rankreq = np.empty(1, dtype='i')
        for (smd_chunk, step_chunk, calibconst_pkt) in self.smdr_man.chunks():
            # Creates a chunk from smd and epics data to send to EventBuilder node 
            # Anatomy of a chunk (pf=packet_footer):
            # [ [smd0][smd1][smd2][pf] ][ [epics0][epics1][epics2][pf] ][ pf ]
            #   ----- smd_chunk ------    ---------epics_chunk------- 
            # -------------------------- chunk ------------------------------
            
            # Read new epics data as available in the queue
            # then send only unseen portion of data to the evtbuilder rank.
            if not smd_chunk: break
            
            st_req = time.time()
            self.comms.smd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            en_req = time.time()
            
            # Check missing steps for the current client
            missing_step_views = self.step_hist.get_buffer(rankreq[0])
            self.calib_hist.extend_buffers([calibconst_pkt], rankreq[0], as_calib=True)
            missing_calib_view = self.calib_hist.get_buffer(rankreq[0])[0]

            # Update step buffers after getting the missing steps
            step_pf = PacketFooter(view=step_chunk)
            step_views = step_pf.split_packets()
            self.step_hist.extend_buffers(step_views, rankreq[0])

            smd_extended = repack_for_eb(smd_chunk, missing_step_views, self.configs, missing_calib_view)
            
            self.comms.smd_comm.Send(smd_extended, dest=rankreq[0])
        
            # sending data to prometheus
            self.c_sent.labels('evts', rankreq[0]).inc(self.smdr_man.got_events)
            self.c_sent.labels('batches', rankreq[0]).inc()
            self.c_sent.labels('MB', rankreq[0]).inc(memoryview(smd_extended).nbytes/1e6)
            self.c_sent.labels('seconds', rankreq[0]).inc(en_req - st_req)
            logging.debug(f'node.py: Smd0 sent {self.smdr_man.got_events} events to {rankreq[0]} (waiting for this rank took {en_req-st_req:.5f} seconds)')
        
        for i in range(self.comms.n_eb_nodes):
            self.comms.smd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            self.comms.smd_comm.Send(bytearray(), dest=rankreq[0])


class EventBuilderNode(object):
    """Handles both smd_0 and bd_nodes
    Receives blocks of smds from smd_0 then assembles
    offsets and dgramsizes into a numpy array. Sends
    this np array to bd_nodes that are registered to it."""
    def __init__(self, comms, configs, dsparms):
        self.comms      = comms
        self.configs    = configs
        self.dsparms    = dsparms
        self.step_hist  = StepHistory(self.comms.bd_size, len(self.configs))
        self.calib_hist = StepHistory(self.comms.bd_size, 1)
        self.waiting_bds= []
        self.c_sent     = self.dsparms.prom_man.get_metric('psana_eb_sent')


    def pack(self, *args):
        pf = PacketFooter(len(args))
        batch = bytearray()
        for i, arg in enumerate(args):
            pf.set_size(i, memoryview(arg).shape[0])
            batch += arg
        batch += pf.footer
        return batch


    def _send_to_dest(self, dest_rank, smd_batch_dict, step_batch_dict, eb_man, calibconst_pkt):
        bd_comm = self.comms.bd_comm
        smd_batch, _ = smd_batch_dict[dest_rank]
        
        missing_step_views = self.step_hist.get_buffer(dest_rank)
        missing_calib_view = self.calib_hist.get_buffer(dest_rank)[0]
        
        batch = repack_for_bd(smd_batch, missing_step_views, self.configs, missing_calib_view, dest_rank)
        bd_comm.Send(batch, dest=dest_rank)
        del smd_batch_dict[dest_rank] # done sending
        
        step_batch, _ = step_batch_dict[dest_rank]
        if eb_man.eb.nsteps > 0 and memoryview(step_batch).nbytes > 0:  
            step_pf = PacketFooter(view=step_batch)
            self.step_hist.extend_buffers(step_pf.split_packets(), dest_rank, as_event=True)
        del step_batch_dict[dest_rank] # done adding

    def _request_rank(self, rankreq):
        st_req = time.time()
        self.comms.bd_comm.Recv(rankreq, source=MPI.ANY_SOURCE)
        en_req = time.time()
        self.c_sent.labels('seconds',rankreq[0]).inc(en_req-st_req)
        logging.debug("node.py: EventBuilder %d got BigData %d (request took %.5f seconds)"%(self.comms.smd_rank, rankreq[0], (en_req-st_req)))

    @s_eb_wait_smd0.time()
    def _request_data(self, smd_comm):
        smd_comm.Send(np.array([self.comms.smd_rank], dtype='i'), dest=0)
        info = MPI.Status()
        smd_comm.Probe(source=0, status=info)
        count = info.Get_elements(MPI.BYTE)
        smd_chunk = bytearray(count)
        smd_comm.Recv(smd_chunk, source=0)
        logging.debug(f"node.py: EventBuilder {self.comms.smd_rank} received {count/1e6:.5f} MB from Smd0")
        return smd_chunk

    def run_mpi(self):
        rankreq = np.empty(1, dtype='i')
        smd_comm   = self.comms.smd_comm
        n_bd_nodes = self.comms.bd_comm.Get_size() - 1
        bd_comm    = self.comms.bd_comm
        smd_rank   = self.comms.smd_rank
        
        while True:
            smd_extended = self._request_data(smd_comm)
            if not smd_extended:
                break
           
            # Unpack smd_chunk for calibconst_pkt
            smd_extended_pf = PacketFooter(view=smd_extended)
            smd_chunk, calibconst_pkt = smd_extended_pf.split_packets()
            eb_man = EventBuilderManager(smd_chunk, self.configs, self.dsparms) 
            self.calib_hist.extend_buffers([calibconst_pkt], 0, as_calib=True) # no need to update send_history
        
            # Build batch of events
            for smd_batch_dict, step_batch_dict  in eb_man.batches():
                
                # If single item and dest_rank=0, send to any bigdata nodes.
                if 0 in smd_batch_dict.keys():
                    smd_batch, _ = smd_batch_dict[0]
                    step_batch, _ = step_batch_dict[0]
                    self._request_rank(rankreq)
                    
                    missing_step_views = self.step_hist.get_buffer(rankreq[0])
                    
                    missing_calib_view = self.calib_hist.get_buffer(rankreq[0])[0]
                    logging.debug(f"node.py: run_mpi extend_buffers for {rankreq[0]} missing_calib_view={memoryview(missing_calib_view).nbytes}")
                    
                    batch = repack_for_bd(smd_batch, missing_step_views, self.configs, missing_calib_view, rankreq[0])
                    bd_comm.Send(batch, dest=rankreq[0])
                    
                    # sending data to prometheus
                    logging.debug('node.py: EventBuilder sent %d events (%.5f MB) to rank %d'%(eb_man.eb.nevents, memoryview(batch).nbytes/1e6, rankreq[0]))
                    self.c_sent.labels('evts', rankreq[0]).inc(eb_man.eb.nevents)
                    self.c_sent.labels('batches', rankreq[0]).inc()
                    self.c_sent.labels('MB', rankreq[0]).inc(memoryview(batch).nbytes/1e6)
                    
                    if eb_man.eb.nsteps > 0 and memoryview(step_batch).nbytes > 0:  
                        step_pf = PacketFooter(view=step_batch)
                        self.step_hist.extend_buffers(step_pf.split_packets(), rankreq[0], as_event=True)
                    
                          
                # With > 1 dest_rank, start looping until all dest_rank batches
                # have been sent.
                else:
                    # Check if destinations are valid 
                    destinations = np.asarray(list(smd_batch_dict.keys()))
                    if any(destinations > n_bd_nodes):
                        print(f"Found invalid destination ({destinations}). Must be <= {n_bd_nodes} (#big data nodes)")
                        break

                    while smd_batch_dict:
                        sent = False
                        if self.waiting_bds: # Check first if there are bd nodes waiting
                            copied_waiting_bds = self.waiting_bds[:]
                            for dest_rank in copied_waiting_bds:
                                if dest_rank in smd_batch_dict:
                                    self._send_to_dest(dest_rank, smd_batch_dict, step_batch_dict, eb_man, calibconst_pkt)
                                    self.waiting_bds.remove(dest_rank)
                                    sent = True
                        
                        if not sent:
                            self._request_rank(rankreq)
                            dest_rank = rankreq[0]
                            if dest_rank in smd_batch_dict:
                                self._send_to_dest(dest_rank, smd_batch_dict, step_batch_dict, eb_man, calibconst_pkt)
                            else:
                                self.waiting_bds.append(dest_rank)
                


        # Done 
        # - kill idling nodes
        for dest_rank in self.waiting_bds:
            bd_comm.Send(bytearray(), dest=dest_rank)
        
        # - kill all other nodes
        for i in range(n_bd_nodes-len(self.waiting_bds)):
            self._request_rank(rankreq)
            bd_comm.Send(bytearray(), dest=rankreq[0])


class BigDataNode(object):
    def __init__(self, comms, configs, dsparms, dm):
        self.comms      = comms
        self.configs    = configs
        self.dsparms    = dsparms
        self.dm         = dm
        self.calib_store= []

    def run_mpi(self):
        @s_bd_wait_eb.time()
        def get_smd():
            bd_comm = self.comms.bd_comm
            bd_rank = self.comms.bd_rank
            bd_comm.Send(np.array([bd_rank], dtype='i'), dest=0)
            info = MPI.Status()
            bd_comm.Probe(source=0, tag=MPI.ANY_TAG, status=info)
            count = info.Get_elements(MPI.BYTE)
            if count == 0:
                return bytearray()

            extended_chunk = bytearray(count)
            st = time.time()
            bd_comm.Recv(extended_chunk, source=0)
            en = time.time()
            
            pf = PacketFooter(view=extended_chunk)
            chunk, calibconst_pkt = pf.split_packets()
            calib_pf = PacketFooter(view=calibconst_pkt)
            for calib_bytes in calib_pf.split_packets():
                self.calib_store.append(pickle.loads(calib_bytes))

            logging.debug(f"node.py: BigData{bd_rank} got {count/1e6:.5f}MB (waiting time: {en-st:.5f} s) calib store has {len(self.calib_store)} items")
            return chunk
        
        events = Events(self.configs, self.dm, self.dsparms.prom_man, 
                filter_callback=self.dsparms.filter, get_smd=get_smd)
        
        for evt in events:
            if evt.service() == TransitionId.BeginRun:
                for calib_const in self.calib_store:
                    for key, _ in calib_const.items():
                        det_name = key
                        break
                    runnum = calib_const[det_name]['pedestals'][1]['run']
                    if evt._dgrams[0].runinfo[0].runinfo.runnum == runnum:
                        self.dsparms.calibconst = calib_const
                        break
            yield evt
