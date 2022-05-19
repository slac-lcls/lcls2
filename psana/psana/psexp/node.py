from sys import byteorder
import numpy as np
from psana.psexp import *
from psana.dgram import Dgram
import os

from psana.psexp.tools import mode
if mode == 'mpi':
    from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

import time

s_eb_wait_smd0 = PrometheusManager.get_metric('psana_eb_wait_smd0')

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
        self.n_smd_nodes = PS_EB_NODES

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
        self.state = True

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

    # change variable keeps track of whether there are any transitions
    # in the current batch. Is so we need to re-partition the region next
    # time around and pack the new transition data.
    # If there are no new transitions then we re-use the existing region
    def extend_buffers_state(self, views, client_id):
        idx = client_id - 1
        change = self.state
        self.state = False
        for i_smd, view in enumerate(views):
            self.bufs[i_smd].extend(view)
            if view.nbytes:
                self.state = True
        return change

    def get_buffer(self, client_id, smd0=False):
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

    def get_buffer_only(self, client_id):
        """ Returns new step data (if any) for this client
        then updates the sent record."""
        views = []
        if self.n_smds: # do nothing if no step data found
            indexed_id = client_id - 1 # keep indexing the same
            views = [bytearray() for i in range(self.n_smds)]
            for i, buf in enumerate(self.bufs):
                current_buf = self.bufs[i]
                current_offset = self.send_history[indexed_id][i]
                current_buf_size = memoryview(current_buf).nbytes
                if current_offset < current_buf_size:
                    views[i].extend(current_buf[current_offset:])
                    #self.send_history[indexed_id][i] = current_buf_size
        return views


def repack_for_bd(smd_batch, step_views, configs, client=-1):
    """ EventBuilder Node uses this to prepend missing step views 
    to the smd_batch. This output chunk contains list of pre-built events."""
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

def repack_with_step_dg(smd_batch, step_views, configs):
    """ EventBuilder Node uses this to prepend missing step datagrams
    to the smd_batch. This output chunk contains list of pre-built events."""
    pf = PacketFooter(view=step_views)
    if pf and pf.get_size(0):
        batch_pf = PacketFooter(view=smd_batch)
        # initialize offsets array to 0
        offsets = [0]*pf.n_packets
        # get the chunks from SMD0
        chunks = pf.split_packets()
        # Create bytearray containing a list of events from step_views
        steps = bytearray()
        # number of files
        n_smds = pf.n_packets
        step_sizes = []
        n_steps = 0
        while offsets[0] < pf.get_size(0):
            step_pf = PacketFooter(n_packets=n_smds)
            step_size = 0
            # organize dgrams vertically
            for i, chunk in enumerate(chunks):
                d = Dgram(view=chunk,config=configs[i],offset=offsets[i])
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
    return smd_batch

def repack_with_mstep_dg(smd_batch, step_views, configs, nviews):
    """ EventBuilder Node uses this to prepend missing step datagrams
    to the smd_batch. This output chunk contains list of pre-built events."""
    return smd_batch
    pf = PacketFooter(view=step_views, num_views=nviews)
    if pf == None:
        return smd_batch
    if pf.get_size(0) == 0:
        return smd_batch
    batch_pf = PacketFooter(view=smd_batch)
    # Create bytearray containing a list of events from step_views

    # number of files
    n_smds = pf.n_packets
    # Create bytearray containing a list of events from step_views
    step_sizes = []
    n_steps = 0
    steps = []
    for j, chunks in enumerate(pf.split_multiple_packets()):
        steps[j] = bytearray()
        # initialize offsets array to 0
        offsets = [0]*pf.n_packets
        while offsets[0] < pf.get_size(0):
            step_pf = PacketFooter(n_packets=n_smds)
            step_size = 0
            # organize dgrams vertically
            for i, chunk in enumerate(chunks):
                d = Dgram(view=chunk,config=configs[i],offset=offsets[i])
                steps[j].extend(d)
                offsets[i] += d._size
                step_size += d._size
                step_pf.set_size(i, d._size)

            steps[j].extend(step_pf.footer)
            # total size of the vertical transition + it's footer
            # file=2, [d0,d1, d0_size, d1_size, num_files], [d0,d1,d0_size,d1_size,num_files]
            # step_sizes [size_0][size_1]
            #step_sizes.append(step_size + memoryview(step_pf.footer).nbytes)
            step_sizes.insert(0,step_size + memoryview(step_pf.footer).nbytes)
            n_steps += 1

    # Create new batch with total_events = smd_batch_events + step_events
    new_batch_pf = PacketFooter(n_packets = batch_pf.n_packets + n_steps)
    for i in range(n_steps):
        new_batch_pf.set_size(i, step_sizes[i])

    for i in range(n_steps, new_batch_pf.n_packets):
        new_batch_pf.set_size(i, batch_pf.get_size(i-n_steps))

    new_batch = bytearray()
    # the order is reversed with legion subregions
    for i in steps:
        new_batch.extend(steps[len(steps)-i+1])

    new_batch.extend(smd_batch[:memoryview(smd_batch).nbytes-memoryview(batch_pf.footer).nbytes])
    new_batch.extend(new_batch_pf.footer)
    return new_batch

def wait_for(requests):
    status = [MPI.Status() for i in range(len(requests))]
    MPI.Request.Waitall(requests, status)

class Smd0(object):
    """ Sends blocks of smds to eb nodes
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp.
    """
    def __init__(self, comms, configs, smdr_man, dsparms):
        self.comms = comms
        self.smdr_man = smdr_man
        self.configs = configs
        self.step_hist = StepHistory(self.comms.smd_size, len(self.configs))
        
        # Collecting Smd0 performance using prometheus
        self.c_sent = dsparms.prom_man.get_metric('psana_smd0_sent')
        
    def start(self):
        rankreq = np.empty(1, dtype='i')
        waiting_ebs = []
        requests = [MPI.REQUEST_NULL for i in range(self.comms.smd_size - 1)]

        # Indentify viewing windows. SmdReaderManager has starting index and block size
        # that it needs to share later when data are packaged for sending to EventBuilders.
        
        # Need this for async MPI to prevent overwriting send buffer
        repack_smds = {}
        
        for i_chunk in self.smdr_man.chunks():
            st_req = time.monotonic()
            logger.debug(f'RANK{self.comms.world_rank} 1. SMD0GOTCHUNK {st_req}')

            req = self.comms.smd_comm.Irecv(rankreq, source=MPI.ANY_SOURCE)
            req.Wait()
            en_req = time.monotonic()
            logger.debug(f'RANK{self.comms.world_rank} 2. SMD0GOTEB{rankreq[0]} {en_req}')
            
            # Check missing steps for the current client
            missing_step_views = self.step_hist.get_buffer(rankreq[0], smd0=True)
            logger.debug(f'RANK{self.comms.world_rank} 2.1 SMD0GOTSTEPHIST {time.monotonic()}')

            # Update step buffers (after getting the missing steps
            step_views = [self.smdr_man.smdr.show(i, step_buf=True) for i in range(self.smdr_man.n_files)]
            self.step_hist.extend_buffers(step_views, rankreq[0])
            logger.debug(f'RANK{self.comms.world_rank} 2.2 SMD0STEPHISTUPDATED {time.monotonic()}')

            # Prevent race condition by making a copy of data
            repack_smds[rankreq[0]] = self.smdr_man.smdr.repack_parallel(missing_step_views, rankreq[0])
            
            logger.debug(f'RANK{self.comms.world_rank} 3. SMD0GOTREPACK {time.monotonic()}')
            
            requests[rankreq[0]-1] = self.comms.smd_comm.Isend(repack_smds[rankreq[0]], dest=rankreq[0])
            
            logger.debug(f'RANK{self.comms.world_rank} 4. SMD0DONEWITHEB{rankreq[0]} {time.monotonic()}')
        
            # sending data to prometheus
            self.c_sent.labels('evts', rankreq[0]).inc(self.smdr_man.got_events)
            self.c_sent.labels('batches', rankreq[0]).inc()
            self.c_sent.labels('MB', rankreq[0]).inc(memoryview(repack_smds[rankreq[0]]).nbytes/1e6)
            self.c_sent.labels('seconds', rankreq[0]).inc(en_req - st_req)
            logger.debug(f'node: smd0 sent {self.smdr_man.got_events} events to {rankreq[0]} (waiting for this rank took {en_req-st_req:.5f} seconds)')
            
            found_endrun = self.smdr_man.smdr.found_endrun()
            if found_endrun: 
                logger.debug("node: smd0 found_endrun")
                break
        # end for (smd_chunk, step_chunk)
        wait_for(requests)

        # check if there are missing steps to be sent 
        requests = [MPI.REQUEST_NULL for i in range(self.comms.smd_size - 1)]
        for i in range(self.comms.n_smd_nodes):
            req = self.comms.smd_comm.Irecv(rankreq, source=MPI.ANY_SOURCE)
            req.Wait()
            missing_step_views = self.step_hist.get_buffer(rankreq[0], smd0=True)
            repack_smds[rankreq[0]] = self.smdr_man.smdr.repack_parallel(missing_step_views, rankreq[0], only_steps=1)
            if memoryview(repack_smds[rankreq[0]]).nbytes > 0:
                requests[rankreq[0]-1] = self.comms.smd_comm.Isend(repack_smds[rankreq[0]], dest=rankreq[0])
            else:
                waiting_ebs.append(rankreq[0])
        wait_for(requests)
        
        # kill waiting bd nodes
        requests = [MPI.REQUEST_NULL for i in range(self.comms.smd_size - 1)]
        for dest_rank in waiting_ebs:
            requests[dest_rank-1] = self.comms.smd_comm.Isend(bytearray(), dest=dest_rank)
        wait_for(requests)

        requests = [MPI.REQUEST_NULL for i in range(self.comms.smd_size - 1)]
        for i in range(self.comms.n_smd_nodes-len(waiting_ebs)):
            req = self.comms.smd_comm.Irecv(rankreq, source=MPI.ANY_SOURCE)
            req.Wait()
            requests[rankreq[0]-1] = self.comms.smd_comm.Isend(bytearray(), dest=rankreq[0])
        wait_for(requests)
    

class EventBuilderNode(object):
    """Handles both smd_0 and bd_nodes
    Receives blocks of smds from smd_0 then assembles
    offsets and dgramsizes into a numpy array. Sends
    this np array to bd_nodes that are registered to it."""
    def __init__(self, comms, configs, dsparms, dm):
        self.comms      = comms
        self.configs    = configs
        self.dsparms    = dsparms
        self.dm         = dm
        self.step_hist  = StepHistory(self.comms.bd_size, len(self.configs))
        # Collecting Smd0 performance using prometheus
        self.c_sent     = dsparms.prom_man.get_metric('psana_eb_sent')
        self.requests   = []
    
    def _init_requests(self):
        self.requests = [MPI.REQUEST_NULL for i in range(self.comms.bd_size - 1)]

    def pack(self, *args):
        pf = PacketFooter(len(args))
        batch = bytearray()
        for i, arg in enumerate(args):
            pf.set_size(i, memoryview(arg).shape[0])
            batch += arg
        batch += pf.footer
        return batch


    def _send_to_dest(self, dest_rank, smd_batch_dict, step_batch_dict, eb_man, batches):
        bd_comm = self.comms.bd_comm
        smd_batch, _ = smd_batch_dict[dest_rank]
        missing_step_views = self.step_hist.get_buffer(dest_rank)
        batches[dest_rank] = repack_for_bd(smd_batch, missing_step_views, self.configs, client=dest_rank)
        self.requests[dest_rank-1] = bd_comm.Isend(batches[dest_rank], dest=dest_rank)
        del smd_batch_dict[dest_rank] # done sending
        
        step_batch, _ = step_batch_dict[dest_rank]
        if eb_man.eb.nsteps > 0 and memoryview(step_batch).nbytes > 0:  
            step_pf = PacketFooter(view=step_batch)
            self.step_hist.extend_buffers(step_pf.split_packets(), dest_rank, as_event=True)
        del step_batch_dict[dest_rank] # done adding

    def _request_rank(self, rankreq):
        st_req = time.monotonic()
        req = self.comms.bd_comm.Irecv(rankreq, source=MPI.ANY_SOURCE)
        req.Wait()
        en_req = time.monotonic()
        self.c_sent.labels('seconds',rankreq[0]).inc(en_req-st_req)
        logger.debug("node: eb%d got bd %d (request took %.5f seconds)"%(self.comms.smd_rank, rankreq[0], (en_req-st_req)))

    @s_eb_wait_smd0.time()
    def _request_data(self, smd_comm):
        logger.debug(f'RANK{self.comms.world_rank} 5. EB{self.comms.world_rank}SENDREQTOSMD0 {time.monotonic()}')
        smd_comm.Isend(np.array([self.comms.smd_rank], dtype='i'), dest=0)
        logger.debug(f'RANK{self.comms.world_rank} 6. EB{self.comms.world_rank}DONESENDREQ {time.monotonic()}')
        info = MPI.Status()
        smd_comm.Probe(source=0, status=info)
        count = info.Get_elements(MPI.BYTE)
        smd_chunk = bytearray(count)
        req = smd_comm.Irecv(smd_chunk, source=0)
        req.Wait()
        logger.debug(f'RANK{self.comms.world_rank} 7. EB{self.comms.world_rank}RECVDATA {time.monotonic()}')
        logger.debug(f"node: eb{self.comms.smd_rank} received {count/1e6:.5f} MB from smd0")
        return smd_chunk

    def start(self):
        rankreq = np.empty(1, dtype='i')
        smd_comm   = self.comms.smd_comm
        n_bd_nodes = self.comms.bd_comm.Get_size() - 1
        bd_comm    = self.comms.bd_comm
        smd_rank   = self.comms.smd_rank
        waiting_bds   = []
        
        # Initialize Non-blocking Send Requests with Null
        self._init_requests()
        
        while True:
            smd_chunk = self._request_data(smd_comm)
            if not smd_chunk:
                break

            eb_man = EventBuilderManager(smd_chunk, self.configs, self.dsparms, self.dm.get_run())
            logger.debug(f'RANK{self.comms.world_rank} 8. EB{self.comms.world_rank}DONEBUILDINGEVENTS {time.monotonic()}')
        
            # Build batches of events
            
            # Need this for async MP to prevent overwriting send buffer
            # The key of batches dict is the bd rank.
            batches = {} 

            for smd_batch_dict, step_batch_dict  in eb_man.batches():
                
                # If single item and dest_rank=0, send to any bigdata nodes.
                if 0 in smd_batch_dict.keys():
                    smd_batch, _ = smd_batch_dict[0]
                    step_batch, _ = step_batch_dict[0]

                    logger.debug(f'RANK{self.comms.world_rank} 9. EB{self.comms.world_rank}REQBD {time.monotonic()}')
                    if waiting_bds:
                        logger.debug(f'before waiting_bds={waiting_bds}')
                        rankreq[0] = waiting_bds.pop()
                        logger.debug(f'after pop waiting_bds={waiting_bds}')
                        logger.debug(f'RANK{self.comms.world_rank} 10. EB{self.comms.world_rank}GOTBD{rankreq[0]+1}FROMQUEUE {time.monotonic()}')
                    else:
                        self._request_rank(rankreq)
                        logger.debug(f'RANK{self.comms.world_rank} 10. EB{self.comms.world_rank}GOTBD{rankreq[0]+1}FROMREQ {time.monotonic()}')
                    
                    missing_step_views = self.step_hist.get_buffer(rankreq[0])
                    batches[rankreq[0]] = repack_for_bd(smd_batch, missing_step_views, self.configs, client=rankreq[0])
                    
                    logger.debug(f'RANK{self.comms.world_rank} 11. EB{self.comms.world_rank}SENDDATATOBD{rankreq[0]+1} {time.monotonic()}')
                    self.requests[rankreq[0]-1] = bd_comm.Isend(batches[rankreq[0]], dest=rankreq[0])
                    logger.debug(f'RANK{self.comms.world_rank} 12. EB{self.comms.world_rank}DONESENDDATATOBD{rankreq[0]+1} {time.monotonic()}')
                    
                    # sending data to prometheus
                    logger.debug(f'node: eb{self.comms.smd_rank} sent {eb_man.eb.nevents} events ({memoryview(smd_batch).nbytes} bytes) to bd{rankreq[0]}')
                    self.c_sent.labels('evts', rankreq[0]).inc(eb_man.eb.nevents)
                    self.c_sent.labels('batches', rankreq[0]).inc()
                    self.c_sent.labels('MB', rankreq[0]).inc(memoryview(batches[rankreq[0]]).nbytes/1e6)
                    
                    if eb_man.eb.nsteps > 0 and memoryview(step_batch).nbytes > 0:  
                        step_pf = PacketFooter(view=step_batch)
                        self.step_hist.extend_buffers(step_pf.split_packets(), rankreq[0], as_event=True)
                    
                          
                # With > 1 dest_rank, start looping until all dest_rank batches
                # have been sent.
                else: # if 0 in smd_batch_dict ...
                    # Check if destinations are valid 
                    destinations = np.asarray(list(smd_batch_dict.keys()))
                    if any(destinations > n_bd_nodes):
                        logger.debug(f"Found invalid destination ({destinations}). Must be <= {n_bd_nodes} (#big data nodes)")
                        break

                    while smd_batch_dict:
                        if waiting_bds: # Check first if there are bd nodes waiting
                            copied_waiting_bds = waiting_bds[:]
                            for dest_rank in copied_waiting_bds:
                                if dest_rank in smd_batch_dict:
                                    self._send_to_dest(dest_rank, smd_batch_dict, step_batch_dict, eb_man, batches)
                                    waiting_bds.remove(dest_rank)
                        
                        if smd_batch_dict:
                            self._request_rank(rankreq)
                            dest_rank = rankreq[0]
                            if dest_rank in smd_batch_dict:
                                self._send_to_dest(dest_rank, smd_batch_dict, step_batch_dict, eb_man, batches)
                            else:
                                waiting_bds.append(dest_rank)
                    # end while smd_batch_dict
               
                # end else -> if 0 in smd_batch_dict.keys() 
            
            # end for smd_batch_dict in ...
            logger.debug(f'RANK{self.comms.world_rank} 12.1 EB{self.comms.world_rank}DONEALLBATCHES {time.monotonic()}')

        # end While True
        wait_for(self.requests)

        batches = {}

        # Check if any of the waiting bds need missing steps from the last batch
        self._init_requests()
        copied_waiting_bds = waiting_bds[:]
        for dest_rank in copied_waiting_bds:
            missing_step_views = self.step_hist.get_buffer(dest_rank)
            batches[dest_rank] = repack_for_bd(bytearray(), missing_step_views, self.configs, client=dest_rank)
            if batches[dest_rank]:
                logger.debug(f'RANK{self.comms.world_rank} 12.2 EB{self.comms.world_rank}SENDMISSINGSTEPTOBD{dest_rank+1} {time.monotonic()}')
                self.requests[dest_rank-1] = bd_comm.Isend(batches[dest_rank], dest_rank)
                logger.debug(f'RANK{self.comms.world_rank} 12.3 EB{self.comms.world_rank}SENDMISSINGSTEPTOBD{dest_rank+1} {time.monotonic()}')
                waiting_bds.remove(dest_rank)
                logger.debug(f'after remove waiting_bds={waiting_bds}')
        wait_for(self.requests)
        
        logger.debug(f'RANK{self.comms.world_rank} 12.4 EB{self.comms.world_rank}DONEMISSSTEPS {time.monotonic()}')

        # Check if the rest of bds need missing steps from the last batch
        self._init_requests()
        for i in range(n_bd_nodes-len(waiting_bds)):
            logger.debug(f'i={i} n_bd_nodes={n_bd_nodes} len(waiting_bds)={len(waiting_bds)}')
            self._request_rank(rankreq)
            missing_step_views = self.step_hist.get_buffer(rankreq[0])
            batches[rankreq[0]] = repack_for_bd(bytearray(), missing_step_views, self.configs, client=rankreq[0])
            if batches[rankreq[0]]:
                logger.debug(f'RANK{self.comms.world_rank} 12.5 EB{self.comms.world_rank}SENDMISSINGSTEPTOBD{rankreq[0]+1} {time.monotonic()}')
                self.requests[rankreq[0]-1] = bd_comm.Isend(batches[rankreq[0]], dest=rankreq[0])
                logger.debug(f'RANK{self.comms.world_rank} 12.6 EB{self.comms.world_rank}SENDMISSINGSTEPTOBD{rankreq[0]+1} {time.monotonic()}')
            else:
                waiting_bds.append(rankreq[0])
                logger.debug(f'after append waiting_bds={waiting_bds}')
        wait_for(self.requests)

        logger.debug(f'RANK{self.comms.world_rank} 12.7 EB{self.comms.world_rank}DONE {time.monotonic()}')

        
        # end While True: done - kill idling nodes
        self._init_requests()
        for dest_rank in waiting_bds:
            self.requests[dest_rank-1] = bd_comm.Isend(bytearray(), dest=dest_rank)
            logger.debug(f"node: eb{self.comms.smd_rank} send null byte to bd{dest_rank}")
        wait_for(self.requests)

        # - kill all other nodes
        self._init_requests()
        for i in range(n_bd_nodes-len(waiting_bds)):
            self._request_rank(rankreq)
            self.requests[rankreq[0]-1] = bd_comm.Isend(bytearray(), dest=rankreq[0])
            logger.debug(f"node: eb{self.comms.smd_rank} send null byte to bd{rankreq[0]}")
        wait_for(self.requests)

class BigDataNode(object):
    def __init__(self, comms, configs, dsparms, dm):
        self.comms      = comms
        self.configs    = configs
        self.dsparms    = dsparms
        self.dm         = dm
        self.bd_wait_eb = PrometheusManager.get_metric('psana_bd_wait_eb')

    def start(self):
        
        def get_smd():
            bd_comm = self.comms.bd_comm
            bd_rank = self.comms.bd_rank
            logger.debug(f'RANK{self.comms.world_rank} 13. BD{self.comms.world_rank}SENDREQTOEB {time.monotonic()}')
            req = bd_comm.Isend(np.array([bd_rank], dtype='i'), dest=0)
            req.Wait()
            logger.debug(f'RANK{self.comms.world_rank} 14. BD{self.comms.world_rank}DONESENDREQTOEB {time.monotonic()}')
            info = MPI.Status()
            bd_comm.Probe(source=0, tag=MPI.ANY_TAG, status=info)
            count = info.Get_elements(MPI.BYTE)
            chunk = bytearray(count)
            st_req = time.monotonic()
            req = bd_comm.Irecv(chunk, source=0)
            req.Wait()
            logger.debug(f'RANK{self.comms.world_rank} 15. BD{self.comms.world_rank}RECVDATA {time.monotonic()}')
            en_req = time.monotonic()
            self.bd_wait_eb.labels('seconds', self.comms.world_rank).inc(en_req - st_req)
            return chunk
        
        events = Events(self.configs, self.dm, self.dsparms, 
                filter_callback=self.dsparms.filter, get_smd=get_smd)

        for evt in events:
            yield evt
