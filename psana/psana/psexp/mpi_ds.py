import sys
import os
import numpy as np
from mpi4py import MPI
from psana.psexp import *
from psana import dgram
from psana.dgrammanager import DgramManager
import logging
import time

class InvalidEventBuilderCores(Exception): pass



class RunParallel(Run):
    """ Yields list of events from multiple smd/bigdata files using > 3 cores."""

    def __init__(self, comms, run_evt, events, configs, dsparms):
        super(RunParallel, self).__init__(dsparms)

        self.comms      = comms
        psana_comm      = comms.psana_comm # TODO tjl and cpo to review
    
        rank            = psana_comm.Get_rank()
        size            = psana_comm.Get_size()
        
        self._evt       = run_evt
        self.beginruns  = run_evt._dgrams
        self._evt_iter  = events
        self.configs    = configs
        
        self._get_runinfo()
        self.esm = EnvStoreManager(self.configs)

    def events(self):
        for evt in self._evt_iter:
            if evt.service() != TransitionId.L1Accept:
                self.esm.update_by_event(evt)
                if evt.service() == TransitionId.EndRun: return
                continue
            st = time.time()
            yield evt
            en = time.time()
            self.c_ana.labels('seconds','None').inc(en-st)
            self.c_ana.labels('batches','None').inc()
        
    def steps(self):
        for evt in self._evt_iter:
            if evt.service() == TransitionId.EndRun: return
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, self._evt_iter, self.esm)
    

class MPIDataSource(DataSourceBase):

    def __init__(self, comms, *args, **kwargs):
        super(MPIDataSource, self).__init__(**kwargs)


        self.comms = comms
        comm = self.comms.psana_comm # todo could be better
        rank = comm.Get_rank()
        size = comm.Get_size()
        

        g_ts = self.prom_man.get_metric("psana_timestamp")

        
        # setup xtc files
        if rank == 0:
            self._setup_xtcs()
        else:
            self.xtc_files = None
            self.smd_files = None
        self.xtc_files = comm.bcast(self.xtc_files, root=0)
        self.smd_files = comm.bcast(self.smd_files, root=0)


        # create and broadcast configs
        if rank == 0:
            self.smd_fds  = np.array([os.open(smd_file, os.O_RDONLY) for smd_file in self.smd_files], dtype=np.int32)
            self.smdr_man = SmdReaderManager(self.smd_fds, self.dsparms)
            self._configs = self.smdr_man.get_next_dgrams()
            self._setup_det_class_table()
            self._set_configinfo()
            self.smdr_man.set_configs(self._configs)
            g_ts.labels("first_event").set(time.time())
            nbytes = np.array([memoryview(config).shape[0] for config in self._configs], \
                    dtype='i')
        else:
            self._configs = None
            nbytes = np.empty(len(self.smd_files), dtype='i')
        comm.Bcast(nbytes, root=0) # no. of bytes is required for mpich
        if rank > 0:
            self._configs = [np.empty(nbyte, dtype='b') for nbyte in nbytes]
        for i in range(len(self._configs)):
            comm.Bcast([self._configs[i], nbytes[i], MPI.BYTE], root=0)
        if rank > 0:
            self._configs = [dgram.Dgram(view=config, offset=0) for config in self._configs]
            g_ts.labels("first_event").set(time.time())
            self._setup_det_class_table()
            self._set_configinfo()
        

        self.dm = DgramManager(self.xtc_files, configs=self._configs)


        # check if no. of ranks is enough
        nsmds = int(os.environ.get('PS_EB_NODES', 1)) # No. of smd cores
        if not (size > (nsmds + 1)):
            print('ERROR Too few MPI processes. MPI size must be more than '
                  ' no. of all workers. '
                  '\n\tTotal psana size: %d'
                  '\n\tPS_EB_NODES:     %d' % (size, nsmds))
            sys.stdout.flush() # make sure error is printed
            MPI.COMM_WORLD.Abort()
       

        if self.destination:
            if nsmds > 1:
                raise(InvalidEventBuilderCores("Invalid no. of eventbuilder cores: %d. There must be only one eventbuilder core when destionation callback is set."%(nsmds)))

        
        self._start_prometheus_client(mpi_rank=rank)

    def runs(self):
        events = self.run_node()
        for evt in events:
            logging.debug(f"mpi_ds.py: rank {self.comms.psana_comm.Get_rank()} got evt.service={evt.service()}")
            if evt.service() == TransitionId.BeginRun:
                run = RunParallel(self.comms, evt, events, self._configs, self.dsparms)
                yield run 
        
        self._end_prometheus_client()
        
        if self.comms.psana_comm.Get_rank() == 0:
            for smd_fd in self.smd_fds:
                os.close(smd_fd)

    def run_node(self):
        if self.comms._nodetype == 'smd0':
            Smd0(self.comms, self._configs, self.smdr_man)
        elif self.comms._nodetype == 'eb':
            eb_node = EventBuilderNode(self.comms, self._configs, self.dsparms)
            eb_node.run_mpi()
        elif self.comms._nodetype == 'bd':
            bd_node = BigDataNode(self.comms, self._configs, self.dsparms, self.dm)
            for evt in bd_node.run_mpi():
                yield evt
        elif self.comms._nodetype == 'srv':
            # tell the iterator to do nothing
            return
