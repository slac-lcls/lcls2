import sys
import os
import numpy as np
from mpi4py import MPI

from .tools import mode
from .ds_base import DataSourceBase
from .run import Run

from psana import dgram
from psana.dgrammanager import DgramManager
from psana.psexp.envstore_manager import EnvStoreManager
from psana.psexp.event_manager import TransitionId
from psana.psexp.node import Smd0, SmdNode, BigDataNode
from psana.psexp.smdreader_manager import SmdReaderManager
import logging
import time

class InvalidEventBuilderCores(Exception): pass



class RunParallel(Run):
    """ Yields list of events from multiple smd/bigdata files using > 3 cores."""

    def __init__(self, comms, exp, run_no, run_src, **kwargs):
        """ Parallel read requires that rank 0 does the file system works.
        Configs and calib constants are sent to other ranks by MPI.
        
        Note that destination callback only works with RunParallel.
        """
        super(RunParallel, self).__init__(exp, run_no, 
                max_events      = kwargs['max_events'], 
                batch_size      = kwargs['batch_size'], 
                filter_callback = kwargs['filter_callback'], 
                destination     = kwargs['destination'],
                prom_man        = kwargs['prom_man'])
        xtc_files, smd_files, other_files = run_src

        self.comms = comms
        psana_comm = comms.psana_comm # TODO tjl and cpo to review
    
        rank = psana_comm.Get_rank()
        size = psana_comm.Get_size()

        if rank == 0:
            # get Configure and BeginRun using SmdReader
            self.smd_fds = np.array([os.open(smd_file, os.O_RDONLY) for smd_file in smd_files], dtype=np.int32)
            self.smdr_man = SmdReaderManager(self)
            self.configs = self.smdr_man.get_next_dgrams()
            self.beginruns = self.smdr_man.get_next_dgrams(configs=self.configs)
            
            self._get_runinfo()

            self.smd_dm = DgramManager(smd_files, configs=self.configs, run=self)
            self.dm = DgramManager(xtc_files, configs=self.smd_dm.configs, run=self)

            nbytes = np.array([memoryview(config).shape[0] for config in self.configs], \
                            dtype='i')
            
            super()._set_configinfo()
            super()._set_calibconst()
            self.bcast_packets = {'calibconst': self.calibconst, \
                    'expt': self.expt, 'runnum': self.runnum, 'timestamp': self.timestamp}
            
        else:
            self.smd_dm = None
            self.dm = None
            self.configs = None
            nbytes = np.empty(len(smd_files), dtype='i')
            self.bcast_packets = None
        
        # Send configs without pickling
        psana_comm.Bcast(nbytes, root=0) # no. of bytes is required for mpich
        if rank > 0:
            self.configs = [np.empty(nbyte, dtype='b') for nbyte in nbytes]
       
        for i in range(len(self.configs)):
            psana_comm.Bcast([self.configs[i], nbytes[i], MPI.BYTE], root=0)
        
        # Send other small things using small-case bcast
        self.bcast_packets = psana_comm.bcast(self.bcast_packets, root=0)
        if rank > 0:
            self.configs = [dgram.Dgram(view=config, offset=0) for config in self.configs]
            self.dm = DgramManager(xtc_files, configs=self.configs, run=self)
            super()._set_configinfo() # after creating a dgrammanger, we can setup config info
            self.calibconst = self.bcast_packets['calibconst']
            self.expt = self.bcast_packets['expt']
            self.runnum = self.bcast_packets['runnum']
            self.timestamp = self.bcast_packets['timestamp']
        
        self.esm = EnvStoreManager(self.configs, 'epics', 'scan')
    
    def _get_runinfo(self):
        if not self.beginruns : return

        beginrun_dgram = self.beginruns[0]
        if hasattr(beginrun_dgram, 'runinfo'): # some xtc2 do not have BeginRun
            self.expt = beginrun_dgram.runinfo[0].runinfo.expt 
            self.runnum = beginrun_dgram.runinfo[0].runinfo.runnum
            self.timestamp = beginrun_dgram.timestamp()
    
    def events(self):
        for evt in self.run_node():
            if evt.service() != TransitionId.L1Accept: continue
            st = time.time()
            yield evt
            en = time.time()
            self.c_ana.labels('seconds','None').inc(en-st)
            self.c_ana.labels('calls','None').inc()

    def steps(self):
        self.scan = True
        for step in self.run_node():
            yield step

    def run_node(self):
        if self.comms._nodetype == 'smd0':
            Smd0(self)
        elif self.comms._nodetype == 'smd':
            smd_node = SmdNode(self)
            smd_node.run_mpi()
        elif self.comms._nodetype == 'bd':
            bd_node = BigDataNode(self)
            for result in bd_node.run_mpi():
                yield result
        elif self.comms._nodetype == 'srv':
            # tell the iterator to do nothing
            return


class MPIDataSource(DataSourceBase):

    def __init__(self, comms, *args, **kwargs):
        super(MPIDataSource, self).__init__(**kwargs)

        self.comms = comms
        comm = self.comms.psana_comm # todo could be better
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            exp, run_dict = super(MPIDataSource, self)._setup_xtcs()
        else:
            exp, run_dict = None, None

        nsmds = int(os.environ.get('PS_SMD_NODES', 1)) # No. of smd cores
        if not (size > (nsmds + 1)):
            print('ERROR Too few MPI processes. MPI size must be more than '
                  ' no. of all workers. '
                  '\n\tTotal psana size: %d'
                  '\n\tPS_SMD_NODES:     %d' % (size, nsmds))
            sys.stdout.flush() # make sure error is printed
            MPI.COMM_WORLD.Abort()
        
        if self.destination:
            if nsmds > 1:
                raise(InvalidEventBuilderCores("Invalid no. of eventbuilder cores: %d. There must be only one eventbuilder core when destionation callback is set."%(nsmds)))

        exp = comm.bcast(exp, root=0)
        run_dict = comm.bcast(run_dict, root=0)

        self.exp = exp
        self.run_dict = run_dict
        
        super()._start_prometheus_client(mpi_rank=rank)


    def runs(self):
        for run_no in self.run_dict:
            run = RunParallel(self.comms, self.exp, run_no, self.run_dict[run_no], 
                        max_events      = self.max_events, 
                        batch_size      = self.batch_size, 
                        filter_callback = self.filter, 
                        destination     = self.destination,
                        prom_man        = self.prom_man)
            self.run = run # FIXME: provide support for cctbx code (ds.Detector). will be removed in next cctbx update.
            yield run
        
        super()._end_prometheus_client(mpi_rank=self.comms.psana_comm.Get_rank()) 


