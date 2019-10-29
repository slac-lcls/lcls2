import os
import numpy as np
from mpi4py import MPI

from .tools import mode
from .ds_base import DataSourceBase
from .run import Run

from psana import dgram
from psana.dgrammanager import DgramManager
from psana.psexp.stepstore_manager import StepStoreManager
from psana.psexp.event_manager import TransitionId
from psana.psexp.node import Smd0, SmdNode, BigDataNode

class InvalidEventBuilderCores(Exception): pass



class RunParallel(Run):
    """ Yields list of events from multiple smd/bigdata files using > 3 cores."""

    def __init__(self, comms, exp, run_no, run_src, **kwargs):
        """ Parallel read requires that rank 0 does the file system works.
        Configs and calib constants are sent to other ranks by MPI.
        
        Note that destination callback only works with RunParallel.
        """
        super(RunParallel, self).__init__(exp, run_no, max_events=kwargs['max_events'], \
                batch_size=kwargs['batch_size'], filter_callback=kwargs['filter_callback'], \
                destination=kwargs['destination'])
        xtc_files, smd_files, other_files = run_src

        self.comms = comms
        psana_comm = comms.psana_comm # TODO tjl and cpo to review
    
        rank = psana_comm.Get_rank()
        size = psana_comm.Get_size()

        if rank == 0:
            self.smd_dm = DgramManager(smd_files)
            self.dm = DgramManager(xtc_files, configs=self.smd_dm.configs)
            self.configs = self.dm.configs
            nbytes = np.array([memoryview(config).shape[0] for config in self.configs], \
                            dtype='i')
            self.calibs = {}
            for det_name in self.detnames:
                self.calibs[det_name] = super(RunParallel, self)._get_calib(det_name)
            
        else:
            self.smd_dm = None
            self.dm = None
            self.configs = None
            self.calibs = None
            nbytes = np.empty(len(smd_files), dtype='i')
        
        psana_comm.Bcast(nbytes, root=0) # no. of bytes is required for mpich
        
        # create empty views of known size
        if rank > 0:
            self.configs = [np.empty(nbyte, dtype='b') for nbyte in nbytes]
        
        for i in range(len(self.configs)):
            psana_comm.Bcast([self.configs[i], nbytes[i], MPI.BYTE], root=0)
        
        self.calibs = psana_comm.bcast(self.calibs, root=0)

        if rank > 0:
            # Create dgram objects using views from rank 0 (no disk operation).
            self.configs = [dgram.Dgram(view=config, offset=0) for config in self.configs]
            self.dm = DgramManager(xtc_files, configs=self.configs)
        
        self.ssm = StepStoreManager(self.configs, 'epics', 'scan')
    
    def events(self):
        for evt in self.run_node():
            if evt._dgrams[0].seq.service() != TransitionId.L1Accept: continue
            yield evt

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
        assert size > (nsmds + 1) # MPI size must be more than no. of all workers
        
        if self.destination:
            if nsmds > 1:
                raise(InvalidEventBuilderCores("Invalid no. of eventbuilder cores: %d. There must be only one eventbuilder core when destionation callback is set."%(nsmds)))

        exp = comm.bcast(exp, root=0)
        run_dict = comm.bcast(run_dict, root=0)

        self.exp = exp
        self.run_dict = run_dict


    def runs(self):
        for run_no in self.run_dict:
            run = RunParallel(self.comms, self.exp, run_no, self.run_dict[run_no], \
                        max_events=self.max_events, batch_size=self.batch_size, \
                        filter_callback=self.filter, destination=self.destination)
            self.run = run # FIXME: provide support for cctbx code (ds.Detector). will be removed in next cctbx update.
            yield run


