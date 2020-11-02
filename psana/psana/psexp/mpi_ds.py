import sys
import os
import numpy as np
from mpi4py import MPI
from psana.psexp import *
from psana import dgram
from psana.dgrammanager import DgramManager
from psana.smalldata import SmallData
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
                if evt.service() == TransitionId.EndRun: 
                    return
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
    
class NullRun(object):
    def __init__(self):
        self.expt = None
        self.runnum = None
    def Detector(self, *args):
        return None
    def events(self):
        return iter([])
    def steps(self):
        return iter([])

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

        
        # prepare comms for running SmallData
        PS_SRV_NODES = int(os.environ.get('PS_SRV_NODES', 0))
        if PS_SRV_NODES > 0:
            self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        else:
            self.smalldata_obj = None

        self._start_prometheus_client(mpi_rank=rank)
        
        if self.comms._nodetype == 'smd0':
            smd0 = Smd0(self.comms, self._configs, self.smdr_man)
            smd0.run_mpi()
            for smd_fd in self.smd_fds:
                os.close(smd_fd)
        elif self.comms._nodetype == 'eb':
            eb_node = EventBuilderNode(self.comms, self._configs, self.dsparms)
            eb_node.run_mpi()
        elif self.comms._nodetype == 'bd':
            self.bd_node = BigDataNode(self.comms, self._configs, self.dsparms, self.dm)
            self.events  = Events(self._configs, self.dm, self.dsparms.prom_man, 
                                  filter_callback = self.dsparms.filter, 
                                  get_smd         = self.bd_node.get_smd)

    def __del__(self):
        self._end_prometheus_client(mpi_rank=self.comms.psana_comm.Get_rank())

    def runs(self):
        if self.comms._nodetype == 'bd': 
            for evt in self.events:
                if evt.service() == TransitionId.BeginRun:
                    for calib_const in self.bd_node.calib_store:
                        runnum = None
                        for det_name, det_dict in calib_const.items():
                            for det_dtype, dtype_list in calib_const[det_name].items():
                                for dtype_dict in dtype_list:
                                    if not isinstance(dtype_dict, dict): continue
                                    for field_name, field_val in dtype_dict.items():
                                        if field_name == 'run':
                                            runnum = field_val
                                            break
                                        if runnum: break
                                    if runnum: break
                                if runnum: break
                            if runnum: break
                        
                        if runnum:
                            if evt._dgrams[0].runinfo[0].runinfo.runnum == runnum:
                                self.dsparms.calibconst = calib_const
                                break

                    run = RunParallel(self.comms, evt, self.events, self._configs, self.dsparms)
                    yield run 
        else:
            run = NullRun()
            yield run

