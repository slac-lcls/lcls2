import sys
import os
import numpy as np
from mpi4py import MPI
from psana.psexp import *
from psana import dgram
from psana.event import Event
from psana.dgrammanager import DgramManager
from psana.smalldata import SmallData
import time
from psana.psexp.tools import Logging as logging

class InvalidEventBuilderCores(Exception): pass

nodetype = None
comm = MPI.COMM_WORLD

class RunParallel(Run):
    """ Yields list of events from multiple smd/bigdata files using > 3 cores."""

    def __init__(self, ds, run_evt):
        super(RunParallel, self).__init__(ds.dsparms)
        self.ds         = ds
        self.comms      = ds.comms
        self._evt       = run_evt
        self.beginruns  = run_evt._dgrams
        self.configs    = ds._configs
        
        self._get_runinfo()
        self.esm = EnvStoreManager(self.configs)

    def events(self):
        evt_iter = self.start()
        for evt in evt_iter:
            if evt.service() != TransitionId.L1Accept:
                self.esm.update_by_event(evt)
                continue
            st = time.time()
            yield evt
            en = time.time()
            self.c_ana.labels('seconds','None').inc(en-st)
            self.c_ana.labels('batches','None').inc()
        
    def steps(self):
        evt_iter = self.start()
        for evt in evt_iter:
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, evt_iter, self.esm)

    def start(self):
        """ Request data for this run"""
        if nodetype == 'smd0':
            self.ds.smd0.start()
        elif nodetype == 'eb':
            self.ds.eb_node.start()
        elif nodetype == 'bd':
            for evt in self.ds.bd_node.start():
                yield evt
        elif nodetype == 'srv':
            return
    
def safe_mpi_abort(msg):
    print(msg)
    sys.stdout.flush() # make sure error is printed
    MPI.COMM_WORLD.Abort()

class MPIDataSource(DataSourceBase):

    def __init__(self, comms, *args, **kwargs):
        super(MPIDataSource, self).__init__(**kwargs)

        self.comms = comms
        comm = self.comms.psana_comm # todo could be better
        rank = comm.Get_rank()
        size = comm.Get_size()
        global nodetype
        nodetype = self.comms.node_type()

        self.smd_fds = None
        
        # prepare comms for running SmallData
        PS_SRV_NODES = int(os.environ.get('PS_SRV_NODES', 0))
        if PS_SRV_NODES > 0:
            self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        else:
            self.smalldata_obj = None
        
        # check if no. of ranks is enough
        nsmds = int(os.environ.get('PS_EB_NODES', 1)) # No. of smd cores
        if not (size > (nsmds + 1)):
            msg = f"""ERROR Too few MPI processes. MPI size must be more than 
                   no. of all workers. 
                  \n\tTotal psana size:{size}
                  \n\tPS_EB_NODES:     {nsmds}"""
            safe_mpi_abort(msg)
        
        # can only have 1 EventBuilder when running with destination
        if self.destination and nsmds > 1:
            msg = 'ERROR Too many EventBuilder cores with destination callback'
            safe_mpi_abort(msg)
        
        # setup runnum list
        if nodetype == 'smd0':
            super()._setup_runnum_list()
        else:
            self.runnum_list= None
            self.xtc_path   = None
        self.runnum_list = comm.bcast(self.runnum_list, root=0)
        self.xtc_path    = comm.bcast(self.xtc_path, root=0)
        self.runnum_list_index = 0

        self._start_prometheus_client(mpi_rank=rank)
        self._setup_run()

    def __del__(self):
        if nodetype == 'smd0':
            super()._close_opened_smd_files()
        self._end_prometheus_client(mpi_rank=self.comms.psana_comm.Get_rank())

    def _setup_configs(self):
        """ Creates and broadcasts configs
        only called by _setup_run()
        """
        g_ts = self.prom_man.get_metric("psana_timestamp")
        if nodetype == 'smd0':
            super()._close_opened_smd_files()
            self.smd_fds  = np.array([os.open(smd_file, os.O_RDONLY) for smd_file in self.smd_files], dtype=np.int32)
            logging.info(f'mpi_ds: smd0 opened smd_fds: {self.smd_fds}')
            self.smdr_man = SmdReaderManager(self.smd_fds, self.dsparms)
            self._configs = self.smdr_man.get_next_dgrams()
            super()._setup_det_class_table()
            super()._set_configinfo()
            g_ts.labels("first_event").set(time.time())
            nbytes = np.array([memoryview(config).shape[0] for config in self._configs], \
                    dtype='i')
        else:
            self._configs = None
            nbytes = np.empty(len(self.smd_files), dtype='i')
        
        self.comms.psana_comm.Bcast(nbytes, root=0) # no. of bytes is required for mpich
        if nodetype != 'smd0':
            self._configs = [np.empty(nbyte, dtype='b') for nbyte in nbytes]
        
        for i in range(len(self._configs)):
            self.comms.psana_comm.Bcast([self._configs[i], nbytes[i], MPI.BYTE], root=0)
        
        if nodetype != 'smd0':
            self._configs = [dgram.Dgram(view=config, offset=0) for config in self._configs]
            g_ts.labels("first_event").set(time.time())
            self._setup_det_class_table()
            self._set_configinfo()

    def _setup_run(self):
        if self.runnum_list_index == len(self.runnum_list):
            return False

        runnum = self.runnum_list[self.runnum_list_index]
        self.runnum_list_index += 1
        
        if nodetype == 'smd0':
            super()._setup_run_files(runnum)
            super()._apply_detector_selection()
        else:
            self.xtc_files = None
            self.smd_files = None
        self.xtc_files = self.comms.psana_comm.bcast(self.xtc_files, root=0)
        self.smd_files = self.comms.psana_comm.bcast(self.smd_files, root=0)
        
        self._setup_configs()
        self.dm = DgramManager(self.xtc_files, configs=self._configs)
        
        if nodetype == 'smd0':
            self.smd0 = Smd0(self.comms, self._configs, self.smdr_man, self.dsparms)
        elif nodetype == 'eb':
            self.eb_node = EventBuilderNode(self.comms, self._configs, self.dsparms)
        elif nodetype == 'bd':
            self.bd_node = BigDataNode(self.comms, self._configs, self.dsparms, self.dm)

        return True

    def _setup_beginruns(self):
        """ Determines if there is a next run as
        1) New run found in the same smalldata files
        2) New run found in the new smalldata files
        """
        while True:
            if nodetype == 'smd0':
                dgrams = self.smdr_man.get_next_dgrams() 
                nbytes = np.zeros(len(self.smd_files), dtype='i')
                if dgrams is not None:
                    nbytes = np.array([memoryview(d).shape[0] for d in dgrams], dtype='i')
            else:
                dgrams = None
                nbytes = np.empty(len(self.smd_files), dtype='i')
            
            self.comms.psana_comm.Bcast(nbytes, root=0) 

            if np.sum(nbytes) == 0: return False

            if nodetype != 'smd0':
                dgrams = [np.empty(nbyte, dtype='b') for nbyte in nbytes]
            
            for i in range(len(dgrams)):
                self.comms.psana_comm.Bcast([dgrams[i], nbytes[i], MPI.BYTE], root=0)
            
            if nodetype != 'smd0':
                dgrams = [dgram.Dgram(view=d, config=config, offset=0) \
                        for d, config in zip(dgrams,self._configs)]

            if dgrams[0].service() == TransitionId.BeginRun:
                self.beginruns = dgrams
                return True
        # end while True

    def _setup_run_calibconst(self):
        if nodetype == 'smd0':
            super()._setup_run_calibconst()
        else: 
            self.dsparms.calibconst = None

        self.dsparms.calibconst = self.comms.psana_comm.bcast(self.dsparms.calibconst, root=0)

    def _start_run(self):
        if self._setup_beginruns():   # try to get next run from current files
            self._setup_run_calibconst()
            return True
        elif self._setup_run():       # try to get next run from next files 
            if self._setup_beginruns():
                self._setup_run_calibconst()
                return True

    def runs(self):
        while self._start_run():
            run = RunParallel(self, Event(dgrams=self.beginruns))
            yield run

