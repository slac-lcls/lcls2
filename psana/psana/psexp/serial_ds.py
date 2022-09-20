from psana.psexp import RunSerial, DataSourceBase
from psana.psexp import Events, TransitionId, SmdReaderManager
from psana.event import Event
from psana.dgrammanager import DgramManager
import numpy as np
import os
from psana.smalldata import SmallData

import logging
logger = logging.getLogger(__name__)

class SerialDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(SerialDataSource, self).__init__(**kwargs)
        super()._setup_runnum_list()
        self.runnum_list_index = 0
        self.smd_fds = None
        
        self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        self._setup_run()
        super()._start_prometheus_client()

    def __del__(self):
        super()._close_opened_smd_files()
        super()._end_prometheus_client()

    def _get_configs(self):
        super()._close_opened_smd_files()
        self.smd_fds  = np.array([os.open(smd_file, os.O_RDONLY) for smd_file in self.smd_files], dtype=np.int32)
        logger.debug(f'serial_ds: opened smd_fds: {self.smd_fds}')
        self.smdr_man = SmdReaderManager(self.smd_fds, self.dsparms)
        # Reading configs (first dgram of the smd files)
        return self.smdr_man.get_next_dgrams()
    
    def _setup_run(self):
        if self.runnum_list_index == len(self.runnum_list):
            return False
        
        runnum = self.runnum_list[self.runnum_list_index]
        self.runnum_list_index += 1
        super()._setup_run_files(runnum)
        super()._apply_detector_selection()
        configs = self._get_configs()
        self.dm = DgramManager(self.xtc_files, configs=configs, config_consumers=[self.dsparms])
        return True

    def _setup_beginruns(self):
        """ Determines if there is a next run as
        1) New run found in the same smalldata files
        2) New run found in the new smalldata files
        """
        dgrams = self.smdr_man.get_next_dgrams() 
        while dgrams is not None:
            if dgrams[0].service() == TransitionId.BeginRun:
                self.beginruns = dgrams
                return True
            dgrams = self.smdr_man.get_next_dgrams() 
        return False

    def _start_run(self):
        found_next_run = False
        if self._setup_beginruns():   # try to get next run from current files
            super()._setup_run_calibconst()
            found_next_run = True
        elif self._setup_run():       # try to get next run from next files 
            if self._setup_beginruns():
                super()._setup_run_calibconst()
                found_next_run = True
        return found_next_run

    def runs(self):
        while self._start_run():
            run = RunSerial(self, Event(dgrams=self.beginruns))
            yield run

    def is_mpi(self):
        return False
