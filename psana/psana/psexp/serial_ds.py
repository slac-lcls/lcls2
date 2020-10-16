from psana.psexp import RunSerial, DataSourceBase
from psana.psexp import Events, TransitionId, SmdReaderManager
from psana.dgrammanager import DgramManager
import numpy as np
import os

class SerialDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(SerialDataSource, self).__init__(**kwargs)
        self._setup_xtcs()
        self.smd_fds  = np.array([os.open(smd_file, os.O_RDONLY) for smd_file in self.smd_files], dtype=np.int32)
        self.smdr_man = SmdReaderManager(self.smd_fds, self.dsparms)
        self._configs = self.smdr_man.get_next_dgrams()
        self.smdr_man.set_configs(self._configs)
        self._setup_det_class_table()
        self._set_configinfo()
        self._start_prometheus_client()
        self.dm = DgramManager(self.xtc_files, configs=self._configs)
    
    def runs(self):
        events = Events(self._configs, self.dm, self.prom_man, filter_callback=self.filter, smdr_man=self.smdr_man)
        for evt in events:
            if evt.service() == TransitionId.BeginRun:
                run = RunSerial(evt, events, self.dsparms)
                yield run 
        
        self._end_prometheus_client()

        for smd_fd in self.smd_fds:
            os.close(smd_fd)

