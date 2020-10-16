from psana.psexp import DataSourceBase, RunShmem
from psana.dgrammanager import DgramManager
from psana.psexp import Events, TransitionId

class ShmemDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(ShmemDataSource, self).__init__(**kwargs)
        self._setup_xtcs()
        self.dm = DgramManager(['shmem'], tag=self.tag)
        self._configs = self.dm.configs
        self._setup_det_class_table()
        self._set_configinfo()
    
    def runs(self):
        events = Events(self._configs, self.dm, self.prom_man, filter_callback=self.filter)
        for evt in events:
            if evt.service() == TransitionId.BeginRun:
                run = RunShmem(evt, events, self.dsparms)
                yield run 


