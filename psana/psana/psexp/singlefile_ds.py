from psana.psexp import DataSourceBase, RunSingleFile
from psana.psexp import Events, TransitionId
from psana.dgrammanager import DgramManager

class SingleFileDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(SingleFileDataSource, self).__init__(**kwargs)
        self._setup_xtcs()
        self.dm = DgramManager(self.files)
        self._configs = self.dm.configs
        self._setup_det_class_table()
        self._set_configinfo()
    
    def runs(self):
        events = Events(self._configs, self.dm, self.prom_man, filter_callback=self.filter)
        for evt in events:
            if evt.service() == TransitionId.BeginRun:
                run = RunSingleFile(evt, events, self.dsparms)
                yield run 


