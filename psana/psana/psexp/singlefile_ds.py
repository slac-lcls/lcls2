from psana.psexp import DataSourceBase, RunSingleFile
from psana.psexp import Events, TransitionId
from psana.dgrammanager import DgramManager
from psana.event import Event

class SingleFileDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(SingleFileDataSource, self).__init__(**kwargs)
        self.runnum_list = list(range(len(self.files))) 
        self.runnum_list_index = 0
        self._setup_run()
        super(). _start_prometheus_client()

    def __del__(self):
        super(). _end_prometheus_client()

    def _setup_run(self):
        if self.runnum_list_index == len(self.runnum_list):
            return False
        
        runnum = self.runnum_list[self.runnum_list_index]
        self.dm = DgramManager(self.files[self.runnum_list_index])
        self._configs = self.dm.configs
        super()._setup_det_class_table()
        super()._set_configinfo()
        self.runnum_list_index += 1
        return True
    
    def _setup_beginruns(self):
        for evt in self.dm:
            if evt.service() == TransitionId.BeginRun:
                self.beginruns = evt._dgrams
                return True
        return False

    def _start_run(self):
        found_next_run = False
        if self._setup_beginruns():   # try to get next run from the current file
            super()._setup_run_calibconst()
            found_next_run = True
        elif self._setup_run():       # try to get next run from next files 
            if self._setup_beginruns():
                super()._setup_run_calibconst()
                found_next_run = True
        return found_next_run

    def runs(self):
        while self._start_run():
            run = RunSingleFile(self, Event(dgrams=self.beginruns))
            yield run
