from psana.psexp import DataSourceBase, RunShmem
from psana.dgrammanager import DgramManager
from psana.psexp import Events, TransitionId
from psana.event import Event
from psana.smalldata import SmallData

class ShmemDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(ShmemDataSource, self).__init__(**kwargs)
        self.tag = self.shmem
        self.runnum_list = [0] 
        self.runnum_list_index = 0
        
        self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        self._setup_run()
        super(). _start_prometheus_client()

    def __del__(self):
        super(). _end_prometheus_client()

    def _setup_run(self):
        if self.runnum_list_index == len(self.runnum_list):
            return False
        
        runnum = self.runnum_list[self.runnum_list_index]
        self.dm = DgramManager(['shmem'], tag=self.tag)
        self._configs = self.dm.configs
        super()._setup_det_class_table()
        super()._set_configinfo()
        self.runnum_list_index += 1
        return True
    
    def _setup_beginruns(self):
        try: 
            beginrun_evt = next(self.dm)
        except StopIteration:
            return False
        
        self.beginruns = beginrun_evt._dgrams 
        return True

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
            run = RunShmem(self, Event(dgrams=self.beginruns))
            yield run
    


