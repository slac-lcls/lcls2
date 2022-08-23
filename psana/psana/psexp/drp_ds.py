from psana.psexp import DataSourceBase, RunDrp
from psana.dgrammanager import DgramManager, dgSize
from psana.psexp import Events, TransitionId
from psana.event import Event
from psana.smalldata import SmallData
from psana.dgramedit import DgramEdit, PyDgram
from psana import dgram
class DrpDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(DrpDataSource, self).__init__(**kwargs)
        self.tag = self.drp
        self.runnum_list = [0] 
        self.runnum_list_index = 0
        self.curr_dgramedit = None
        self.config_dgramedit = None
        self._ed_detectors = []
        self._ed_detectors_handles = {}
        self._edtbl_config = True
        self._config_ts = None

        self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        self._setup_run()
        super(). _start_prometheus_client()

    def __del__(self):
        super(). _end_prometheus_client()

    def _setup_run(self):
        if self.runnum_list_index == len(self.runnum_list):
            return False
        
        runnum = self.runnum_list[self.runnum_list_index]
        self.dm = DgramManager(['drp'], tag=self.tag, config_consumers=[self.dsparms])
        self.curr_dgramedit = DgramEdit(
            PyDgram(self._configs[-1].get_dgram_ptr(), self.dm.shm_size)
        )
        self.runnum_list_index += 1
        return True
    
    def _setup_beginruns(self):
        for evt in self.dm:
            self.curr_dgramedit = DgramEdit(
                PyDgram(evt._dgrams[0].get_dgram_ptr(), self.dm.shm_size),
                config=self.config_dgramedit
            )
            self.curr_dgramedit.save(self.dm.shm_send)
            print("DEBUG: Saving")
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
        self._edtbl_config = False
        self.curr_dgramedit.save(self.dm.shm_send)
        self.config_dgramedit = self.curr_dgramedit
        while self._start_run():
            run = RunDrp(self, Event(dgrams=self.beginruns))
            yield run

    def is_mpi(self):
        return False

    def add_detector(self, detdef, algdef, datadef):
        if self._edtbl_config:
            return self.curr_dgramedit.Detector(detdef, algdef, datadef)
        else:
            raise RuntimeError(
                "Cannot edit the configuration after starting iteration over events"
            )

    def adddata(self, data):
        if not self._edtbl_config:
            return self.curr_dgramedit.adddata(data)
        else:
            raise RuntimeError(
                "Cannot add data to events before starting iteration over events"
            )

    def removedata(self, det_name, alg):
        if not self._edtbl_config:
            return self.curr_dgramedit.removedata(det_name, alg)
        else:
            raise RuntimeError(
                "Cannot remove data from events before starting iteration over events"
            )