from psana.psexp import DataSourceBase, RunDrp
from psana.dgrammanager import DgramManager, dgSize
from psana.psexp import Events, TransitionId
from psana.event import Event
from psana.smalldata import SmallData
from psana.dgramedit import DgramEdit, PyDgram
from psana import dgram
from psdaq.configdb.pub_server import pub_send
from psdaq.configdb.sub_client import sub_recv

import sys
worker_num = int(sys.argv[7])
is_publisher = True if worker_num == 0 else False

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
        self.config_dgramedit = DgramEdit(
            PyDgram(self._configs[-1].get_dgram_ptr(), self.dm.transition_bufsize)
        )
        self.curr_dgramedit = self.config_dgramedit
        self.runnum_list_index += 1
        return True
    
    def _setup_beginruns(self):
        for evt in self.dm:
            if evt.service() == TransitionId.L1Accept:
                buffer_size = self.dm.pebble_bufsize
            else:
                buffer_size = self.dm.transition_bufsize
            self.curr_dgramedit = DgramEdit(
                PyDgram(evt._dgrams[0].get_dgram_ptr(), buffer_size),
                config=self.config_dgramedit
            )
            self.curr_dgramedit.save(self.dm.shm_res_mv)
            if evt.service() == TransitionId.BeginRun:
                self.beginruns = evt._dgrams
                return True
        return False
    
    def _setup_run_calibconst(self):
        if is_publisher:
            super()._setup_run_calibconst()
            pub_send(self.dsparms.calibconst)
        else: 
            self.dsparms.calibconst = sub_recv()
        print(f"[Python - Thread {worker_num} done broadcast {self.dsparms.calibconst}]")

    def _start_run(self):
        found_next_run = False
        if self._setup_beginruns():   # try to get next run from the current file 
            self._setup_run_calibconst()
            found_next_run = True
        elif self._setup_run():       # try to get next run from next files 
            if self._setup_beginruns():
                self._setup_run_calibconst()
                found_next_run = True
        return found_next_run

    def runs(self):
        self._edtbl_config = False
        self.curr_dgramedit.save(self.dm.shm_res_mv)
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
