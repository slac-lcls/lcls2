from psana.psexp import DataSourceBase, RunDrp
from psana.dgrammanager import DgramManager, dgSize
from psana.psexp import Events, TransitionId
from psana.event import Event
from psana.smalldata import SmallData
from psana.dgramedit import DgramEdit
from psana import dgram
from psana.psexp.zmq_utils import pub_send, sub_recv

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

        self.worker_num = kwargs["drp"].worker_num
        self._is_publisher = True if self.worker_num == 0 else False

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
        self.config_dgramedit = DgramEdit(self._configs[-1], bufsize=self.dm.transition_bufsize)
        self.curr_dgramedit = self.config_dgramedit
        self.runnum_list_index += 1
        return True

    def _setup_beginruns(self):
        for evt in self.dm:
            if not evt:
                return None
            if evt.service() == TransitionId.L1Accept:
                buffer_size = self.dm.pebble_bufsize
            else:
                buffer_size = self.dm.transition_bufsize
            self.curr_dgramedit = DgramEdit(
                 evt._dgrams[0],
                 config_dgramedit=self.config_dgramedit,
                 bufsize=buffer_size
            )
            self.curr_dgramedit.save(self.dm.shm_res_mv)
            if evt.service() == TransitionId.BeginRun:
                self.beginruns = evt._dgrams
                return True

    def _setup_run_calibconst(self):
        if self._is_publisher:
            super()._setup_run_calibconst()
            pub_send(self.dsparms.calibconst)
        else:
            self.dsparms.calibconst = sub_recv()
        # Verbose print: print(f"[Python - Worker {self.worker_num}] Done broadcast {self.dsparms.calibconst}]")

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

    def add_detector(self, detdef, algdef, datadef,
                     nodeId=None, namesId=None, segment=None):
        if self._edtbl_config:
            return self.curr_dgramedit.Detector(detdef, algdef, datadef,
                                                nodeId, namesId, segment)
        else:
            raise RuntimeError(
                "[Python - Worker {self.worker_num}] Cannot edit the configuration "
                "after starting iteration over events"
            )

    def adddata(self, data):
        return self.curr_dgramedit.adddata(data)

    def removedata(self, det_name, alg):
        if not self._edtbl_config:
            return self.curr_dgramedit.removedata(det_name, alg)
        else:
            raise RuntimeError(
                "[Python - Worker {self.worker_num}] Cannot remove data from events "
                "before starting iteration over events"
            )
