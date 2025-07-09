from psana2.dgramedit import DgramEdit
from psana2.dgrammanager import DgramManager
from psana2.event import Event
from psana2.psexp import TransitionId
from psana2.psexp.ds_base import DataSourceBase
from psana2.psexp.run import RunDrp
from psana2.psexp.zmq_utils import PubSocket, SubSocket
from psana2.smalldata import SmallData


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

        self.det_segment = kwargs["drp"].det_segment
        self._is_supervisor = kwargs["drp"].is_supervisor
        self.worker_num = kwargs["drp"].worker_num
        self._is_publisher = True if self.worker_num == 0 else False

        if self._is_supervisor:
            if self._is_publisher:
                self._tcp_pub_socket = PubSocket(kwargs["drp"].tcp_socket_name)
                self._ipc_pub_socket = PubSocket(kwargs["drp"].ipc_socket_name)
            else:
                self._ipc_sub_socket = SubSocket(kwargs["drp"].ipc_socket_name)
        else:
            if self._is_publisher:
                self._tcp_sub_socket = SubSocket(kwargs["drp"].tcp_socket_name)
                self._ipc_pub_socket = PubSocket(kwargs["drp"].ipc_socket_name)
            else:
                self._ipc_sub_socket = SubSocket(kwargs["drp"].ipc_socket_name)

        self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        self._setup_run()
        super()._start_prometheus_client(
            prom_cfg_dir=kwargs["drp"].prom_cfg_dir
        )  # Use http exporter

    def __del__(self):
        super()._end_prometheus_client()

    def _setup_run(self):
        if self.runnum_list_index == len(self.runnum_list):
            return False
        self.dm = DgramManager(["drp"], tag=self.tag, config_consumers=[self.dsparms])
        self.config_dgramedit = DgramEdit(
            self._configs[-1], bufsize=self.dm.transition_bufsize
        )
        self.curr_dgramedit = self.config_dgramedit
        self.runnum_list_index += 1
        return True

    def _setup_beginruns(self):
        for evt in self.dm:
            if not evt:
                return None
            if TransitionId.isEvent(evt.service()):
                buffer_size = self.dm.pebble_bufsize
            else:
                buffer_size = self.dm.transition_bufsize
            self.curr_dgramedit = DgramEdit(
                evt._dgrams[0],
                config_dgramedit=self.config_dgramedit,
                bufsize=buffer_size,
            )
            self.curr_dgramedit.save(self.dm.shm_res_mv)
            if evt.service() == TransitionId.BeginRun:
                self.beginruns = evt._dgrams
                return True

    def _setup_run_calibconst(self):
        if self._is_supervisor:
            if self._is_publisher:
                super()._setup_run_calibconst()
                self._tcp_pub_socket.send(self.dsparms.calibconst)
                # This is for drp_python
                self._ipc_pub_socket.send(self.dsparms.calibconst)
            else:
                self.dsparms.calibconst = self._ipc_sub_socket.recv()
        else:
            if self._is_publisher:
                self.dsparms.calibconst = self._tcp_sub_socket.recv()
                self._ipc_pub_socket.send(self.dsparms.calibconst)
            else:
                self.dsparms.calibconst = self._ipc_sub_socket.recv()

        # Verbose print: print(f"[Python - Worker {self.worker_num}] Done broadcast {self.dsparms.calibconst}]")

    def _start_run(self):
        found_next_run = False
        if self._setup_beginruns():  # try to get next run from the current file
            self._setup_run_calibconst()
            found_next_run = True
        elif self._setup_run():  # try to get next run from next files
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

        # Delete sockets to avoid getting 'Address already in use' when recreating them
        # Its unclear why this doesn't work when done in __del__()
        if self._is_supervisor:
            if self._is_publisher:
                del self._tcp_pub_socket
                del self._ipc_pub_socket
            else:
                del self._ipc_sub_socket
        else:
            if self._is_publisher:
                del self._tcp_sub_socket
                del self._ipc_pub_socket
            else:
                del self._ipc_sub_socket

    def is_mpi(self):
        return False

    def add_detector(
        self, detdef, algdef, datadef, nodeId=None, namesId=None, segment=None
    ):
        if self._edtbl_config:
            return self.curr_dgramedit.Detector(
                detdef, algdef, datadef, nodeId, namesId, segment
            )
        else:
            raise RuntimeError(
                "[Python - Worker {self.worker_num}] Cannot edit the configuration "
                "after starting iteration over events"
            )

    def add_data(self, data):
        return self.curr_dgramedit.adddata(data)

    def remove_data(self, det_name, alg):
        if not self._edtbl_config:
            return self.curr_dgramedit.removedata(det_name, alg)
        else:
            raise RuntimeError(
                "[Python - Worker {self.worker_num}] Cannot remove data from events "
                "before starting iteration over events"
            )
