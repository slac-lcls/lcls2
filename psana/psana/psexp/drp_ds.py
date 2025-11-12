from psana.dgrammanager import DgramManager
from psana.psexp import TransitionId
from psana.psexp.ds_base import DataSourceBase
from psana.psexp.run import RunDrp
from psana.psexp.zmq_utils import PubSocket, SubSocket
from psana.smalldata import SmallData
from psana import utils


class DrpDataSource(DataSourceBase):
    def __init__(self, *args, **kwargs):
        super(DrpDataSource, self).__init__(**kwargs)
        self.tag = self.drp
        self.runnum_list = [0]
        self.dsparms.update_smd_state([None], [False] * len(self.runnum_list))  # disable SMDs
        self.runnum_list_index = 0

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
        self.runnum_list_index += 1
        return True

    def _setup_beginruns(self):
        for dgrams in self.dm:
            if not dgrams:
                return False
            service = utils.first_service(dgrams)
            if service == TransitionId.BeginRun:
                self.beginruns = dgrams
                return True

    def _start_run(self):
        found_next_run = False
        if self._setup_beginruns():  # try to get next run from the current file
            found_next_run = True
        elif self._setup_run():  # try to get next run from next files
            if self._setup_beginruns():
                found_next_run = True
        return found_next_run

    def runs(self):
        while self._start_run():
            # Extra kwargs for RunDrp
            kwargs = {'drp_is_supervisor': self._is_supervisor,
                      'drp_is_publisher': self._is_publisher,
                      'drp_tcp_pub_socket': self._tcp_pub_socket if self._is_supervisor and self._is_publisher else None,
                      'drp_ipc_pub_socket': self._ipc_pub_socket if self._is_publisher else None,}
            # Pull (expt, runnum, ts) from the BeginRun dgrams
            expt, runnum, ts = self._get_runinfo()
            run = RunDrp(
                expt,                 # experiment string
                runnum,               # run number (int)
                ts,                   # begin-run timestamp
                self.dsparms,         # shared parameters / config tables
                self.dm,              # DgramManager
                None,                 # SmdReaderManager (None RunSingleFile & RunShmem)
                self._configs,        # configs for this run
                self.beginruns,       # beginrun dgrams
                **kwargs              # extra drp args
            )
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
