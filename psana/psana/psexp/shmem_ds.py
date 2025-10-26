
from psana import utils
from psana.dgrammanager import DgramManager
from psana.event import Event
from psana.psexp import TransitionId
from psana.psexp.ds_base import DataSourceBase
from psana.psexp.run import RunShmem
from psana.psexp.zmq_utils import PubSocket, SubSocket
from psana.smalldata import SmallData


class ShmemDataSource(DataSourceBase):
    def __init__(self, *args, **kwargs):
        super(ShmemDataSource, self).__init__(**kwargs)
        self.tag = self.shmem
        self.runnum_list = [0]
        self.runnum_list_index = 0

        self.logger = utils.get_logger(name=utils.get_class_name(self))

        # Setup socket for calibration constant broadcast if supervisor
        # is set (1=I am supervisor, 0=I am not supervisor).
        self.supervisor = -1
        if "supervisor" in kwargs:
            self.supervisor = kwargs["supervisor"]
            socket_name = f"tcp://{kwargs['supervisor_ip_addr']}"
            if self.supervisor == 1:
                self._pub_socket = PubSocket(socket_name)
            else:
                self._sub_socket = SubSocket(socket_name)

        self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        self._setup_run()
        super()._start_prometheus_client()

    def __del__(self):
        super()._end_prometheus_client()

    def _setup_run(self):
        if self.runnum_list_index == len(self.runnum_list):
            return False

        self.dm = DgramManager(["shmem"], tag=self.tag, config_consumers=[self.dsparms])
        self.runnum_list_index += 1
        return True

    def _setup_beginruns(self):
        for dgrams in self.dm:
            if utils.first_service(dgrams) == TransitionId.BeginRun:
                self.beginruns = dgrams
                return True
        return False

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
            kwargs = {'shmem_supervisor': self.supervisor,
                      'shmem_pub_socket': self._pub_socket if self.supervisor == 1 else None,
                      'shmem_sub_socket': self._sub_socket if self.supervisor == 0 else None}
            run = RunShmem(self, Event(dgrams=self.beginruns), **kwargs)
            yield run

    def is_mpi(self):
        return False
