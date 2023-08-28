from psana.psexp import DataSourceBase, RunShmem
from psana.dgrammanager import DgramManager
from psana.psexp import Events, TransitionId
from psana.event import Event
from psana.smalldata import SmallData
from psana.psexp.zmq_utils import PubSocket, SubSocket

class ShmemDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(ShmemDataSource, self).__init__(**kwargs)
        self.tag = self.shmem
        self.runnum_list = [0]
        self.runnum_list_index = 0

        # Setup socket for calibration constant broadcast if supervisor
        # is set (1=I am supervisor, 0=I am not supervisor).
        self.supervisor = -1
        if 'supervisor' in kwargs:
            self.supervisor = kwargs['supervisor']
            socket_name = f"tcp://{kwargs['supervisor_ip_addr']}"
            if self.supervisor == 1:
                self._pub_socket = PubSocket(socket_name)
            else:
                self._sub_socket = SubSocket(socket_name)

        self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        self._setup_run()
        super(). _start_prometheus_client()

    def __del__(self):
        super(). _end_prometheus_client()

    def _setup_run(self):
        if self.runnum_list_index == len(self.runnum_list):
            return False

        runnum = self.runnum_list[self.runnum_list_index]
        self.dm = DgramManager(['shmem'], tag=self.tag, config_consumers=[self.dsparms])
        self.runnum_list_index += 1
        return True

    def _setup_beginruns(self):
        for evt in self.dm:
            if evt.service() == TransitionId.BeginRun:
                self.beginruns = evt._dgrams
                return True
        return False

    def _setup_run_calibconst(self):
        if self.supervisor:
            super()._setup_run_calibconst()
            if self.supervisor == 1:
                self._pub_socket.send(self.dsparms.calibconst)

        else:
            self.dsparms.calibconst = self._sub_socket.recv()

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
        while self._start_run():
            run = RunShmem(self, Event(dgrams=self.beginruns))
            yield run

    def is_mpi(self):
        return False



