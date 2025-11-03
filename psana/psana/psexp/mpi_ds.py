import os
import sys
import time

import numpy as np
from contextlib import contextmanager

from psana import dgram
from psana import utils
from psana.dgrammanager import DgramManager
from psana.event import Event
from psana.psexp import TransitionId
from psana.psexp.ds_base import DataSourceBase
from psana.psexp.node import BigDataNode, EventBuilderNode, Smd0
from psana.psexp.run import Run
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.step import Step
from psana.psexp.tools import mode
from psana.smalldata import SmallData
from psana.psexp.prometheus_manager import get_prom_manager

if mode == "mpi":
    from mpi4py import MPI


class InvalidEventBuilderCores(Exception):
    pass


nodetype = None


class RunParallel(Run):
    """Yields list of events from multiple smd/bigdata files using > 3 cores."""

    def __init__(self, ds, run_evt):
        super(RunParallel, self).__init__(ds)
        self.ds = ds
        self.comms = ds.comms
        self._evt = run_evt
        self.beginruns = run_evt._dgrams
        self.configs = ds._configs

        self.logger = utils.get_logger(level=ds.dsparms.log_level, logfile=ds.dsparms.log_file, name=utils.get_class_name(self))

        super()._setup_envstore()

        if nodetype == "smd0":
            self.smd0 = Smd0(ds)
        elif nodetype == "eb":
            self.eb_node = EventBuilderNode(ds)
        elif nodetype == "bd":
            self.bd_node = BigDataNode(ds, self)
            self.ana_t_gauge = get_prom_manager().get_metric("psana_bd_ana_rate")

        self._setup_run_calibconst()

    def _setup_run_calibconst(self):
        if nodetype == "smd0":
            super()._setup_run_calibconst()
        else:
            # _setup_run_calibconst runs this
            self._clear_calibconst()

        self._calib_const = self.comms.psana_comm.bcast(
            self._calib_const, root=0
        )

        # workaround for archon crash on rank0 from Gabriel
        #if nodetype != "smd0":
        #    # smd0 already did this in _setup_run_calibconst
        #    self._create_weak_calibconst()
        self._create_weak_calibconst()


    def events(self):
        evt_iter = self.start()
        st = time.time()
        for i, evt in enumerate(evt_iter):
            if not TransitionId.isEvent(evt.service()):
                continue
            yield evt
            if i % 1000 == 0:
                en = time.time()
                ana_rate = 1000 / (en - st)
                self.logger.debug(f"ANARATE {ana_rate:.2f} Hz")
                self.ana_t_gauge.set(ana_rate)
                st = time.time()

    def steps(self):
        evt_iter = self.start()
        for evt in evt_iter:
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, evt_iter)

    def start(self):
        """Request data for this run"""
        if nodetype == "smd0":
            self.smd0.start()
        elif nodetype == "eb":
            self.eb_node.start()
        elif nodetype == "bd":
            yield from self.bd_node.start()
        elif nodetype == "srv":
            return

    @contextmanager
    def build_table(self):
        """
        Context manager for building timestamp-offset table.
        Returns True only on BigDataNode if the table was successfully built.

        Requires PS_EB_NODES=1 for broadcast mode.
        """
        if os.environ.get("PS_EB_NODES", "1") != "1":
            raise RuntimeError("build_table() currently supports only PS_EB_NODES=1")

        success = False
        if nodetype == "smd0":
            self.smd0.start()
        elif nodetype == "eb":
            self.eb_node.start_broadcast()
        elif nodetype == "bd":
            self.bd_node.start_smdonly()
            success = bool(self._ts_table)
        yield success

    def event(self, ts):
        offsets = self._ts_table.get(ts)
        if offsets is None:
            raise ValueError(f"Timestamp {ts} not found in offset table.")

        dgrams = [None] * len(self.configs)
        for i, (offset, size) in offsets.items():
            buf = os.pread(self.ds.dm.fds[i], size, offset)
            dgrams[i] = dgram.Dgram(config=self.ds.dm.configs[i], view=buf)

        return Event(dgrams=dgrams, run=self)

def safe_mpi_abort(msg):
    print(msg)
    sys.stdout.flush()  # make sure error is printed
    MPI.COMM_WORLD.Abort()


class MPIDataSource(DataSourceBase):
    def __init__(self, comms, *args, **kwargs):
        # Check if an I/O-friendly numpy file storing timestamps is given by the user
        if "timestamps" in kwargs:
            if isinstance(kwargs["timestamps"], str):
                kwargs["mpi_ts"] = 1

        # Initialize base class
        super(MPIDataSource, self).__init__(**kwargs)
        self.smd_fds = None

        # Set up the MPI communication
        self.comms = comms
        comm = self.comms.psana_comm  # todo could be better
        rank = comm.Get_rank()
        size = comm.Get_size()
        global nodetype
        nodetype = self.comms.node_type()

        # prepare comms for running SmallData
        PS_SRV_NODES = int(os.environ.get("PS_SRV_NODES", 0))
        if PS_SRV_NODES > 0:
            self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        else:
            self.smalldata_obj = None

        # check if no. of ranks is enough
        nsmds = int(os.environ.get("PS_EB_NODES", 1))  # No. of smd cores
        if not (size > (nsmds + 1)):
            msg = f"""ERROR Too few MPI processes. MPI size must be more than
                   no. of all workers.
                  \n\tTotal psana size:{size}
                  \n\tPS_EB_NODES:     {nsmds}"""
            safe_mpi_abort(msg)

        # can only have 1 EventBuilder when running with destination
        if self.destination and nsmds > 1:
            msg = "ERROR Too many EventBuilder cores with destination callback"
            safe_mpi_abort(msg)

        # Load timestamp files on EventBuilder Node
        if "mpi_ts" in kwargs and nodetype == "eb":
            self.dsparms.timestamps = self.get_filter_timestamps(self.timestamps)

        # setup runnum list
        if nodetype == "smd0":
            super()._setup_runnum_list()
        else:
            self.runnum_list = None
            self.xtc_path = None
        self.runnum_list = comm.bcast(self.runnum_list, root=0)
        self.xtc_path = comm.bcast(self.xtc_path, root=0)
        self.runnum_list_index = 0

        self._start_prometheus_client(mpi_rank=rank)
        self._setup_run()

    def __del__(self):
        if nodetype == "smd0":
            super()._close_opened_smd_files()
        self._end_prometheus_client()

    def terminate(self):
        self.comms.terminate()
        super().terminate()

    def _get_configs(self):
        """Creates and broadcasts configs
        only called by _setup_run()
        """
        if nodetype == "smd0":
            super()._close_opened_smd_files()
            self.smd_fds = np.array(
                [os.open(smd_file, os.O_RDONLY) for smd_file in self.smd_files],
                dtype=np.int32,
            )
            self.logger.debug(f"smd0 opened smd_fds: {self.smd_fds}")
            self.smdr_man = SmdReaderManager(self.smd_fds, self.dsparms)
            configs = self.smdr_man.get_next_dgrams()
            nbytes = np.array(
                [memoryview(config).shape[0] for config in configs], dtype="i"
            )
        else:
            configs = None
            nbytes = np.empty(len(self.smd_files), dtype="i")

        self.comms.psana_comm.Bcast(
            nbytes, root=0
        )  # no. of bytes is required for mpich
        if nodetype != "smd0":
            configs = [np.empty(nbyte, dtype="b") for nbyte in nbytes]

        for i in range(len(configs)):
            self.comms.psana_comm.Bcast([configs[i], nbytes[i], MPI.BYTE], root=0)

        if nodetype != "smd0":
            configs = [dgram.Dgram(view=config, offset=0) for config in configs]
        return configs

    def _setup_run(self):
        if self.runnum_list_index == len(self.runnum_list):
            return False

        runnum = self.runnum_list[self.runnum_list_index]
        self.runnum_list_index += 1

        if nodetype == "smd0":
            super()._setup_run_files(runnum)
            super()._apply_detector_selection()
        else:
            self.xtc_files = None
            self.smd_files = None
            self.dsparms.use_smds = None
        self.xtc_files = self.comms.psana_comm.bcast(self.xtc_files, root=0)
        self.smd_files = self.comms.psana_comm.bcast(self.smd_files, root=0)
        self.dsparms.use_smds = self.comms.psana_comm.bcast(
            self.dsparms.use_smds, root=0
        )

        configs = self._get_configs()
        self.dm = DgramManager(
            self.xtc_files, configs=configs, config_consumers=[self.dsparms]
        )
        return True

    def _setup_beginruns(self):
        """Determines if there is a next run as
        1) New run found in the same smalldata files
        2) New run found in the new smalldata files
        """
        while True:
            if nodetype == "smd0":
                dgrams = self.smdr_man.get_next_dgrams()
                nbytes = np.zeros(len(self.smd_files), dtype="i")
                if dgrams is not None:
                    nbytes = np.array(
                        [memoryview(d).shape[0] for d in dgrams], dtype="i"
                    )
            else:
                dgrams = None
                nbytes = np.empty(len(self.smd_files), dtype="i")

            self.comms.psana_comm.Bcast(nbytes, root=0)

            if np.sum(nbytes) == 0:
                return False

            if nodetype != "smd0":
                dgrams = [np.empty(nbyte, dtype="b") for nbyte in nbytes]

            for i in range(len(dgrams)):
                self.comms.psana_comm.Bcast([dgrams[i], nbytes[i], MPI.BYTE], root=0)

            if nodetype != "smd0":
                dgrams = [
                    dgram.Dgram(view=d, config=config, offset=0)
                    for d, config in zip(dgrams, self._configs)
                ]

            if dgrams[0].service() == TransitionId.BeginRun:
                self.beginruns = dgrams
                return True
        # end while True

    def _start_run(self):
        if self._setup_beginruns():  # try to get next run from current files
            return True
        elif self._setup_run():  # try to get next run from next files
            if self._setup_beginruns():
                return True

    def runs(self):
        while self._start_run():
            run = RunParallel(self, Event(dgrams=self.beginruns))
            yield run

    def is_mpi(self):
        return True
