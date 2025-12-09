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
from psana.psexp.calib_xtc import load_calib_xtc_from_buffer
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.step import Step
from psana.psexp.tools import mode
from psana.smalldata import SmallData
from psana.psexp.prometheus_manager import get_prom_manager
from psana.psexp.calib_xtc import CalibXtcConverter

if mode == "mpi":
    from mpi4py import MPI


class InvalidEventBuilderCores(Exception):
    pass


nodetype = None


class RunParallel(Run):
    """Yields list of events from multiple smd/bigdata files using > 3 cores."""
    def __init__(self, expt, runnum, timestamp, dsparms, dm, smdr_man, configs, begingrun_dgrams, comms=None):
        super(RunParallel, self).__init__(expt, runnum, timestamp, dsparms, dm, smdr_man, begingrun_dgrams)
        self.configs = configs
        self.comms = comms

        self.logger = utils.get_logger(name=utils.get_class_name(self))

        super()._setup_envstore()

        if nodetype == "smd0":
            self.smd0 = Smd0(comms, smdr_man, configs)
        elif nodetype == "eb":
            self.eb_node = EventBuilderNode(comms, configs, dsparms)
        elif nodetype == "bd":
            self.bd_node = BigDataNode(comms, configs, dm, dsparms, self.shared_state)
            self.ana_t_gauge = get_prom_manager().get_metric("psana_bd_ana_rate")

        self._setup_run_calibconst()

    def _setup_run_calibconst(self):
        if nodetype == "smd0":
            super()._setup_run_calibconst()
            self.build_xtc_buffer(self.get_filtered_detinfo())
            if not getattr(self, "_calib_xtc_buffer", None):
                raise RuntimeError("Failed to build calibration xtc buffer on smd0")
        else:
            self._clear_calibconst()

        self._distribute_calib_xtc()

    def build_xtc_buffer(self, det_info):
        if not self._calib_const:
            self._calib_xtc_buffer = None
            return
        det_info = det_info or {}
        try:
            converter = CalibXtcConverter(det_info)
            config_bytes, data_bytes = converter.convert_to_bytes(self._calib_const)
            blob = bytearray(len(config_bytes) + len(data_bytes))
            blob[: len(config_bytes)] = config_bytes
            blob[len(config_bytes) :] = data_bytes
            self._calib_xtc_buffer = blob
            self.logger.debug(
                "Built calibration xtc buffer from pickle (%d bytes)",
                len(self._calib_xtc_buffer),
            )
        except Exception as exc:
            self.logger.warning(f"Failed to build calibration xtc buffer: {exc}")

    def _distribute_calib_xtc(self):
        node_comm = self.comms.get_node_comm()
        leader_comm = self.comms.get_node_leader_comm()
        if node_comm is None:
            raise RuntimeError("Node communicator unavailable for shared calibration distribution")

        is_leader = self.comms.is_node_leader()
        leader_buffer = None
        total_bytes = None

        if leader_comm != MPI.COMM_NULL:
            size_arr = np.array([0], dtype=np.int64) if is_leader else np.empty(1, dtype=np.int64)
            if is_leader and nodetype == "smd0":
                leader_buffer = np.frombuffer(self._calib_xtc_buffer, dtype=np.uint8)
                size_arr[0] = leader_buffer.size

            leader_comm.Bcast(size_arr, root=0)
            total_bytes = int(size_arr[0])

        total_bytes = node_comm.bcast(total_bytes, root=0)
        if total_bytes <= 0:
            if self.logger:
                self.logger.warning("RunParallel: shared xtc broadcast reported zero bytes; clearing calibration")
            self._calib_const = {}
            self.dsparms.calibconst = self._calib_const
            return

        if leader_comm != MPI.COMM_NULL and is_leader:
            if nodetype != "smd0":
                leader_buffer = np.empty(total_bytes, dtype=np.uint8)
            elif leader_buffer.size != total_bytes:
                leader_buffer = leader_buffer[:total_bytes]
            leader_comm.Bcast(leader_buffer, root=0)
            if self.logger:
                self.logger.debug(
                    f"RunParallel: node leader rank {self.comms.psana_comm.Get_rank()} received {total_bytes} bytes from smd0"
                )

        win = MPI.Win.Allocate_shared(total_bytes if is_leader else 0, 1, comm=node_comm)
        buf, _ = win.Shared_query(0)
        shared_array = np.ndarray(buffer=buf, dtype=np.uint8, shape=(total_bytes,))
        if is_leader:
            if leader_buffer is None:
                raise RuntimeError("Leader buffer missing during shared memory population")
            shared_array[:] = leader_buffer
            if self.logger:
                self.logger.debug(
                    f"RunParallel: node leader rank {self.comms.psana_comm.Get_rank()} populated shared memory ({total_bytes} bytes)"
                )
        node_comm.Barrier()

        shared_view = memoryview(shared_array)
        calib_const, owner = load_calib_xtc_from_buffer(shared_view)
        self._calib_const = calib_const
        self._calib_xtc_buffer = owner
        self._calib_xtc_shared = shared_array
        self._calib_xtc_win = win
        self.dsparms.calibconst = self._calib_const
        if self.logger:
            path = "shared memory leader" if is_leader else "shared memory follower"
            self.logger.debug(
                f"RunParallel: rank {self.comms.psana_comm.Get_rank()} loaded calibration via {path}"
            )

    def events(self):
        evt_iter = self.start()
        st = time.time()
        for i, dgrams in enumerate(evt_iter):
            if self._handle_transition(dgrams):
                continue  # swallow non-L1 transitions in events() stream
            # L1Accept: construct Event at the Run level
            yield Event(dgrams=dgrams, run=self._run_ctx)
            if i % 1000 == 0:
                en = time.time()
                ana_rate = 1000 / (en - st)
                self.logger.debug(f"ANARATE {ana_rate:.2f} Hz")
                self.ana_t_gauge.set(ana_rate)
                st = time.time()

    def steps(self):
        evt_iter = self.start()
        for dgrams in evt_iter:
            svc = utils.first_service(dgrams)
            if TransitionId.isEvent(svc):
                # steps() only yields on BeginStep transitions; ignore L1
                continue
            # Update envstore for every transition
            self._update_envstore_from_dgrams(dgrams)

            if svc == TransitionId.BeginStep:
                yield Step(Event(dgrams=dgrams, run=self._run_ctx), evt_iter, self._run_ctx)

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
            self._ts_table = self.bd_node.start_smdonly()
            success = bool(self._ts_table)
        yield success

    def event(self, ts):
        offsets = self._ts_table.get(ts)
        if offsets is None:
            raise ValueError(f"Timestamp {ts} not found in offset table.")

        dgrams = [None] * len(self.configs)
        for i, (offset, size) in offsets.items():
            buf = os.pread(self.dm.fds[i], size, offset)
            dgrams[i] = dgram.Dgram(config=self.dm.configs[i], view=buf)

        return Event(dgrams=dgrams, run=self)

    def terminate(self):
        self.comms.terminate()
        super().terminate()


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
            self.smdr_man = None  # Only smd0 uses SmdReaderManager
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
            # Pull (expt, runnum, ts) from the BeginRun dgrams
            expt, runnum, ts = self._get_runinfo()
            run = RunParallel(
                expt,                 # experiment string
                runnum,               # run number (int)
                ts,                   # begin-run timestamp
                self.dsparms,         # shared parameters / config tables
                self.dm,              # DgramManager
                self.smdr_man,        # SmdReaderManager (may be None for non-SMD modes)
                self._configs,        # configs for this run
                self.beginruns,       # beginrun dgrams
                comms=self.comms,     # MPI communications
            )
            yield run

    def is_mpi(self):
        return True
