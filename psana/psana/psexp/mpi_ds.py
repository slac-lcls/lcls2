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
from psana.psexp.node import (
    BigDataNode,
    EventBuilderNode,
    MarchingBigDataNode,
    MarchingEventBuilderNode,
    Smd0,
)
from psana.psexp.run import Run
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.step import Step
from psana.psexp.tools import mode
from psana.psexp.marching_shmem import MarchingSharedMemory
from psana.smalldata import SmallData
from psana.psexp.prometheus_manager import get_prom_manager

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

        self._init_marching_shared_buffers()

        marching_enabled = (
            self.dsparms.marching_read
            and getattr(self.comms, "marching_enabled", False)
            and getattr(self.comms, "march_shared_mem", None) is not None
        )

        if nodetype == "smd0":
            self.smd0 = Smd0(comms, smdr_man, configs)
        elif nodetype == "eb":
            if marching_enabled:
                self.eb_node = MarchingEventBuilderNode(comms, configs, dsparms)
            else:
                self.eb_node = EventBuilderNode(comms, configs, dsparms)
        elif nodetype == "bd":
            if marching_enabled:
                self.bd_node = MarchingBigDataNode(
                    comms, configs, dm, dsparms, self.shared_state
                )
            else:
                self.bd_node = BigDataNode(
                    comms, configs, dm, dsparms, self.shared_state
                )
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
        self.dsparms.calibconst = self._calib_const

    def _init_marching_shared_buffers(self):
        if not (
            self.dsparms.marching_read
            and getattr(self.comms, "marching_enabled", False)
            and getattr(self.comms, "march_shm_comm", None) not in (None, MPI.COMM_NULL)
        ):
            return
        if getattr(self.comms, "march_shared_mem", None) is not None:
            return
        shared_mem = MarchingSharedMemory(shm_comm=self.comms.march_shm_comm)
        params = self._marching_params()
        slot_shape = (params["n_slots"], params["max_events"], len(self.configs))
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_bd_offsets", slot_shape, np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_bd_sizes", slot_shape, np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_smd_offsets", slot_shape, np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_smd_sizes", slot_shape, np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_services", slot_shape, np.int32)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_cutoff_flags", slot_shape, np.int8)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_new_chunk_ids", slot_shape, np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_slot_events", (params["n_slots"],), np.int32)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_slot_chunk_ids", (params["n_slots"],), np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_slot_states", (params["n_slots"],), np.int32)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_slot_next_evt", (params["n_slots"],), np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_slot_consumers_done", (params["n_slots"],), np.int32)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_chunk_sizes", (params["n_slots"],), np.int32)
        self._alloc_marching_array(
            shared_mem,
            f"{params['prefix']}_chunk_bytes",
            (params["n_slots"], params["max_chunk_bytes"]),
            np.uint8,
        )
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_shutdown", (1,), np.int32)
        self.comms.march_shared_mem = shared_mem
        self.comms.march_params = params

    def _alloc_marching_array(self, shared_mem, name, shape, dtype):
        if not shared_mem.has_array(name):
            shared_mem.allocate_array(name, shape, dtype)

    def _marching_params(self):
        prefix = os.environ.get("PS_MARCH_PREFIX", "march")
        n_slots = int(os.environ.get("PS_MARCH_SLOTS", "2"))
        n_slots = max(n_slots, 1)
        default_max_events = int(os.environ.get("PS_SMD_N_EVENTS", "20000"))
        env_max_events = os.environ.get("PS_MARCH_MAX_EVENTS")
        if env_max_events is not None:
            max_events = min(int(env_max_events), default_max_events)
        else:
            max_events = default_max_events
        max_events = max(max_events, 1)
        max_chunk_bytes = int(os.environ.get("PS_MARCH_MAX_CHUNK_MB", "32"))
        max_chunk_bytes = max_chunk_bytes * 1024 * 1024
        return {
            "prefix": prefix,
            "n_slots": n_slots,
            "max_events": max_events,
            "max_chunk_bytes": max_chunk_bytes,
        }

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
                yield Step(
                    Event(dgrams=dgrams, run=self._run_ctx),
                    evt_iter,
                    self._run_ctx,
                    esm=self.esm,
                )

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
        if self.dsparms.marching_read:
            raise RuntimeError("build_table() is not supported in marching mode")
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
