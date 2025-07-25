import abc
import gc
import glob
import json
import logging
import os
import re
import socket
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from kafka import KafkaProducer

import psana.pscalib.calib.MDBWebUtils as wu
from psana import utils
from psana.app.psplot_live.utils import MonitorMsgType
from psana.pscalib.app.calib_prefetch import calib_utils
from psana.psexp.prometheus_manager import PrometheusManager
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.tools import MODE, mode
from psana.psexp.zmq_utils import ClientSocket

if mode == "mpi":
    pass

class InvalidDataSourceArgument(Exception):
    pass


def _log_file_list(logger, title, files):
    if not files:
        logger.debug(f"{title}: (empty)")
        return

    dir_path = Path(files[0]).parent
    filenames = [Path(f).name for f in files]
    logger.debug(f"{title}:\n  {dir_path}/\n    " + "\n    ".join(filenames))


@dataclass
class DsParms:
    batch_size: int
    max_events: int
    filter: int
    destination: int
    prom_man: int
    max_retries: int
    live: bool
    smd_inprogress_converted: int
    timestamps: np.ndarray
    intg_det: str
    intg_delta_t: int
    log_level: str
    log_file: str
    smd_callback: int = 0
    terminate_flag: bool = False

    def set_det_class_table(
        self, det_classes, xtc_info, det_info_table, det_stream_id_table
    ):
        self.det_classes = det_classes
        self.xtc_info = xtc_info
        self.det_info_table = det_info_table
        self.det_stream_id_table = det_stream_id_table

    def set_use_smds(self, use_smds):
        self.use_smds = use_smds

    @property
    def intg_stream_id(self):
        # We only set detector related fields later (setup run files) so there
        # is a chance that the stream id table is not created yet.
        stream_id = -1
        if hasattr(self, "det_stream_id_table"):
            if self.intg_det in self.det_stream_id_table:
                stream_id = self.det_stream_id_table[self.intg_det]
        return stream_id

class DataSourceBase(abc.ABC):
    """
    Base class for data sources in the psana2 framework.

    Keyword Arguments:
    ------------------
    exp : str
        Experiment ID (e.g., 'xpptut13').
    run : int
        Run number to process.
    dir : str
        Manual path to XTC files.
    files : str or list
        XTC2 file path(s).
    shmem : str
        Shared memory identifier (for live mode).
    drp : str
        DRP-specific parameters (not currently used).
    filter : callable
        A function that takes an event and returns True/False.
    batch_size : int
        Number of events per batch sent to bigdata core (default: 1000).
    max_events : int
        Max number of events to read.
    detectors : list
        User-selected detector names.
    xdetectors : list
        Detectors to explicitly exclude.
    det_name : str
        Deprecated (use 'detectors').
    destination : callable or int
        Callback that returns destination rank (used in event builder).
    live : bool
        Enable live data mode (default: False).
    smalldata_kwargs : dict
        Arguments forwarded to the SmallData interface.
    monitor : bool
        Enable Prometheus monitoring client (default: False).
    small_xtc : list
        Detectors to use bigdata files in place of SMD files.
    timestamps : np.ndarray
        List of user-selected timestamps to filter on.
    dbsuffix : str
        Suffix to use for private calibration constants.
    intg_det : str
        Name of integrating detector for timestamp alignment.
    intg_delta_t : float
        Integration delay in seconds.
    smd_callback : callable or int
        Callback for SMD event handling.
    psmon_publish : psmon.publish
        Enable publishing to psmon (default: None).
    prom_jobid : str
        Prometheus job ID tag.
    skip_calib_load : str
        List of detectors that skip calibration constant loading.
    use_calib_cache : bool
        Enable calibration constant saving and loading in shared memory (default: False).
    fetch_calib_cache_max_retries: int
        Max number of retries reading calibration constant from shared memory (default: 60).
    cached_detectors : list, optional
        List of detector names to load cached calibration attributes for
        when use_calib_cache is True. Default is [].
    mpi_ts : bool
        Used internally to avoid repeated file I/O in MPI contexts.
    log_level : int
        Python logging level (e.g., logging.DEBUG, logging.INFO). Default is logging.INFO.
    """

    def __init__(self, **kwargs):
        # Setup logger
        log_level = kwargs.get("log_level", logging.INFO)
        log_file = kwargs.get("log_file", None)
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper(), logging.INFO)

        # Default values
        self.filter = kwargs.get("filter", 0)
        self.batch_size = kwargs.get("batch_size", 1000)
        self.max_events = kwargs.get("max_events", 0)
        self.detectors = kwargs.get("detectors", [])
        self.xdetectors = kwargs.get("xdetectors", [])
        self.exp = kwargs.get("exp", None)
        self.runnum = kwargs.get("run", None)
        self.live = kwargs.get("live", False)
        self.dir = kwargs.get("dir", None)
        self.files = kwargs.get("files", None)
        self.shmem = kwargs.get("shmem", None)
        self.destination = kwargs.get("destination", 0)
        self.monitor = kwargs.get("monitor", False)
        self.small_xtc = kwargs.get("small_xtc", [])
        self.timestamps = kwargs.get("timestamps", np.empty(0, dtype=np.uint64))
        self.dbsuffix = kwargs.get("dbsuffix", "")
        self.intg_det = kwargs.get("intg_det", "")
        self.intg_delta_t = kwargs.get("intg_delta_t", 0)
        self.current_retry_no = 0
        self.smd_callback = kwargs.get("smd_callback", 0)
        self.psmon_publish = kwargs.get("psmon_publish", None)
        self.prom_jobid = kwargs.get("prom_jobid", None)
        self.skip_calib_load = kwargs.get("skip_calib_load", [])
        self.use_calib_cache = kwargs.get("use_calib_cache", False)
        self.fetch_calib_cache_max_retries = kwargs.get("fetch_calib_cache_max_retries", 60)
        self.cached_detectors = kwargs.get("cached_detectors", [])
        self.smalldata_kwargs = kwargs.get("smalldata_kwargs", {})
        self.files = [self.files] if isinstance(self.files, str) else self.files

        # Calibration constant structures
        # Held by _calib_const, _cc_weak holds weak references to the sub-dicts
        # dsparms.calibconst holds a weak reference to _cc_weak
        self._calib_const = None
        """Holds calibration constants for all detectors."""
        self._cc_weak = utils.WeakDict({})
        """Holds weakrefs to each detectors constants. Passed via weakref in dsparms."""

        # Retry config
        self.max_retries = int(os.environ.get("PS_R_MAX_RETRIES", "60")) if self.live else 0
        if not self.live:
            os.environ["PS_R_MAX_RETRIES"] = "0"

        # Reset batch_size if a custom destination is set
        if self.destination != 0:
            self.logger.debug("Custom destination set. Resetting batch_size to 1.")
            self.batch_size = 1

        # Load timestamps unless MPI context handles it
        if not kwargs.get("mpi_ts", False):
            self.timestamps = self.get_filter_timestamps(self.timestamps)

        # Final sanity check
        assert self.batch_size > 0, "batch_size must be greater than 0"

        # Create Prometheus manager
        self.prom_man = PrometheusManager(job=self.prom_jobid)

        # Package up DataSource parameters
        self.dsparms = DsParms(
            self.batch_size,
            self.max_events,
            self.filter,
            self.destination,
            self.prom_man,
            self.max_retries,
            self.live,
            self.smd_inprogress_converted,
            self.timestamps,
            self.intg_det,
            self.intg_delta_t,
            log_level,
            log_file,
            smd_callback=self.smd_callback,
        )

        self.logger = utils.get_logger(dsparms=self.dsparms, name=utils.get_class_name(self))

        # Warn about unrecognized kwargs
        known_keys = {
            "exp", "run", "dir", "files", "shmem", "drp", "filter", "batch_size",
            "max_events", "detectors", "xdetectors", "det_name", "destination",
            "live", "smalldata_kwargs", "monitor", "small_xtc", "timestamps",
            "dbsuffix", "intg_det", "intg_delta_t", "smd_callback",
            "psmon_publish", "prom_jobid", "skip_calib_load", "use_calib_cache",
            "fetch_calib_cache_max_retries", "cached_detectors", "mpi_ts", "mode",
            "log_level", "log_file"
        }
        for k in kwargs:
            if k not in known_keys:
                self.logger.debug(f"Unrecognized kwarg={k}")

    def get_filter_timestamps(self, timestamps):
        # Returns a sorted numpy array
        # Input: timestamps
        #   str         : load the input file (e.g. .npy) and return sorted array with uint64 dtype
        #   np.ndarray  : return sorted array with np.uint64 dtype
        formatted_timestamps = np.empty(0, dtype=np.uint64)
        if isinstance(timestamps, str):
            with open(timestamps, "rb") as ts_npy_f:
                formatted_timestamps = np.load(ts_npy_f)
        elif isinstance(timestamps, np.ndarray):
            formatted_timestamps = timestamps
        else:
            self.logger.info(
                f"Warning: No timestamp filtering, unrecognized format of the input filter timestamp ({type(timestamps)}). Allowed formats are .npy file or numpy.ndarray)."
            )
        return np.asarray(np.sort(formatted_timestamps), dtype=np.uint64)

    def setup_psplot_live(self):
        # Send run info to psplotdb server using kafka (default).
        # Note that you can use zmq instead by specifying zmq server
        # in the env var. below.
        PSPLOT_LIVE_ZMQ_SERVER = os.environ.get("PSPLOT_LIVE_ZMQ_SERVER", "")
        if getattr(self, "psmon_publish", None) is not None:
            publish = self.psmon_publish
            publish.init()
            # Send fully qualified hostname
            fqdn_host = socket.getfqdn()
            info = {
                "node": fqdn_host,
                "exp": self.exp,
                "runnum": self.runnum,
                "port": publish.port,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", os.getpid()),
            }
            if PSPLOT_LIVE_ZMQ_SERVER == "":
                KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "psplot_live")
                KAFKA_BOOTSTRAP_SERVER = os.environ.get(
                    "KAFKA_BOOTSTRAP_SERVER", "172.24.5.240:9094"
                )
                # Connect to kafka server
                producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVER,
                    value_serializer=lambda m: json.JSONEncoder()
                    .encode(m)
                    .encode("utf-8"),
                )
                producer.send(KAFKA_TOPIC, info)
            else:
                sub = ClientSocket(PSPLOT_LIVE_ZMQ_SERVER)
                info["msgtype"] = MonitorMsgType.PSPLOT
                sub.send(info)

    def smd_inprogress_converted(self):
        """Returns a list of True/False if smd.xtc2.inprogress has been
        converted to smd.xtc2.

        Only applies to smalldata files. Bigdata gets read-in at offsets
        and thus doesn't care about the end of file checking.

        For each xtc file, returns True ONLY IF using .inprogress files and .xtc2 files
        are found on disk. We return False in the case where .inprogress
        files are not used because this callback is used to check if we
        should wait for more data on the xtc files. The read-retry will not
        happen if this callback returns True and there's 0 byte read out.

        See https://github.com/monarin/psana-nersc/blob/master/psana2/write_then_move.sh
        to mimic a live run for testing this feature.
        """
        found_flags = [False] * self.n_files

        for i_file, smd_file in enumerate(self.smd_files):
            if smd_file.find(".xtc2.inprogress") >= 0:
                if os.path.isfile(os.path.splitext(smd_file)[0]):
                    found_flags[i_file] = True
        return found_flags

    @abc.abstractmethod
    def runs(self):
        return

    @abc.abstractmethod
    def is_mpi(self):
        return

    def terminate(self):
        """Sets terminate flag

        The Events iterators of all Run types check this flag
        to see if they need to stop (raise StopIterator for
        RunSerial, RunShmem, & RunSinglefile or skip events
        for RunParallel (see BigDataNode implementation).

        Note that mpi_ds implements this differently by adding
        Isend prior to setting this flag.
        """
        self.dsparms.terminate_flag = True

    def unique_user_rank(self):
        """Only applicable to MPIDataSource
        All other types of DataSource always return True.
        For MPIDataSource, only the last bigdata rank returns True."""
        if not self.is_mpi():
            return True

        n_srv_ranks = int(os.environ.get("PS_SRV_NODES", 0))
        my_rank = self.comms.world_rank
        world_size = self.comms.world_size

        if my_rank == (world_size - n_srv_ranks - 1):
            return True
        else:
            return False

    def is_srv(self):
        """Only NullDataSource is the srv node."""
        return False

    # to be added at a later date...
    # @abc.abstractmethod
    # def steps(self):
    #    retur

    def _check_file_exist_with_retry(self, xtc_file):
        """Returns isfile flag and the true xtc file name (.xtc2 or .inprogress)."""
        # Reconstruct .inprogress file name
        dirname = os.path.dirname(xtc_file)
        fn_only = os.path.splitext(os.path.basename(xtc_file))[0]
        inprogress_file = os.path.join(dirname, fn_only + ".xtc2.inprogress")

        # Check if either one exists
        file_found = os.path.isfile(xtc_file)
        true_xtc_file = xtc_file
        if not file_found:
            file_found = os.path.isfile(inprogress_file)
            if not file_found:
                true_xtc_file = inprogress_file

        # Retry if live mode is set
        while self.current_retry_no < self.dsparms.max_retries:
            file_found = os.path.isfile(xtc_file)
            if not file_found:
                file_found = os.path.isfile(inprogress_file)
                if file_found:
                    true_xtc_file = inprogress_file
            if file_found:
                break
            self.current_retry_no += 1
            self.logger.info(f"Waiting for {xtc_file} ...(#retry:{self.current_retry_no})")
            time.sleep(1)
        return file_found, true_xtc_file

    def _get_file_info_from_db(self, runnum):
        """Returns a list of xtc2 files as shown in the db.

        Note that the requested url below returns list of all files (all streams and
        all chunks) from the database. For psana2, we only need the first chunk (-c)
        of each stream so the original list is filtered down.

        Example of names returned by the requests:
        /tmo/tmoc00118/xtc/tmoc00118-r0222-s000-c000.xtc2
        /tmo/tmoc00118/xtc/tmoc00118-r0222-s000-c001.xtc2
        /tmo/tmoc00118/xtc/tmoc00118-r0222-s002-c000.xtc2

        Returns
        file_info['xtc_files']
        tmoc00118-r0222-s000-c000.xtc2
        tmoc00118-r0222-s002-c000.xtc2
        file_info['dirname']
        /tmo/tmoc00118/xtc/
        """
        url = f"https://pswww.slac.stanford.edu/ws/lgbk/lgbk/{self.exp}/ws/{runnum}/files_for_live_mode"
        resp = requests.get(url)
        try:
            resp.raise_for_status()
            all_xtc_files = resp.json()["value"]
        except Exception:
            self.logger.info(f"Warning: unable to connect to {url}")
            all_xtc_files = []

        file_info = {}
        if all_xtc_files:
            # Only take chunk 0 xtc files (matched with *-c*0.)
            xtc_files = [
                os.path.basename(xtc_file)
                for xtc_file in all_xtc_files
                if re.search(r"-c000\.", xtc_file)
            ]
            if xtc_files:
                file_info["xtc_files"] = xtc_files
                file_info["dirname"] = os.path.dirname(all_xtc_files[0])
        return file_info

    def _setup_run_files(self, runnum):
        """
        Generate list of smd and xtc files given a run number.

        Allowed extentions include .xtc2 and .xtc2.inprogress.
        """
        file_info = self._get_file_info_from_db(runnum)

        # If we get the list of stream ids from db (some test cases
        # like xpptut15 run 1 are not registered in the db), these streams take
        # priority over what found on disk. We support two cases:
        #
        # * Non-live mode: exit when expected stream is not found
        # * Live mode    : wait for files and time out when max wait (s) is reached.
        #
        # If we don't get anything from the db, use what exist on disk.
        if file_info:
            # Builds a list of expected smd files
            xtc_files_from_db = file_info["xtc_files"]
            smd_files = [
                os.path.join(
                    self.xtc_path,
                    "smalldata",
                    os.path.splitext(xtc_file_from_db)[0] + ".smd.xtc2",
                )
                for xtc_file_from_db in xtc_files_from_db
            ]

            self.current_retry_no = 0
            for i_smd, smd_file in enumerate(smd_files):
                flag_found, true_xtc_file = self._check_file_exist_with_retry(smd_file)
                if not flag_found:
                    raise FileNotFoundError(true_xtc_file)
                smd_files[i_smd] = true_xtc_file

        else:
            smd_dir = os.path.join(self.xtc_path, "smalldata")
            smd_files = sorted(
                glob.glob(
                    os.path.join(smd_dir, "*r%s-s*.smd.xtc2" % (str(runnum).zfill(4)))
                )
                + glob.glob(
                    os.path.join(
                        smd_dir, "*r%s-s*.smd.xtc2.inprogress" % (str(runnum).zfill(4))
                    )
                )
            )

        self.n_files = len(smd_files)
        assert (
            self.n_files > 0
        ), f"No smalldata files found from this path: {os.path.join(self.xtc_path, 'smalldata')}"

        # Look for matching bigdata files - MUST match all.
        # We start by looking for smd basename with .inprogress extension.
        # If this name is not found, try .xtc2.
        xtc_files = [
            os.path.join(
                self.xtc_path, os.path.basename(smd_file).split(".smd")[0] + ".xtc2"
            )
            for smd_file in smd_files
        ]
        for i_xtc, xtc_file in enumerate(xtc_files):
            flag_found, true_xtc_file = self._check_file_exist_with_retry(xtc_file)
            if not flag_found:
                raise FileNotFoundError(true_xtc_file)
            xtc_files[i_xtc] = true_xtc_file

        self.smd_files = smd_files
        self.xtc_files = xtc_files

        _log_file_list(self.logger, "smd_files", self.smd_files)
        _log_file_list(self.logger, "xtc_files", self.xtc_files)

        # Set default flag for replacing smalldata with bigda files.
        self.dsparms.set_use_smds([False] * self.n_files)

    def _setup_runnum_list(self):
        """
        Requires:
        1) exp
        2) exp, run=N or [N, M, ...]

        Build runnum_list for the experiment.
        """

        if not self.exp:
            raise InvalidDataSourceArgument("Missing exp or files input")

        if self.dir:
            self.xtc_path = self.dir
        else:
            xtc_dir = os.environ.get("SIT_PSDM_DATA", "/reg/d/psdm")
            self.xtc_path = os.path.join(xtc_dir, self.exp[:3], self.exp, "xtc")

        if self.runnum is None:
            run_list = [
                int(
                    os.path.splitext(os.path.basename(_dummy))[0]
                    .split("-r")[1]
                    .split("-")[0]
                )
                for _dummy in glob.glob(os.path.join(self.xtc_path, "*-r*.xtc2"))
            ]
            assert run_list
            run_list = list(set(run_list))
            run_list.sort()
            self.runnum_list = run_list
        elif isinstance(self.runnum, list):
            self.runnum_list = self.runnum
        elif isinstance(self.runnum, int):
            self.runnum_list = [self.runnum]
        else:
            raise InvalidDataSourceArgument(
                "run accepts only int or list. Leave out run arugment to process all available runs."
            )

    def smalldata(self, **kwargs):
        if MODE == "PARALLEL":
            PS_SRV_NODES = int(os.environ.get("PS_SRV_NODES", 0))
            if not PS_SRV_NODES:
                msg = f"Smalldata requires at least one SRV core ({MODE=}). Try setting PS_SRV_NODES=1 or more."
                raise Exception(msg)
        self.smalldata_obj.setup_parms(**kwargs)
        return self.smalldata_obj

    def _start_prometheus_client(self, mpi_rank=0, prom_cfg_dir=None):
        if not self.monitor:
            self.logger.debug("RUN W/O PROMETHEUS CLENT")
        elif prom_cfg_dir is None:  # Use push gateway
            self.prom_man.rank = mpi_rank
            self.logger.debug(
                f"START PROMETHEUS CLIENT (JOB:{self.prom_man.job} RANK:{self.prom_man.rank})"
            )
            self.e = threading.Event()
            self.t = threading.Thread(
                name="PrometheusThread%s" % (mpi_rank),
                target=self.prom_man.push_metrics,
                args=(self.e,),
                daemon=True,
            )
            self.t.start()
        else:  # Use http exposer
            self.logger.debug(
                "START PROMETHEUS CLIENT (PROM_CFG_DIR: %s)" % (prom_cfg_dir)
            )
            self.e = None
            self.prom_man.create_exposer(prom_cfg_dir)

    def _end_prometheus_client(self, mpi_rank=0):
        if not self.monitor:
            return

        if self.e is not None:  # Push gateway case only
            self.logger.debug(
                "END PROMETHEUS CLIENT (JOB:%s)" % (self.prom_man.job)
            )
            self.e.set()

    @property
    def _configs(self):
        """Returns configs from DgramManager"""
        return self.dm.configs

    def _apply_detector_selection(self):
        """
        Handles two arguments
        1) detectors=['detname', 'detname_0']
            Reduce no. of smd/xtc files to only those with given 'detname' or
            'detname:segment_id'.
        2) xdetectors=['detname', 'detname_0']
            Reduce no. of smd/xtc files to only *NOT* in this list.
        3) small_xtc=['detname', 'detname_0']
            Swap out smd files with these given detectors with bigdata files
        """
        n_smds = len(self.smd_files)
        use_smds = [False] * n_smds
        if self.detectors or self.small_xtc or self.xdetectors:
            # Get tmp configs using SmdReader
            # this smd_fds, configs, and SmdReader will not be used later
            smd_fds = np.array(
                [os.open(smd_file, os.O_RDONLY) for smd_file in self.smd_files],
                dtype=np.int32,
            )
            self.logger.debug(f"smd0 opened tmp smd_fds for detection selection: {smd_fds}")
            smdr_man = SmdReaderManager(smd_fds, self.dsparms)
            all_configs = smdr_man.get_next_dgrams()

            # Keeps list of selected files and configs
            xtc_files = []
            smd_files = []
            configs = []
            det_dict = {}

            # Get list of detectors from all configs in the format of detname
            # and detname:segment_id
            all_det_dict = {i: [] for i in range(n_smds)}
            for i, config in enumerate(all_configs):
                det_list = all_det_dict[i]
                for detname, seg_dict in config.software.__dict__.items():
                    det_list.append(detname)
                    for seg_id, _ in seg_dict.items():
                        det_list.append(f"{detname}_{seg_id}")
                self.logger.debug(f"Stream:{i} detectors:{all_det_dict[i]}")

            # Apply detector selection exclusion
            if self.detectors or self.xdetectors:
                include_set = set(self.detectors)
                exclude_set = set(self.xdetectors)
                self.logger.debug("Applying detector selection/exclusion")
                self.logger.debug(f"  Included: {include_set}")
                self.logger.debug(f"  Excluded: {exclude_set}")

                # This will become 'key' to the new det_dict
                cn_keeps = 0
                for i in range(n_smds):
                    flag_keep = True
                    exist_set = set(all_det_dict[i])
                    self.logger.debug(f"  Stream:{i}")
                    if self.detectors:
                        if not include_set.intersection(exist_set):
                            flag_keep = False
                            self.logger.debug("  |-- Discarded, not matched given detectors")
                    if self.xdetectors and flag_keep:
                        matched_set = exclude_set.intersection(exist_set)
                        if matched_set:
                            flag_keep = False
                            self.logger.debug(
                                "  |-- Discarded, matched with excluded detectors"
                            )
                            # We only warn users in the case where we exclude a detector
                            # and there're more than one detectors in the file.
                            if len(exist_set) > len(matched_set):
                                self.logger.info(
                                    f"Warning: Stream-{i} has one or more detectors matched with the excluded set. All detectors in this stream will be excluded."
                                )

                    if flag_keep:
                        if self.xtc_files:
                            xtc_files.append(self.xtc_files[i])
                        if self.smd_files:
                            smd_files.append(self.smd_files[i])
                        configs.append(config)
                        det_dict[cn_keeps] = all_det_dict[i]
                        cn_keeps += 1
                        self.logger.debug("  |-- Kept")
            else:
                xtc_files = self.xtc_files[:]
                smd_files = self.smd_files[:]
                configs = all_configs
                det_dict = all_det_dict

            use_smds = [False] * len(smd_files)
            if self.small_xtc:
                s1 = set(self.small_xtc)
                self.logger.debug("Applying smalldata replacement")
                self.logger.debug(f"  Smalldata: {self.small_xtc}")
                for i in range(len(smd_files)):
                    exist_set = set(det_dict[i])
                    self.logger.debug(f" Stream:{i}")
                    if s1.intersection(exist_set):
                        smd_files[i] = xtc_files[i]
                        use_smds[i] = True
                        self.logger.debug("  |-- Replaced with smalldata")
                    else:
                        self.logger.debug("  |-- Kept with bigdata")

            self.xtc_files = xtc_files
            self.smd_files = smd_files

            for smd_fd in smd_fds:
                os.close(smd_fd)
            self.logger.debug(f"close tmp smd fds:{smd_fds}")

        self.dsparms.set_use_smds(use_smds)

    def _clear_calibconst(self):
        """Explicitly clear calibration constants and force garbage collection.

        Helps to prevent leakage of calibration constants between runs.
        """
        if self._calib_const is not None:
            self._calib_const = None
            gc.collect()
            self._calib_const = utils.WeakDict({})
            self._cc_weak = utils.WeakDict({})

    def _create_weak_calibconst(self):
        """Setup weak references to calibration constants.

        Pass along only weak references to large calibration data structures
        since references are held in many places downstream making garbage
        collection unreliable if any one object leaks/has circular references.
        """
        if self._calib_const is not None:
            for key in self._calib_const:
                self._cc_weak[key] = weakref.WeakValueDictionary(self._calib_const[key])
        # Only use dsparms.calibconst to pass to runs, detectors etc.
        self.dsparms.calibconst = weakref.proxy(self._cc_weak)

    def _setup_run_calibconst(self):
        """
        Initialize the `dsparms.calibconst` dictionary with calibration constants.

        If `use_calib_cache` is enabled, attempts to load calibration constants from
        shared memory (e.g., `/dev/shm`) using `ensure_valid_calibconst`. Otherwise,
        fetches calibration constants directly from the database for each detector.

        This function supports skipping selected detectors via `skip_calib_load`.

        Args:
            None

        Returns:
            None
        """
        runinfo = self._get_runinfo()
        if not runinfo:
            return
        expt, runnum, _ = runinfo
        self.logger.debug(f"_setup_run_calibconst called {expt=} {runnum=}")

        self._clear_calibconst()

        if self.use_calib_cache:
            self.logger.debug("using calibration constant from shared memory, if exists")
            calib_const, existing_info, max_retry = (None, None, 0)
            latest_info = {k: v.uniqueid for k, v in self.dsparms.configinfo_dict.items()}
            while not calib_const and max_retry < self.fetch_calib_cache_max_retries and existing_info != latest_info:
                try:
                    loaded_data = calib_utils.try_load_data_from_file(self.logger)
                except Exception as e:
                    self.logger.warning(f"failed to retrieve calib_const: {e}")
                if loaded_data is not None:
                    self._calib_const = utils.make_weak_refable(loaded_data.get("calib_const"))
                    existing_info = loaded_data.get("det_info")
                if existing_info != latest_info:
                    self._calib_const = utils.WeakDict({})
                    self.logger.warning("det_info in pickle file does not matched with the latest run")
                elif self._calib_const:
                    self.logger.debug(f"received calib_const for {','.join(self._calib_const.keys())}")
                    break
                max_retry += 1
                self.logger.debug(f"try to read calib_const from shared memory (retry: {max_retry}/{self.fetch_calib_cache_max_retries})")
                time.sleep(1)
        else:
            if self._calib_const is None:
                self._calib_const = utils.WeakDict({})
            for det_name, configinfo in self.dsparms.configinfo_dict.items():
                if expt:
                    if expt == "xpptut15":
                        det_uniqueid = "cspad_detnum1234"
                    else:
                        det_uniqueid = configinfo.uniqueid
                    if hasattr(self, "skip_calib_load") and (det_name in self.skip_calib_load or self.skip_calib_load=="all"):
                        self._calib_const[det_name] = utils.WeakDict({})
                        continue
                    st = time.monotonic()
                    calib_const = wu.calib_constants_all_types(
                        det_uniqueid, exp=expt, run=runnum, dbsuffix=self.dbsuffix
                    )
                    en = time.monotonic()
                    self.logger.debug(f"received calibconst for {det_name} in {en-st:.4f}s.")
                    self._calib_const[det_name] = utils.make_weak_refable(calib_const)
                else:
                    self.logger.info(
                        "Warning: cannot access calibration constant (exp is None)"
                    )
                    self._calib_const[det_name] = utils.WeakDict({})
        self._create_weak_calibconst()

    def _get_runinfo(self):
        expt, runnum, timestamp = (None, None, None)

        if self.beginruns:
            beginrun_dgram = self.beginruns[0]
            if hasattr(beginrun_dgram, "runinfo"):  # some xtc2 do not have BeginRun
                # BeginRun in all streams have the same information.
                # We just need to get one from the first segment/ dgram.
                segment_id = list(beginrun_dgram.runinfo.keys())[0]
                expt = beginrun_dgram.runinfo[segment_id].runinfo.expt
                runnum = beginrun_dgram.runinfo[segment_id].runinfo.runnum
                timestamp = beginrun_dgram.timestamp()
        return expt, runnum, timestamp

    def _close_opened_smd_files(self):
        # Make sure to close all smd files opened by previous run
        if self.smd_fds is not None:
            for fd in self.smd_fds:
                os.close(fd)
