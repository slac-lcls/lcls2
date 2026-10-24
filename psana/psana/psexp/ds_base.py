"""Base class and shared parameters of the psana data sources: `DataSourceBase`, `DsParms` and
`InvalidDataSourceArgument`.
"""
import abc
import glob
import json
import logging
import os
import re
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
from kafka import KafkaProducer

from psana import utils
from psana.app.psplot_live.utils import MonitorMsgType
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.tools import MODE, mode
from psana.psexp.zmq_utils import ClientSocket
from psana.psexp.prometheus_manager import ensure_pusher, stop_pusher, get_prom_manager


if mode == "mpi":
    pass

class InvalidDataSourceArgument(Exception):
    """Exception raised by `DataSourceBase._setup_runnum_list` when `exp` is missing or `run` is neither an int nor a list."""
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
    """Data source parameters shared with runs, readers and event builders.

    The dataclass fields hold the batch size, maximum number of events, read retries, live flag,
    filter timestamps, integrating detector settings, calibration cache settings, database suffix,
    smalldata callback, and the smd file list with its per-file flags. Other code adds attributes:
    for example `DgramManager.set_configs` sets `det_classes`, `configinfo_dict` and other detector
    tables on it, and runs set `calibconst`.
    """
    batch_size: int
    max_events: int
    max_retries: int
    live: bool
    timestamps: np.ndarray
    intg_det: str
    intg_delta_t: int
    use_calib_cache: bool
    cached_detectors: list
    fetch_calib_cache_max_retries: int
    skip_calib_load: list
    dbsuffix: str
    smd_callback: int = 0
    smd_files: list[str] = field(default_factory=list)
    use_smds: list[bool] = field(default_factory=list)

    def set_det_class_table(
        self, det_classes, xtc_info, det_info_table, det_stream_id_table
    ):
        """Store `det_classes`, `xtc_info`, `det_info_table` and `det_stream_id_table` as attributes of the same names."""
        self.det_classes = det_classes
        self.xtc_info = xtc_info
        self.det_info_table = det_info_table
        self.det_stream_id_table = det_stream_id_table

    def update_smd_state(self, smd_files, use_smds):
        """Set `smd_files` and the per-file flags `use_smds`.

        `DataSourceBase._apply_detector_selection` sets a flag to True where a bigdata file is used in
        place of the smalldata file.
        """
        self.smd_files = smd_files
        self.use_smds = use_smds

    @property
    def intg_stream_id(self):
        # We only set detector related fields later (setup run files) so there
        # is a chance that the stream id table is not created yet.
        """Stream index of the integrating detector `intg_det`, or -1.

        Looked up in `det_stream_id_table`; -1 if that table is not set yet or does not contain
        `intg_det`.
        """
        stream_id = -1
        if hasattr(self, "det_stream_id_table"):
            if self.intg_det in self.det_stream_id_table:
                stream_id = self.det_stream_id_table[self.intg_det]
        return stream_id

class DataSourceBase(abc.ABC):
    """Base class for data sources in the psana2 framework.

    `__init__` reads the options below from its keyword arguments; any other keyword is logged as a
    warning ("Unrecognized kwarg").

    Other Parameters
    ----------------
    exp : str
        Experiment ID (e.g., 'xpptut13').
    run : int or list of int
        Run number or list of run numbers to process (used with `exp`). If None (the default), every
        run that has a `*-r*.xtc2` file in the xtc directory is used; any other type raises
        InvalidDataSourceArgument.
    dir : str
        Manual path to XTC files.
    files : str or list of str
        XTC2 file path(s); a single str is wrapped in a list.
    shmem : str
        Shared memory identifier (for live mode).
    drp : object
        DRP parameters object. `DataSource` creates a `DrpDataSource` when `drp` is given without
        `shmem`, `exp` or `files`; that class reads attributes such as `det_segment`, `is_supervisor`
        and `worker_num` from it. This base class does not read it.
    batch_size : int
        Number of events per batch sent to bigdata core (default: 1000). Must be greater than 0
        (asserted).
    max_events : int
        Max number of events to read (default: 0).
    detectors : list
        User-selected detector names (default: []).
    xdetectors : list
        Detectors to explicitly exclude (default: []).
    det_name : str
        Deprecated (use 'detectors'). Accepted without a warning but not read.
    live : bool
        Enable live data mode (default: False).
    smalldata_kwargs : dict
        Arguments forwarded to the SmallData interface (default: {}).
    monitor : bool
        Enable Prometheus monitoring client (default: False).
    small_xtc : list
        Detectors to use bigdata files in place of SMD files (default: []).
    timestamps : np.ndarray or str
        User-selected timestamps to filter on, as an array or the path of a .npy file; stored sorted
        as uint64 (default: empty array). Any other type is logged and gives no filtering.
    dbsuffix : str
        Suffix to use for private calibration constants (default: "").
    intg_det : str
        Name of integrating detector for timestamp alignment (default: "").
    intg_delta_t : int
        Integration time window in nanoseconds; the smd reader adds it to each integrating-detector
        timestamp when it builds batches (default: 0).
    smd_callback : callable or int
        Callback for SMD event handling (default: 0).
    psmon_publish : psmon.publish
        Enable publishing to psmon (default: None).
    prom_jobid : str
        Prometheus job ID tag (default: None). Stored as the `prom_jobid` attribute; no code in this
        repository reads it.
    skip_calib_load : list of str or str
        Names of detectors that skip calibration constant loading, or "all" to skip it for every
        detector (default: []).
    use_calib_cache : bool
        Enable calibration constant saving and loading in shared memory (default: False).
    fetch_calib_cache_max_retries : int
        Max number of retries reading calibration constant from shared memory (default: 60).
    cached_detectors : list, optional
        List of detector names to load cached calibration attributes for
        when use_calib_cache is True. Default is [].
    mpi_ts : bool
        Used internally to avoid repeated file I/O in MPI contexts: when True, `timestamps` is not
        loaded or sorted here.
    log_level : int or str
        Python logging level (e.g., logging.DEBUG, logging.INFO) or its name (e.g., "DEBUG"; an
        unknown name gives INFO). Default is logging.INFO.
    log_file : str
        Path of a rotating log file written in addition to the console output on stderr (default:
        None, console only). The rank is inserted before the file extension, e.g. `run.log` becomes
        `run.rank0.log`; the rank is the MPI world rank when PS_PARALLEL is "mpi" (the default) and
        0 otherwise.
    auto_tune : bool
        Stored as the `auto_tune` attribute (default: False); no code in this repository reads it.
    """

    def __init__(self, **kwargs):
        # Setup logger
        log_level = kwargs.get("log_level", logging.INFO)
        log_file = kwargs.get("log_file", None)
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper(), logging.INFO)
        utils.configure_logging(level=log_level,
                        logfile=log_file,
                        timestamp=False)
        self.logger = utils.get_logger(name=utils.get_class_name(self))

        # Default values
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
        self.auto_tune = kwargs.get("auto_tune", False)
        # Retry config
        self.max_retries = int(os.environ.get("PS_R_MAX_RETRIES", "60")) if self.live else 0
        if not self.live:
            os.environ["PS_R_MAX_RETRIES"] = "0"

        # Load timestamps unless MPI context handles it
        if not kwargs.get("mpi_ts", False):
            self.timestamps = self.get_filter_timestamps(self.timestamps)

        # Final sanity check
        assert self.batch_size > 0, "batch_size must be greater than 0"

        # Package up DataSource parameters
        self.dsparms = DsParms(
            self.batch_size,
            self.max_events,
            self.max_retries,
            self.live,
            self.timestamps,
            self.intg_det,
            self.intg_delta_t,
            self.use_calib_cache,
            self.cached_detectors,
            self.fetch_calib_cache_max_retries,
            self.skip_calib_load,
            self.dbsuffix,
            smd_callback=self.smd_callback,
        )

        # Warn about unrecognized kwargs
        known_keys = {
            "exp", "run", "dir", "files", "shmem", "drp", "batch_size",
            "max_events", "detectors", "xdetectors", "det_name",
            "live", "smalldata_kwargs", "monitor", "small_xtc", "timestamps",
            "dbsuffix", "intg_det", "intg_delta_t", "smd_callback",
            "psmon_publish", "prom_jobid", "skip_calib_load", "use_calib_cache",
            "fetch_calib_cache_max_retries", "cached_detectors", "mpi_ts",
            "log_level", "log_file", 'auto_tune'
        }
        for k in kwargs:
            if k not in known_keys:
                self.logger.warning(f"Unrecognized kwarg={k}")


    def get_filter_timestamps(self, timestamps):
        # Returns a sorted numpy array
        # Input: timestamps
        #   str         : load the input file (e.g. .npy) and return sorted array with uint64 dtype
        #   np.ndarray  : return sorted array with np.uint64 dtype
        """Return the filter timestamps as a sorted numpy uint64 array.

        A str is the path of a .npy file to load; a numpy array is used as given; any other type logs a
        message at info level and gives an empty array.
        """
        formatted_timestamps = np.empty(0, dtype=np.uint64)
        if isinstance(timestamps, str):
            with open(timestamps, "rb") as ts_npy_f:
                formatted_timestamps = np.load(ts_npy_f)
        elif isinstance(timestamps, np.ndarray):
            formatted_timestamps = timestamps
        else:
            self.logger.info(
                "Warning: No timestamp filtering. Unrecognized input type "
                f"({type(timestamps)}). Allowed formats are .npy or numpy.ndarray."
            )
        return np.asarray(np.sort(formatted_timestamps), dtype=np.uint64)

    def setup_psplot_live(self):
        # Send run info to psplotdb server using kafka (default).
        # Note that you can use zmq instead by specifying zmq server
        # in the env var. below.
        """Publish this job's run info for psplot_live when `psmon_publish` is set.

        Calls `psmon_publish.init()` and sends a dict with the fully qualified host name, `exp`,
        `runnum`, the publish port and SLURM_JOB_ID (or the process id): to the Kafka topic KAFKA_TOPIC
        (default "psplot_live") at KAFKA_BOOTSTRAP_SERVER (default 172.24.5.240:9094), or, when
        PSPLOT_LIVE_ZMQ_SERVER is set, through a zmq REQ socket to that address with `msgtype` set to
        `MonitorMsgType.PSPLOT`. Does nothing if `psmon_publish` is None.
        """
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

    @abc.abstractmethod
    def runs(self):
        """Abstract method: subclasses yield their run objects. The base body returns None."""
        return

    @abc.abstractmethod
    def is_mpi(self):
        """Abstract method: subclasses return whether they use MPI parallel mode. The base body returns None."""
        return

    def unique_user_rank(self):
        """Return whether this rank is the unique user rank.

        When `is_mpi()` is False this base method returns True; that is the case for `SerialDataSource`,
        `SingleFileDataSource`, `ShmemDataSource` and `DrpDataSource`. `NullDataSource` overrides it to
        return False. For `MPIDataSource` only world rank `world_size - PS_SRV_NODES - 1` (the highest rank
        that is not a srv rank; PS_SRV_NODES defaults to 0) returns True.
        """
        if not self.is_mpi():
            return True

        n_srv_ranks = int(os.environ.get("PS_SRV_NODES", 0))
        my_rank = self.comms.world_rank
        world_size = self.comms.world_size

        if my_rank == (world_size - n_srv_ranks - 1):
            return True
        else:
            return False

    def is_bd(self):
        """Only applicable to MPIDataSource
        For MPIDataSource, returns True for bigdata ranks only.
        Bigdata ranks are > PS_EB_NODES and < world_size - PS_SRV_NODES."""
        if not self.is_mpi():
            return True

        n_eb_ranks = int(os.environ.get("PS_EB_NODES", "1"))
        n_srv_ranks = int(os.environ.get("PS_SRV_NODES", "0"))
        my_rank = self.comms.world_rank
        world_size = self.comms.world_size

        return (my_rank > n_eb_ranks) and (my_rank < (world_size - n_srv_ranks))

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

    def _missing_xtc_files_error(self):
        if self.dsparms.max_retries > 0:
            timeout = self.dsparms.max_retries
            return FileNotFoundError(
                "Timed out waiting for XTC files for exp=%s run=%s in dir=%s (timeout=%ss). "
                "Checked for both final and .inprogress filenames."
                % (self.exp, self.runnum, self.xtc_path, timeout)
            )
        return FileNotFoundError(
            "No XTC files found for exp=%s run=%s in dir=%s. "
            "Checked for both final and .inprogress filenames."
            % (self.exp, self.runnum, self.xtc_path)
        )

    def _file_info_url(self, runnum):
        return f"https://pswww.slac.stanford.edu/ws/lgbk/lgbk/{self.exp}/ws/{runnum}/current_files_for_live_mode"

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
        url = self._file_info_url(runnum)
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
            xtc_files = sorted(
                os.path.basename(xtc_file)
                for xtc_file in all_xtc_files
                if re.search(r"-c000\.", xtc_file)
            )
            if xtc_files:
                file_info["xtc_files"] = xtc_files
                file_info["dirname"] = os.path.dirname(all_xtc_files[0])
        return file_info

    def _file_info_key(self, file_info):
        if not file_info:
            return None
        return (
            file_info.get("dirname"),
            tuple(sorted(file_info.get("xtc_files", []))),
        )

    def _file_info_stable_polls(self):
        try:
            stable_polls = int(os.environ.get("PS_LIVE_FILE_LIST_STABLE_POLLS", "3"))
        except ValueError:
            stable_polls = 3
        return max(1, stable_polls)

    def _get_file_info_from_db_with_retry(self, runnum):
        stable_poll_target = self._file_info_stable_polls()
        file_info = self._get_file_info_from_db(runnum)
        file_info_key = self._file_info_key(file_info)
        candidate_file_info_key = None
        stable_polls = 0
        retry_no = 0
        while self.live:
            if file_info_key:
                if file_info_key == candidate_file_info_key:
                    stable_polls += 1
                else:
                    candidate_file_info_key = file_info_key
                    stable_polls = 1
                if stable_polls >= stable_poll_target:
                    return file_info
            elif candidate_file_info_key is None:
                stable_polls = 0

            if retry_no >= self.dsparms.max_retries:
                break

            retry_no += 1
            xtc_files = file_info.get("xtc_files", []) if file_info else []
            self.logger.info(
                "Waiting for file list from %s to settle "
                "...(#retry:%s stable:%s/%s nfiles:%s)",
                self._file_info_url(runnum),
                retry_no,
                stable_polls,
                stable_poll_target,
                len(xtc_files),
            )
            time.sleep(2)
            file_info = self._get_file_info_from_db(runnum)
            file_info_key = self._file_info_key(file_info)
        return {}

    def _missing_file_info_error(self, runnum):
        if self.dsparms.max_retries > 0:
            return FileNotFoundError(
                "Timed out waiting for stable logbook file list for live mode "
                "exp=%s run=%s from %s (retries=%s, interval=2s). live=True does not scan "
                "the directory for stream discovery."
                % (
                    self.exp,
                    runnum,
                    self._file_info_url(runnum),
                    self.dsparms.max_retries,
                )
            )
        return FileNotFoundError(
            "No stable logbook file list found for live mode exp=%s run=%s from %s. "
            "live=True does not scan the directory for stream discovery."
            % (self.exp, runnum, self._file_info_url(runnum))
        )

    def _scan_run_files_on_disk(self, runnum):
        smd_dir = os.path.join(self.xtc_path, "smalldata")
        return sorted(
            glob.glob(
                os.path.join(smd_dir, "*r%s-s*.smd.xtc2" % (str(runnum).zfill(4)))
            )
            + glob.glob(
                os.path.join(
                    smd_dir, "*r%s-s*.smd.xtc2.inprogress" % (str(runnum).zfill(4))
                )
            )
        )

    def _setup_run_files(self, runnum):
        """
        Generate list of smd and xtc files given a run number.

        Allowed extentions include .xtc2 and .xtc2.inprogress.
        """
        if self.live:
            file_info = self._get_file_info_from_db_with_retry(runnum)
        else:
            file_info = self._get_file_info_from_db(runnum)

        # If we get the list of stream ids from db, these streams take
        # priority over what is found on disk. In live mode the DB stream
        # list is required; non-live mode keeps the disk fallback for old
        # test data that is not registered in the db.
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
                    if self.dir and not self.live:
                        self.logger.info(
                            "Expected DB stream file %s not found under explicit dir=%s; "
                            "falling back to on-disk stream discovery for this directory.",
                            true_xtc_file,
                            self.xtc_path,
                        )
                        smd_files = self._scan_run_files_on_disk(runnum)
                        break
                    raise self._missing_xtc_files_error()
                smd_files[i_smd] = true_xtc_file

        else:
            if self.live:
                raise self._missing_file_info_error(runnum)
            smd_files = self._scan_run_files_on_disk(runnum)

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
                raise self._missing_xtc_files_error()
            xtc_files[i_xtc] = true_xtc_file

        self.smd_files = smd_files
        self.xtc_files = xtc_files

        _log_file_list(self.logger, "smd_files", self.smd_files)
        _log_file_list(self.logger, "xtc_files", self.xtc_files)

        self.dsparms.update_smd_state(self.smd_files, [False] * self.n_files)

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
        """Set up and return this data source's `SmallData` object.

        Calls `self.smalldata_obj.setup_parms(**kwargs)` (filename, batch_size, cache_size, callbacks,
        swmr_mode); on smalldata server ranks that call runs the server receive loop until all clients
        are done. The `smalldata_obj` attribute must have been created by the subclass.

        Raises
        ------
        Exception
            In parallel mode (`psana.psexp.tools.MODE` "PARALLEL") if PS_SRV_NODES is unset or 0.
        """
        if MODE == "PARALLEL":
            PS_SRV_NODES = int(os.environ.get("PS_SRV_NODES", 0))
            if not PS_SRV_NODES:
                msg = f"Smalldata requires at least one SRV core ({MODE=}). Try setting PS_SRV_NODES=1 or more."
                raise Exception(msg)
        self.smalldata_obj.setup_parms(**kwargs)
        return self.smalldata_obj

    def _start_prometheus_client(self, mpi_rank=0, prom_cfg_dir=None):
        """Start Prometheus either via push gateway (default) or HTTP exposer.
        With the singleton, this is safe to call multiple times."""
        if not self.monitor:
            if mpi_rank == 0:
                self.logger.debug("RUN W/O PROMETHEUS CLIENT")
            return

        if prom_cfg_dir is None:  # Push-gateway mode
            pm = ensure_pusher(rank=mpi_rank)   # starts pusher once per process
            if mpi_rank == 0:
                self.logger.debug(
                    f"START PROMETHEUS CLIENT (JOB:{pm.job} RANK:{pm.rank})"
                )
        else:  # HTTP exposer mode (DAQ-style scrape)
            pm = get_prom_manager()
            self.logger.debug(
                f"START PROMETHEUS HTTP EXPOSER (PROM_CFG_DIR:{prom_cfg_dir})"
            )
            pm.create_exposer(prom_cfg_dir)      # safe: only starts once per process

    def _end_prometheus_client(self):
        """Stop the push-gateway pusher if running. Exposer stays up (matches previous behavior)."""
        if not self.monitor:
            return
        try:
            stop_pusher()  # no-op if nothing running
        except Exception as ex:
            self.logger.debug(f"END PROMETHEUS CLIENT: stop_pusher() raised {ex!r}")

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
                                    "Warning: Stream-%s has detectors in the excluded set; "
                                    "excluding entire stream.",
                                    i,
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

        self.dsparms.update_smd_state(self.smd_files, use_smds)

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
