import weakref
import os
import glob
import abc
import numpy as np
import pathlib
import requests
import re
import time

from psana.dgrammanager import DgramManager

from psana.psexp import PrometheusManager, SmdReaderManager
import threading
from psana import dgram

from psana.detector import detectors
import psana.pscalib.calib.MDBWebUtils as wu

import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass

class InvalidDataSourceArgument(Exception): pass

@dataclass
class DsParms:
    batch_size:         int
    max_events:         int
    filter:             int
    destination:        int
    prom_man:           int
    max_retries:        int
    live:               bool
    smd_inprogress_converted: int
    timestamps:         np.ndarray
    intg_det:           str
    smd_callback:       int = 0
    terminate_flag:     bool = False

    def set_det_class_table(self, det_classes, xtc_info, det_info_table, det_stream_id_table):
        self.det_classes        = det_classes
        self.xtc_info           = xtc_info
        self.det_info_table     = det_info_table
        self.det_stream_id_table= det_stream_id_table

    def set_use_smds(self, use_smds):
        self.use_smds = use_smds

    def read_ts_npy_file(self):
        with open(self.timestamps, 'rb') as ts_npy_f:
            return(np.asarray(np.load(ts_npy_f), dtype=np.uint64))

    def set_timestamps(self):
        if isinstance(self.timestamps, str):
            self.timestamps = self.read_ts_npy_file()

    @property
    def intg_stream_id(self):
        # We only set detector related fields later (setup run files) so there
        # is a chance that the stream id table is not created yet.
        stream_id = -1
        if hasattr(self, "det_stream_id_table"):
            if self.intg_det in self.det_stream_id_table:
                stream_id=self.det_stream_id_table[self.intg_det]
        return stream_id


class DataSourceBase(abc.ABC):
    def __init__(self, **kwargs):
        """Initializes datasource base"""
        self.filter      = 0         # callback that takes an evt and return True/False.
        self.batch_size  = 1000      # no. of events per batch sent to a bigdata core
        self.max_events  = 0         # no. of maximum events
        self.detectors   = []        # user-selected detector names
        self.xdetectors  = []        # user-selected excluded detector names
        self.exp         = None      # experiment id (e.g. xpptut13)
        self.runnum      = None      # run no.
        self.live        = False     # turns live mode on/off
        self.dir         = None      # manual entry for path to xtc files
        self.files       = None      # xtc2 file path
        self.shmem       = None
        self.destination = 0         # callback that returns rank no. (used by EventBuilder)
        self.monitor     = False     # turns prometheus monitoring client of/off
        self.small_xtc   = []        # swap smd file(s) with bigdata files for these detetors
        self.timestamps  = np.empty(0, dtype=np.uint64)
                                     # list of user-selected timestamps
        self.dbsuffix  = ''          # calibration database name extension for private constants
        self.intg_det  = ''          # integrating detector name (contains marker ts for a batch)
        self.current_retry_no = 0    # global counting var for no. of read attemps
        self.smd_callback= 0

        if kwargs is not None:
            self.smalldata_kwargs = {}
            keywords = ('exp',
                    'dir',
                    'files',
                    'shmem',
                    'drp',
                    'filter',
                    'batch_size',
                    'max_events',
                    'detectors',
                    'xdetectors',
                    'det_name',
                    'destination',
                    'live',
                    'smalldata_kwargs',
                    'monitor',
                    'small_xtc',
                    'timestamps',
                    'dbsuffix',
                    'intg_det',
                    'smd_callback',
                    'psmon_publish',
                    'psplotdb_server',
                    )

            for k in keywords:
                if k in kwargs:
                    if k == 'timestamps':
                        msg = 'Numpy array or .npy filename is required for timestamps argument'
                        assert isinstance(kwargs[k], (np.ndarray, str)), msg
                        # For numpy array, format timestamps to uint64 (for correct search result)
                        if isinstance(kwargs[k], (np.ndarray,)):
                            setattr(self, k, np.asarray(kwargs[k], dtype=np.uint64))
                        else:
                            setattr(self, k, kwargs[k])
                    else:
                        setattr(self, k, kwargs[k])

            if self.destination != 0:
                self.batch_size = 1 # reset batch_size to prevent L1 transmitted before BeginRun (FIXME?: Mona)

            if 'run' in kwargs:
                setattr(self, 'runnum', kwargs['run'])

            if 'files' in kwargs:
                if isinstance(self.files, str):
                    self.files = [self.files]

            if 'dbsuffix' in kwargs:
                setattr(self, 'dbsuffix', kwargs['dbsuffix'])

            max_retries = 0
            if self.live:
                max_retries = int(os.environ.get('PS_R_MAX_RETRIES', '60'))
            else:
                os.environ['PS_R_MAX_RETRIES'] = '0'

        assert self.batch_size > 0

        self.prom_man = PrometheusManager(os.environ['PS_PROMETHEUS_JOBID'])
        self.dsparms  = DsParms(self.batch_size,
                self.max_events,
                self.filter,
                self.destination,
                self.prom_man,
                max_retries,
                self.live,
                self.smd_inprogress_converted,
                self.timestamps,
                self.intg_det,
                self.smd_callback,
                )

        if 'mpi_ts' not in kwargs:
            self.dsparms.set_timestamps()
        else:
            if kwargs['mpi_ts'] == 0:
                self.dsparms.set_timestamps()

    def smd_inprogress_converted(self):
        """ Returns a list of True/False if smd.xtc2.inprogress has been
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
        """ Sets terminate flag

        The Events iterators of all Run types check this flag
        to see if they need to stop (raise StopIterator for
        RunSerial, RunShmem, & RunSinglefile or skip events
        for RunParallel (see BigDataNode implementation).

        Note that mpi_ds implements this differently by adding
        Isend prior to setting this flag.
        """
        self.dsparms.terminate_flag = True

    def unique_user_rank(self):
        """ Only applicable to MPIDataSource
        All other types of DataSource always return True.
        For MPIDataSource, only the last bigdata rank returns True."""
        if not self.is_mpi():
            return True

        n_srv_ranks = int(os.environ.get('PS_SRV_NODES', 0))
        my_rank = self.comms.world_rank
        world_size = self.comms.world_size

        if my_rank == (world_size - n_srv_ranks - 1):
            return True
        else:
            return False

    def is_srv(self):
        """ Only NullDataSource is the srv node. """
        return False

    # to be added at a later date...
    #@abc.abstractmethod
    #def steps(self):
    #    retur

    def _check_file_exist_with_retry(self, xtc_file):
        """ Returns isfile flag and the true xtc file name (.xtc2 or .inprogress).
        """
        # Reconstruct .inprogress file name
        dirname = os.path.dirname(xtc_file)
        fn_only = os.path.splitext(os.path.basename(xtc_file))[0]
        inprogress_file = os.path.join(dirname, fn_only+'.xtc2.inprogress')

        # Check if either one exists
        file_found = os.path.isfile(xtc_file)
        true_xtc_file = xtc_file
        if file_found == False:
            file_found = os.path.isfile(inprogress_file)
            if file_found == True:
                true_xtc_file = inprogress_file

        # Retry if live mode is set
        while self.current_retry_no < self.dsparms.max_retries:
            file_found = os.path.isfile(xtc_file)
            if file_found == False:
                file_found = os.path.isfile(inprogress_file)
                if file_found == True:
                    true_xtc_file = inprogress_file
            if file_found:
                break
            self.current_retry_no += 1
            print(f'Waiting for {xtc_file} ...(#retry:{self.current_retry_no})')
            time.sleep(1)
        return file_found, true_xtc_file

    def _get_file_info_from_db(self, runnum):
        """ Returns a list of xtc2 files as shown in the db.

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
            all_xtc_files=resp.json()["value"]
        except Exception:
            print(f'Warning: unable to connect to {url}')
            all_xtc_files = []

        file_info = {}
        if all_xtc_files:
            # Only take chunk 0 xtc files (matched with *-c*0.)
            xtc_files = [os.path.basename(xtc_file) for xtc_file in all_xtc_files if re.search(r'-c(.+?)0\.', xtc_file)]
            if xtc_files:
                file_info['xtc_files'] = xtc_files
                file_info['dirname'] = os.path.dirname(all_xtc_files[0])
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
            xtc_files_from_db   = file_info['xtc_files']
            smd_files   = [os.path.join(self.xtc_path, 'smalldata',
                    os.path.splitext(xtc_file_from_db)[0]+'.smd.xtc2')
                    for xtc_file_from_db in xtc_files_from_db]

            self.current_retry_no = 0
            for i_smd, smd_file in enumerate(smd_files):
                flag_found, true_xtc_file = self._check_file_exist_with_retry(smd_file)
                if not flag_found:
                    raise FileNotFoundError(true_xtc_file)
                smd_files[i_smd] = true_xtc_file

        else:
            smd_dir = os.path.join(self.xtc_path, 'smalldata')
            smd_files = sorted(glob.glob(os.path.join(smd_dir, '*r%s-s*.smd.xtc2'%(str(runnum).zfill(4)))) + \
                    glob.glob(os.path.join(smd_dir, '*r%s-s*.smd.xtc2.inprogress'%(str(runnum).zfill(4)))))

        self.n_files   = len(smd_files)
        assert self.n_files > 0 , f"No smalldata files found from this path: {os.path.join(self.xtc_path, 'smalldata')}"

        # Look for matching bigdata files - MUST match all.
        # We start by looking for smd basename with .inprogress extension.
        # If this name is not found, try .xtc2.
        xtc_files = [os.path.join(self.xtc_path, os.path.basename(smd_file).split('.smd')[0] + ".xtc2")
                for smd_file in smd_files]
        for i_xtc, xtc_file in enumerate(xtc_files):
            flag_found, true_xtc_file = self._check_file_exist_with_retry(xtc_file)
            if not flag_found:
                raise FileNotFoundError(true_xtc_file)
            xtc_files[i_xtc] = true_xtc_file

        self.smd_files = smd_files
        self.xtc_files = xtc_files

        logger.debug('smd_files:\n'+'\n'.join(self.smd_files))
        logger.debug('xtc_files:\n'+'\n'.join(self.xtc_files))

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
            xtc_dir = os.environ.get('SIT_PSDM_DATA', '/reg/d/psdm')
            self.xtc_path = os.path.join(xtc_dir, self.exp[:3], self.exp, 'xtc')

        if self.runnum is None:
            run_list = [int(os.path.splitext(os.path.basename(_dummy))[0].split('-r')[1].split('-')[0]) \
                    for _dummy in glob.glob(os.path.join(self.xtc_path, '*-r*.xtc2'))]
            assert run_list
            run_list = list(set(run_list))
            run_list.sort()
            self.runnum_list = run_list
        elif isinstance(self.runnum, list):
            self.runnum_list = self.runnum
        elif isinstance(self.runnum, int):
            self.runnum_list = [self.runnum]
        else:
            raise InvalidDataSourceArgument("run accepts only int or list. Leave out run arugment to process all available runs.")

    def smalldata(self, **kwargs):
        self.smalldata_obj.setup_parms(**kwargs)
        return self.smalldata_obj

    def _start_prometheus_client(self, mpi_rank=0, prom_cfg_dir=None):
        if not self.monitor:
            logger.debug('ds_base: RUN W/O PROMETHEUS CLENT')
        elif prom_cfg_dir is None: # Use push gateway
            logger.debug('ds_base: START PROMETHEUS CLIENT (JOBID:%s RANK: %d)'%(self.prom_man.jobid, mpi_rank))
            self.e = threading.Event()
            self.t = threading.Thread(name='PrometheusThread%s'%(mpi_rank),
                    target=self.prom_man.push_metrics,
                    args=(self.e, mpi_rank),
                    daemon=True)
            self.t.start()
        else:                      # Use http exposer
            logger.debug('ds_base: START PROMETHEUS CLIENT (PROM_CFG_DIR: %s)'%(prom_cfg_dir))
            self.e = None
            self.prom_man.create_exposer(prom_cfg_dir)

    def _end_prometheus_client(self, mpi_rank=0):
        if not self.monitor:
            return

        if self.e is not None:     # Push gateway case only
            logger.debug('ds_base: END PROMETHEUS CLIENT (JOBID:%s RANK: %d)'%(self.prom_man.jobid, mpi_rank))
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
            smd_fds  = np.array([os.open(smd_file, os.O_RDONLY) for smd_file in self.smd_files], dtype=np.int32)
            logger.debug(f'smd0 opened tmp smd_fds for detection selection: {smd_fds}')
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
                        det_list.append(f'{detname}_{seg_id}')
                logger.debug(f'Stream:{i} detectors:{all_det_dict[i]}')

            # Apply detector selection exclusion
            if self.detectors or self.xdetectors:
                include_set = set(self.detectors)
                exclude_set = set(self.xdetectors)
                logger.debug(f'Applying detector selection/exclusion')
                logger.debug(f'  Included: {include_set}')
                logger.debug(f'  Excluded: {exclude_set}')

                # This will become 'key' to the new det_dict
                cn_keeps = 0
                for i in range(n_smds):
                    flag_keep = True
                    exist_set = set(all_det_dict[i])
                    logger.debug(f'  Stream:{i}')
                    if self.detectors:
                        if not include_set.intersection(exist_set):
                            flag_keep = False
                            logger.debug(f'  |-- Discarded, not matched given detectors')
                    if self.xdetectors and flag_keep:
                        matched_set = exclude_set.intersection(exist_set)
                        if matched_set:
                            flag_keep = False
                            logger.debug(f'  |-- Discarded, matched with excluded detectors')
                            # We only warn users in the case where we exclude a detector
                            # and there're more than one detectors in the file.
                            if len(exist_set) > len(matched_set):
                                print(f'Warning: Stream-{i} has one or more detectors matched with the excluded set. All detectors in this stream will be excluded.')

                    if flag_keep:
                        if self.xtc_files: xtc_files.append(self.xtc_files[i])
                        if self.smd_files: smd_files.append(self.smd_files[i])
                        configs.append(config)
                        det_dict[cn_keeps] = all_det_dict[i]
                        cn_keeps += 1
                        logger.debug(f'  |-- Kept')
            else:
                xtc_files = self.xtc_files[:]
                smd_files = self.smd_files[:]
                configs = all_configs
                det_dict = all_det_dict

            use_smds = [False] * len(smd_files)
            if self.small_xtc:
                s1 = set(self.small_xtc)
                logger.debug(f'Applying smalldata replacement')
                logger.debug(f'  Smalldata: {self.small_xtc}')
                for i in range(len(smd_files)):
                    exist_set = set(det_dict[i])
                    logger.debug(f' Stream:{i}')
                    if s1.intersection(exist_set):
                        smd_files[i] = xtc_files[i]
                        use_smds[i] = True
                        logger.debug(f'  |-- Replaced with smalldata')
                    else:
                        logger.debug(f'  |-- Kept with bigdata')

            self.xtc_files = xtc_files
            self.smd_files = smd_files

            for smd_fd in smd_fds:
                os.close(smd_fd)
            logger.debug(f"ds_base: close tmp smd fds:{smd_fds}")

        self.dsparms.set_use_smds(use_smds)

    def _setup_run_calibconst(self):
        """
        note: calibconst is set differently in MPIDataSource and DrpDataSource
        """
        runinfo = self._get_runinfo()
        if not runinfo:
            return
        expt, runnum, _ = runinfo

        self.dsparms.calibconst = {}
        for det_name, configinfo in self.dsparms.configinfo_dict.items():
            if expt:
                if expt == "cxid9114": # mona: hack for cctbx
                    det_uniqueid = "cspad_0002"
                elif expt == "xpptut15":
                    det_uniqueid = "cspad_detnum1234"
                else:
                    det_uniqueid = configinfo.uniqueid
                calib_const = wu.calib_constants_all_types(det_uniqueid, exp=expt, run=runnum, dbsuffix=self.dbsuffix)
                self.dsparms.calibconst[det_name] = calib_const
            else:
                print(f"ds_base: Warning: cannot access calibration constant (exp is None)")
                self.dsparms.calibconst[det_name] = None

    def _get_runinfo(self):
        expt, runnum, timestamp = (None, None, None)

        if self.beginruns:
            beginrun_dgram = self.beginruns[0]
            if hasattr(beginrun_dgram, 'runinfo'): # some xtc2 do not have BeginRun
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
            logger.debug(f'ds_base: close smd fds: {self.smd_fds}')


