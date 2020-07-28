import weakref
import os
import glob
import abc
import numpy as np
import pathlib

from psana.dgrammanager import DgramManager
from psana.smalldata import SmallData

from psana.psexp.prometheus_manager import PrometheusManager
import threading
import logging

class InvalidFileType(Exception): pass
class XtcFileNotFound(Exception): pass

class DataSourceBase(abc.ABC):
    filter      = 0         # callback that takes an evt and return True/False.
    batch_size  = 1         # length of batched offsets
    max_events  = 0         # no. of maximum events
    detectors   = []        # user-selected detector names
    exp         = None      # experiment id (e.g. xpptut13)
    run_num     = -1        # run no. 
    live        = False     # turns live mode on/off 
    dir         = None      # manual entry for path to xtc files
    files       = None      # list of xtc files 
    shmem       = None
    run_dict    = {}
    destination = 0         # callback that returns rank no. (used by EventBuilder)
    monitor     = False      # turns prometheus monitoring client of/off

    def __init__(self, **kwargs):
        """Initializes datasource base"""
        if kwargs is not None:
            self.smalldata_kwargs = {}
            keywords = ('exp', 
                    'dir', 
                    'files', 
                    'shmem', 
                    'filter', 
                    'batch_size', 
                    'max_events', 
                    'detectors', 
                    'det_name',
                    'destination',
                    'live',
                    'smalldata_kwargs', 
                    'monitor')
            
            for k in keywords:
                if k in kwargs:
                    setattr(self, k, kwargs[k])

            if 'run' in kwargs:
                setattr(self, 'run_num', int(kwargs['run']))

            if not self.live:
                os.environ['PS_SMD_MAX_RETRIES'] = '0' # do not retry when not in live mode
            else:
                os.environ['PS_SMD_MAX_RETRIES'] = '30'

        assert self.batch_size > 0
        
        self.prom_man = PrometheusManager(os.environ['PS_PROMETHEUS_JOBID'])
        

    def events(self):
        for run in self.runs():
            for evt in run.events(): yield evt

    @abc.abstractmethod
    def runs(self):
        return

    # to be added at a later date...
    #@abc.abstractmethod
    #def steps(self):
    #    return
    
    def _setup_xtcs(self):
        exp = None
        run_dict = {} # stores list of runs with corresponding xtc_files, smd_files, and epic file

        if self.shmem:
            self.tag = self.shmem
            run_dict[-1] = (['shmem'], None, None)
            return exp, run_dict

        # Reading xtc files in one of these two ways
        assert self.exp != self.files
        
        read_exp = False
        if self.exp:
            read_exp = True
        elif self.files:
            if isinstance(self.files, (str)):
                f = pathlib.Path(self.files)
                if not f.exists():
                    err = f"File {self.files} not found" 
                    raise XtcFileNotFound(err)
                xtc_files = [self.files]
            elif isinstance(self.files, (list, np.ndarray)):
                for xtc_file in self.files:
                    f = pathlib.Path(xtc_file)
                    if not f.exists():
                        err = f"File {xtc_file} not found"
                        raise XtcFileNotFound(err)
                xtc_files = self.files
            else:
                raise InvalidFileType("Only accept filename string or list of files.")

            # In case of reading file(s), user negative integers for the index.
            # If files is a list, separate each file to an individual run.
            for num, xtc_file in enumerate(xtc_files):
                run_dict[-1*(num+1)] = ([xtc_file], None, None)

        # Reads list of xtc files from experiment folder
        if read_exp:
            if self.dir:
                xtc_path = self.dir
            else:
                xtc_dir = os.environ.get('SIT_PSDM_DATA', '/reg/d/psdm')
                xtc_path = os.path.join(xtc_dir, self.exp[:3], self.exp, 'xtc')

            # Get a list of runs (or just one run if user specifies it) then
            # setup corresponding xtc_files and smd_files for each run in run_dict
            run_list = []
            if self.run_num > -1:
                run_list = [self.run_num]
            else:
                run_list = [int(os.path.splitext(os.path.basename(_dummy))[0].split('-r')[1].split('-')[0]) \
                        for _dummy in glob.glob(os.path.join(xtc_path, '*-r*.xtc2'))]
                run_list.sort()

            smd_dir = os.path.join(xtc_path, 'smalldata')
            for r in run_list:
                all_smd_files = glob.glob(os.path.join(smd_dir, '*r%s-s*.smd.xtc2'%(str(r).zfill(4))))
                if self.detectors:
                    # Create a dgrammanager to access the configs. This will be
                    # done on only core 0.
                    s1 = set(self.detectors)
                    smd_dm = DgramManager(all_smd_files)
                    smd_files = [all_smd_files[i] for i in range(len(all_smd_files)) \
                                if s1.intersection(set(smd_dm.configs[i].__dict__.keys()))]
                else:
                    smd_files = all_smd_files
                    
                xtc_files = [os.path.join(xtc_path, \
                             os.path.basename(smd_file).split('.smd')[0] + '.xtc2') \
                             for smd_file in smd_files \
                             if os.path.isfile(os.path.join(xtc_path, \
                             os.path.basename(smd_file).split('.smd')[0] + '.xtc2'))]
                all_files = glob.glob(os.path.join(xtc_path, '*r%s-*.xtc2'%(str(r).zfill(4))))
                other_files = [f for f in all_files if f not in xtc_files]
                run_dict[r] = (xtc_files, smd_files, other_files)
        
        return self.exp, run_dict


    def smalldata(self, **kwargs):
        return SmallData(**self.smalldata_kwargs, **kwargs)

    def _start_prometheus_client(self, mpi_rank=0):
        if not self.monitor:
            logging.debug('not monitoring performance with prometheus')
        else:
            logging.debug('START PROMETHEUS CLIENT (JOBID:%s RANK: %d)'%(self.prom_man.jobid, mpi_rank))
            self.e = threading.Event()
            self.t = threading.Thread(name='PrometheusThread%s'%(mpi_rank),
                    target=self.prom_man.push_metrics,
                    args=(self.e, mpi_rank),
                    daemon=True)
            self.t.start()

    def _end_prometheus_client(self, mpi_rank=0):
        if not self.monitor:
            return

        logging.debug('END PROMETHEUS CLIENT (JOBID:%s RANK: %d)'%(self.prom_man.jobid, mpi_rank))
        self.e.set()

