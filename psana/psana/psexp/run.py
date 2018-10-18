from psana.psexp.node import run_node
import numpy as np
from psana.dgrammanager import DgramManager
from psana import dgram
from psana.detector.detector import Detector
from psana.psexp.legion_node import analyze
from psana.psexp.tools import run_from_id, RunHelper
import os

from psana.psexp.tools import mode
MPI = None
if mode == 'mpi':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

class Run(object):
    exp = None
    run_no = None
    dm = None
    configs = None
    smd_dm = None
    smd_configs = None
    max_events = None
    batch_size = None
    filter_callback = None
    
    def __init__(self, exp, run_no, max_events=0, batch_size=1, filter_callback=0):
        self.exp = exp
        self.run_no = run_no
        self.max_events = max_events
        self.batch_size = batch_size
        self.filter_callback = filter_callback

    def run(self):
        """ Returns integer representaion of run no.
        default: (when no run is given) is set to -1"""
        return self.run_no

    def Detector(self, det_name):
        calib = self._get_calib(det_name)
        self.Detector = Detector(self.configs, calib=calib)

    def _get_calib(self, det_name):
        gain_mask, pedestals, geometry_string, common_mode = None, None, None, None
        if self.exp and det_name:
            calib_dir = os.environ.get('PS_CALIB_DIR')
            if calib_dir:
                if os.path.exists(os.path.join(calib_dir,'gain_mask.pickle')):
                    gain_mask = pickle.load(open(os.path.join(calib_dir,'gain_mask.pickle'), 'r'))

            from psana.pscalib.calib.MDBWebUtils import calib_constants
            det = eval('self.configs[0].software.%s'%(det_name))

            # calib_constants takes det string (e.g. cspad_0001) with requested calib type.
            # as a hack (until detid in xtc files have been changed
            det_str = det.dettype + '_' + det.detid
            pedestals, _ = calib_constants(det_str, exp=self.exp, ctype='pedestals', run=self.run_no)
            geometry_string, _ = calib_constants(det_str, exp=self.exp, ctype='geometry', run=self.run_no)

            # python2 sees geometry_string as unicode (use str to check for compatibility py2/py3)
            # - convert to str accordingly
            if not isinstance(geometry_string, str) and geometry_string is not None:
                import unicodedata
                geometry_string = unicodedata.normalize('NFKD', geometry_string).encode('ascii','ignore')
            common_mode, _ = calib_constants(det_str, exp=self.exp, ctype='common_mode', run=self.run_no)

        calib = {'gain_mask': gain_mask,
                 'pedestals': pedestals,
                 'geometry_string': geometry_string,
                 'common_mode': common_mode}
        return calib


class RunSerial(Run):

    def __init__(self, exp, run_no, xtc_files, max_events, filter_callback):
        super().__init__(exp, run_no, max_events=max_events, filter_callback=filter_callback)
        self.dm = DgramManager(xtc_files)
        self.configs = self.dm.configs

    def events(self):
        for evt in self.dm: yield evt

class RunParallel(Run):
    nodetype = None
    nsmds = None

    def __init__(self, exp, run_no, xtc_files, smd_files, nodetype, nsmds,
            max_events, batch_size, filter_callback):
        """ Parallel read requires that rank 0 does the file system works.
        Configs and calib constants are sent to other ranks by MPI."""
        super().__init__(exp, run_no, max_events=max_events, \
                batch_size=batch_size, filter_callback=filter_callback)
        self.nodetype = nodetype
        self.nsmds = nsmds
        
        if rank == 0:
            self.dm = DgramManager(xtc_files)
            self.configs = self.dm.configs
            self.smd_dm = DgramManager(smd_files)
            self.smd_configs = self.smd_dm.configs
            nbytes = np.array([memoryview(config).shape[0] for config in self.configs], \
                            dtype='i')
            smd_nbytes = np.array([memoryview(config).shape[0] for config in self.smd_configs], \
                            dtype='i')
        else:
            self.dm = None
            self.configs = [dgram.Dgram() for i in range(len(xtc_files))]
            self.smd_dm = None
            self.smd_configs = [dgram.Dgram() for i in range(len(smd_files))]
            nbytes = np.empty(len(xtc_files), dtype='i')
            smd_nbytes = np.empty(len(smd_files), dtype='i')
        
        comm.Bcast(smd_nbytes, root=0) # no. of bytes is required for mpich
        for i in range(len(smd_files)):
            comm.Bcast([self.smd_configs[i], smd_nbytes[i], MPI.BYTE], root=0)
        comm.Bcast(nbytes, root=0) # no. of bytes is required for mpich
        for i in range(len(xtc_files)):
            comm.Bcast([self.configs[i], nbytes[i], MPI.BYTE], root=0)
        
        # This creates dgrammanager without reading config from disk
        self.dm = DgramManager(xtc_files, configs=self.configs)

    
    def events(self):
        for evt in run_node(self, self.nodetype, self.nsmds, self.max_events, \
                self.batch_size, self.filter_callback):
            yield evt

    def Detector(self, det_name):
        if rank == 0:
            calib = super()._get_calib(det_name)
        else:
            calib = None
        calib = comm.bcast(calib, root=0)
        self.Detector = Detector(self.configs, calib=calib)

class RunLegion(Run):

    def __init__(self, exp, run_no, xtc_files, smd_files, 
            max_events, batch_size, filter_callback):
        """ Parallel read requires that rank 0 does the file system works.
        Configs and calib constants are sent to other ranks by MPI."""
        super().__init__(exp, run_no, max_events=max_events, \
                batch_size=batch_size, filter_callback=filter_callback)
        
        self.dm = DgramManager(xtc_files)
        self.configs = self.dm.configs
        self.smd_dm = DgramManager(smd_files)
        self.smd_configs = self.smd_dm.configs
        RunHelper(self) # assigns a unique id to the run
        
    def Detector(self, det_name):
        calib = super()._get_calib(det_name)

    def analyze(self, **kwargs):
        analyze(self, **kwargs)

    def __reduce__(self):
        return (run_from_id, (self.id,))
