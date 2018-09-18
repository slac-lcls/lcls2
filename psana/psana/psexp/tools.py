import numpy as np
import glob, os
from psana.dgrammanager import DgramManager
from psana import dgram
import pickle
import weakref

from psana.psexp.node import mode
MPI = None
if mode == 'mpi':
    from mpi4py import MPI


class Error(Exception):
    pass

class InputError(Error):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

class DataSourceHelper(object):
    """ initializes datasource"""

    # Every DataSource is assigned an ID. This permits DataSource to be
    # pickled and sent across the network, as long as every node has the same
    # DataSource under the same ID. (This should be true as long as the client
    # code initializes DataSources in a deterministic order.)
    next_ds_id = 0
    ds_by_id = weakref.WeakValueDictionary()

    def __init__(self, expstr, ds):
        ds.id = self.next_ds_id
        self.next_ds_id += 1
        self.ds_by_id[ds.id] = ds

        ds.nodetype = 'bd'
        rank = ds.mpi.rank
        size = ds.mpi.size
        comm = ds.mpi.comm

        if rank == 0 or mode == 'legion':
            exp, run_dict = self.parse_expstr(expstr)
        else:
            exp, run_dict = None, None

        if size == 1 and mode != 'legion':
            pass
        else:
            exp = comm.bcast(exp, root=0)
            run_dict = comm.bcast(run_dict, root=0)

        ds.exp = exp
        ds.run_dict = run_dict

        # No. of smd nodes (default is 1)
        ds.nsmds = int(os.environ.get('PS_SMD_NODES', 1))
        if rank == 0:
            ds.nodetype = 'smd0'
        elif rank < ds.nsmds + 1:
            ds.nodetype = 'smd'

    def parse_expstr(self, expstr):
        exp = None
        run_dict = {} # stores list of runs with corresponding xtc_files and smd_files

        # Check if we are reading file(s) or an experiment
        read_exp = False
        if isinstance(expstr, (str)):
            if expstr.find("exp") == -1:
                xtc_files = [expstr]
                smd_files = None
                run_dict[-1] = (xtc_files, smd_files)
            else:
                read_exp = True
        elif isinstance(expstr, (list, np.ndarray)):
            xtc_files = expstr
            smd_files = None
            run_dict[-1] = (xtc_files, smd_files)

        # Reads list of xtc files from experiment folder
        if read_exp:
            opts = expstr.split(':')
            exp_dict = {}
            for opt in opts:
                items = opt.split('=')
                assert len(items) == 2
                exp_dict[items[0]] = items[1]

            assert 'exp' in exp_dict
            exp = exp_dict['exp']

            if 'dir' in exp_dict:
                xtc_path = exp_dict['dir']
            else:
                xtc_dir = os.environ.get('SIT_PSDM_DATA', '/reg/d/psdm')
                xtc_path = os.path.join(xtc_dir, exp_dict['exp'][:3], exp_dict['exp'], 'xtc')
            
            run_num = -1
            if 'run' in exp_dict:
                run_num = int(exp_dict['run'])
            
            # get a list of runs (or just one run if user specifies it) then
            # setup corresponding xtc_files and smd_files for each run in run_dict
            run_list = []
            if run_num > -1:
                run_list = [run_num]
            else:
                run_list = [int(os.path.splitext(os.path.basename(_dummy))[0].split('-r')[1].split('-')[0]) \
                        for _dummy in glob.glob(os.path.join(xtc_path, '*-r*.xtc'))]
                run_list.sort()

            smd_dir = os.path.join(xtc_path, 'smalldata')
            for r in run_list:
                xtc_files = glob.glob(os.path.join(xtc_path, '*r%s*.xtc'%(str(r).zfill(4))))
                smd_files = [os.path.join(smd_dir,
                             os.path.splitext(os.path.basename(xtc_file))[0] + '.smd.xtc')
                             for xtc_file in xtc_files]
                run_dict[r] = (xtc_files, smd_files)

        return exp, run_dict


class RunHelper(object):
    def __init__(self, ds, run_no):
        """Setups per-run required objects
        Creates per-run dgrammanager and calib then send them
        to all client ranks."""
        self.ds = ds
        self.run_no = run_no
        xtc_files, smd_files = self.ds.run_dict[run_no]
        size = ds.mpi.size
        rank = ds.mpi.rank
        comm = ds.mpi.comm

        if size == 1 and mode != 'legion':
            # This is one-core read.
            self._setup_dgrammanager()
            self._setup_calib()
        else:
            # This is parallel read.
            # Rank 0 creates dgrammanager (+configs) and calib dict.
            # then sends them to all other ranks. Note that only rank0
            # owns smd dgrammanger (see Smd0 and Smd implementations in node.py)
            if rank == 0 or mode == 'legion':
                self._setup_dgrammanager()
                self._setup_calib()
                smd_nbytes = np.array([memoryview(config).shape[0] for config in ds.smd_configs], \
                            dtype='i')
                nbytes = np.array([memoryview(config).shape[0] for config in ds.configs], \
                                                        dtype='i')
            else:
                ds.smd_dm = None
                ds.smd_configs = [dgram.Dgram() for i in range(len(smd_files))]
                smd_nbytes = np.empty(len(smd_files), dtype='i')
                ds.dm = None
                ds.configs = [dgram.Dgram() for i in range(len(xtc_files))]
                nbytes = np.empty(len(xtc_files), dtype='i')
                ds.calib = None

            if mode != 'legion':
                comm.Bcast(smd_nbytes, root=0) # no. of bytes is required for mpich
                for i in range(len(smd_files)):
                    comm.Bcast([ds.smd_configs[i], smd_nbytes[i], MPI.BYTE], root=0)
                comm.Bcast(nbytes, root=0) # no. of bytes is required for mpich
                for i in range(len(xtc_files)):
                    comm.Bcast([ds.configs[i], nbytes[i], MPI.BYTE], root=0)

                ds.calib = comm.bcast(ds.calib, root=0)

            # This creates dgrammanager without reading config from disk 
            ds.dm = DgramManager(xtc_files, configs=ds.configs)

    def _setup_dgrammanager(self):
        """Uses run_no to access xtc_files, smd_files, and configs
        """
        ds, run_no = self.ds, self.run_no
        xtc_files, smd_files = ds.run_dict[run_no]
        if smd_files is not None:
            ds.smd_dm = DgramManager(smd_files)
            ds.smd_configs = ds.smd_dm.configs # FIXME: there should only be one type of config
        
        # This creates a dgrammanager while reading the config from disk
        ds.dm = DgramManager(xtc_files)
        ds.configs = ds.dm.configs

    def _setup_calib(self):
        """ Creates dictionary object that stores per-run calibration constants.
        Retrieves calibration constants from webapi and store them
        in dictionary object. Note that calibration constants can only be retrieved
        when expid and det_name are given."""
        ds, run_no = self.ds, self.run_no
        gain_mask, pedestals, geometry_string, common_mode = None, None, None, None
        if ds.exp and ds.det_name:
            calib_dir = os.environ.get('PS_CALIB_DIR')
            if calib_dir:
                if os.path.exists(os.path.join(calib_dir,'gain_mask.pickle')):
                    gain_mask = pickle.load(open(os.path.join(calib_dir,'gain_mask.pickle'), 'r'))

            from psana.pscalib.calib.MDBWebUtils import calib_constants
            det = eval('ds._configs[0].software.%s'%(ds.det_name))
            
            # calib_constants takes det string (e.g. cspad_0001) with requested calib type.
            # as a hack (until detid in xtc files have been changed
            det_str = det.dettype + '_' + det.detid
            pedestals = calib_constants(det_str, exp=ds.exp, ctype='pedestals', run=run_no)
            geometry_string = calib_constants(det_str, exp=ds.exp, ctype='geometry', run=run_no)
            
            # python2 sees geometry_string as unicode (use str to check for compatibility py2/py3)
            # - convert to str accordingly
            if not isinstance(geometry_string, str) and geometry_string is not None:
                import unicodedata
                geometry_string = unicodedata.normalize('NFKD', geometry_string).encode('ascii','ignore')
            common_mode = calib_constants(det_str, exp=ds.exp, ctype='common_mode', run=run_no)
        
        calib = {'gain_mask': gain_mask,
                 'pedestals': pedestals,
                 'geometry_string': geometry_string,
                 'common_mode': common_mode}
        ds.calib = calib

    
def datasource_from_id(ds_id):
    return DataSourceHelper.ds_by_id[ds_id]

class MpiComm(object):
    rank = 0
    size = 1
    comm = None

    def __init__(self):
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        except ImportError:
            pass # do single core read if no mpi

        if self.size > 1:
            # need at least 3 cores for parallel processing
            if self.size < 3:
                raise InputError(self.size, "Parallel processing needs at least 3 cores.")

