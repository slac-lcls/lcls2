import numpy as np
import glob, os
from psana.dgrammanager import DgramManager
from psana import dgram
from mpi4py import MPI
import pickle

FILENAME_LEN = 200
PERCENT_SMD = .25

class Error(Exception):
    pass

class InputError(Error):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

class DataSourceHelper(object):
    """ initializes datasource"""
    def __init__(self, expstr, ds):
        ds.nodetype = 'bd'
        ds.run = -1
        rank = ds.mpi.rank
        size = ds.mpi.size
        comm = ds.mpi.comm
        
        # Check if we are reading file(s) or an experiment
        read_exp = False
        if isinstance(expstr, (str)):
            if expstr.find("exp") == -1:
                xtc_files = np.array([expstr], dtype='U%s'%FILENAME_LEN)
                smd_files = None
            else:
                read_exp = True
        elif isinstance(expstr, (list, np.ndarray)):
            xtc_files = np.asarray(expstr, dtype='U%s'%FILENAME_LEN)
            smd_files = None
        
        # Reads list of xtc files from experiment folder 
        if read_exp:
            if rank == 0:
                opts = expstr.split(':')
                exp = {}
                for opt in opts:
                    items = opt.split('=')
                    assert len(items) == 2
                    exp[items[0]] = items[1]

                run = -1
                if 'dir' in exp:
                    xtc_path = exp['dir']
                else:
                    xtc_dir = os.environ.get('SIT_PSDM_DATA', '/reg/d/psdm')
                    xtc_path = os.path.join(xtc_dir, exp['exp'][:3], exp['exp'], 'xtc')
                
                if 'run' in exp:
                    run = int(exp['run'])

                if run > -1:
                    xtc_files = np.array(glob.glob(os.path.join(xtc_path, '*r%s*.xtc'%(str(run).zfill(4)))), dtype='U%s'%FILENAME_LEN)
                else:
                    xtc_files = np.array(glob.glob(os.path.join(xtc_path, '*.xtc')), dtype='U%s'%FILENAME_LEN)

                ds.run = run
                xtc_files.sort()
                
                # smd files are needed only for parallel read
                if size > 1:
                    smd_files = np.empty(len(xtc_files), dtype='U%s'%FILENAME_LEN)
                    smd_dir = os.path.join(xtc_path, 'smalldata')
                    for i, xtc_file in enumerate(xtc_files):
                        smd_file = os.path.join(smd_dir,
                                os.path.splitext(os.path.basename(xtc_file))[0] + '.smd.xtc')
                        if os.path.isfile(smd_file):
                            smd_files[i] = smd_file
                        else:
                            raise InputError(smd_file, "File not found.")

                nfiles = np.array([len(xtc_files)], dtype='i')
                if nfiles[0] == 0:
                    raise InputError(nfiles[0]==0, "No file found for the given experiment and run no.")
            else:
                # Do nothing on other ranks
                nfiles = np.zeros(1, dtype='i')
        
        # Setup DgramManager, Configs, and Calib
        if size == 1:
            # This is one-core read.
            ds.dm = DgramManager(xtc_files)
            ds.configs = ds.dm.configs
            ds.calib = self.get_calib_dict(run_no=ds.run)
        else:
            # This is parallel read. 
            comm.Bcast(nfiles, root=0)

            if rank > 0:
                xtc_files = np.empty(nfiles[0], dtype='U%s'%FILENAME_LEN)
                smd_files = np.empty(nfiles[0], dtype='U%s'%FILENAME_LEN)

            comm.Bcast([xtc_files, MPI.CHAR], root=0)
            comm.Bcast([smd_files, MPI.CHAR], root=0)

            # Send configs
            if rank == 0:
                ds.smd_dm = DgramManager(smd_files)
                ds.smd_configs = ds.smd_dm.configs # FIXME: there should only be one type of config
                smd_nbytes = np.array([memoryview(config).shape[0] for config in ds.smd_configs], \
                            dtype='i')
                ds.dm = DgramManager(xtc_files)
                ds.configs = ds.dm.configs
                nbytes = np.array([memoryview(config).shape[0] for config in ds.configs], \
                                                        dtype='i')
            else:
                ds.smd_dm = None
                ds.smd_configs = [dgram.Dgram() for i in range(nfiles[0])]
                smd_nbytes = np.empty(nfiles[0], dtype='i')
                ds.dm = None
                ds.configs = [dgram.Dgram() for i in range(nfiles[0])]
                nbytes = np.empty(nfiles[0], dtype='i')

            comm.Bcast(smd_nbytes, root=0) # no. of bytes is required for mpich
            for i in range(nfiles[0]):
                comm.Bcast([ds.smd_configs[i], smd_nbytes[i], MPI.BYTE], root=0)
            comm.Bcast(nbytes, root=0) # no. of bytes is required for mpich
            for i in range(nfiles[0]):
                comm.Bcast([ds.configs[i], nbytes[i], MPI.BYTE], root=0)
                
            # Send calib
            if rank == 0:
                ds.calib = self.get_calib_dict(run_no=ds.run)
            else:
                ds.calib = None
            ds.calib = comm.bcast(ds.calib, root=0)
            # Assign node types
            ds.nsmds = int(os.environ.get('PS_SMD_NODES', np.ceil((size-1)*PERCENT_SMD)))
            if rank == 0:
                ds.nodetype = 'smd0'
            elif rank < ds.nsmds + 1:
                ds.nodetype = 'smd'

            if ds.nodetype == 'bd':
                ds.dm = DgramManager(xtc_files, configs=ds.configs)

    def get_calib_dict(self, run_no=-1):
        """ Creates dictionary object that stores calibration constants.
        This routine will be replaced with calibration reading (psana2 style)"""
        calib_dir = os.environ.get('PS_CALIB_DIR')
        calib = None
        if calib_dir:
            gain_mask = None
            pedestals = None
            if os.path.exists(os.path.join(calib_dir,'gain_mask.pickle')):
                gain_mask = pickle.load(open(os.path.join(calib_dir,'gain_mask.pickle'), 'r'))
            
            # Find corresponding pedestals
            if run_no > -1: # Do not fetch pedestals when run_no is not given
                if os.path.exists(os.path.join(calib_dir,'pedestals.npy')):
                    pedestals = np.load(os.path.join(calib_dir,'pedestals.npy'))
                else:
                    files = glob.glob(os.path.join(calib_dir,"*-end.npy"))
                    darks = np.sort([int(os.path.basename(file_name).split('-')[0]) for file_name in files])
                    sel_darks = darks[(darks < run_no)]
                    if sel_darks.size > 0:
                        if os.path.exists(os.path.join(calib_dir,'%s-end.npy'%sel_darks[0])):
                            pedestals = np.load(os.path.join(calib_dir, '%s-end.npy'%sel_darks[0]))
            calib = {'gain_mask': gain_mask,
                     'pedestals': pedestals}
        return calib


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

