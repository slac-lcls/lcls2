import sys, os
from psana import dgram
from psana.dgrammanager import DgramManager
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def reader(ds, analyze, filter):
    smd_dm = DgramManager(ds.smd_files)
    dm = DgramManager(ds.xtc_files)
    for offset_evt in smd_dm:
        if filter: 
            if not filter(offset_evt): continue
        offsets = np.asarray([d.info.offsetAlg.intOffset for d in offset_evt], dtype='i')
        evt = dm.next(offsets=offsets)
        
        analyze(evt)
        
        if os.environ.get('PSANA_DEBUG'):
            print("Debug Mode")
            test_vals = [d.xpphsd.fex.intFex==42 for d in evt]
            assert all(test_vals)

def server(ds, filter):
    dm = DgramManager(ds.smd_files)
    byterank = bytearray(32)
    offsets = np.zeros(len(dm.configs), dtype='i')
    for evt in dm:
        if filter: 
            if not filter(evt): continue
        comm.Recv(byterank, source=MPI.ANY_SOURCE)
        rankreq = int.from_bytes(byterank, byteorder=sys.byteorder)
        offsets = np.asarray([d.info.offsetAlg.intOffset for d in evt], dtype='i')
        comm.Send(offsets, dest=rankreq)
    for rankreq in range(size-1):
        comm.Recv(byterank, source=MPI.ANY_SOURCE)
        rankreq = int.from_bytes(byterank, byteorder=sys.byteorder)
        comm.Send(offsets*0-1, dest=rankreq)

def client(ds, configs, analyze):
    dm = DgramManager(ds.xtc_files, configs=configs)
    while True:
        comm.Send(rank.to_bytes(32, byteorder=sys.byteorder), dest=0)
        offsets = np.empty(ds.nfiles, dtype='i')
        comm.Recv(offsets, source=0)
        if all(offsets == -1): break
        
        evt = dm.next(offsets=offsets)
        analyze(evt)
        
        if os.environ.get('PSANA_DEBUG'):
            print("Debug Mode")
            test_vals = [d.xpphsd.fex.intFex==42 for d in evt]
            assert all(test_vals)


class DataSource(object):
    """ Uses DgramManager to read XTC files  """ 
    def __init__(self, expstr):
        if size == 1:
            self.xtc_files = np.array(['data.xtc', 'data_1.xtc'], dtype='U25')
            self.smd_files = np.array(['smd.xtc', 'smd_1.xtc'], dtype='U25')
        else:
            self.nfiles = 2
            if rank == 0:
                self.xtc_files = np.array(['data.xtc','data_1.xtc'], dtype='U25')
                self.smd_files = np.array(['smd.xtc', 'smd_1.xtc'], dtype='U25')
            else:
                self.xtc_files = np.empty(self.nfiles, dtype='U25')
                self.smd_files = np.empty(self.nfiles, dtype='U25')
            
            comm.Bcast([self.xtc_files, MPI.CHAR], root=0)
            comm.Bcast([self.smd_files, MPI.CHAR], root=0)
 
    def start(self, analyze, filter=None):
        if size == 1:
            reader(self, analyze, filter)
            return

        if rank == 0:
            dm = DgramManager(self.xtc_files) # todo: remove this server only works with sml ds
            configs = dm.configs
            nbytes = np.array([memoryview(config).nbytes for config in configs], dtype='i')
        else:
            dm = None
            configs = [dgram.Dgram() for i in range(self.nfiles)]
            nbytes = np.empty(self.nfiles, dtype='i')
    
        comm.Bcast(nbytes, root=0) # no. of bytes is required for mpich
        for i in range(self.nfiles): 
            comm.Bcast([configs[i], nbytes[i], MPI.BYTE], root=0)
    
        if rank == 0:
            server(self, filter)
        else:
            client(self, configs, analyze)
