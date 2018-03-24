import sys, os
from psana import dgram
from psana.dgrammanager import DgramManager
import numpy as np

rank = 0
size = 1
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    pass # do single core read if no mpi

FN_L = 100

def debug(evt):
    if os.environ.get('PSANA_DEBUG'):
        print("Debug Mode")
        test_vals = [d.xpphsd.fex.intFex==42 for d in evt]
        assert all(test_vals)

def process_batch(dm, offset_batch, analyze):
    if (offset_batch == -1).all(): return False
    for offsets in offset_batch:
        if all(offsets == -1): break
        evt = dm.next(offsets=offsets)
        analyze(evt)
        debug(evt)
    return True

def reader(ds, analyze, filter):
    smd_dm = DgramManager(ds.smd_files)
    dm = DgramManager(ds.xtc_files, configs=smd_dm.configs)
    for offset_evt in smd_dm:
        if filter: 
            if not filter(offset_evt): continue
        offsets = np.asarray([d.info.offsetAlg.intOffset for d in offset_evt], dtype='i')
        evt = dm.next(offsets=offsets)
        
        analyze(evt)
        debug(evt)
        
def server(ds, filter):
    byterank = bytearray(32)
    offset_batch = np.ones([ds.lbatch, ds.nfiles], dtype='i') * -1
    nevent = 0
    for evt in ds.dm:
        if filter: 
            if not filter(evt): continue
        offset_batch[nevent % ds.lbatch, :] = [d.info.offsetAlg.intOffset for d in evt]
        if nevent % ds.lbatch == ds.lbatch - 1:
            comm.Recv(byterank, source=MPI.ANY_SOURCE)
            rankreq = int.from_bytes(byterank, byteorder=sys.byteorder)
            comm.Send(offset_batch, dest=rankreq)
            offset_batch = offset_batch * 0 -1
        nevent += 1

    if not (offset_batch==-1).all():
        comm.Recv(byterank, source=MPI.ANY_SOURCE)
        rankreq = int.from_bytes(byterank, byteorder=sys.byteorder)
        comm.Send(offset_batch, dest=rankreq)

    for rankreq in range(size-1):
        comm.Recv(byterank, source=MPI.ANY_SOURCE)
        rankreq = int.from_bytes(byterank, byteorder=sys.byteorder)
        comm.Send(offset_batch * 0 -1, dest=rankreq)

def client(ds, analyze):
    while True:
        comm.Send(rank.to_bytes(32, byteorder=sys.byteorder), dest=0)
        offset_batch = np.empty([ds.lbatch, ds.nfiles], dtype='i')
        comm.Recv(offset_batch, source=0)
        if not process_batch(ds.dm, offset_batch, analyze): break

class DataSource(object):
    """ Uses DgramManager to read XTC files  """ 
    def __init__(self, expstr):
        if size == 1:
            self.xtc_files = np.array(['data.xtc', 'data_1.xtc'], dtype='U%s'%FN_L)
            self.smd_files = np.array(['smd.xtc', 'smd_1.xtc'], dtype='U%s'%FN_L)
        else:
            self.nfiles = 2
            if rank == 0:
                self.xtc_files = np.array(['data.xtc','data_1.xtc'], dtype='U%s'%FN_L)
                self.smd_files = np.array(['smd.xtc', 'smd_1.xtc'], dtype='U%s'%FN_L)
            else:
                self.xtc_files = np.empty(self.nfiles, dtype='U%s'%FN_L)
                self.smd_files = np.empty(self.nfiles, dtype='U%s'%FN_L)
            
            comm.Bcast([self.xtc_files, MPI.CHAR], root=0)
            comm.Bcast([self.smd_files, MPI.CHAR], root=0)
 
    def start(self, analyze, filter=None, lbatch=1):
        assert lbatch > 0
        self.lbatch = lbatch
        if size == 1:
            reader(self, analyze, filter)
            return

        if rank == 0:
            self.dm = DgramManager(self.smd_files) 
            configs = self.dm.configs
            nbytes = np.array([memoryview(config).nbytes for config in configs], dtype='i')
        else:
            self.dm = None
            configs = [dgram.Dgram() for i in range(self.nfiles)]
            nbytes = np.empty(self.nfiles, dtype='i')
    
        comm.Bcast(nbytes, root=0) # no. of bytes is required for mpich
        for i in range(self.nfiles): 
            comm.Bcast([configs[i], nbytes[i], MPI.BYTE], root=0)
    
        if rank == 0:
            server(self, filter)
        else:
            self.dm = DgramManager(self.xtc_files, configs=configs)
            client(self, analyze)
