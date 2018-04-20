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
    test_vals = [d.xpphsd.fex.intFex==42 for d in evt]
    print("-debug mode- rank: %d %s"%(rank, test_vals))
    assert all(test_vals)

def reader(ds):
    """ reads dgram sequentially """
    if ds.smd_files is None:
        for evt in ds.dm:
            yield evt
    else:
        smd_dm = DgramManager(ds.smd_files)
        dm = DgramManager(ds.xtc_files, configs=smd_dm.configs)
        for offset_evt in smd_dm:
            if ds.filter: 
                if not ds.filter(offset_evt): continue
            offsets_and_sizes = np.asarray([[d.info.offsetAlg.intOffset, d.info.offsetAlg.intDgramSize] for d in offset_evt], dtype='i')
            evt = dm.next(offsets=offsets_and_sizes[:,0], 
                      sizes=offsets_and_sizes[:,1], read_chunk=False)
            debug(evt)
            yield evt
        
def server(ds):
    """ 
    server reads small data for offsets and dgram sizes 
    then hands them out clients in a batch (batch default size=1)
    """
    rankreq = np.empty(1, dtype='i')
    offset_batch = np.ones([ds.nfiles, 2, ds.lbatch], dtype='i') * -1
    nevent = 0
    for evt in ds.dm:
        if ds.filter: 
            if not ds.filter(evt): continue
        offset_batch[:,:,nevent % ds.lbatch] = [[d.info.offsetAlg.intOffset, d.info.offsetAlg.intDgramSize] \
                                               for d in evt]
        if nevent % ds.lbatch == ds.lbatch - 1:
            comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            comm.Send(offset_batch, dest=rankreq[0])
            offset_batch = offset_batch * 0 -1
        nevent += 1

    if not (offset_batch==-1).all():
        comm.Recv(rankreq, source=MPI.ANY_SOURCE)
        comm.Send(offset_batch, dest=rankreq[0])

    for i in range(size-1):
        comm.Recv(rankreq, source=MPI.ANY_SOURCE)
        comm.Send(offset_batch * 0 -1, dest=rankreq[0])

def client(ds):
    """
    client uses offsets and dgram sizes to read
    big data
    """
    is_done = False
    while not is_done:
        comm.Send(np.array([rank], dtype='i'), dest=0)
        offset_batch = np.empty([ds.nfiles, 2, ds.lbatch], dtype='i')
        comm.Recv(offset_batch, source=0)
        for i in range(offset_batch.shape[2]):
            offsets_and_sizes = offset_batch[:,:,i]
            if (offsets_and_sizes == -1).all(): 
                is_done = True
                break
            evt = ds.dm.next(offsets=offsets_and_sizes[:,0],
                          sizes=offsets_and_sizes[:,1], read_chunk=False)
            debug(evt)
            yield evt

def read_event(ds, event_type=None):
    """ 
    reads an event (parallel read when size > 1)
    : this def is used to go throuh events
    using offsets from smd files. event_type is
    used to determined config updates or other
    types of event.
    """
    if size == 1:
        for evt in reader(ds): yield evt       # safe for python2
    else:
        if rank == 0:
            server(ds)
        else:
            for evt in client(ds): yield evt 

class Run(object):
    """ Run generator """
    def __init__(self, ds):
        self.ds = ds

    def events(self): 
        for evt in read_event(self.ds): yield evt
            
    def configUpdates(self):
        for i in range(1):
            yield ConfigUpdate(self)

class ConfigUpdate(object):
    """ ConfigUpdate generator """
    def __init__(self, run):
        self.run = run

    def events(self):
        for evt in read_event(self.run.ds, event_type="config"):
            yield evt

class DataSource(object):
    """ Uses DgramManager to read XTC files  """ 
    def __init__(self, expstr, filter=filter, lbatch=1):
        assert lbatch > 0
        self.lbatch = lbatch
        self.filter = filter
        
        is_test = False
        if isinstance(expstr, (str)):
            if expstr.find("exp") == -1:
                self.xtc_files = np.array([expstr], dtype='U%s'%FN_L)
                self.smd_files = None
            else:
                is_test = True
        elif isinstance(expstr, (list, np.ndarray)):
            self.xtc_files = np.asarray(expstr, dtype='U%s'%FN_L)
            self.smd_files = None
        
        if not is_test:
            self.dm = DgramManager(self.xtc_files)
        else:
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
        
                if rank == 0:
                    self.dm = DgramManager(self.smd_files) 
                    configs = self.dm.configs
                    nbytes = np.array([memoryview(config).shape[0] for config in configs], dtype='i')
                else:
                    self.dm = None
                    configs = [dgram.Dgram() for i in range(self.nfiles)]
                    nbytes = np.empty(self.nfiles, dtype='i')
    
                comm.Bcast(nbytes, root=0) # no. of bytes is required for mpich
                for i in range(self.nfiles): 
                    comm.Bcast([configs[i], nbytes[i], MPI.BYTE], root=0)
            
                if rank > 0:
                    self.dm = DgramManager(self.xtc_files, configs=configs)
 
    def runs(self): 
        nruns = 1
        for run_no in range(nruns):
            yield Run(self)

    def events(self): 
        for run in self.runs():
            for evt in run.events(): yield evt


