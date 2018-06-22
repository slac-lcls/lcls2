import sys, os, glob
from psana import dgram
from psana.dgrammanager import DgramManager
from psana.detector.detector import Detector
from psana.smdreader import SmdReader
from psana.event import Event
from psana.eventbuilder import EventBuilder
import numpy as np

class Error(Exception):
    pass

class InputError(Error):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

rank = 0
size = 1
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    pass # do single core read if no mpi

if size > 1:
    # need at least 3 cores for parallel processing
    if size < 3:
        raise InputError(size, "Parallel processing needs at least 3 cores.")

FN_L = 200
n_events = 10000
PERCENT_SMD = .25

def debug(evt):
    test_vals = [d.xpphsd.fex.intFex==42 for d in evt]
    print("-debug mode- rank: %d %s"%(rank, test_vals))
    assert all(test_vals)

def smd_0(fds, n_smd_nodes):
    """ Sends blocks of smds to smd_node
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp to an smd_node.
    """
    assert len(fds) > 0
    smdr = SmdReader(fds)
    got_events = -1
    rankreq = np.empty(1, dtype='i')

    while got_events != 0:
        smdr.get(n_events)
        got_events = smdr.got_events
        views = bytearray()
        for i in range(len(fds)):
            view = smdr.view(i)
            if view != 0:
                views.extend(view)
                if i < len(fds) - 1:
                    views.extend(b'endofstream')

        if views:
            comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            comm.Send(views, dest=rankreq[0], tag=12)

    for i in range(n_smd_nodes):
        comm.Recv(rankreq, source=MPI.ANY_SOURCE)
        comm.Send(bytearray(b'eof'), dest=rankreq[0], tag=12)

def smd_node(configs, n_bd_nodes, batch_size=1, filter=0):
    """Handles both smd_0 and bd_nodes
    Receives blocks of smds from smd_0 then assembles
    offsets and dgramsizes into a numpy array. Sends
    this np array to bd_nodes that are registered to it."""
    rankreq = np.empty(1, dtype='i')
    view = 0
    while True:
        if not view:
            # handles requests from smd_0
            comm.Send(np.array([rank], dtype='i'), dest=0)
            info = MPI.Status()
            comm.Probe(source=0, tag=12, status=info)
            count = info.Get_elements(MPI.BYTE)
            view = bytearray(count)
            comm.Recv(view, source=0, tag=12)
            if view.startswith(b'eof'):
                break

        else:
            # send offset/size(s) to bd_node
            views = view.split(b'endofstream')

            n_views = len(views)
            assert n_views == len(configs)

            # build batch of events
            eb = EventBuilder(views, configs)
            batch = eb.build(batch_size=batch_size, filter=filter)
            
            while eb.nevents:
                comm.Recv(rankreq, source=MPI.ANY_SOURCE, tag=13)
                comm.Send(batch, dest=rankreq[0])
                batch = eb.build(batch_size=batch_size, filter=filter)
            
            view = 0

    for i in range(n_bd_nodes):
        comm.Recv(rankreq, source=MPI.ANY_SOURCE, tag=13)
        comm.Send(bytearray(b'eof'), dest=rankreq[0])

def bd_node(ds, smd_node_id):
    configs = ds.configs
    while True:
        comm.Send(np.array([rank], dtype='i'), dest=smd_node_id, tag=13)
        info = MPI.Status()
        comm.Probe(source=smd_node_id, tag=MPI.ANY_TAG, status=info)
        count = info.Get_elements(MPI.BYTE)
        view = bytearray(count)
        comm.Recv(view, source=smd_node_id)
        if view.startswith(b'eof'):
            break

        views = view.split(b'endofevt')
        for event_bytes in views:
            if event_bytes:
                evt = Event.from_bytes(configs, event_bytes)
                # get big data
                ofsz = np.asarray([[d.info.offsetAlg.intOffset, d.info.offsetAlg.intDgramSize] \
                        for d in evt])
                bd_evt = ds.dm.next(offsets=ofsz[:,0], sizes=ofsz[:,1], read_chunk=False)
                yield bd_evt


def reader(ds):
    """Reads dgram sequentially """
    if ds.smd_files is None:
        for evt in ds.dm:
            yield evt
    else:
        for offset_evt in ds.smd_dm:
            if ds.filter: 
                if not ds.filter(offset_evt): continue
            offsets_and_sizes = np.asarray([[d.info.offsetAlg.intOffset, d.info.offsetAlg.intDgramSize] for d in offset_evt], dtype='i')
            evt = ds.dm.next(offsets=offsets_and_sizes[:,0], 
                      sizes=offsets_and_sizes[:,1], read_chunk=False)
            yield evt
        
def read_event(ds, event_type=None):
    """Reads an event.

    Keywords:
    ds         -- DataSource with access to the smd or xtc files
    event_type -- used to determined config updates or other
                  types of event.
    """
    if size == 1:
        for evt in reader(ds): yield evt       # safe for python2
    else:
        if ds.nodetype == 'smd0':
            smd_0(ds._file_descriptors, ds.nsmds)
        elif ds.nodetype == 'smd':
            bd_nodes = (np.arange(size)[ds.nsmds+1:] % ds.nsmds) + 1
            smd_node(ds.configs, len(bd_nodes[bd_nodes==rank]), \
                    batch_size=ds.batch_size, filter=ds.filter)
        elif ds.nodetype == 'bd':
            smd_node_id = (rank % ds.nsmds) + 1
            for evt in bd_node(ds, smd_node_id): 
                yield evt

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
    def __init__(self, expstr, filter=filter, batch_size=1):
        """Initializes datasource.
        
        Keyword arguments:
        expstr     -- experiment string (eg. exp=xpptut13:run=1) or 
                      a file or list of files (eg. 'data.xtc' or ['data0.xtc','dataN.xtc'])
        batch_size -- length of batched offsets
        
        Supports:
        reading file(s) -- DgramManager handles those files (single core)
        reading an exp  -- Single core owns both smd and xtc DgramManagers
                           Multiple cores: rank 0 owns smd DgramManager
                           rank 1 owns xtc DgramManager
        """

        assert batch_size > 0
        self.batch_size = batch_size
        self.filter = filter
        self.nodetype = 'bd'

        # Check if we are reading file(s) or an experiment
        self.read_files = False
        if isinstance(expstr, (str)):
            if expstr.find("exp") == -1:
                self.xtc_files = np.array([expstr], dtype='U%s'%FN_L)
                self.smd_files = None
            else:
                self.read_files = True
        elif isinstance(expstr, (list, np.ndarray)):
            self.xtc_files = np.asarray(expstr, dtype='U%s'%FN_L)
            self.smd_files = None
        
        if not self.read_files:
            self.dm = DgramManager(self.xtc_files)
            self.configs = self.dm.configs
        else:
            # Fetch xtc files
            if rank == 0:
                opts = expstr.split(':')
                exp = {}
                for opt in opts:
                    items = opt.split('=')
                    assert len(items) == 2
                    exp[items[0]] = items[1]
                
                run = ''
                if 'dir' in exp:
                    xtc_path = exp['dir']
                else:
                    xtc_dir = os.environ.get('SIT_PSDM_DATA', '/reg/d/psdm')
                    xtc_path = os.path.join(xtc_dir, exp['exp'][:3], exp['exp'], 'xtc')
                    if 'run' in exp:
                        run = exp['run']
            
                if run:
                    self.xtc_files = np.array(glob.glob(os.path.join(xtc_path, '*r%s*.xtc'%(run.zfill(4)))), dtype='U%s'%FN_L)
                else:
                    self.xtc_files = np.array(glob.glob(os.path.join(xtc_path, '*.xtc')), dtype='U%s'%FN_L)

                self.xtc_files.sort()
                self.smd_files = np.empty(len(self.xtc_files), dtype='U%s'%FN_L)
                smd_dir = os.path.join(xtc_path, 'smalldata')
                for i, xtc_file in enumerate(self.xtc_files):
                    smd_file = os.path.join(smd_dir, 
                            os.path.splitext(os.path.basename(xtc_file))[0] + '.smd.xtc')
                    if os.path.isfile(smd_file):
                        self.smd_files[i] = smd_file
                    else:
                        raise InputError(smd_file, "File not found.")
                
                self.nfiles = np.array([len(self.xtc_files)], dtype='i')
                assert self.nfiles[0] > 0
            else:
                self.nfiles = np.zeros(1, dtype='i')

            if size == 1:
                self.smd_dm = DgramManager(self.smd_files) 
                self.dm = DgramManager(self.xtc_files, configs=self.smd_dm.configs)
                self.configs = self.smd_dm.configs
            else:
                # Send filenames
                comm.Bcast(self.nfiles, root=0)

                if rank > 0:
                    self.xtc_files = np.empty(self.nfiles[0], dtype='U%s'%FN_L)
                    self.smd_files = np.empty(self.nfiles[0], dtype='U%s'%FN_L)
            
                comm.Bcast([self.xtc_files, MPI.CHAR], root=0)
                comm.Bcast([self.smd_files, MPI.CHAR], root=0)
                
                # Send configs
                if rank == 0:
                    self.dm = DgramManager(self.smd_files) 
                    self.configs = self.dm.configs
                    nbytes = np.array([memoryview(config).shape[0] for config in self.configs], \
                            dtype='i')
                else:
                    self.dm = None
                    self.configs = [dgram.Dgram() for i in range(self.nfiles[0])]
                    nbytes = np.empty(self.nfiles[0], dtype='i')
    
                comm.Bcast(nbytes, root=0) # no. of bytes is required for mpich
                for i in range(self.nfiles[0]): 
                    comm.Bcast([self.configs[i], nbytes[i], MPI.BYTE], root=0)
                 
                # Assign node types
                self.nsmds = int(os.environ.get('PS_SMD_NODES', np.ceil((size-1)*PERCENT_SMD)))
                if rank == 0:
                    self.nodetype = 'smd0'
                elif rank < self.nsmds + 1:
                    self.nodetype = 'smd'
                
                # Big-data nodes own big data
                if self.nodetype == 'bd':
                    self.dm = DgramManager(self.xtc_files, configs=self.configs)
        
        self.Detector = Detector(self.configs)

    def runs(self): 
        nruns = 1
        for run_no in range(nruns):
            yield Run(self)

    def events(self): 
        for run in self.runs():
            for evt in run.events(): yield evt
    
    @property
    def _configs(self):
        """ Returns configs 
        reading file(s) -- returns configs of these files
        reading an exp  -- returns configs of xtc DgramManager for both
                           single and multiple core read.
        """
        assert len(self.configs) > 0
        return self.configs
    
    @property
    def _file_descriptors(self):
        if self.dm:
            return self.dm.fds
        else:
            return 0
