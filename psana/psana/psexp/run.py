import os
import pickle
import inspect
import numpy as np
from copy import copy

from psana import dgram
from psana.dgrammanager import DgramManager
from psana.psexp.tools import run_from_id, RunHelper
from psana.psexp import legion_node
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.eventbuilder_manager import EventBuilderManager
from psana.psexp.event_manager import EventManager
from psana.psexp.epicsstore import EpicsStore
from psana.psexp.epicsreader import EpicsReader


from psana.psexp.tools import mode
MPI = None
if mode == 'mpi':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if size > 1:
        from psana.psexp.node import run_node # only import node when running in parallel
        

def _enumerate_attrs(obj):
    state = []
    found = []

    def mygetattr(obj):
        children = [attr for attr in dir(obj) if not attr.startswith('_')]

        for child in children:
            childobj = getattr(obj,child)

            if len([attr for attr in dir(childobj) if not attr.startswith('_')]) == 0:
                found.append( '.'.join(state + [child]) )
            elif type(childobj) == property:
                found.append( '.'.join(state + [child]) )
            elif not inspect.isclass(childobj) and callable(childobj):
                found.append( '.'.join(state + [child]) )
            else:
                state.append(child)
                mygetattr(childobj)
                state.pop()

    mygetattr(obj)
    return found

# FIXME: to support run.ds.Detector in cctbx. to be removed.
class DsContainer(object):

    def __init__(self, run):
        self.run = run

    def Detector(self, dummy):
        return self.run._det # return pre-created detector

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
    epics_store = None
    
    def __init__(self, exp, run_no, max_events=0, batch_size=1, filter_callback=0, destination=0):
        self.exp = exp
        self.run_no = run_no
        self.max_events = max_events
        self.batch_size = batch_size
        self.filter_callback = filter_callback
        self.destination = destination
        self.ds = DsContainer(self) # FIXME: to support run.ds.Detector in cctbx. to be removed.
        RunHelper(self)

    def run(self):
        """ Returns integer representaion of run no.
        default: (when no run is given) is set to -1"""
        return self.run_no

    def Detector(self,name):
        class Container:
            def __init__(self):
                pass
        det = Container
        # instantiate the detector xface for this detector/drp_class
        # (e.g. (xppcspad,raw) or (xppcspad,fex)
        # make them attributes of the container
        flag_found = False
        for (det_name,drp_class_name),drp_class in self.dm.det_class_table.items():
            if det_name == name:
                setattr(det,drp_class_name,drp_class(det_name, drp_class_name, self.configs, self.calibs))
                flag_found = True
        
        # If no detector found, try epics 
        # Epics det is identified by its keywords (e.g. 'XPP:VARS:FLOAT:02', etc).
        # Here, epics_store object is passed to drp_class so that the keyword
        # can be looked-up when evt is given (e.g. det(evt) returns value of
        # the given epics keyword.
        if not flag_found:
            alg = self.epics_store.alg_from_variable(name)
            if alg:
                det_name = 'xppepics'
                drp_class_name = alg
                drp_class = self.epics_dm.det_class_table[(det_name, drp_class_name)]
                det = drp_class(name, drp_class_name, self.configs, self.calibs, self.epics_store)

        return det

    @property
    def detnames(self):
        return set([x[0] for x in self.dm.det_class_table.keys()])

    @property
    def detinfo(self):
        info = {}
        for ((detname,det_xface_name),det_xface_class) in self.dm.det_class_table.items():
            info[(detname,det_xface_name)] = _enumerate_attrs(det_xface_class)
        return info

    @property
    def epicsinfo(self):
        return self.epics_store.epics_info
    
    @property
    def xtcinfo(self):
        return self.dm.xtc_info

    def _get_calib(self, det_name):
        return {} # temporary cpo hack while database is down
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

    def analyze(self, event_fn=None, det=None):
        for event in self.events():
            if event_fn is not None:
                event_fn(event, det)

    def __reduce__(self):
        return (run_from_id, (self.id,))

class RunShmem(Run):
    """ Yields list of events from a shared memory client (no event building routine). """
    
    def __init__(self, exp, run_no, xtc_files, max_events, batch_size, filter_callback, tag):
        super(RunShmem, self).__init__(exp, run_no, max_events=max_events, batch_size=batch_size, filter_callback=filter_callback)
        self.dm = DgramManager(xtc_files,tag=tag)
        self.configs = self.dm.configs
        self.calibs = {}
        for det_name in self.detnames:
            self.calibs[det_name] = self._get_calib(det_name)
        self.dm.calibs = self.calibs

    def events(self):
        for evt in self.dm: yield evt

class RunSingleFile(Run):
    """ Yields list of events from a single bigdata file. """
    
    def __init__(self, exp, run_no, run_src, **kwargs):
        super(RunSingleFile, self).__init__(exp, run_no, \
                max_events=kwargs['max_events'], batch_size=kwargs['batch_size'], \
                filter_callback=kwargs['filter_callback'])
        xtc_files, smd_files, epics_file = run_src
        self.dm = DgramManager(xtc_files)
        self.configs = self.dm.configs
        self.calibs = {}
        for det_name in self.detnames:
            self.calibs[det_name] = self._get_calib(det_name)

    def events(self):
        for evt in self.dm: yield evt


class RunSerial(Run):
    """ Yields list of events from multiple smd/bigdata files using single core."""

    def __init__(self, exp, run_no, run_src, **kwargs):
        super(RunSerial, self).__init__(exp, run_no, \
                max_events=kwargs['max_events'], batch_size=kwargs['batch_size'], \
                filter_callback=kwargs['filter_callback'])
        xtc_files, smd_files, other_files = run_src
        self.dm = DgramManager(xtc_files)
        self.configs = self.dm.configs
        self.smd_dm = DgramManager(smd_files)
        self.epics_dm = DgramManager(other_files, pulse_id=0xffff)
        self.epics_reader = EpicsReader(self.epics_dm.fds)
        self.epics_store = EpicsStore(self.epics_dm.configs)
        self.calibs = {}
        for det_name in self.detnames:
            self.calibs[det_name] = self._get_calib(det_name)
        
    def events(self):
        smd_configs = self.smd_dm.configs
        ev_man = EventManager(smd_configs, self.dm, \
                filter_fn=self.filter_callback)

        #get smd chunks
        smdr_man = SmdReaderManager(self.smd_dm.fds, self.max_events)
        eb_man = EventBuilderManager(smd_configs, batch_size=self.batch_size, filter_fn=self.filter_callback)
        for chunk in smdr_man.chunks():
            # Update epics_store for each chunk
            # This update checks for new data in epics_reader's queue
            # and rebuild the store accordingly.
            self.epics_store.update(self.epics_reader.read())
            
            for batch_dict in eb_man.batches(chunk):
                batch, _ = batch_dict[0] # there's only 1 dest_rank for serial run
                for evt in ev_man.events(batch):
                    yield evt
    
    def epicsStore(self, evt):
        """ Returns Epics event """
        epics_dicts = self.epics_store.checkout_by_events([evt])
        if not epics_dicts:
            return None

        return epics_dicts[0]


class RunParallel(Run):
    """ Yields list of events from multiple smd/bigdata files using > 3 cores."""

    def __init__(self, exp, run_no, run_src, **kwargs):
        """ Parallel read requires that rank 0 does the file system works.
        Configs and calib constants are sent to other ranks by MPI.
        
        Note that destination callback only works with RunParallel.
        """
        super(RunParallel, self).__init__(exp, run_no, max_events=kwargs['max_events'], \
                batch_size=kwargs['batch_size'], filter_callback=kwargs['filter_callback'], \
                destination=kwargs['destination'])
        xtc_files, smd_files, other_files = run_src
        if rank == 0:
            self.dm = DgramManager(xtc_files)
            self.configs = self.dm.configs
            self.smd_dm = DgramManager(smd_files)
            self.smd_configs = self.smd_dm.configs
            nbytes = np.array([memoryview(config).shape[0] for config in self.configs], \
                            dtype='i')
            smd_nbytes = np.array([memoryview(config).shape[0] for config in self.smd_configs], \
                            dtype='i')
            self.calibs = {}
            for det_name in self.detnames:
                self.calibs[det_name] = super(RunParallel, self)._get_calib(det_name)
            
            self.epics_dm = DgramManager(other_files, pulse_id=0xffff)
            self.epics_reader = EpicsReader(self.epics_dm.fds)
            self.epics_configs = self.epics_dm.configs
            epics_files = self.epics_dm.xtc_files
            epics_nbytes = np.array([memoryview(config).shape[0] for config in self.epics_configs], \
                            dtype='i')
        else:
            self.dm = None
            self.calibs = None
            self.smd_dm = None
            self.epics_dm = None
            nbytes = np.empty(len(xtc_files), dtype='i')
            smd_nbytes = np.empty(len(smd_files), dtype='i')
            epics_files = None
            epics_nbytes = None
        
        comm.Bcast(smd_nbytes, root=0) # no. of bytes is required for mpich
        comm.Bcast(nbytes, root=0) 
        epics_files = comm.bcast(epics_files, root=0)
        epics_nbytes = comm.bcast(epics_nbytes, root=0)
        
        # create empty views of known size
        if rank > 0:
            self.smd_configs = [np.empty(smd_nbyte, dtype='b') for smd_nbyte in smd_nbytes]
            self.configs = [np.empty(nbyte, dtype='b') for nbyte in nbytes]
            self.epics_configs = [np.empty(epics_nbyte, dtype='b') for epics_nbyte in epics_nbytes]
        
        for i in range(len(self.smd_configs)):
            comm.Bcast([self.smd_configs[i], smd_nbytes[i], MPI.BYTE], root=0)

        for i in range(len(self.configs)):
            comm.Bcast([self.configs[i], nbytes[i], MPI.BYTE], root=0)
        
        for i in range(len(self.epics_configs)):
            comm.Bcast([self.epics_configs[i], epics_nbytes[i], MPI.BYTE], root=0)

        self.calibs = comm.bcast(self.calibs, root=0)

        if rank > 0:
            # Create dgram objects using views from rank 0 (no disk operation).
            self.smd_configs = [dgram.Dgram(view=smd_config, offset=0) for smd_config in self.smd_configs]
            self.configs = [dgram.Dgram(view=config, offset=0) for config in self.configs]
            self.dm = DgramManager(xtc_files, configs=self.configs)
            self.epics_configs = [dgram.Dgram(view=config, offset=0) for config in self.epics_configs]
            self.epics_dm = DgramManager(epics_files, configs=self.epics_configs)
        self.epics_store = EpicsStore(self.epics_configs)   
    
    def events(self):
        for evt in run_node(self):
            yield evt
    
    def epicsStore(self, evt):
        """ Returns Epics event """
        epics_dicts = self.epics_store.checkout_by_events([evt])
        if not epics_dicts:
            return None

        return epics_dicts[0]
    
class RunLegion(Run):

    def __init__(self, exp, run_no, run_src, **kwargs):
        """ Parallel read using Legion """
        super(RunLegion, self).__init__(exp, run_no, max_events=kwargs['max_events'], \
                batch_size=kwargs['batch_size'], filter_callback=kwargs['filter_callback'])
        xtc_files, smd_files, other_files = run_src
        self.dm = DgramManager(xtc_files)
        self.configs = self.dm.configs
        self.smd_dm = DgramManager(smd_files)
        self.smd_configs = self.smd_dm.configs
        self.epics_dm = DgramManager(other_files, pulse_id=0xffff)
        self.epics_reader = EpicsReader(self.epics_dm.fds)
        self.epics_store = EpicsStore(self.epics_dm.configs)
        self.calibs = {}
        for det_name in self.detnames:
            self.calibs[det_name] = super(RunLegion, self)._get_calib(det_name)

    def analyze(self, **kwargs):
        return legion_node.analyze(self, **kwargs)
