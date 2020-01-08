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
from psana.psexp.event_manager import EventManager
from psana.psexp.envstore_manager import EnvStoreManager
from psana.psexp.packet_footer import PacketFooter
from psana.psexp.step import Step
from psana.psexp.event_manager import TransitionId
from psana.psexp.events import Events
import psana.pscalib.calib.MDBWebUtils as wu

from psana.psexp.tools import mode

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
#class DsContainer(object):

    #def __init__(self, run):
    #    self.run = run

    #def Detector(self, dummy):
    #    return self.run._det # return pre-created detector

class Run(object):
    exp = None # FIXME: consolidate with below
    run_no = None
    expt = 0
    runnum = 0 
    dm = None
    configs = None
    smd_dm = None
    max_events = None
    batch_size = None
    filter_callback = None
    nfiles = 0
    scan = False # True when looping over steps
    
    def __init__(self, exp, run_no, max_events=0, batch_size=1, filter_callback=0, destination=0):
        self.exp = exp
        self.run_no = run_no
        self.max_events = max_events
        self.batch_size = batch_size
        self.filter_callback = filter_callback
        self.destination = destination
        #self.ds = DsContainer(self) # FIXME: to support run.ds.Detector in cctbx. to be removed.
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
                setattr(det,drp_class_name,drp_class(det_name, drp_class_name, self.configs, self.calibconst[det_name]))
                setattr(det,'_configs', self.configs)
                setattr(det,'_calibconst', self.calibconst[det_name])
                setattr(det,'_dettype', self.dm.det_info_table[det_name][0])
                setattr(det,'_detid', self.dm.det_info_table[det_name][1])
                flag_found = True
        
        # If no detector found, try EnvStore.
        # Environment values are identified by variable names (e.g. 'XPP:VARS:FLOAT:02', 'motor1').
        # First we search for the store that has this name and pass it to drp_class so that the 
        # varialbe name can be looked-up when evt is given (e.g. det(evt) returns value of
        # the given epics keyword.)
        # Current, there are two algorithms (epics and scan).  
        # d.epics[0].epics.HX2:DVD:GCC:01:PMON = 41.0
        # d.epics[0].epics.HX2:DVD:GPI:01:PMON = 'Test String'
        if not flag_found:
            #for alg, store in self.esm.stores.items():
            #alg = self.esm.stores['epics'].alg_from_variable(name)
            alg = self.esm.alg_from_variable(name)
            if alg:
                det_name = alg
                var_name = name
                drp_class_name = alg
                drp_class = self.dm.det_class_table[(det_name, drp_class_name)]
                det = drp_class(det_name, var_name, drp_class_name, self.dm.configs, self.calibconst, self.esm.stores[alg])

        return det

    @property
    def detnames(self):
        return set([x[0] for x in self.dm.det_class_table.keys()])

    @property
    def detinfo(self):
        info = {}
        for ((detname,det_xface_name),det_xface_class) in self.dm.det_class_table.items():
#            info[(detname,det_xface_name)] = _enumerate_attrs(det_xface_class)
            info[(detname,det_xface_name)] = _enumerate_attrs(getattr(self.Detector(detname),det_xface_name))
        return info

    @property
    def epicsinfo(self):
        return self.esm.get_info('epics')
    
    @property
    def scaninfo(self):
        return self.esm.get_info('scan')
    
    @property
    def xtcinfo(self):
        return self.dm.xtc_info

    def _set_calibconst(self):
        self.calibconst = {}
        for det_name, (dettype, detid) in self.dm.det_info_table.items():
            det_str = dettype + '_' + detid
            if self.expt:
                self.calibconst[det_name] = wu.calib_constants_all_types(det_name, exp=self.expt, run=self.runnum)
            else:
                self.calibconst[det_name] = None


    def analyze(self, event_fn=None, det=None):
        for event in self.events():
            if event_fn is not None:
                event_fn(event, det)

    def __reduce__(self):
        return (run_from_id, (self.id,))

    def _get_runinfo(self):
        """ Gets runinfo from BeginRun event"""
        beginrun_evt = None
        if hasattr(self.dm.configs[0].software, 'smdinfo'):
            # This run has smd files - use offset to get BeginRun dgram
            smd_beginrun_evt = next(self.smd_dm)
            ofsz = np.asarray([[d.smdinfo[0].offsetAlg.intOffset, \
                    d.smdinfo[0].offsetAlg.intDgramSize] for d in smd_beginrun_evt])
            beginrun_evt = self.dm.jump(ofsz[:,0], ofsz[:,1])
        elif hasattr(self.dm.configs[0].software, 'runinfo'):
            beginrun_evt = next(self.dm)
        
        if not beginrun_evt: return

        if hasattr(beginrun_evt._dgrams[0], 'runinfo'): # some xtc2 do not have BeginRun
            self.expt = beginrun_evt._dgrams[0].runinfo[0].runinfo.expt 
            self.runnum = beginrun_evt._dgrams[0].runinfo[0].runinfo.runnum
            self.timestamp = beginrun_evt.timestamp


class RunShmem(Run):
    """ Yields list of events from a shared memory client (no event building routine). """
    
    def __init__(self, exp, run_no, xtc_files, max_events, batch_size, filter_callback, tag):
        super(RunShmem, self).__init__(exp, run_no, max_events=max_events, batch_size=batch_size, filter_callback=filter_callback)
        self.dm = DgramManager(xtc_files,tag=tag)
        self.configs = self.dm.configs 
        super()._get_runinfo()
        super()._set_calibconst()
        self.dm.calibconst = self.calibconst
        self.esm = EnvStoreManager(self.dm.configs, 'epics', 'scan')

    def events(self):
        for evt in self.dm:
            if evt._dgrams[0].service() != TransitionId.L1Accept: continue
            yield evt

class RunSingleFile(Run):
    """ Yields list of events from a single bigdata file. """
    
    def __init__(self, exp, run_no, run_src, **kwargs):
        super(RunSingleFile, self).__init__(exp, run_no, \
                max_events=kwargs['max_events'], batch_size=kwargs['batch_size'], \
                filter_callback=kwargs['filter_callback'])
        xtc_files, smd_files, epics_file = run_src
        self.dm = DgramManager(xtc_files)
        self.configs = self.dm.configs
        super()._get_runinfo()
        super()._set_calibconst()
        self.esm = EnvStoreManager(self.dm.configs, 'epics', 'scan')

    def events(self):
        for evt in self.dm:
            if evt._dgrams[0].service() != TransitionId.L1Accept: continue
            yield evt

    def steps(self):
        for evt in self.dm:
            if evt._dgrams[0].service() == TransitionId.BeginStep: 
                yield Step(evt, self.dm)
    
class RunSerial(Run):
    """ Yields list of events from multiple smd/bigdata files using single core."""

    def __init__(self, exp, run_no, run_src, **kwargs):
        super(RunSerial, self).__init__(exp, run_no, \
                max_events=kwargs['max_events'], batch_size=kwargs['batch_size'], \
                filter_callback=kwargs['filter_callback'])
        xtc_files, smd_files, other_files = run_src
        self.smd_dm = DgramManager(smd_files)
        self.dm = DgramManager(xtc_files, configs=self.smd_dm.configs)
        self.configs = self.dm.configs
        super()._get_runinfo()
        super()._set_calibconst()
        self.esm = EnvStoreManager(self.smd_dm.configs, 'epics', 'scan')
        
    def events(self):
        events = Events(self)
        for evt in events:
            if evt._dgrams[0].service() == TransitionId.L1Accept:
                yield evt
    
    def steps(self):
        """ Generates events between steps. """
        events = Events(self)
        for evt in events:
            if evt._dgrams[0].service() == TransitionId.BeginStep:
                yield Step(evt, events)

class RunLegion(Run):

    def __init__(self, exp, run_no, run_src, **kwargs):
        """ Parallel read using Legion """
        super(RunLegion, self).__init__(exp, run_no, max_events=kwargs['max_events'], \
                batch_size=kwargs['batch_size'], filter_callback=kwargs['filter_callback'])
        xtc_files, smd_files, other_files = run_src
        self.smd_dm = DgramManager(smd_files)
        self.dm = DgramManager(xtc_files, configs=self.smd_dm.configs)
        self.configs = self.dm.configs
        super()._get_runinfo()
        super()._set_calibconst()
        self.esm = EnvStoreManager(self.configs, 'epics', 'scan')

    def analyze(self, **kwargs):
        return legion_node.analyze(self, **kwargs)
