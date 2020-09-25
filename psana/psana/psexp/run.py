import os, sys
import pickle
import inspect
import numpy as np
import time
from copy import copy
from psana import dgram
from psana.dgrammanager import DgramManager
import psana.pscalib.calib.MDBWebUtils as wu
from psana.detector.detector_impl import MissingDet
from psana.psexp import *
import logging


class DetectorNameError(Exception): pass


def _enumerate_attrs(obj):
    state = []
    found = []

    def mygetattr(obj):
        children = [attr for attr in dir(obj) if not attr.startswith('_') and attr != "dtype"]

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

class StepEvent(object):
    def __init__(self, env_store):
        self.env_store = env_store
    def dgrams(self, evt):
        return self.env_store.get_step_dgrams_of_event(evt)
    def docstring(self, evt) -> str:
        env_values = self.env_store.values([evt], 'step_docstring')
        return env_values[0]
    def value(self, evt) -> float:
        env_values = self.env_store.values([evt], 'step_value')
        return env_values[0]

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
    smd_fds = None
    
    def __init__(self, exp, run_no, 
            max_events=0, batch_size=1, 
            filter_callback=0, destination=0, 
            prom_man=None):
        self.exp                = exp
        self.run_no             = run_no
        self.max_events         = max_events
        self.batch_size         = batch_size
        self.filter_callback    = filter_callback
        self.destination        = destination
        self.prom_man           = prom_man
        self.c_ana              = self.prom_man.get_metric('psana_bd_ana')
        RunHelper(self)

    def close(self):
        """ Closing all xtcfiles 
        This is called when StopIteration is raised in Events Iterator 
        (BigData cores for parallel run) and when Smd0 or EventBuilders
        close their connections with clients.
        
        For SmallData
        - RunSingleFile, RunShmem, close files opened by DgramManager.
        - RunSerial, RunParallel close smd_fds opened by Run.
        """
        if self.smd_fds is not None:
            for fd in self.smd_fds:
                os.close(fd)
            
        if self.smd_dm:
            self.smd_dm.close()

        if self.dm:
            self.dm.close()


    def run(self):
        """ Returns integer representaion of run no.
        default: (when no run is given) is set to -1"""
        return self.run_no
    
    def _set_configinfo(self):
        """ From configs, we generate a dictionary lookup with det_name as a key.
        The information stored the value field contains:
        
        - configs specific to that detector
        - sorted_segment_ids
          used by Detector cls for checking if an event has correct no. of segments
        - detid_dict
          has segment_id as a key
        - dettype
        - uniqueid
        """
        self.configinfo_dict = {}

        for _, det_class in self.dm.det_classes.items(): # det_class is either normal or envstore
            for (det_name, _), _ in det_class.items():
                # Create a copy of list of configs for this detector
                det_configs = [dgram.Dgram(view=config) for config in self.dm.configs \
                        if hasattr(config.software, det_name)]
                sorted_segment_ids = []
                # a dictionary of the ids (a.k.a. serial-number) of each segment
                detid_dict = {}
                dettype = ""
                uniqueid = ""
                for config in det_configs:
                    seg_dict = getattr(config.software, det_name)
                    sorted_segment_ids += list(seg_dict.keys())
                    for segment, det in seg_dict.items():
                        detid_dict[segment] = det.detid
                        dettype = det.dettype
                
                sorted_segment_ids.sort()
                
                uniqueid = dettype
                for segid in sorted_segment_ids:
                    uniqueid += '_'+detid_dict[segid]

                self.configinfo_dict[det_name] = type("ConfigInfo", (), {\
                        "configs": det_configs, \
                        "sorted_segment_ids": sorted_segment_ids, \
                        "detid_dict": detid_dict, \
                        "dettype": dettype, \
                        "uniqueid": uniqueid})

    def Detector(self, name, accept_missing=False):
        if name not in self.configinfo_dict and self.esm.env_from_variable(name) is None:
            if not accept_missing:
                err_msg = f"Cannot find {name} detector in configs. Available detectors: {','.join(list(self.configinfo_dict.keys()))}"
                raise DetectorNameError(err_msg)
            else:
                return MissingDet()

        class Container:
            def __init__(self):
                pass
        
        det = Container()
        # instantiate the detector xface for this detector/drp_class
        # (e.g. (xppcspad,raw) or (xppcspad,fex)
        # make them attributes of the container
        flag_found = False
        for (det_name,drp_class_name),drp_class in self.dm.det_classes['normal'].items():
            if det_name == name:
                # Detetors with cfgscan also owns an EnvStore
                env_store = None
                var_name  = None
                if det_name in self.esm.stores:
                    env_store = self.esm.stores[det_name]
                    setattr(det, "step", StepEvent(env_store))

                setattr(det,drp_class_name,drp_class(det_name, drp_class_name, self.configinfo_dict[det_name], self.calibconst[det_name], env_store, var_name))
                setattr(det,'_configs', self.configs)
                setattr(det,'calibconst', self.calibconst[det_name])
                setattr(det,'_dettype', self.dm.det_info_table[det_name][0])
                setattr(det,'_detid', self.dm.det_info_table[det_name][1])
                setattr(det,'_det_name', det_name)
                flag_found = True

        # If no detector found, EnvStore variable is assumed to have been passed in.
        # Environment values are identified by variable names (e.g. 'XPP:VARS:FLOAT:02').
        # From d.epics[0].raw.HX2:DVD:GCC:01:PMON = 41.0
        # (d.env_name[0].alg.variable),
        # EnvStoreManager searches all the stores assuming that the variable name is
        # unique and returns env_name ('epics' or 'scan') and algorithm. 
        if not flag_found:
            found = self.esm.env_from_variable(name) # assumes that variable name is unique
            if found is not None:
                env_name, alg = found
                det_name = env_name
                var_name = name
                drp_class_name = alg
                det_class_table = self.dm.det_classes[det_name]
                drp_class = det_class_table[(det_name, drp_class_name)]
                det = drp_class(det_name, drp_class_name, self.configinfo_dict[det_name], self.calibconst[det_name], self.esm.stores[env_name], var_name)
                setattr(det, '_det_name', det_name)

        return det

    @property
    def detnames(self):
        return set([x[0] for x in self.dm.det_classes['normal'].keys()])

    @property
    def detinfo(self):
        info = {}
        for ((detname,det_xface_name),det_xface_class) in self.dm.det_classes['normal'].items():
#            info[(detname,det_xface_name)] = _enumerate_attrs(det_xface_class)
            info[(detname,det_xface_name)] = _enumerate_attrs(getattr(self.Detector(detname),det_xface_name))
        return info

    @property
    def epicsinfo(self):
        return self.esm.stores['epics'].get_info()
    
    @property
    def scaninfo(self):
        return self.esm.stores['scan'].get_info()

    @property
    def stepinfo(self):
        return self.esm.get_stepinfo()
    
    @property
    def xtcinfo(self):
        return self.dm.xtc_info

    def _set_calibconst(self):
        self.calibconst = {}
        for det_name, configinfo in self.configinfo_dict.items():
            if self.expt: 
                if self.expt == "cxid9114": # mona: hack for cctbx
                    det_uniqueid = "cspad_0002"
                elif self.expt == "xpptut15":
                    det_uniqueid = "cspad_detnum1234"
                else:
                    det_uniqueid = configinfo.uniqueid
                calib_const = wu.calib_constants_all_types(det_uniqueid, exp=self.expt, run=self.runnum)
                
                # mona - hopefully this will be removed once the calibconst
                # db all use uniqueid as an identifier
                if not calib_const:
                    calib_const = wu.calib_constants_all_types(det_name, exp=self.expt, run=self.runnum)
                self.calibconst[det_name] = calib_const
            else:
                logging.debug(f"Cannot access calibration constants without experiment code: self.expt={self.expt}")
                self.calibconst[det_name] = None




    def analyze(self, event_fn=None, det=None):
        for event in self.events():
            if event_fn is not None:
                event_fn(event, det)

    def __reduce__(self):
        return (run_from_id, (self.id,))

    def _get_runinfo(self):
        """ Gets runinfo from BeginRun event"""

        if not self.configs:
            err = "Empty configs. It's likely that the location of xtc files is incorrect. Check path of the dir argument in DataSource."
            raise XtcFileNotFound(err)

        config = self.configs[0]
        beginrun_evt = None
        if hasattr(config.software, 'smdinfo'):
            # This run has smd files 
            beginrun_evt = next(self.smd_dm) 
        elif hasattr(config.software, 'runinfo'):
            beginrun_evt = next(self.dm)
        
        if not beginrun_evt: return

        beginrun_dgram = beginrun_evt._dgrams[0]
        if hasattr(beginrun_dgram, 'runinfo'): # some xtc2 do not have BeginRun
            self.expt = beginrun_dgram.runinfo[0].runinfo.expt 
            self.runnum = beginrun_dgram.runinfo[0].runinfo.runnum
            self.timestamp = beginrun_evt.timestamp


class RunShmem(Run):
    """ Yields list of events from a shared memory client (no event building routine). """
    
    def __init__(self, exp, run_no, xtc_files, max_events, batch_size, filter_callback, tag, prom_man):
        super(RunShmem, self).__init__(exp, run_no, max_events=max_events, batch_size=batch_size, filter_callback=filter_callback, prom_man=prom_man)
        self.dm = DgramManager(xtc_files,tag=tag)
        self.configs = self.dm.configs 
        super()._get_runinfo()
        super()._set_configinfo()
        super()._set_calibconst()
        self.dm.calibconst = self.calibconst
        self.esm = EnvStoreManager(self.dm.configs)

    def events(self):
        events = Events(self, dm=self.dm)
        for evt in events:
            if evt.service() == TransitionId.L1Accept:
                yield evt
    
    def steps(self):
        """ Generates events between steps. """
        events = Events(self, dm=self.dm)
        for evt in events:
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, events)

class RunSingleFile(Run):
    """ Yields list of events from a single bigdata file. """
    
    def __init__(self, exp, run_no, run_src, **kwargs):
        super(RunSingleFile, self).__init__(exp, run_no, \
                max_events=kwargs['max_events'], batch_size=kwargs['batch_size'], \
                filter_callback=kwargs['filter_callback'], 
                prom_man=kwargs['prom_man'])
        xtc_files, smd_files, epics_file = run_src
        self.dm = DgramManager(xtc_files)
        self.configs = self.dm.configs
        super()._get_runinfo()
        super()._set_configinfo()
        super()._set_calibconst()
        self.esm = EnvStoreManager(self.dm.configs)

    def events(self):
        events = Events(self, dm=self.dm)
        for evt in events:
            if evt.service() == TransitionId.L1Accept:
                yield evt
    
    def steps(self):
        """ Generates events between steps. """
        events = Events(self, dm=self.dm)
        for evt in events:
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, events)
    
class RunSerial(Run):
    """ Yields list of events from multiple smd/bigdata files using single core."""

    def __init__(self, exp, run_no, run_src, **kwargs):
        super(RunSerial, self).__init__(exp, run_no, 
                max_events      = kwargs['max_events'], 
                batch_size      = kwargs['batch_size'], 
                filter_callback = kwargs['filter_callback'],
                prom_man        = kwargs['prom_man'])
        
        xtc_files, smd_files, other_files = run_src

        # get Configure and BeginRun using SmdReader
        self.smd_fds    = np.array([os.open(smd_file, os.O_RDONLY) for smd_file in smd_files], dtype=np.int32)
        self.smdr_man   = SmdReaderManager(self)
        self.configs    = self.smdr_man.get_next_dgrams()
        self.beginruns  = self.smdr_man.get_next_dgrams(configs=self.configs)

        # setup DgramManagers for both small- and bigdata
        self.smd_dm     = DgramManager(smd_files, configs=self.configs, fds=self.smd_fds)
        self.dm         = DgramManager(xtc_files, configs=self.smd_dm.configs)
        self.esm        = EnvStoreManager(self.configs)
        self._get_runinfo()
        super()._set_configinfo()
        super()._set_calibconst()
    
    def _get_runinfo(self):
        if not self.beginruns : return

        beginrun_dgram = self.beginruns[0]
        if hasattr(beginrun_dgram, 'runinfo'): # some xtc2 do not have BeginRun
            self.expt = beginrun_dgram.runinfo[0].runinfo.expt 
            self.runnum = beginrun_dgram.runinfo[0].runinfo.runnum
            self.timestamp = beginrun_dgram.timestamp()
    
    def events(self):
        events = Events(self)
        for evt in events:
            if evt.service() == TransitionId.L1Accept:
                st = time.time()
                yield evt
                en = time.time()
                self.c_ana.labels('seconds','None').inc(en-st)
                self.c_ana.labels('batches','None').inc()
        self.close()

    
    def steps(self):
        """ Generates events between steps. """
        events = Events(self)
        for evt in events:
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, events)
        self.close()

class RunLegion(Run):

    def __init__(self, exp, run_no, run_src, **kwargs):
        """ Parallel read using Legion """
        super(RunLegion, self).__init__(exp, run_no, 
                max_events      = kwargs['max_events'], 
                batch_size      = kwargs['batch_size'], 
                filter_callback = kwargs['filter_callback'],
                prom_man        = kwargs['prom_man'])
        xtc_files, smd_files, other_files = run_src
        
        # get Configure and BeginRun using SmdReader
        self.smd_fds = np.array([os.open(smd_file, os.O_RDONLY) for smd_file in smd_files], dtype=np.int32)
        self.smdr_man = SmdReaderManager(self)
        self.configs = self.smdr_man.get_next_dgrams()
        self.beginruns = self.smdr_man.get_next_dgrams(configs=self.configs)
        
        self._get_runinfo()
        self.smd_dm = DgramManager(smd_files, configs=self.configs, fds=self.smd_fds)
        self.dm = DgramManager(xtc_files, configs=self.smd_dm.configs)
        super()._set_configinfo()
        super()._set_calibconst()
        self.esm = EnvStoreManager(self.configs)

    def _get_runinfo(self):
        if not self.beginruns : return

        beginrun_dgram = self.beginruns[0]
        if hasattr(beginrun_dgram, 'runinfo'): # some xtc2 do not have BeginRun
            self.expt = beginrun_dgram.runinfo[0].runinfo.expt 
            self.runnum = beginrun_dgram.runinfo[0].runinfo.runnum
            self.timestamp = beginrun_dgram.timestamp()

    def analyze(self, **kwargs):
        return legion_node.analyze(self, **kwargs)

