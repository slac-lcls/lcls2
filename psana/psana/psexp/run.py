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
    expt    = 0
    runnum  = 0 
    dm      = None
    configs = None
    smd_dm  = None
    nfiles  = 0
    scan    = False # True when looping over steps
    smd_fds = None
    
    def __init__(self, ds):
        self.dsparms = ds.dsparms
        self.c_ana   = self.dsparms.prom_man.get_metric('psana_bd_ana')
        if hasattr(ds, "dm"): ds.dm.set_run(self)
        if hasattr(ds, "smdr_man"): ds.smdr_man.set_run(self)
        RunHelper(self)
        self._dets   = {}

    def run(self):
        """ Returns integer representaion of run no.
        default: (when no run is given) is set to -1"""
        return self.run_no
    
    def _check_empty_calibconst(self, det_name):
        # Some detectors do not have calibration constant - set default value to None
        if not hasattr(self.dsparms, 'calibconst'):
            self.dsparms.calibconst = {det_name: None}
        else:
            if det_name not in self.dsparms.calibconst:
                self.dsparms.calibconst[det_name]  = None

    def Detector(self, name, accept_missing=False):
        if name in self._dets:
            return self._dets[name]

        if name not in self.dsparms.configinfo_dict and self.esm.env_from_variable(name) is None:
            if not accept_missing:
                err_msg = f"Cannot find {name} detector in configs. Available detectors: {','.join(list(self.dsparms.configinfo_dict.keys()))}"
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
        for (det_name,drp_class_name),drp_class in self.dsparms.det_classes['normal'].items():
            if det_name == name:
                # Detetors with cfgscan also owns an EnvStore
                env_store = None
                var_name  = None
                if det_name in self.esm.stores:
                    env_store = self.esm.stores[det_name]
                    setattr(det, "step", StepEvent(env_store))

                self._check_empty_calibconst(det_name)

                setattr(det,drp_class_name,drp_class(det_name, drp_class_name, self.dsparms.configinfo_dict[det_name], self.dsparms.calibconst[det_name], env_store, var_name))
                setattr(det,'_configs', self.configs)
                setattr(det,'calibconst', self.dsparms.calibconst[det_name])
                setattr(det,'_dettype', self.dsparms.det_info_table[det_name][0])
                setattr(det,'_detid', self.dsparms.det_info_table[det_name][1])
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
                det_class_table = self.dsparms.det_classes[det_name]
                drp_class = det_class_table[(det_name, drp_class_name)]

                self._check_empty_calibconst(det_name)
                
                det = drp_class(det_name, drp_class_name, self.dsparms.configinfo_dict[det_name], self.dsparms.calibconst[det_name], self.esm.stores[env_name], var_name)
                setattr(det, '_det_name', det_name)

        self._dets[name] = det
        return det

    @property
    def detnames(self):
        return set([x[0] for x in self.dsparms.det_classes['normal'].keys()])

    @property
    def detinfo(self):
        info = {}
        for ((detname,det_xface_name),det_xface_class) in self.dsparms.det_classes['normal'].items():
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
        return self.dsparms.xtc_info

    def analyze(self, event_fn=None, det=None):
        for event in self.events():
            if event_fn is not None:
                event_fn(event, det)

    def __reduce__(self):
        return (run_from_id, (self.id,))

    def _get_runinfo(self):
        if not self.beginruns : return

        beginrun_dgram = self.beginruns[0]
        if hasattr(beginrun_dgram, 'runinfo'): # some xtc2 do not have BeginRun
            self.expt = beginrun_dgram.runinfo[0].runinfo.expt 
            self.runnum = beginrun_dgram.runinfo[0].runinfo.runnum
            self.timestamp = beginrun_dgram.timestamp()
    
class RunShmem(Run):
    """ Yields list of events from a shared memory client (no event building routine). """
    
    def __init__(self, ds, run_evt):
        super(RunShmem, self).__init__(ds)
        self._evt      = run_evt
        self.beginruns = run_evt._dgrams
        self.configs   = ds._configs
        super()._get_runinfo()
        self.esm = EnvStoreManager(self.configs)
        self._evt_iter = Events(self.configs, ds.dm, ds.dsparms, 
                filter_callback=ds.dsparms.filter)
    
    def events(self):
        for evt in self._evt_iter:
            if evt.service() != TransitionId.L1Accept:
                self.esm.update_by_event(evt)
                if evt.service() == TransitionId.EndRun: return
                continue
            st = time.time()
            yield evt
            en = time.time()
            self.c_ana.labels('seconds','None').inc(en-st)
            self.c_ana.labels('batches','None').inc()
        
    def steps(self):
        for evt in self._evt_iter:
            if evt.service() == TransitionId.EndRun: return
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, self._evt_iter, self.esm)

class RunSingleFile(Run):
    """ Yields list of events from a single bigdata file. """
    
    def __init__(self, ds, run_evt):
        super(RunSingleFile, self).__init__(ds)
        self._evt      = run_evt
        self.beginruns = run_evt._dgrams
        self.configs   = ds._configs
        super()._get_runinfo()
        self.esm = EnvStoreManager(self.configs)
        self._evt_iter = Events(self.configs, ds.dm, ds.dsparms, 
                filter_callback=ds.dsparms.filter)
    
    def events(self):
        for evt in self._evt_iter:
            if evt.service() != TransitionId.L1Accept:
                self.esm.update_by_event(evt)
                if evt.service() == TransitionId.EndRun: return
                continue
            st = time.time()
            yield evt
            en = time.time()
            self.c_ana.labels('seconds','None').inc(en-st)
            self.c_ana.labels('batches','None').inc()
        
    def steps(self):
        for evt in self._evt_iter:
            if evt.service() == TransitionId.EndRun: return
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, self._evt_iter, self.esm)

class RunSerial(Run):
    """ Yields list of events from multiple smd/bigdata files using single core."""

    def __init__(self, ds, run_evt):
        super(RunSerial, self).__init__(ds)
        self._evt      = run_evt
        self.beginruns = run_evt._dgrams
        self.configs   = ds._configs
        super()._get_runinfo()
        self.esm = EnvStoreManager(self.configs)
        self._evt_iter = Events(self.configs, ds.dm, ds.dsparms, 
                filter_callback=ds.dsparms.filter, smdr_man=ds.smdr_man)
    
    def events(self):
        for evt in self._evt_iter:
            if evt.service() != TransitionId.L1Accept:
                self.esm.update_by_event(evt)
                if evt.service() == TransitionId.EndRun: return
                continue
            st = time.time()
            yield evt
            en = time.time()
            self.c_ana.labels('seconds','None').inc(en-st)
            self.c_ana.labels('batches','None').inc()
        
    def steps(self):
        for evt in self._evt_iter:
            if evt.service() == TransitionId.EndRun: return
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, self._evt_iter, self.esm)

class RunLegion(Run):
    def __init__(self, ds, run_evt):
        self.dsparms = ds.dsparms
        self.c_ana   = self.dsparms.prom_man.get_metric('psana_bd_ana')
        RunHelper(self)
        self._evt       = run_evt
        self.beginruns  = run_evt._dgrams
        self.smdr_man   = ds.smdr_man
        self.dm         = ds.dm
        self.configs    = ds._configs
        self.esm        = EnvStoreManager(self.configs)
        super()._get_runinfo()
    
    def analyze(self, **kwargs):
        return legion_node.analyze(self, **kwargs)
    
    
