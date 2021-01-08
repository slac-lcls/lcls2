from psana.psexp import EnvStore, TransitionId
from psana.dgram import Dgram
import copy

class EnvStoreManager(object):
    """ Manages envStore.
    Stores list of EnvStore (defaults are epics and scan).

    For detectors with cfgscan, also create corresponding EnvStore.
    E.g. configs[0].tmoopal[0].cfgscan.user.black_level, then
    this detector also owns and EnvStore('tmoopal'). The value of
    the leaf node can be accessed by calling 
    
    det = run.Detector("tmoopal")
    val = det.cfgscan.user.black_level(evt)
    """
    stores = {}
    
    def __init__(self, configs):
        self.configs    = configs
        envstore_names  = ['epics', 'scan', 'step']

        # Locate detectors with cfgscan in DrpClassName
        for detname, segments in self.configs[0].software.__dict__.items():
            for segid, segment in segments.items():
                if 'config' in segment.__dict__:
                    if detname not in envstore_names: 
                        envstore_names.append(detname)

        for envstore_name in envstore_names:
            self.stores[envstore_name] = EnvStore(configs, envstore_name)
    
    def _update_config(self, objori, objupd):
        for key, _ in objupd.__dict__.items():
            if hasattr(getattr(objupd, key), "__dict__"):
                self._update_config(getattr(objori, key), getattr(objupd, key))
            else:
                setattr(objori, key, getattr(objupd, key))

    def update_by_event(self, evt):
        if not evt:
            return
        for i, d in enumerate(evt._dgrams):
            if not d: continue

            # This releases the original dgram object (friendly
            # with shared memory which has limited capacity).
            new_d = Dgram(view=d, config=self.configs[i], offset=0)
            for key, val in d.__dict__.items():
                if key in self.stores:
                    self.stores[key].add_to(new_d, i)

                if new_d.service() == TransitionId.BeginStep:
                    # Always store step - FIXME: mona this creates an extra ref.
                    # of beginstep in case 'scan' is also available. Find a way
                    # to not store this extra ref.
                    self.stores['step'].add_to(new_d, i)

                    # For BeginStep, checks if self.configs need to be updated.
                    # Only apply fields w/o leading "_" and exist in the 
                    # original config
                    if key.startswith("_") or not hasattr(self.configs[i], key): continue
                    cfgold = getattr(self.configs[i], key)

                    for segid, segment in getattr(new_d, key).items():
                        # Only apply to fields with .config
                        if hasattr(segment, "config"):
                            self._update_config(cfgold[segid].config, getattr(segment, "config"))

    def env_from_variable(self, variable_name):
        for env_name, store in self.stores.items():
            found = store.locate_variable(variable_name)
            if found is not None:
                alg, _ = found
                return env_name, alg
        return None

    def get_stepinfo(self):
        config_stores = set(self.stores.keys()).difference({'epics', 'scan', 'step'})
        stepinfo = {}
        for config_store in config_stores:
            stepinfo[(config_store, 'step')] = ['value', 'docstring'] 
        return stepinfo

    


        
        
