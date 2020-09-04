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
        envstore_names  = ['epics', 'scan']

        # Locate detectors with cfgscan in DrpClassName
        for detname, segments in self.configs[0].software.__dict__.items():
            for segid, segment in segments.items():
                if 'raw' in segment.__dict__:
                    if detname not in envstore_names: 
                        envstore_names.append(detname)

        for envstore_name in envstore_names:
            self.stores[envstore_name] = EnvStore(configs, envstore_name)
    
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

                # For BeginStep, checks if self.configs need to be updated.
                if new_d.service() == TransitionId.BeginStep:
                    # Only apply fields w/o leading "_" and exist in the 
                    # original config
                    if key.startswith("_") or not hasattr(self.configs[i], key): continue
                    cfgold = getattr(self.configs[i], key)

                    for segid, segment in getattr(new_d, key).items():
                        # Only apply fiedls with .config
                        if not hasattr(segment, "config"): continue
                        cfgold[segid] = copy.deepcopy(segment) 

    def update_by_views(self, views):
        if not views:
            return
        for i in range(len(views)):
            view = bytearray(views[i])
            offset = 0
            while offset < memoryview(view).shape[0]:
                d = Dgram(view=view, config=self.configs[i], offset=offset)
                for key, val in d.__dict__.items():
                    if key in self.stores:
                        self.stores[key].add_to(d, i)
                    
                offset += d._size
                    
    def env_from_variable(self, variable_name):
        for env_name, store in self.stores.items():
            found = store.locate_variable(variable_name)
            if found is not None:
                alg, _ = found
                return env_name, alg
        return None

    


        
        
