from psana.psexp import EnvStore, TransitionId
from psana.dgram import Dgram
import copy

class EnvStoreManager(object):
    """ Manages envStore.
    Stores list of envStores (created according to given keywords e.g 'epics')
    and update the stores with list of views.
    """
    stores = {}
    
    def __init__(self, configs, *args):
        self.configs = configs
        for arg in args:
            self.stores[arg] = EnvStore(configs, arg)
    
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

                # For BeginStep, checks if self.configs need to
                # be updated.
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

    


        
        
