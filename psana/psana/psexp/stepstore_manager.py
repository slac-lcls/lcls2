from psana.psexp.stepstore import StepStore
from psana.dgram import Dgram

class StepStoreManager(object):
    """ Manages stepStore.
    Stores list of stepStores (created according to given keywords e.g 'epics')
    and update the stores with list of views.
    """
    stores = {}
    
    def __init__(self, configs, *args):
        self.configs = configs
        for arg in args:
            self.stores[arg] = StepStore(configs, arg)
    
    def update(self, views):
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

        
        
