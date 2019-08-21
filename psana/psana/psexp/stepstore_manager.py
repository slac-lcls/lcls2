from psana.psexp.stepstore import StepStore

class StepStoreManager(object):
    """ Manages stepStore.
    Stores list of stepStores (created according to given keywords e.g 'epics')
    and update the stores with list of views.
    """
    stores = {}
    
    def __init__(self, configs, *args):
        for arg in args:
            self.stores[arg] = StepStore(configs, arg)
    
    def update(self, views):
        for step_name, store in self.stores.items():
            store.update(views)
        