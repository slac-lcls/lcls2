from psana.psexp import TransitionId

class RunSmallData():
    """ Yields list of smalldata events
    
    This class is created by SmdReaderManager and used exclusively by EventBuilder.
    There's no DataSource class associated with it. This class makes step and 
    event generator available to user in smalldata callback. It does minimal work
    and doesn't require the Run baseclass.
    """
    def __init__(self, run, eb):
        self._evt = run._evt
        self.configs = run.configs
        self.eb = eb
        
        # SmdDataSource and BatchIterator share this list. SmdDataSource automatically
        # adds transitions to this list (skip yield and so hidden from smd_callback).
        # BatchIterator adds user-selected L1Accept to the list (default is add all).
        self.proxy_events = []

    def events(self):
        for evt in self.eb.events():
            if evt.service() != TransitionId.L1Accept: 
                self.proxy_events.append(evt._proxy_evt)
                continue
            yield evt
    
