from psana.psexp import TransitionId

class SmdDataSource():

    def __init__(self, configs, eb, run=None):
        self.configs = configs
        self.eb = eb
        self.run = run
        
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




