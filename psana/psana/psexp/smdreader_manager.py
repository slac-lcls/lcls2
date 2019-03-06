from psana.smdreader import SmdReader
from psana.psexp.packet_footer import PacketFooter
import os

class SmdReaderManager(object):

    def __init__(self, fds, max_events):
        self.n_files = len(fds)
        assert self.n_files > 0
        self.smdr = SmdReader(fds)
        self.n_events = int(os.environ.get('PS_SMD_N_EVENTS', 1000))
        self.max_events = max_events
        self.processed_events = 0
        if self.max_events:
            if self.max_events < self.n_events:
                self.n_events = self.max_events

    def chunks(self):
        got_events = -1
        while got_events != 0:
            self.smdr.get(self.n_events)
            got_events = self.smdr.got_events
            self.processed_events += got_events
            view = bytearray()
            pf = PacketFooter(n_packets=self.n_files)
            for i in range(self.n_files):
                if self.smdr.view(i) != 0:
                    view.extend(self.smdr.view(i))
                    pf.set_size(i, memoryview(self.smdr.view(i)).shape[0])

            if view:
                view.extend(pf.footer) # attach footer 
                yield view

            if self.max_events:
                if self.processed_events >= self.max_events:
                    break
    
    @property
    def min_ts(self):
        return self.smdr.min_ts

    @property
    def max_ts(self):
        return self.smdr.max_ts
