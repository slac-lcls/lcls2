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
        """ Generates a tuple of smd and update dgrams """
        got_events = -1
        while got_events != 0:
            self.smdr.get(self.n_events)
            got_events = self.smdr.got_events
            self.processed_events += got_events
            
            smd_view = bytearray()
            smd_pf = PacketFooter(n_packets=self.n_files)
            update_view = bytearray()
            update_pf = PacketFooter(n_packets=self.n_files)
            
            for i in range(self.n_files):
                _smd_view = self.smdr.view(i)
                if _smd_view != 0:
                    smd_view.extend(_smd_view)
                    smd_pf.set_size(i, memoryview(_smd_view).shape[0])
                
                _update_view = self.smdr.view(i, update=True)
                if _update_view != 0:
                    update_view.extend(_update_view)
                    update_pf.set_size(i, memoryview(_update_view).shape[0])

            if smd_view or update_view:
                if smd_view:
                    smd_view.extend(smd_pf.footer)
                if update_view:
                    update_view.extend(update_pf.footer)
                yield (smd_view, update_view)

            if self.max_events:
                if self.processed_events >= self.max_events:
                    break
    
    @property
    def min_ts(self):
        return self.smdr.min_ts

    @property
    def max_ts(self):
        return self.smdr.max_ts
