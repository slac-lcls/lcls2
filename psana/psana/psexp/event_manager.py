from psana.event import Event
from psana import dgram
from psana.psexp.packet_footer import PacketFooter
import numpy as np
import os

class EventManager(object):

    def __init__(self, smd_configs, dm, filter_fn=0, fuzzy_es=None):
        self.smd_configs = smd_configs
        self.dm = dm
        self.n_smd_files = len(self.smd_configs)
        self.filter_fn = filter_fn
        self.fuzzy_es = fuzzy_es

    def events(self, view):
        pf = PacketFooter(view=view)
        views = pf.split_packets()
        
        # Keeps offsets and sizes for all events in the batch
        # for batch reading (if filter_fn is not given).
        ofsz_batch = np.zeros((pf.n_packets, self.n_smd_files, 2), dtype=np.intp)
        for i, event_bytes in enumerate(views):
            if event_bytes:
                evt = Event._from_bytes(self.smd_configs, event_bytes)
                
                # get big data
                ofsz = np.asarray([[d.info.offsetAlg.intOffset, d.info.offsetAlg.intDgramSize] \
                        for d in evt])
                ofsz_batch[i,:,:] = ofsz

                # Only get big data one event at a time when filter is off
                if self.filter_fn:
                    bd_evt = self.dm.jump(ofsz[:,0], ofsz[:,1])
                    if self.fuzzy_es:
                        fuzzy_evt = self.fuzzy_es.generate(bd_evt)
                    yield bd_evt

        if self.filter_fn == 0:
            # Read chunks of 'size' bytes and store them in views
            views = [None] * self.n_smd_files
            view_sizes = np.zeros(self.n_smd_files)
            for i in range(self.n_smd_files):
                # If no data were filtered, we can assume that all bigdata
                # dgrams starting from the first offset are stored consecutively
                # in the file. We read a chunk of sum(all dgram sizes) and
                # store in a view.
                offset = ofsz_batch[0, i, 0]
                size = np.sum(ofsz_batch[:, i, 1])
                view_sizes[i] = size
                
                os.lseek(self.dm.fds[i], offset, 0)
                views[i] = os.read(self.dm.fds[i], size)
            
            # Build each event from these views
            dgrams = [None] * self.n_smd_files
            offsets = [0] * self.n_smd_files
            for i in range(pf.n_packets):
                for j in range(self.n_smd_files):
                    if offsets[j] >= view_sizes[j]:
                        continue

                    size = ofsz_batch[i, j, 1]
                    if size:
                        dgrams[j] = dgram.Dgram(view=views[j], config=self.dm.configs[j], offset=offsets[j])
                        offsets[j] += size
                
                bd_evt = Event(dgrams)
                if self.fuzzy_es:
                    fuzzy_evt = self.fuzzy_es.generate(bd_evt)
                yield bd_evt

