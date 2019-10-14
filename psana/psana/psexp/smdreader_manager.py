from psana.smdreader import SmdReader
from psana.psexp.packet_footer import PacketFooter
from psana.eventbuilder import EventBuilder
import os

class BatchIterator(object):
    """ Iterates over batches of events.

    SmdReaderManager returns this object when a chunk is read.
    """
    def __init__(self, views, batch_size=1, filter_fn=0, destination=0):
        self.batch_size = batch_size
        self.filter_fn = filter_fn
        self.destination = destination
        self.eb = None
        if all(views) == 0:
            self.eb = None
        else:
            self.eb = EventBuilder(views)

    def __iter__(self):
        return self

    def __next__(self):
        # With batch_size known, smditer returns a batch_dict,
        # {rank:[bytearray, evt_size_list], ...} for each next 
        # while updating offsets of each smd memoryview
        if not self.eb: raise StopIteration

        batch_dict = self.eb.build(batch_size=self.batch_size, filter_fn=self.filter_fn, \
                destination=self.destination)
        if self.eb.nevents == 0: raise StopIteration
        return batch_dict

class SmdReaderManager(object):

    def __init__(self, run):
        self.n_files = len(run.smd_dm.fds)
        assert self.n_files > 0
        self.run = run
        self.smdr = SmdReader(run.smd_dm.fds)
        self.n_events = int(os.environ.get('PS_SMD_N_EVENTS', 1000))
        self.processed_events = 0
        if self.run.max_events:
            if self.run.max_events < self.n_events:
                self.n_events = self.run.max_events
        self.got_events = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.got_events == 0: raise StopIteration

        self.smdr.get(self.n_events)
        self.got_events = self.smdr.got_events
        self.processed_events += self.got_events
        views = [self.smdr.view(i) for i in range(self.n_files)]
        batch_iter = BatchIterator(views, batch_size=self.run.batch_size, \
                filter_fn=self.run.filter_callback, destination=self.run.destination)
        
        if self.run.max_events:
            if self.processed_events >= self.run.max_events:
                self.got_events = 0
        
        return batch_iter

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

            if self.run.max_events:
                if self.processed_events >= self.run.max_events:
                    break
    
    @property
    def min_ts(self):
        return self.smdr.min_ts

    @property
    def max_ts(self):
        return self.smdr.max_ts
