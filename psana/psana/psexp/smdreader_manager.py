from psana.smdreader import SmdReader
from psana.psexp.packet_footer import PacketFooter
from psana.eventbuilder import EventBuilder
import os, time

class BatchIterator(object):
    """ Iterates over batches of events.

    SmdReaderManager returns this object when a chunk is read.
    """
    def __init__(self, views, batch_size=1, filter_fn=0, destination=0):
        self.batch_size = batch_size
        self.filter_fn = filter_fn
        self.destination = destination
        
        empty_view = True
        for view in views:
            if view:
                empty_view = False
                break

        if empty_view:
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
        
        self.n_events = int(os.environ.get('PS_SMD_N_EVENTS', 1000))
        if self.run.max_events:
            if self.run.max_events < self.n_events:
                self.n_events = self.run.max_events
        
        self.chunksize = int(os.environ.get('PS_SMD_CHUNKSIZE', 0x100000))
        self.smdr = SmdReader(run.smd_dm.fds, self.chunksize, self.n_events)
        self.processed_events = 0
        self.got_events = -1

    def __iter__(self):
        return self

    def _read(self):
        max_retries = int(os.environ.get('PS_SMD_MAX_RETRIES', '5'))
        sleep_secs = int(os.environ.get('PS_SMD_SLEEP_SECS', '1'))
        
        self.smdr.get()
        
        cn_retries = 0
        while self.smdr.got_events==0:
            self.smdr.retry()
            cn_retries += 1
            if cn_retries == max_retries:
                break
            time.sleep(sleep_secs)

        self.got_events = self.smdr.got_events
        self.processed_events += self.got_events
        
    def __next__(self):
        self._read()
        
        if self.got_events == 0: raise StopIteration

        views =[]
        for i in range(self.n_files):
            view = self.smdr.view(i)
            if view:
                views.append(view)
            else:
                views.append(memoryview(bytearray()))
        
        batch_iter = BatchIterator(views, batch_size=self.run.batch_size, \
                filter_fn=self.run.filter_callback, destination=self.run.destination)
        
        if self.run.max_events:
            if self.processed_events >= self.run.max_events:
                self.got_events = 0
        
        return batch_iter

    def chunks(self):
        """ Generates a tuple of smd and step dgrams """
        self._read()
        while self.got_events > 0:
            smd_view = bytearray()
            smd_pf = PacketFooter(n_packets=self.n_files)
            step_view = bytearray()
            step_pf = PacketFooter(n_packets=self.n_files)
            
            for i in range(self.n_files):
                _smd_view = self.smdr.view(i)
                if _smd_view != 0:
                    smd_view.extend(_smd_view)
                    smd_pf.set_size(i, memoryview(_smd_view).shape[0])
                
                _step_view = self.smdr.view(i, step=True)
                if _step_view != 0:
                    step_view.extend(_step_view)
                    step_pf.set_size(i, memoryview(_step_view).shape[0])

            if smd_view or step_view:
                if smd_view:
                    smd_view.extend(smd_pf.footer)
                if step_view:
                    step_view.extend(step_pf.footer)
                yield (smd_view, step_view)

            if self.run.max_events:
                if self.processed_events >= self.run.max_events:
                    break

            self._read()
    
    @property
    def min_ts(self):
        return self.smdr.min_ts

    @property
    def max_ts(self):
        return self.smdr.max_ts
