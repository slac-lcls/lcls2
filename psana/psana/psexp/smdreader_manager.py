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

        batch_dict, step_dict = self.eb.build(batch_size=self.batch_size, filter_fn=self.filter_fn, \
                destination=self.destination)
        if self.eb.nevents == 0 and self.eb.nsteps == 0: raise StopIteration
        return batch_dict, step_dict

class SmdReaderManager(object):

    def __init__(self, run):
        self.n_files = len(run.smd_dm.fds)
        assert self.n_files > 0
        self.run = run
        
        self.batch_size = int(os.environ.get('PS_SMD_N_EVENTS', 10000))
        if self.run.max_events:
            if self.run.max_events < self.batch_size:
                self.batch_size = self.run.max_events
        
        self.chunksize = int(os.environ.get('PS_SMD_CHUNKSIZE', 0x1000000))
        self.smdr = SmdReader(run.smd_dm.fds, self.chunksize)
        self.processed_events = 0
        self.got_events = -1
        self.max_retries = int(os.environ['PS_SMD_MAX_RETRIES'])
        self.sleep_secs = int(os.environ.get('PS_SMD_SLEEP_SECS', '1'))

    def __iter__(self):
        return self
    
    def __next__(self):
        """
        Returns a batch of events as an iterator object.
        This is used by non-parallel run. Parallel run uses chunks
        generator that yields chunks of raw smd data and steps (no
        event building). 
        
        The iterator stops reading under two conditions. Either there's
        no more data or max_events reached.
        """
        if self.run.max_events and self.processed_events >= self.run.max_events:
            raise StopIteration
        
        if not self.smdr.is_complete():
            self.smdr.get()
            cn_retries = 0
            while not self.smdr.is_complete():
                if self.max_retries > 0:
                    time.sleep(self.sleep_secs)
                    self.smdr.get()
                    cn_retries += 1
                    if cn_retries > self.max_retries:
                        raise StopIteration 
                else:
                    raise StopIteration
        
        mmrv_bufs, _ = self.smdr.view(batch_size=self.batch_size)
        batch_iter = BatchIterator(mmrv_bufs, batch_size=self.run.batch_size, \
                filter_fn=self.run.filter_callback, destination=self.run.destination)
        self.got_events = self.smdr.view_size
        self.processed_events += self.got_events
        return batch_iter
        

    def chunks(self):
        """ Generates a tuple of smd and step dgrams """
        is_done = False
        while not is_done:
            if self.smdr.is_complete():
                mmrv_bufs, mmrv_step_bufs = self.smdr.view(batch_size=self.batch_size)
                self.got_events = self.smdr.view_size
                self.processed_events += self.got_events
                if self.run.max_events and self.processed_events >= self.run.max_events:
                    is_done = True
                
                smd_view = bytearray()
                smd_pf = PacketFooter(n_packets=self.n_files)
                step_view = bytearray()
                step_pf = PacketFooter(n_packets=self.n_files)
                
                for i, (mmrv_buf, mmrv_step_buf) in enumerate(zip(mmrv_bufs, mmrv_step_bufs)):
                    if mmrv_buf != 0:
                        smd_view.extend(mmrv_buf)
                        smd_pf.set_size(i, memoryview(mmrv_buf).nbytes)
                    
                    if mmrv_step_buf != 0:
                        step_view.extend(mmrv_step_buf)
                        step_pf.set_size(i, memoryview(mmrv_step_buf).nbytes)

                if smd_view or step_view:
                    if smd_view:
                        smd_view.extend(smd_pf.footer)
                    if step_view:
                        step_view.extend(step_pf.footer)
                    yield (smd_view, step_view)

            else:
                self.smdr.get()
                cn_retries = 0
                while not self.smdr.is_complete():
                    if self.max_retries > 0:
                        time.sleep(self.sleep_secs)
                        self.smdr.get()
                        cn_retries += 1
                        if cn_retries > self.max_retries:
                            is_done = True
                            break
                    else:
                        is_done = True
                        break


    
    @property
    def min_ts(self):
        return self.smdr.min_ts

    @property
    def max_ts(self):
        return self.smdr.max_ts
