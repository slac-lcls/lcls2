from psana.smdreader import SmdReader
from psana.eventbuilder import EventBuilder
import os, time
from psana import dgram
from psana.event import Event
from .run import RunSmallData

import logging
logger = logging.getLogger(__name__)


class BatchIterator(object):
    """ Iterates over batches of events.

    SmdReaderManager returns this object when a chunk is read.
    """
    def __init__(self, views, configs, run, dsparms):
        self.dsparms = dsparms
        
        # Requires all views
        empty_view = True
        for view in views:
            if view:
                empty_view = False
                break

        if empty_view:
            self.eb = None
        else:
            self.eb = EventBuilder(views, configs, 
                    dsparms=dsparms,
                    run=run,
                    prometheus_counter=None)
            self.run_smd = RunSmallData(run, self.eb)

    def __iter__(self):
        return self

    def __next__(self):
        # With batch_size known, smditer returns a batch_dict of this format:
        # {rank:[bytearray, evt_size_list], ...} 
        # for each next while updating offsets of each smd memoryview
        if not self.eb: 
            raise StopIteration
        
        # Collects list of proxy events to be converted to bigdata batches (batch_size).
        # Note that we are persistently calling smd_callback until there's nothing
        # left in all views used by EventBuilder. From this while/for loops, we 
        # either gets transitions from SmdDataSource and/or L1 from the callback.
        if self.dsparms.smd_callback == 0:
            batch_dict, step_dict = self.eb.build()
            if self.eb.nevents == 0:
                raise StopIteration
        else:
            while self.run_smd.proxy_events == [] and self.eb.has_more():
                for evt in self.dsparms.smd_callback(self.run_smd):
                    self.run_smd.proxy_events.append(evt._proxy_evt)
        
            if not self.run_smd.proxy_events:
                raise StopIteration

            # Generate a bytearray representation of all the proxy events.
            # Note that setting run_serial=True allow EventBuilder to combine
            # L1Accept and transitions into one batch (key=0). Here, step_dict
            # is always an empty bytearray.
            batch_dict, step_dict = self.eb.gen_bytearray_batch(self.run_smd.proxy_events, run_serial=True)
            self.run_smd.proxy_events = []

        return batch_dict, step_dict



class SmdReaderManager(object):
    def __init__(self, smd_fds, dsparms, configs=None):
        self.n_files = len(smd_fds)
        self.dsparms = dsparms
        self.configs = configs
        assert self.n_files > 0
        
        # Sets no. of events Smd0 sends to each EventBuilder core. This gets
        # overridden by max_events set by DataSource if max_events is smaller.
        self.smd0_n_events = int(os.environ.get('PS_SMD_N_EVENTS', 1000))
        if self.dsparms.max_events:
            if self.dsparms.max_events < self.smd0_n_events:
                self.smd0_n_events = self.dsparms.max_events
        
        # Sets the memory size for smalldata buffer for each stream file.
        self.chunksize = int(os.environ.get('PS_SMD_CHUNKSIZE', 0x10000000))

        self.smdr = SmdReader(smd_fds, self.chunksize, self.dsparms.max_retries)
        self.processed_events = 0
        self.got_events = -1
        self._run = None
        
        # Collecting Smd0 performance using prometheus
        self.c_read = self.dsparms.prom_man.get_metric('psana_smd0_read')

    def _get(self):
        st = time.monotonic()
        self.smdr.get(self.dsparms.smd_inprogress_converted)
        en = time.monotonic()
        logger.debug(f'read {self.smdr.got/1e6:.3f} MB took {en-st}s. rate: {self.smdr.got/(1e6*(en-st))} MB/s')
        self.c_read.labels('MB', 'None').inc(self.smdr.got/1e6)
        self.c_read.labels('seconds', 'None').inc(en-st)
        
        if self.smdr.chunk_overflown > 0:
            msg = f"SmdReader found dgram ({self.smdr.chunk_overflown} MB) larger than chunksize ({self.chunksize/1e6} MB)"
            raise ValueError(msg)

    def get_next_dgrams(self):
        """ Returns list of dgrams as appeared in the current offset of the smd chunks.

        Currently used to retrieve Configure and BeginRun. This allows read with wait
        for these two types of dgram.
        """
        if self.dsparms.max_events > 0 and \
                self.processed_events >= self.dsparms.max_events:
            logger.debug(f'max_events={self.dsparms.max_events} reached')
            return None

        dgrams = None
        if not self.smdr.is_complete():
            self._get()
         
        if self.smdr.is_complete():
            # Get chunks with only one dgram each. There's no need to set
            # integrating stream id here since Configure and BeginRun
            # must exist in this stream too. 
            self.smdr.view(batch_size=1)

            # For configs, we need to copy data from smdreader's buffers
            # This prevents it from getting overwritten by other dgrams.
            bytearray_bufs = [bytearray(self.smdr.show(i)) for i in range(self.n_files)]
            
            if self.configs is None:
                dgrams = [dgram.Dgram(view=ba_buf, offset=0) for ba_buf in bytearray_bufs]
                self.configs = dgrams
                self.smdr.set_configs(self.configs)
            else:
                dgrams = [dgram.Dgram(view=ba_buf, config=config, offset=0) for ba_buf, config in zip(bytearray_bufs, self.configs)]
        return dgrams


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
        intg_stream_id = self.dsparms.intg_stream_id
        
        if self.dsparms.max_events and self.processed_events >= self.dsparms.max_events:
            raise StopIteration
        
        if not self.smdr.is_complete():
            self._get()
            if not self.smdr.is_complete():
                raise StopIteration
        self.smdr.view(batch_size=self.smd0_n_events, intg_stream_id=intg_stream_id)
        mmrv_bufs = [self.smdr.show(i) for i in range(self.n_files)]
        batch_iter = BatchIterator(mmrv_bufs, self.configs, self._run, self.dsparms)
        self.got_events = self.smdr.view_size
        self.processed_events += self.got_events
        return batch_iter
        

    def chunks(self):
        """ Generates a tuple of smd and step dgrams """
        is_done = False
        d_view, d_read = 0, 0
        cn_chunks = 0
        while not is_done:
            logger.debug(f'SMD0 1. STARTCHUNK {time.monotonic()}')
            st_view, en_view, st_read, en_read = 0,0,0,0

            l1_size = 0
            tr_size = 0
            got_events = 0
            if self.smdr.is_complete():

                st_view = time.monotonic()

                # Gets the next batch of already read-in data. 
                self.smdr.view(batch_size=self.smd0_n_events, intg_stream_id=self.dsparms.intg_stream_id)
                self.got_events = self.smdr.view_size
                got_events = self.got_events
                self.processed_events += self.got_events
                
                # sending data to prometheus
                logger.debug('got %d events'%(self.got_events))

                if self.dsparms.max_events and self.processed_events >= self.dsparms.max_events:
                    logger.debug(f'max_events={self.dsparms.max_events} reached')
                    is_done = True
                
                en_view = time.monotonic()
                d_view += en_view - st_view
                logger.debug(f'SMD0 2. DONECREATEVIEW {time.monotonic()}')

                if self.got_events:
                    cn_chunks += 1
                    yield cn_chunks

            else: # if self.smdr.is_complete()
                st_read = time.monotonic()
                self._get()
                en_read = time.monotonic()
                logger.debug(f'SMD0 3. DONEREAD {time.monotonic()}')
                d_read += en_read - st_read
                if not self.smdr.is_complete():
                    is_done = True
                    break

    @property
    def min_ts(self):
        return self.smdr.min_ts


    @property
    def max_ts(self):
        return self.smdr.max_ts

    def set_run(self, run):
        self._run = run

    def get_run(self):
        return self._run

