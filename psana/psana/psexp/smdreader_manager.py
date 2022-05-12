from psana.smdreader import SmdReader
from psana.eventbuilder import EventBuilder
from psana.psexp import *
import os, time
from psana import dgram
from psana.event import Event

import logging
logger = logging.getLogger(__name__)


class BatchIterator(object):
    """ Iterates over batches of events.

    SmdReaderManager returns this object when a chunk is read.
    """
    def __init__(self, views, configs, run, 
            batch_size=1, filter_fn=0, 
            destination=0, timestamps=0):
        self.batch_size     = batch_size
        self.filter_fn      = filter_fn
        self.destination    = destination
        self.timestamps     = timestamps
        self.run            = run 
        
        empty_view = True
        for view in views:
            if view:
                empty_view = False
                break

        if empty_view:
            self.eb = None
        else:
            self.eb = EventBuilder(views, configs)


    def __iter__(self):
        return self


    def __next__(self):
        # With batch_size known, smditer returns a batch_dict,
        # {rank:[bytearray, evt_size_list], ...} for each next 
        # while updating offsets of each smd memoryview
        if not self.eb: 
            raise StopIteration

        batch_dict, step_dict = self.eb.build(self.timestamps,
                batch_size=self.batch_size, 
                filter_fn=self.filter_fn, 
                destination=self.destination,
                run=self.run,
                )
        if self.eb.nevents == 0 and self.eb.nsteps == 0: 
            raise StopIteration
        return batch_dict, step_dict



class SmdReaderManager(object):
    def __init__(self, smd_fds, dsparms, configs=None):
        self.n_files = len(smd_fds)
        self.dsparms = dsparms
        self.configs = configs
        assert self.n_files > 0
        
        self.smd0_n_events = int(os.environ.get('PS_SMD_N_EVENTS', 1000))
        if self.dsparms.max_events:
            if self.dsparms.max_events < self.smd0_n_events:
                self.smd0_n_events = self.dsparms.max_events
        
        self.chunksize = int(os.environ.get('PS_SMD_CHUNKSIZE', 0x1000000))
        is_legion = 0
        mode = os.environ.get('PS_PARALLEL')
        if mode and mode == 'legion':
            is_legion = 1
        self.smdr = SmdReader(smd_fds, self.chunksize, self.dsparms.max_retries, is_legion)
        self.processed_events = 0
        self.got_events = -1
        self._run = None
        
        # Collecting Smd0 performance using prometheus
        self.c_read = self.dsparms.prom_man.get_metric('psana_smd0_read')

    def _get(self):
        st = time.monotonic()
        self.smdr.get(self.dsparms.found_xtc2_callback)
        en = time.monotonic()
        logger.debug(f'read {self.smdr.got/1e6:.3f} MB took {en-st}s. rate: {self.smdr.got/(1e6*(en-st))} MB/s')
        self.c_read.labels('MB', 'None').inc(self.smdr.got/1e6)
        self.c_read.labels('seconds', 'None').inc(en-st)
        
        if self.smdr.chunk_overflown > 0:
            msg = f"SmdReader found dgram ({self.smdr.chunk_overflown} MB) larger than chunksize ({self.chunksize/1e6} MB)"
            raise ValueError(msg)

    def get_next_dgrams(self):
        if self.dsparms.max_events > 0 and \
                self.processed_events >= self.dsparms.max_events:
            logger.debug(f'max_events={self.dsparms.max_events} reached')
            return None

        dgrams = None
        if not self.smdr.is_complete():
            self._get()
         
        if self.smdr.is_complete():
            self.smdr.view(batch_size=1)

            # For configs, we need to copy data from smdreader's buffers
            # This prevents it from getting overwritten by other dgrams.
            bytearray_bufs = [bytearray(self.smdr.show(i)) for i in range(self.n_files)]
            
            if self.configs is None:
                dgrams = [dgram.Dgram(view=ba_buf, offset=0) for ba_buf in bytearray_bufs]
                self.configs = dgrams
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
        if self.dsparms.max_events and self.processed_events >= self.dsparms.max_events:
            raise StopIteration
        
        if not self.smdr.is_complete():
            self._get()
            if not self.smdr.is_complete():
                raise StopIteration
        
        self.smdr.view(batch_size=self.smd0_n_events)
        mmrv_bufs = [self.smdr.show(i) for i in range(self.n_files)]
        batch_iter = BatchIterator(mmrv_bufs, self.configs, self._run, 
                batch_size  = self.dsparms.batch_size, 
                filter_fn   = self.dsparms.filter, 
                destination = self.dsparms.destination,
                timestamps  = self.dsparms.timestamps)
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

                #mmrv_bufs, mmrv_step_bufs = self.smdr.view(batch_size=self.smd0_n_events)
                # internally updates the buffeer step/start/blocksize
                self.smdr.view(batch_size=self.smd0_n_events)
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

