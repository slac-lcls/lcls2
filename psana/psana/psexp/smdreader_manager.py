from psana.smdreader import SmdReader
from psana.eventbuilder import EventBuilder
from psana.psexp import *
import os, time
from psana import dgram
from psana.event import Event
import logging
import psana.pscalib.calib.MDBWebUtils as wu
import pickle

s_smd0_disk = PrometheusManager.get_metric('psana_smd0_wait_disk')


class BatchIterator(object):
    """ Iterates over batches of events.

    SmdReaderManager returns this object when a chunk is read.
    """
    def __init__(self, views, configs, batch_size=1, filter_fn=0, destination=0):
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
            self.eb = EventBuilder(views, configs)


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
    def __init__(self, smd_fds, dsparms):
        self.n_files = len(smd_fds)
        assert self.n_files > 0
        self.dsparms = dsparms
        
        self.batch_size = int(os.environ.get('PS_SMD_N_EVENTS', 1000))
        if self.dsparms.max_events:
            if self.dsparms.max_events < self.batch_size:
                self.batch_size = self.dsparms.max_events
        
        self.chunksize = int(os.environ.get('PS_SMD_CHUNKSIZE', 0x1000000))
        self.smdr = SmdReader(smd_fds, self.chunksize)
        self.processed_events = 0
        self.got_events = -1
        
        # Collecting Smd0 performance using prometheus
        self.c_read = self.dsparms.prom_man.get_metric('psana_smd0_read')

    @s_smd0_disk.time()
    def _get(self):
        self.smdr.get()
        logging.debug('SmdReaderManager: read %.5f MB'%(self.smdr.got/1e6))
        self.c_read.labels('MB', 'None').inc(self.smdr.got/1e6)
        
        if self.smdr.chunk_overflown > 0:
            msg = f"smdreader_manager.py: SmdReader found dgram ({self.smdr.chunk_overflown} MB) larger than chunksize ({self.chunksize/1e6} MB)"
            logging.debug(msg)
            raise ValueError(msg)

    def get_next_dgrams(self, configs=None):
        dgrams = None
        if not self.smdr.is_complete():
            self._get()
         
        if self.smdr.is_complete():
            mmrv_bufs, _ = self.smdr.view(batch_size=1)

            # For configs, we need to copy data from smdreader's buffers
            # This prevents it from getting overwritten by other dgrams.
            bytearray_bufs = [bytearray(mmrv_buf) for mmrv_buf in mmrv_bufs]
            
            if configs is None:
                dgrams = [dgram.Dgram(view=ba_buf, offset=0) for ba_buf in bytearray_bufs]
            else:
                dgrams = [dgram.Dgram(view=ba_buf, config=config, offset=0) for ba_buf, config in zip(bytearray_bufs, configs)]
        return dgrams

    def set_configs(self, configs):
        self.configs = configs

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
        
        mmrv_bufs, _ = self.smdr.view(batch_size=self.batch_size)
        batch_iter = BatchIterator(mmrv_bufs, self.configs, 
                batch_size  = self.dsparms.batch_size, 
                filter_fn   = self.dsparms.filter, 
                destination = self.dsparms.destination)
        self.got_events = self.smdr.view_size
        self.processed_events += self.got_events

        # sending data to prometheus
        self.c_read.labels('evts', 'None').inc(self.got_events)
        self.c_read.labels('batches', 'None').inc()

        return batch_iter
        
    def _get_calibconst_bytes(self, expt, runnum):
        calibconst_dict = {}
        for det_name, configinfo in self.dsparms.configinfo_dict.items():
            if expt == "xpptut15":
                det_uniqueid = "cspad_detnum1234"
            else:
                det_uniqueid = configinfo.uniqueid
            calib_const = wu.calib_constants_all_types(det_uniqueid, exp=expt, run=runnum)
            
            # mona - hopefully this will be removed once the calibconst
            # db all use uniqueid as an identifier
            if not calib_const:
                calib_const = wu.calib_constants_all_types(det_name, exp=expt, run=runnum)
            calibconst_dict[det_name] = calib_const
        return pickle.dumps(calibconst_dict, pickle.HIGHEST_PROTOCOL)

    def chunks(self):
        """ Generates a tuple of smd and step dgrams """
        is_done = False
        while not is_done:
            if self.smdr.is_complete():
                mmrv_bufs, mmrv_step_bufs = self.smdr.view(batch_size=self.batch_size)
                self.got_events = self.smdr.view_size
                self.processed_events += self.got_events

                #  if beginrun(s) is found, create calibconst packet for them
                logging.debug(f"smdreader_manager.py: Smd0 got {self.smdr.n_beginruns} beginruns {memoryview(self.smdr.beginrun_view()).nbytes} bytes")
                calibconst_pkt = bytearray()
                beginrun_bytes = bytearray(self.smdr.beginrun_view())
                if memoryview(beginrun_bytes).nbytes > 0:
                    calibconst_pf = PacketFooter(n_packets=self.smdr.n_beginruns)
                    cn_beginrun = 0
                    offset = 0
                    while offset < self.smdr.beginrun_offset:
                        d = dgram.Dgram(view=beginrun_bytes, config=self.configs[0], offset=offset)
                        offset += d._size


                        expt = d.runinfo[0].runinfo.expt 
                        runnum = d.runinfo[0].runinfo.runnum
                        calibconst_bytes = self._get_calibconst_bytes(expt, runnum)

                        logging.debug(f"smdreader_manager.py: Smd0 got beginrun ts={d.timestamp()} service={d.service()} expt={expt} runnum={runnum} calibconst_bytes={memoryview(calibconst_bytes).nbytes}")
                        calibconst_pf.set_size(cn_beginrun, memoryview(calibconst_bytes).nbytes)
                        cn_beginrun += 1
                        calibconst_pkt.extend(calibconst_bytes)

                    calibconst_pkt.extend(calibconst_pf.footer)
                    # Flush
                    self.smdr.reset_beginrun()
                
                
                # sending data to prometheus
                logging.debug('Smd0 got %d events'%(self.got_events))
                self.c_read.labels('evts', 'None').inc(self.got_events)
                self.c_read.labels('batches', 'None').inc()

                if self.dsparms.max_events and self.processed_events >= self.dsparms.max_events:
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
                    yield (smd_view, step_view, calibconst_pkt)

            else:
                self._get()
                if not self.smdr.is_complete():
                    is_done = True
                    break
        

    @property
    def min_ts(self):
        return self.smdr.min_ts


    @property
    def max_ts(self):
        return self.smdr.max_ts
