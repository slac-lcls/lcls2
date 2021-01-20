from psana       import dgram
from psana.event import Event
from psana.psexp import PacketFooter, TransitionId, PrometheusManager
import numpy as np
import os
from psana.psexp.tools import Logging as logging
import time

s_bd_disk = PrometheusManager.get_metric('psana_bd_wait_disk')

class EventManager(object):
    """ Return an event from the received smalldata memoryview (view)

    1) If dm is empty (no bigdata), yield this smd event
    2) If dm is not empty, 
        - with filter fn, fetch one bigdata and yield it.
        - w/o filter fn, fetch one big chunk of bigdata and
          replace smalldata view with the read out bigdata.
          Yield one bigdata event.
    """
    def __init__(self, view, smd_configs, dm,  
            filter_fn=0, prometheus_counter=None, 
            max_retries=0, use_smds=[]):
        if view:
            pf = PacketFooter(view=view)
            self.smd_events = pf.split_packets()
            self.n_events = pf.n_packets
        else:
            self.smd_events = None
            self.n_events = 0

        self.smd_configs = smd_configs
        self.dm = dm
        self.n_smd_files = len(self.smd_configs)
        self.filter_fn = filter_fn
        self.cn_events = 0
        self.prometheus_counter = prometheus_counter
        self.max_retries = max_retries
        self.use_smds = use_smds

        if not self.filter_fn and len(self.dm.xtc_files) > 0:
            self._read_bigdata_in_chunk()

    @s_bd_disk.time()
    def _read_chunks_from_disk(self, fds, offsets, sizes):
        sum_read_nbytes = 0 # for prometheus counter
        st = time.time()
        

        for i_smd in range(self.n_smd_files):
            if self.use_smds[i_smd]: continue # smd data were already copied
            
            offset = offsets[i_smd]
            size = sizes[i_smd]
            chunk = bytearray()
            logging.info(f'event_manager: _read_chunks_from_disk i_smd={i_smd}')
            for _ in range(self.max_retries+1):
                chunk.extend(os.pread(fds[i_smd], size, offset))
                got = memoryview(chunk).nbytes
                if got == sizes[i_smd]:
                    break
                offset += got
                size -= got
            self.bigdata[i_smd].extend(chunk)
            sum_read_nbytes += sizes[i_smd]
        en = time.time()
        rate = 0
        if sum_read_nbytes > 0:
            rate = (sum_read_nbytes/1e6)/(en-st)
        logging.info(f"event_manager: bd reads chunk {sum_read_nbytes/1e6:.5f} MB took {en-st:.2f} s (Rate: {rate:.2f} MB/s)")
        self._inc_prometheus_counter('MB', sum_read_nbytes/1e6)
        return 
    
    @s_bd_disk.time()
    def _read_dgram_from_disk(self, dgram_i, offset_and_size):
        offset = offset_and_size[0,0]
        size   = offset_and_size[0,1]
        st              = time.time()
        dgram           = self.dm.jumps(dgram_i, offset, size)
        en              = time.time()
        rate            = 0
        if dgram._size > 0:
            rate = (dgram._size/1e6)/(en-st)
        logging.info(f"event_manager: bd reads dgram{dgram_i} {dgram._size/1e6:.5f} MB took {en-st:.2f} s (Rate: {rate:.2f} MB/s)")
        return dgram
            
    def _read_bigdata_in_chunk(self):
        """ Read bigdata chunks of 'size' bytes and store them in views
        Note that views here contain bigdata (and not smd) events.
        All non L1 dgrams are copied from smd_events and prepend
        directly to bigdata chunks.
        """
        self.bigdata = []
        for i in range(self.n_smd_files):
            self.bigdata.append(bytearray())
        
        offsets = [0] * self.n_smd_files
        sizes = [0] * self.n_smd_files
        self.ofsz_batch = np.zeros((self.n_events, self.n_smd_files, 2), dtype=np.intp)
        
        # Look for first L1 event - copy all non L1 to bigdata buffers
        first_L1_pos = -1
        for i_evt, event_bytes in enumerate(self.smd_events):
            if event_bytes:
                smd_evt = Event._from_bytes(self.smd_configs, event_bytes, run=self.dm.get_run())
                ofsz = smd_evt.get_offsets_and_sizes() 
                if smd_evt.service() == TransitionId.L1Accept:
                    for i_smd, smd_dgram in enumerate(smd_evt._dgrams):
                        if self.use_smds[i_smd]:
                            # use smd_event, offset indicates index of the first event to be used
                            offsets[i_smd] = i_evt 
                        else:
                            offsets[i_smd] = ofsz[i_smd, 0]
                    first_L1_pos = i_evt
                    break
                else:
                    for smd_id, d in enumerate(smd_evt._dgrams):
                        if not d: continue
                        self.bigdata[smd_id].extend(d)

                if i_evt > 0:
                    self.ofsz_batch[i_evt,:,0] = self.ofsz_batch[i_evt-1,:,0] + self.ofsz_batch[i_evt-1,:,1]
                self.ofsz_batch[i_evt,:,1] = ofsz[:,1]
                
        if first_L1_pos == -1: return

        for i_evt, event_bytes in enumerate(self.smd_events[first_L1_pos:]):
            j_evt = i_evt + first_L1_pos
            if event_bytes:
                smd_evt = Event._from_bytes(self.smd_configs, event_bytes, run=self.dm.get_run())
                for i_smd, smd_dgram in enumerate(smd_evt._dgrams):
                    if self.use_smds[i_smd]:
                        d_size = smd_dgram._size
                        self.bigdata[i_smd].extend(smd_dgram)
                    else:
                        d_size = smd_evt.get_offset_and_size(i_smd)[0,1] # only need size
                    if j_evt > 0:
                        prev_d_offset = self.ofsz_batch[j_evt-1, i_smd, 0]
                        prev_d_size = self.ofsz_batch[j_evt-1, i_smd, 1]
                        d_offset = prev_d_offset + prev_d_size
                    else:
                        d_offset = 0
                    self.ofsz_batch[j_evt, i_smd] = [d_offset, d_size] 
                    sizes[i_smd] += d_size
                       
        # If no data were filtered, we can assume that all bigdata
        # dgrams starting from the first offset are stored consecutively
        # in the file. We read a chunk of sum(all dgram sizes) and
        # store in a view.
        self._read_chunks_from_disk(self.dm.fds, offsets, sizes)
            
    def __iter__(self):
        return self

    def _inc_prometheus_counter(self, unit, value=1):
        if self.prometheus_counter:
            self.prometheus_counter.labels(unit,'None').inc(value)

    def __next__(self):
        if self.cn_events == self.n_events: 
            raise StopIteration
        
        smd_evt = Event._from_bytes(self.smd_configs, self.smd_events[self.cn_events], run=self.dm.get_run())
        if len(self.dm.xtc_files)==0 or smd_evt.service() != TransitionId.L1Accept:
            self.cn_events += 1
            self._inc_prometheus_counter('evts')
            return smd_evt
        
        if self.filter_fn:
            bd_dgrams = []
            read_size = 0
            for smd_i, smd_dgram in enumerate(smd_evt._dgrams):
                if self.use_smds[smd_i]:
                    bd_dgrams.append(smd_dgram)
                else:
                    offset_and_size = smd_evt.get_offset_and_size(smd_i)
                    read_size += offset_and_size[0,1]
                    bd_dgrams.append(self._read_dgram_from_disk(smd_i, offset_and_size))
            bd_evt = Event(dgrams=bd_dgrams, run=self.dm.get_run())
            self.cn_events += 1
            self._inc_prometheus_counter('MB', read_size/1e6)
            self._inc_prometheus_counter('evts')
            return bd_evt
        
        dgrams = [None] * self.n_smd_files
        ofsz = self.ofsz_batch[self.cn_events,:,:]
        for i_smd in range(self.n_smd_files):
            d_offset, d_size = ofsz[i_smd]
            if d_size and d_offset + d_size <= \
                    memoryview(self.bigdata[i_smd]).nbytes:
                dgrams[i_smd] = dgram.Dgram(view=self.bigdata[i_smd], 
                        config=self.dm.configs[i_smd], offset=d_offset)
        bd_evt = Event(dgrams, run=self.dm.get_run())
        self.cn_events += 1
        self._inc_prometheus_counter('evts')
        return bd_evt
        
