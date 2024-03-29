## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from parallelreader cimport Buffer
from cython.parallel import prange
import os
from dgramlite cimport Xtc, Sequence, Dgram
cimport cython
from psana.psexp import TransitionId

cdef class ParallelReader:
    
    def __cinit__(self, int[:] file_descriptors, size_t chunksize, *args, **kwargs):
        # Keyword args that need to be passed in once. To save some of
        # them as cpp class attributes, we need to read them in as PyObject*.
        cdef char* kwlist[2]
        kwlist[0] = "dsparms"
        kwlist[1] = NULL
        if PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, 
                &(self.dsparms)) == False:
            raise RuntimeError, "Invalid kwargs for SmdReader"
        dsparms = <object> self.dsparms
        cdef uint64_t[:] filter_timestamps = dsparms.timestamps
        
        self.file_descriptors   = file_descriptors
        self.chunksize          = chunksize
        self.nfiles             = self.file_descriptors.shape[0]
        self.Configure          = TransitionId.Configure
        self.BeginRun           = TransitionId.BeginRun
        self.L1Accept           = TransitionId.L1Accept
        self.EndRun             = TransitionId.EndRun
        self.bufs               = <Buffer *>malloc(sizeof(Buffer) * self.nfiles)
        self.step_bufs          = <Buffer *>malloc(sizeof(Buffer) * self.nfiles)
        self.tmp_bufs           = <Buffer *>malloc(sizeof(Buffer) * self.nfiles)
        self.got                = 0
        self.chunk_overflown    = 0                         # set to dgram size if it's too big
        self.max_events         = int(self.chunksize / 70)  # guess no. of smd events in one chunk
        self.num_threads        = int(os.environ.get('PS_SMD0_NUM_THREADS', '16'))
        self.zeroedbug_wait_sec = int(os.environ.get('PS_ZEROEDBUG_WAIT_SEC', '3'))
        self.max_retries        = int(os.environ.get('PS_R_MAX_RETRIES', '0'))
        self.gots               = array.array('l', [0]*self.nfiles)
        
        # This is a 1d array containing timestamp is found flags for all streams
        cdef int flag_len = self.nfiles * filter_timestamps.shape[0]
        self.filter_timestamp_flags = array.array('i', [0]*flag_len) 

        self._init_buffers(self.bufs)
        self._init_buffers(self.step_bufs)
        self._init_buffers(self.tmp_bufs)

    def __dealloc__(self):
        self._free_buffers(self.bufs)
        self._free_buffers(self.step_bufs)
        self._free_buffers(self.tmp_bufs)

    cdef void _init_buffers(self, Buffer* bufs):
        cdef Py_ssize_t i
        cdef Buffer* buf
        for i in range(self.nfiles):
            buf                 = &(bufs[i])
            buf.got             = 0
            buf.ready_offset    = 0     # offset of the last event in the buffer
            buf.n_ready_events  = 0     # no. of total events in the buffer 
            buf.seen_offset     = 0     # offset of the event seen (yielded) so far
            buf.n_seen_events   = 0     # no. of seen events
            buf.timestamp       = 0       
            buf.found_endrun    = 0
            buf.endrun_ts       = 0
            buf.force_reread    = 0
            buf.chunk      = <char *>malloc(self.chunksize)
            buf.ts_arr     = <uint64_t *>malloc(sizeof(uint64_t) * self.max_events)
            buf.sv_arr     = <unsigned *>malloc(sizeof(unsigned) * self.max_events)
            buf.st_offset_arr = <uint64_t *>malloc(sizeof(uint64_t) * self.max_events)
            buf.en_offset_arr = <uint64_t *>malloc(sizeof(uint64_t) * self.max_events)
            buf.result_stat= <struct_stat *>malloc(sizeof(struct_stat))

    cdef void _free_buffers(self, Buffer* bufs):
        cdef Py_ssize_t i
        cdef Buffer* buf
        if bufs:
            for i in range(self.nfiles):
                buf = &(bufs[i])
                free(buf.chunk)
                free(buf.ts_arr)
                free(buf.sv_arr)
                free(buf.st_offset_arr)
                free(buf.en_offset_arr)
                free(buf.result_stat)
            free(bufs)

    @cython.boundscheck(False)
    cdef void just_read(self):
        """
        Reads only if the buffer has no more unseen events

        If there's some data left at the bottom of the buffer due to cutoff,
        copy this remaining data to the begining of the buffer then read to 
        fill the rest of the chunk. Sets the following variables when done:
        - got = remaining (from copying) + new got (from reading)
        - ready_offset = offset of the last event that fits in the buffer
        - n_ready_events = no. of total events that fit in the buffer

        """
        cdef Py_ssize_t i       = 0, j=0
        cdef int64_t got        = 0
        cdef int64_t[:] gots    = self.gots
        cdef Dgram* d
        cdef Buffer* buf
        cdef Buffer* step_buf
        cdef Buffer* tmp_buf
        cdef uint64_t payload   = 0
        cdef int err_thread_id  = -1
        cdef int fstat_err      = 0
        self.got                = 0
        cdef int st=0, en=0, sum_found=0
        
        dsparms = <object> self.dsparms
        cdef uint64_t[:] filter_timestamps = dsparms.timestamps
        cdef int[:] filter_timestamp_flags = self.filter_timestamp_flags
        cdef sum_founds = array.array('i', [0] * self.nfiles)
        cdef int[:] sum_found_views = sum_founds
        
        for i in prange(self.nfiles, nogil=True, num_threads=self.num_threads):
            if filter_timestamps.shape[0] > 0:
                st = i*filter_timestamps.shape[0]
                en = st + filter_timestamps.shape[0]
                for j in range(st, en):
                    sum_found_views[i] += filter_timestamp_flags[j]
                if sum_found_views[i] == filter_timestamps.shape[0]:
                    continue
            gots[i] = 0
            buf = &(self.bufs[i])
            step_buf = &(self.step_bufs[i])
            tmp_buf = &(self.tmp_bufs[i])

            # skip reading this buffer if there is/are still some event(s).
            # or when there's a split integrating event.
            if (buf.n_ready_events - buf.n_seen_events > 0) and not buf.force_reread: continue 
            
            # copy remaining data if any -
            # if force_reread is set (intg det), we need to copy data from seen_offset
            # instead of ready_offset.
            if buf.force_reread:
                buf.cp_offset = buf.seen_offset
            else:
                buf.cp_offset = buf.ready_offset
            if buf.got - buf.cp_offset > 0 and buf.cp_offset > 0:
                memcpy(buf.chunk, buf.chunk + buf.cp_offset, buf.got - buf.cp_offset)
            
            # temporary fix for zeroed bug filesystem problem (live-mode only)
            if self.max_retries > 0:
                fstat_err = fstat(self.file_descriptors[i], buf.result_stat)
                gettimeofday(&(buf.t_now), NULL)
                buf.t_delta = (buf.t_now.tv_sec + buf.t_now.tv_usec * 10e-9) - buf.result_stat.st_mtime
                if buf.t_delta > 0 and buf.t_delta < self.zeroedbug_wait_sec:
                    sleep(self.zeroedbug_wait_sec)

            # read more data to fill up the buffer
            gots[i] = read( self.file_descriptors[i], buf.chunk + (buf.got - buf.cp_offset), \
                                        self.chunksize - (buf.got - buf.cp_offset) )

            # summing the size of all the new reads
            self.got += gots[i]
            
            buf.got = (buf.got - buf.cp_offset) + gots[i]
            
            # reset the offsets and no. of events
            buf.ready_offset        = 0
            buf.n_ready_events      = 0
            buf.seen_offset         = 0
            buf.n_seen_events       = 0
            step_buf.ready_offset   = 0
            step_buf.n_ready_events = 0
            step_buf.seen_offset    = 0
            step_buf.n_seen_events  = 0
            tmp_buf.ready_offset   = 0
            tmp_buf.n_ready_events = 0
            tmp_buf.seen_offset    = 0
            tmp_buf.n_seen_events  = 0
            
            # reset force_read flag for integrating event
            buf.force_reread        = 0
            
            while buf.ready_offset < buf.got and buf.n_ready_events < self.max_events:
                if buf.got - buf.ready_offset >= sizeof(Dgram):
                    d = <Dgram *>(buf.chunk + buf.ready_offset)

                    if d.xtc.extent == 0:
                        err_thread_id = i
                        break

                    payload = d.xtc.extent - sizeof(Xtc)

                    # check if this dgram is too big to fit in the chunk
                    if sizeof(Dgram) + payload > self.chunksize:
                        self.chunk_overflown = sizeof(Dgram) + payload

                    if (buf.got - buf.ready_offset) >= sizeof(Dgram) + payload:
                        buf.ts_arr[buf.n_ready_events] = <uint64_t>d.seq.high << 32 | d.seq.low
                        buf.sv_arr[buf.n_ready_events] = (d.env>>24)&0xf
                        buf.st_offset_arr[buf.n_ready_events] = buf.ready_offset
                        buf.en_offset_arr[buf.n_ready_events] = buf.ready_offset + sizeof(Dgram) + payload

                        if filter_timestamps.shape[0] > 0 and buf.sv_arr[buf.n_ready_events] == self.L1Accept:
                            for j in range(filter_timestamps.shape[0]):
                                if filter_timestamps[j] == buf.ts_arr[buf.n_ready_events] and \
                                        filter_timestamp_flags[i * filter_timestamps.shape[0] + j] == 0:
                                    filter_timestamp_flags[i * filter_timestamps.shape[0] + j] = 1
                                    memcpy(tmp_buf.chunk + tmp_buf.ready_offset, d, sizeof(Dgram) + payload)
                                    tmp_buf.ts_arr[tmp_buf.n_ready_events] = buf.ts_arr[buf.n_ready_events]
                                    tmp_buf.st_offset_arr[tmp_buf.n_ready_events] = tmp_buf.ready_offset
                                    tmp_buf.en_offset_arr[tmp_buf.n_ready_events] = tmp_buf.ready_offset + sizeof(Dgram) + payload
                                    tmp_buf.sv_arr[tmp_buf.n_ready_events] = buf.sv_arr[buf.n_ready_events] 
                                    tmp_buf.n_ready_events += 1
                                    tmp_buf.timestamp = buf.ts_arr[buf.n_ready_events] 
                                    tmp_buf.ready_offset += sizeof(Dgram) + payload
                                    break
                        
                        # check if this a non L1
                        if buf.sv_arr[buf.n_ready_events] != self.L1Accept:
                            memcpy(step_buf.chunk + step_buf.ready_offset, d, sizeof(Dgram) + payload)
                            step_buf.ts_arr[step_buf.n_ready_events] = buf.ts_arr[buf.n_ready_events]
                            step_buf.st_offset_arr[step_buf.n_ready_events] = step_buf.ready_offset
                            step_buf.en_offset_arr[step_buf.n_ready_events] = step_buf.ready_offset + sizeof(Dgram) + payload
                            step_buf.sv_arr[step_buf.n_ready_events] = buf.sv_arr[buf.n_ready_events]
                            step_buf.n_ready_events += 1
                            step_buf.ready_offset += sizeof(Dgram) + payload
                            step_buf.timestamp = buf.ts_arr[buf.n_ready_events]

                            if buf.sv_arr[buf.n_ready_events] == self.EndRun:
                                buf.endrun_ts = buf.ts_arr[buf.n_ready_events]

                            if filter_timestamps.shape[0] > 0:
                                memcpy(tmp_buf.chunk + tmp_buf.ready_offset, d, sizeof(Dgram) + payload)
                                tmp_buf.ts_arr[tmp_buf.n_ready_events] = buf.ts_arr[buf.n_ready_events]
                                tmp_buf.st_offset_arr[tmp_buf.n_ready_events] = tmp_buf.ready_offset
                                tmp_buf.en_offset_arr[tmp_buf.n_ready_events] = tmp_buf.ready_offset + sizeof(Dgram) + payload
                                tmp_buf.sv_arr[tmp_buf.n_ready_events] = buf.sv_arr[buf.n_ready_events] 
                                tmp_buf.n_ready_events += 1
                                tmp_buf.timestamp = buf.ts_arr[buf.n_ready_events] 
                                tmp_buf.ready_offset += sizeof(Dgram) + payload
                        
                        buf.timestamp = buf.ts_arr[buf.n_ready_events] 
                        buf.ready_offset += sizeof(Dgram) + payload
                        buf.n_ready_events += 1

                    else: # if (buf.got - buf.ready_offset) >= sizeof(Dgram) + payload
                        break
                else: #if buf.got - buf.ready_offset >= sizeof(Dgram)
                    break
            
            # end while buf.ready_offset < buf.got:

            # check if we need to copy from tmp_buf to buf (timestamp filter)
            if tmp_buf.ready_offset > 0:
                memcpy(buf.chunk, tmp_buf.chunk, tmp_buf.ready_offset)
                for j in range(tmp_buf.n_ready_events):
                    buf.ts_arr[j] = tmp_buf.ts_arr[j]
                    buf.st_offset_arr[j] = tmp_buf.st_offset_arr[j]
                    buf.en_offset_arr[j] = tmp_buf.en_offset_arr[j]
                    buf.sv_arr[j] = tmp_buf.sv_arr[j]
                buf.n_ready_events = tmp_buf.n_ready_events
                buf.timestamp = tmp_buf.timestamp
                buf.ready_offset = tmp_buf.ready_offset

        
        # end for i 
        if err_thread_id >= 0: 
            print(f'Error: found 0 extent in stream {err_thread_id}')
            raise


