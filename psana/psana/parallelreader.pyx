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
        
        self.file_descriptors   = file_descriptors
        self.chunksize          = chunksize
        self.nfiles             = self.file_descriptors.shape[0]
        self.Configure          = TransitionId.Configure
        self.BeginRun           = TransitionId.BeginRun
        self.L1Accept           = TransitionId.L1Accept
        self.EndRun             = TransitionId.EndRun
        self.bufs               = <Buffer *>malloc(sizeof(Buffer) * self.nfiles)
        self.step_bufs          = <Buffer *>malloc(sizeof(Buffer) * self.nfiles)
        self.got                = 0
        self.chunk_overflown    = 0                         # set to dgram size if it's too big
        self.max_events         = int(self.chunksize / 70)  # guess no. of smd events in one chunk
        self.num_threads        = int(os.environ.get('PS_SMD0_NUM_THREADS', '16'))
        self.zeroedbug_wait_sec = int(os.environ.get('PS_ZEROEDBUG_WAIT_SEC', '3'))
        self.max_retries        = int(os.environ.get('PS_R_MAX_RETRIES', '0'))
        self.gots               = array.array('l', [0]*self.nfiles)
        self._init_buffers(self.bufs)
        self._init_buffers(self.step_bufs)

    def __dealloc__(self):
        self._free_buffers(self.bufs)
        self._free_buffers(self.step_bufs)

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
        cdef size_t i           = 0, k=0
        cdef Py_ssize_t j       = 0
        cdef int64_t got        = 0
        cdef int64_t[:] gots    = self.gots
        cdef Dgram* d
        cdef Buffer* buf
        cdef Buffer* step_buf
        cdef uint64_t payload   = 0
        cdef int err_thread_id  = -1
        cdef int fstat_err      = 0
        self.got                = 0
        cdef int st=0, en=0, sum_found=0
        
        for i in prange(self.nfiles, nogil=True, num_threads=self.num_threads):
            gots[i] = 0
            buf = &(self.bufs[i])
            step_buf = &(self.step_bufs[i])

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

                        buf.timestamp = buf.ts_arr[buf.n_ready_events] 
                        buf.ready_offset += sizeof(Dgram) + payload
                        buf.n_ready_events += 1

                    else: # if (buf.got - buf.ready_offset) >= sizeof(Dgram) + payload
                        break
                else: #if buf.got - buf.ready_offset >= sizeof(Dgram)
                    break
            
            # end while buf.ready_offset < buf.got:

        # end for i 
        if err_thread_id >= 0: 
            print(f'Error: found 0 extent in stream {err_thread_id}')
            raise


