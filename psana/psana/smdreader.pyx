## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from parallelreader cimport Buffer, ParallelReader
from libc.stdint cimport uint32_t, uint64_t
from cpython cimport array
import time, os
cimport cython
from psana.psexp import TransitionId
import numpy as np
from cython.parallel import prange
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from psana.dgramedit import DgramEdit


cdef class SmdReader:
    cdef ParallelReader prl_reader
    cdef int        is_legion
    cdef int         winner, n_view_events
    cdef int         max_retries, sleep_secs
    cdef array.array i_starts                   # ¬ used locally for finding boundary of each
    cdef array.array i_ends                     # } stream file (defined here for speed) in view(),  
    cdef array.array i_stepbuf_starts           # } which generates sharing window variables below.
    cdef array.array i_stepbuf_ends             # }
    cdef array.array block_sizes                # }
    cdef array.array i_st_bufs                  # ¬ for sharing viewing windows in show() 
    cdef array.array block_size_bufs            # }
    cdef array.array i_st_stepbufs              # }
    cdef array.array block_size_stepbufs        # }
    cdef array.array repack_offsets             # ¬ for parallel repack 
    cdef array.array repack_step_sizes          # }
    cdef array.array repack_footer              # }
    cdef unsigned    winner_last_sv             # ¬ transition id and ts of the last dgram in winner's chunk
    cdef uint64_t    winner_last_ts             # } 
    cdef uint64_t    _next_fake_ts              # incremented from winner_last_ts (shared by all streams) 
    cdef float       total_time
    cdef int         num_threads
    cdef Buffer*     send_bufs                  # array of customed Buffers (one per EB node)
    cdef Buffer*     send_step_bufs             # array of customed transition Buffers (one per EB node)
    cdef int         sendbufsize                # size of each send buffer
    cdef int         n_eb_nodes
    cdef int         fakestep_flag               
    cdef list        configs
    cdef bytearray   _fakebuf
    cdef unsigned    _fakebuf_maxsize
    cdef unsigned    _fakebuf_size


    def __init__(self, int[:] fds, int chunksize, int max_retries, int is_lg=0):
        assert fds.size > 0, "Empty file descriptor list (fds.size=0)."
        self.prl_reader         = ParallelReader(fds, chunksize)
        self.max_retries        = max_retries                       # no default value (set when creating datasource)
        self.sleep_secs         = 1
        self.total_time         = 0
        self.num_threads        = int(os.environ.get('PS_SMD0_NUM_THREADS', '16'))
        self.n_eb_nodes         = int(os.environ.get('PS_EB_NODES', '1'))
        if is_lg:
            self.is_legion = 1
        else:
            self.is_legion = 0
        self.winner_last_ts     = 0
        self._next_fake_ts      = 0
        self.configs            = []
        self.i_starts           = array.array('L', [0]*fds.size) 
        self.i_ends             = array.array('L', [0]*fds.size) 
        self.i_stepbuf_starts   = array.array('L', [0]*fds.size) 
        self.i_stepbuf_ends     = array.array('L', [0]*fds.size) 
        self.block_sizes        = array.array('L', [0]*fds.size) 
        self.i_st_bufs          = array.array('L', [0]*fds.size) 
        self.block_size_bufs    = array.array('L', [0]*fds.size) 
        self.i_st_stepbufs      = array.array('L', [0]*fds.size) 
        self.block_size_stepbufs= array.array('L', [0]*fds.size) 
        self.repack_offsets     = array.array('L', [0]*fds.size)
        self.repack_step_sizes  = array.array('L', [0]*fds.size)
        self.repack_footer      = array.array('I', [0]*(fds.size+1))# size of all smd chunks plus no. of smds
        self._fakebuf_maxsize   = 0x1000
        self._fakebuf           = bytearray(self._fakebuf_maxsize)
        self._fakebuf_size      = 0

        # Repack footer contains constant (per run) no. of smd files
        self.repack_footer[fds.size] = fds.size 
        
        # For speed, SmdReader puts together smd chunk and missing step info
        # in parallel and store the new data in `send_bufs` (one buffer per stream).
        self.send_bufs          = <Buffer *>malloc(self.n_eb_nodes * sizeof(Buffer))
        self.sendbufsize        = 0x10000000                                

        if self.is_legion > 0:
            self.send_step_bufs  = <Buffer *>malloc(self.n_eb_nodes * sizeof(Buffer))
            self._init_step_buffers()

        self._init_send_buffers()
        
        # Index of the slowest detector or intg_stream_id if given.
        self.winner             = -1
        
        # Sets event frequency that fake EndStep/BeginStep pair is inserted.
        self.fakestep_flag = int(os.environ.get('PS_FAKESTEP_FLAG', 0))


    def __dealloc__(self):
        cdef Py_ssize_t i
        for i in range(self.n_eb_nodes):
            free(self.send_bufs[i].chunk)
        if self.is_legion:
            for i in range(self.n_eb_nodes):
                free(self.send_step_bufs[i].chunk)
    
    cdef void _init_send_buffers(self):
        cdef Py_ssize_t i
        for i in range(self.n_eb_nodes):
            self.send_bufs[i].chunk      = <char *>malloc(self.sendbufsize)

    cdef void _init_step_buffers(self):
        cdef Py_ssize_t i
        for i in range(self.n_eb_nodes):
            self.send_step_bufs[i].chunk      = <char *>malloc(self.sendbufsize)
    
    def is_complete(self):
        """ Checks that all buffers have at least one event 
        """
        cdef int is_complete = 1
        cdef int i

        for i in range(self.prl_reader.nfiles):
            if self.prl_reader.bufs[i].n_ready_events - \
                    self.prl_reader.bufs[i].n_seen_events == 0:
                is_complete = 0
                break

        return is_complete

    def set_configs(self, configs):
        # SmdReaderManager calls view (with batch_size=1)  at the beginning
        # to read config dgrams. It passes the configs to SmdReader for
        # creating any fake dgrams using DgramEdit.
        self.configs = configs

    def get(self, smd_inprogress_converted):
        """SmdReaderManager only calls this function when there's no more event
        in one or more buffers. Reset the indices for buffers that need re-read."""

        # Exit if EndRun is found for all files. This is safe even though
        # we support mulirun in one xtc2 file but this is only for shmem mode,
        # which doesn't use SmdReader.
        if self.found_endrun(): return

        cdef int i=0
        for i in range(self.prl_reader.nfiles):
            buf = &(self.prl_reader.bufs[i])
            if buf.n_ready_events - buf.n_seen_events > 0: continue 
            self.i_starts[i] = 0
            self.i_ends[i] = 0
            self.i_stepbuf_starts[i] = 0
            self.i_stepbuf_ends[i] = 0

        self.prl_reader.just_read()
        
        if self.max_retries > 0:

            cn_retries = 0
            while not self.is_complete():
                flag_founds = smd_inprogress_converted()

                # Only when .inprogress file is used and ALL xtc2 files are found 
                # that this will return a list of all(True). If we have a mixed
                # of True and False, we let ParallelReader decides which file
                # to read but we'll still need to do sleep.
                if all(flag_founds): break

                time.sleep(self.sleep_secs)
                print(f'waiting for an event...retry#{cn_retries+1} (max_retries={self.max_retries}, use PS_R_MAX_RETRIES for different value)')
                self.prl_reader.just_read()
                cn_retries += 1
                if cn_retries >= self.max_retries:
                    break
        
        # Reset winning buffer when we read-in more data
        self.winner = -1

    @cython.boundscheck(False)
    def view(self, int batch_size=1000, int intg_stream_id=-1):
        """ Returns memoryview of the data and step buffers.

        This function is called by SmdReaderManager only when is_complete is True (
        all buffers have at least one event). It returns events of batch_size if
        possible or as many as it has for the buffer.

        Integrating detector:
        When intg_stream_id is set (value > -1)
        """
        st_all = time.monotonic()
        cdef int i=0
        cdef int i_eob=0

        # Find the winning buffer (slowest detector)- skip when no new read.
        cdef uint64_t limit_ts=0
        if intg_stream_id > -1:
            self.winner = intg_stream_id
        else:
            if self.winner == -1:
                for i in range(self.prl_reader.nfiles):
                    if self.prl_reader.bufs[i].timestamp < limit_ts or limit_ts == 0:
                        self.winner = i
                        limit_ts = self.prl_reader.bufs[self.winner].timestamp
        limit_ts = self.prl_reader.bufs[self.winner].timestamp

        # No. of events available in the viewing window
        self.n_view_events = self.prl_reader.bufs[self.winner].n_ready_events - \
                self.prl_reader.bufs[self.winner].n_seen_events
        # Index of the last event in the viewing window
        i_eob = self.prl_reader.bufs[self.winner].n_ready_events - 1
        
        # Apply batch_size- find boundaries (limit ts) of the winning buffer.
        # this is either the nth or the batch_size event.
        if self.n_view_events > batch_size:
            i_eob               = self.prl_reader.bufs[self.winner].n_seen_events - 1 + batch_size
            limit_ts            = self.prl_reader.bufs[self.winner].ts_arr[i_eob]
            self.n_view_events  = batch_size
        
        # Save timestamp and transition id of the last event in batch
        self.winner_last_sv = self.prl_reader.bufs[self.winner].sv_arr[i_eob]      
        self.winner_last_ts = self.prl_reader.bufs[self.winner].ts_arr[i_eob]

        # Reset timestamp and buffer for fake steps (will be calculated lazily)
        self._next_fake_ts = 0
        self._fakebuf_size = 0

        # Locate the viewing window and update seen_offset for each buffer
        cdef Buffer* buf
        #cdef uint64_t[:] buf_ts_arr     
        cdef uint64_t[:] i_starts           = self.i_starts
        cdef uint64_t[:] i_ends             = self.i_ends 
        cdef uint64_t[:] i_stepbuf_starts   = self.i_stepbuf_starts
        cdef uint64_t[:] i_stepbuf_ends     = self.i_stepbuf_ends 
        cdef uint64_t[:] block_sizes        = self.block_sizes 
        cdef uint64_t[:] i_st_bufs          = self.i_st_bufs
        cdef uint64_t[:] block_size_bufs    = self.block_size_bufs
        cdef uint64_t[:] i_st_stepbufs      = self.i_st_stepbufs
        cdef uint64_t[:] block_size_stepbufs= self.block_size_stepbufs

        # Need to convert Python object to c++ data type for the nogil loop 
        cdef unsigned endrun_id = TransitionId.EndRun
        
        st_search = time.monotonic()
        
        # Find the boundary of each buffer using limit_ts
        for i in prange(self.prl_reader.nfiles, nogil=True, num_threads=self.num_threads):
            buf = &(self.prl_reader.bufs[i])
            i_st_bufs[i] = 0
            block_size_bufs[i] = 0
            
            if buf.ts_arr[i_starts[i]] > limit_ts: continue

            i_ends[i] = i_starts[i] 
            if i_ends[i] < buf.n_ready_events:
                if buf.ts_arr[i_ends[i]] != limit_ts:
                    ## Reduce the size of the search by searching from the next unseen event
                    #i_ends[i] = np.searchsorted(buf_ts_arr[buf.n_seen_events:buf.n_ready_events], limit_ts) + buf.n_seen_events

                    ## Note that limit_ts should be at this found index or its left index
                    ## If the found index is 0, then this buffer has nothing to share.
                    #if buf.ts_arr[i_ends[i]] != limit_ts:
                    #    if i_ends[i] == 0:
                    #        i_ends[i] = i_starts[i] - 1
                    #    else:
                    #        i_ends[i] -= 1
                    
                    while buf.ts_arr[i_ends[i] + 1] <= limit_ts \
                            and i_ends[i] < buf.n_ready_events - 1:
                        i_ends[i] += 1
                
                block_sizes[i] = buf.en_offset_arr[i_ends[i]] - buf.st_offset_arr[i_starts[i]]
               
                i_st_bufs[i] = i_starts[i]
                block_size_bufs[i] = block_sizes[i]
                
                buf.seen_offset = buf.en_offset_arr[i_ends[i]]
                buf.n_seen_events =  i_ends[i] + 1
                if buf.sv_arr[i_ends[i]] == endrun_id:
                    buf.found_endrun = 1
                i_starts[i] = i_ends[i] + 1

            
            # Handle step buffers the same way
            buf = &(self.prl_reader.step_bufs[i])
            i_st_stepbufs[i] = 0
            block_size_stepbufs[i] = 0
            
            # Find boundary using limit_ts (omit check for exact match here because it's unlikely
            # for transition buffers.
            i_stepbuf_ends[i] = i_stepbuf_starts[i] 
            if i_stepbuf_ends[i] <  buf.n_ready_events \
                    and buf.ts_arr[i_stepbuf_ends[i]] <= limit_ts: 
                while buf.ts_arr[i_stepbuf_ends[i] + 1] <= limit_ts \
                        and i_stepbuf_ends[i] < buf.n_ready_events - 1:
                    i_stepbuf_ends[i] += 1
                
                block_sizes[i] = buf.en_offset_arr[i_stepbuf_ends[i]] - buf.st_offset_arr[i_stepbuf_starts[i]]
                
                i_st_stepbufs[i] = i_stepbuf_starts[i]
                block_size_stepbufs[i] = block_sizes[i]
                
                buf.seen_offset = buf.en_offset_arr[i_stepbuf_ends[i]]
                buf.n_seen_events = i_stepbuf_ends[i] + 1
                i_stepbuf_starts[i]  = i_stepbuf_ends[i] + 1

        # end for i in ...
        en_all = time.monotonic()

        self.total_time += en_all - st_all

    def get_next_fake_ts(self):
        """Returns next fake timestamp 

        This is calculated from the last event in the current batch of the
        winning stream. The `_next_fake_ts` is reset every time the view()
        function is called. When one of the streams calls this function,
        we remember the value that will be used in other streams. This is
        done so that all fake dgrams in differnt streams share the same
        timestamp.
        """
        if self._next_fake_ts == 0:
            self._next_fake_ts = self.winner_last_ts + 1
        return self._next_fake_ts

    def get_fake_buffer(self):
        """Returns set of fake transitions.

        For inserting in all streams files. These transtion set is the
        same therefore we only need to generate once.
        """
        cdef unsigned i_fake=0, out_offset = 0
        
        # Only create fake buffer when set and the last dgram is a valid transition
        if self.fakestep_flag == 1 and self.winner_last_sv in (TransitionId.L1Accept, TransitionId.SlowUpdate):
            # Only need to create once since fake transition set is the same for all streams.
            if self._fakebuf_size == 0:
                fake_config = DgramEdit(transition_id=TransitionId.Configure, ts=0)
                fake_transitions = [TransitionId.Disable, 
                                    TransitionId.EndStep,
                                    TransitionId.BeginStep,
                                    TransitionId.Enable,
                                   ]
                fake_ts = self.get_next_fake_ts()
                for i_fake, fake_transition in enumerate(fake_transitions):
                    fake_dgram = DgramEdit(transition_id=fake_transition, config=fake_config, ts=fake_ts+i_fake)
                    fake_dgram.save(self._fakebuf, offset=out_offset)
                    out_offset += fake_dgram.size 
                self._fakebuf_size = out_offset
            return self._fakebuf[:self._fakebuf_size]
        else:
            return memoryview(bytearray())

    def show(self, int i_buf, step_buf=False):
        """ Returns memoryview of buffer i_buf at the current viewing
        i_st and block_size.
        
        If fake transition need to be inserted (see conditions below),
        we need to copy smd or step buffers to a new buffer and append
        it with these transitions.
        """

        cdef Buffer* buf
        cdef uint64_t[:] block_size_bufs
        cdef uint64_t[:] i_st_bufs      
        if step_buf:
            buf = &(self.prl_reader.step_bufs[i_buf])
            block_size_bufs = self.block_size_stepbufs
            i_st_bufs = self.i_st_stepbufs
        else:
            buf = &(self.prl_reader.bufs[i_buf])
            block_size_bufs = self.block_size_bufs
            i_st_bufs = self.i_st_bufs
        
        cdef char[:] view
        if block_size_bufs[i_buf] > 0:
            view = <char [:block_size_bufs[i_buf]]> (buf.chunk + buf.st_offset_arr[i_st_bufs[i_buf]])
            # Check if there's any data in fake buffer
            fakebuf = self.get_fake_buffer()
            if self._fakebuf_size > 0:
                outbuf = bytearray()
                outbuf.extend(view)
                outbuf.extend(fakebuf)

                # Increase total no. of events in the viewing window to include
                # fake step transition set (four events: Disable, EndStep,
                # BeginStep Enable) is inserted in all streams but
                # we only need to increase no. of available `events` once.
                #
                # NOTE that it's NOT ideal to do it here but this show() function is 
                # called once for normal buffer by RunSerial and once for step buffer 
                # by RunParallel.
                if i_buf == self.winner:
                    self.n_view_events += 4
                return memoryview(outbuf)
            else:
                return view
        else:
            return memoryview(bytearray()) 
    
    def get_total_view_size(self):
        """ Returns sum of the viewing window sizes for all buffers."""
        cdef int i
        cdef uint64_t total_size = 0
        for i in range(self.prl_reader.nfiles):
            total_size += self.block_size_bufs[i]
        return total_size

    @property
    def view_size(self):
        return self.n_view_events

    @property
    def total_time(self):
        return self.total_time

    @property
    def got(self):
        return self.prl_reader.got

    @property
    def chunk_overflown(self):
        return self.prl_reader.chunk_overflown

    @property
    def winner_last_sv(self):
        return self.winner_last_sv

    @property
    def winner_last_ts(self):
        return self.winner_last_ts

    def found_endrun(self):
        cdef int i
        found = False
        cn_endruns = 0
        for i in range(self.prl_reader.nfiles):
            if self.prl_reader.bufs[i].found_endrun == 1:
                cn_endruns += 1
        if cn_endruns == self.prl_reader.nfiles:
            found = True
        return found

    def repack(self, step_views, eb_node_id, only_steps=False):
        """ Repack step and smd data in one consecutive chunk with footer at end."""
        cdef Buffer* smd_buf
        cdef Py_buffer step_buf
        cdef int i=0, offset=0
        cdef uint64_t smd_size=0, step_size=0, footer_size=0, total_size=0
        cdef char[:] view
        cdef int eb_idx = eb_node_id - 1        # eb rank starts from 1 (0 is Smd0)
        cdef char* send_buf
        send_buf = self.send_bufs[eb_idx].chunk # select the buffer for this eb
        cdef uint32_t* footer = <uint32_t*>self.repack_footer.data.as_voidptr
        
        # Copy step and smd buffers if exist
        for i in range(self.prl_reader.nfiles):
            PyObject_GetBuffer(step_views[i], &step_buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            view_ptr = <char *>step_buf.buf
            step_size = step_buf.len
            if step_size > 0:
                memcpy(send_buf + offset, view_ptr, step_size)
                offset += step_size
            PyBuffer_Release(&step_buf)

            smd_size = 0
            if not only_steps:
                smd_size = self.block_size_bufs[i]
                if smd_size > 0:
                    smd_buf = &(self.prl_reader.bufs[i])
                    memcpy(send_buf + offset, smd_buf.chunk + smd_buf.st_offset_arr[self.i_st_bufs[i]], smd_size)
                    offset += smd_size
            
            footer[i] = step_size + smd_size
            total_size += footer[i]

        # Copy footer
        footer_size = sizeof(unsigned) * (self.prl_reader.nfiles + 1)
        memcpy(send_buf + offset, footer, footer_size) 
        total_size += footer_size
        view = <char [:total_size]> (send_buf) 
        return view

    def repack_parallel(self, step_views, eb_node_id, only_steps=0):
        """ Repack step and smd data in one consecutive chunk with footer at end.
        Memory copying is done is parallel.
        """
        cdef Py_buffer step_buf
        cdef char* ptr_step_bufs[1000]          # FIXME No easy way to fix this yet.
        cdef uint64_t* offsets = <uint64_t*>self.repack_offsets.data.as_voidptr
        cdef uint64_t* step_sizes = <uint64_t*>self.repack_step_sizes.data.as_voidptr
        cdef char[:] view
        cdef int c_only_steps = only_steps
        cdef int eb_idx = eb_node_id - 1        # eb rank starts from 1 (0 is Smd0)
        cdef char* send_buf
        if only_steps and self.is_legion:
            send_buf = self.send_step_bufs[eb_idx].chunk # select the buffer for this eb
        else:
            send_buf = self.send_bufs[eb_idx].chunk # select the buffer for this eb
        cdef int i=0, offset=0
        cdef uint64_t footer_size=0, total_size=0
        
        # Check if we need to append fakestep transition set
        cdef Py_buffer fake_pybuf
        cdef char* fakebuf_ptr
        fakebuf = self.get_fake_buffer()
        PyObject_GetBuffer(fakebuf, &fake_pybuf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        fakebuf_ptr = <char *>fake_pybuf.buf
        PyBuffer_Release(&fake_pybuf)

        # Compute beginning offsets of each chunk and get a list of buffer objects
        # If fakestep_flag is set, we need to append fakestep transition step
        # to the new repacked data.
        for i in range(self.prl_reader.nfiles):
            offsets[i] = offset
            # Move offset and total size to include missing steps
            step_sizes[i]  = memoryview(step_views[i]).nbytes
            total_size += step_sizes[i] 
            offset += step_sizes[i]

            # Move offset and total size to include smd data 
            if only_steps==0:
                total_size += self.block_size_bufs[i]
                offset += self.block_size_bufs[i]
            
            # Move offset and total size to include fakestep transition set (if set)
            total_size += self._fakebuf_size 
            offset += self._fakebuf_size
            # Stores the pointers to missing step buffers for parallel loop below
            PyObject_GetBuffer(step_views[i], &step_buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            ptr_step_bufs[i] = <char *>step_buf.buf
            PyBuffer_Release(&step_buf)

        # Access raw C pointers so they can be used in nogil loop below
        cdef uint64_t* block_size_bufs = <uint64_t*>self.block_size_bufs.data.as_voidptr
        cdef uint64_t* i_st_bufs = <uint64_t*>self.i_st_bufs.data.as_voidptr
        cdef uint32_t* footer = <uint32_t*>self.repack_footer.data.as_voidptr

        # Copy step and smd buffers if exist
        for i in prange(self.prl_reader.nfiles, nogil=True, num_threads=self.num_threads):
            footer[i] = 0
            if step_sizes[i] > 0:
                memcpy(send_buf + offsets[i], ptr_step_bufs[i], step_sizes[i])
                offsets[i] += step_sizes[i]
                footer[i] += step_sizes[i]
            
            if c_only_steps == 0:
                if block_size_bufs[i] > 0:
                    memcpy(send_buf + offsets[i], 
                            self.prl_reader.bufs[i].chunk + self.prl_reader.bufs[i].st_offset_arr[i_st_bufs[i]], 
                            block_size_bufs[i])
                    offsets[i] += block_size_bufs[i]
                    footer[i] += block_size_bufs[i]

            if self._fakebuf_size > 0:
                memcpy(send_buf + offsets[i], fakebuf_ptr, self._fakebuf_size)
                offsets[i] += self._fakebuf_size
                footer[i] += self._fakebuf_size

        # Copy footer 
        footer_size = sizeof(uint32_t) * (self.prl_reader.nfiles + 1)
        memcpy(send_buf + total_size, footer, footer_size) 
        total_size += footer_size
        view = <char [:total_size]> (send_buf) 
        return view
            
    def repack_only_buf(self, eb_node_id):
        """ Repack only buf (no step) smd data in one consecutive chunk with footer at end.
        Memory copying is done is parallel.
        """
        cdef Py_buffer step_buf
        cdef uint64_t* offsets = <uint64_t*>self.repack_offsets.data.as_voidptr
        cdef char[:] view
        cdef int eb_idx = eb_node_id - 1        # eb rank starts from 1 (0 is Smd0)
        cdef char* send_buf
        send_buf = self.send_bufs[eb_idx].chunk # select the buffer for this eb
        cdef int i=0, offset=0
        cdef uint64_t footer_size=0, total_size=0
        
        # Compute beginning offsets of each chunk and get a list of buffer objects
        for i in range(self.prl_reader.nfiles):
            offsets[i] = offset
            # Move offset and total size to include smd data 
            total_size += self.block_size_bufs[i]
            offset += self.block_size_bufs[i]
            
        # Access raw C pointers so they can be used in nogil loop below
        cdef uint64_t* block_size_bufs = <uint64_t*>self.block_size_bufs.data.as_voidptr
        cdef uint64_t* i_st_bufs = <uint64_t*>self.i_st_bufs.data.as_voidptr
        cdef uint32_t* footer = <uint32_t*>self.repack_footer.data.as_voidptr

        # Copy step and smd buffers if exist
        for i in prange(self.prl_reader.nfiles, nogil=True, num_threads=self.num_threads):
            footer[i] = 0
            if block_size_bufs[i] > 0:
                memcpy(send_buf + offsets[i], 
                       self.prl_reader.bufs[i].chunk + self.prl_reader.bufs[i].st_offset_arr[i_st_bufs[i]], 
                       block_size_bufs[i])
                offsets[i] += block_size_bufs[i]
                footer[i] += block_size_bufs[i]

        # Copy footer 
        footer_size = sizeof(uint32_t) * (self.prl_reader.nfiles + 1)
        memcpy(send_buf + total_size, footer, footer_size) 
        total_size += footer_size
        view = <char [:total_size]> (send_buf) 
        return view
