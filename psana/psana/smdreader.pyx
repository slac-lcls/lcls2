## cython: linetrace=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from cpython cimport array
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from parallelreader cimport Buffer, ParallelReader

import os
import time

cimport cython

from cython.parallel import prange

DEF DEBUG_MODULE = "SmdReader"
include "cydebug.pxh"

from psana.psexp import TransitionId

from cpython.buffer cimport (PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE,
                             PyBuffer_Release, PyObject_GetBuffer)

from psana.dgramedit import DgramEdit

from cpython.getargs cimport PyArg_ParseTupleAndKeywords
from cpython.object cimport PyObject
from dgramlite cimport Dgram


cdef uint64_t INVALID_TS = 0xFFFFFFFFFFFFFFFF


cdef class SmdReader:
    cdef ParallelReader prl_reader
    cdef int         winner
    cdef uint64_t    n_view_events
    cdef uint64_t    n_view_L1Accepts
    cdef uint64_t    n_processed_events
    cdef int         max_retries, sleep_secs
    cdef array.array i_st_nextblocks            # ¬ used locally for finding boundary of each
    cdef array.array i_st_blocks                # } which generates sharing window variables below.
    cdef array.array i_en_blocks                # } for sharing viewing windows in show()
    cdef array.array block_sizes                # }
    cdef array.array i_st_step_nextblocks       # }
    cdef array.array i_st_step_blocks           # }
    cdef array.array i_en_step_blocks           # }
    cdef array.array step_block_sizes           # }
    cdef array.array repack_offsets             # ¬ for parallel repack
    cdef array.array repack_step_sizes          # }
    cdef array.array repack_footer              # }
    cdef array.array _i_st_blocks_firstbatch
    cdef array.array _i_st_step_blocks_firstbatch
    cdef array.array _cn_batch_bufs
    cdef array.array _cn_batch_stepbufs
    cdef unsigned    winner_last_sv             # ¬ transition id and ts of the last dgram in winner's chunk
    cdef uint64_t    winner_last_ts             # }
    cdef uint64_t    _next_fake_ts              # incremented from winner_last_ts (shared by all streams)
    cdef float       total_time
    cdef int         num_threads
    cdef Buffer*     send_bufs                  # array of customed Buffers (one per EB node)
    cdef uint64_t    sendbufsize                # size of each send buffer
    cdef int         n_eb_nodes
    cdef int         fakestep_flag
    cdef list        configs
    cdef bytearray   _fakebuf
    cdef unsigned    _fakebuf_maxsize
    cdef unsigned    _fakebuf_size
    cdef PyObject*   dsparms
    cdef uint8_t     L1Accept
    cdef uint8_t     L1Accept_EndOfBatch

    def __init__(self, int[:] fds, int chunksize, *args, **kwargs):
        assert fds.size > 0, "Empty file descriptor list (fds.size=0)."

        # Keyword args that need to be passed in once. To save some of
        # them as cpp class attributes, we need to read them in as PyObject*.
        cdef char* kwlist[2]
        kwlist[0] = "dsparms"
        kwlist[1] = NULL
        if PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &(self.dsparms)) is False:
            raise RuntimeError, "Invalid kwargs for SmdReader"

        dsparms                 = <object> self.dsparms
        self.prl_reader         = ParallelReader(fds, chunksize, dsparms=dsparms)
        self.max_retries        = dsparms.max_retries
        self.sleep_secs         = 1
        self.total_time         = 0
        self.num_threads        = int(os.environ.get('PS_SMD0_NUM_THREADS', '16'))
        self.n_eb_nodes         = int(os.environ.get('PS_EB_NODES', '1'))
        self.winner_last_ts     = 0
        self._next_fake_ts      = 0
        self.configs            = []
        self.i_st_nextblocks    = array.array('L', [0]*fds.size)
        self.i_st_step_nextblocks = array.array('L', [0]*fds.size)
        self.i_st_blocks        = array.array('L', [0]*fds.size)
        self.block_sizes        = array.array('L', [0]*fds.size)
        self.i_en_blocks        = array.array('L', [0]*fds.size)
        self.i_st_step_blocks   = array.array('L', [0]*fds.size)
        self.step_block_sizes   = array.array('L', [0]*fds.size)
        self.i_en_step_blocks   = array.array('L', [0]*fds.size)
        self.repack_offsets     = array.array('L', [0]*fds.size)
        self.repack_step_sizes  = array.array('L', [0]*fds.size)
        self.repack_footer      = array.array('I', [0]*(fds.size+1))  # size of all smd chunks plus no. of smds
        self._i_st_blocks_firstbatch = array.array('L', [0]*fds.size)
        self._i_st_step_blocks_firstbatch = array.array('L', [0]*fds.size)
        self._cn_batch_bufs = array.array('L', [0]*fds.size)
        self._cn_batch_stepbufs = array.array('L', [0]*fds.size)
        self._fakebuf_maxsize   = 0x1000
        self._fakebuf           = bytearray(self._fakebuf_maxsize)
        self._fakebuf_size      = 0
        self.n_processed_events = 0
        self.L1Accept           = TransitionId.L1Accept
        self.L1Accept_EndOfBatch= TransitionId.L1Accept_EndOfBatch

        # Repack footer contains constant (per run) no. of smd files
        self.repack_footer[fds.size] = fds.size

        # For speed, SmdReader puts together smd chunk and missing step info
        # in parallel and store the new data in `send_bufs` (one buffer per stream).
        self.send_bufs          = <Buffer *>malloc(self.n_eb_nodes * sizeof(Buffer))
        self.sendbufsize        = 0x10000000
        self._init_send_buffers()

        # Index of the slowest detector or intg_stream_id if given.
        self.winner             = -1

        # Sets event frequency that fake EndStep/BeginStep pair is inserted.
        self.fakestep_flag = int(os.environ.get('PS_FAKESTEP_FLAG', 0))

    def __dealloc__(self):
        cdef Py_ssize_t i
        for i in range(self.n_eb_nodes):
            free(self.send_bufs[i].chunk)

    cdef void _init_send_buffers(self):
        cdef Py_ssize_t i
        for i in range(self.n_eb_nodes):
            self.send_bufs[i].chunk      = <char *>malloc(self.sendbufsize)

    def set_configs(self, configs):
        # SmdReaderManager calls view (with batch_size=1)  at the beginning
        # to read config dgrams. It passes the configs to SmdReader for
        # creating any fake dgrams using DgramEdit.
        self.configs = configs

    def force_read(self, smd_inprogress_converted):
        """
        Force a full read across all streams.

        Resets buffer indices and reads new data into the ParallelReader buffers.
        In live mode, retries up to max_retries if new data isn't immediately available.
        """
        debug_print("force_read called")

        # Exit if EndRun is found for all files. This is safe even though
        # we support mulirun in one xtc2 file but this is only for shmem mode,
        # which doesn't use SmdReader.
        if self.found_endrun():
            debug_print("Exit - EndRun found")
            return

        # Always reset the per‐buffer “next block” indices before reading,
        # so partial tails always get carried forward.
        for i in range(self.prl_reader.nfiles):
            self.i_st_nextblocks[i]      = 0
            self.i_st_step_nextblocks[i] = 0

        self.prl_reader.force_read()

        # Reset winning buffer when we read-in more data
        self.winner = -1

    def find_limit_ts(self, batch_size, max_events, ignore_transition):
        """ Find the winning stream and the limit_ts.
        This limit_ts is used to find the offset of all the chunks yielded back
        as a memoryview.

        batch_size and max_events only count L1Accept when ignore_transition is set.

        We also set the attributes that keep records of L1Accept and all events per call
        and all events that have been processed from the begining.
        """
        debug_print("find_limit_ts called")

        cdef int i=0
        is_transition = not ignore_transition

        # The winning stream is the one with smallest timestamp and the end of its chunk.
        # We only need to do this once at a new read.
        cdef uint64_t limit_ts=INVALID_TS
        cdef uint64_t buf_ts=0
        if self.winner == -1:
            debug_print("    Find winner:")
            for i in range(self.prl_reader.nfiles):
                if self.prl_reader.bufs[i].n_ready_events == 0:
                    continue
                buf_ts = self.prl_reader.bufs[i].ts_arr[self.prl_reader.bufs[i].n_ready_events-1]
                if limit_ts == INVALID_TS:
                    self.winner = i
                    limit_ts = buf_ts
                else:
                    if buf_ts == limit_ts:
                        if self.winner > -1:
                            if self.prl_reader.bufs[i].n_ready_events > \
                                    self.prl_reader.bufs[self.winner].n_ready_events:
                                self.winner = i
                    elif buf_ts < limit_ts:
                        self.winner = i
                        limit_ts = buf_ts
                debug_print(f"    file[{i}]: n_ready_events={self.prl_reader.bufs[i].n_ready_events} "
                            f"buf_ts={buf_ts} limit_ts={limit_ts} --> winner={self.winner}")
        else:
            if self.prl_reader.bufs[self.winner].n_ready_events > 0:
                limit_ts = self.prl_reader.bufs[self.winner].ts_arr[self.prl_reader.bufs[self.winner].n_ready_events-1]
                debug_print(f"    Known winner: {self.winner} "
                            f"n_ready_events={self.prl_reader.bufs[self.winner].n_ready_events} "
                            f"limit_ts={limit_ts}")

        # Apply batch_size and max_events
        cdef int n_L1Accepts=0
        cdef int n_events=0
        # Index of the first and last event in the viewing window
        cdef int i_bob=0, i_eob=0

        if limit_ts != INVALID_TS:
            i_bob = self.prl_reader.bufs[self.winner].n_seen_events - 1
            i_eob = self.prl_reader.bufs[self.winner].n_ready_events - 1
            debug_print(f"    Apply batch cut-off is_transition={is_transition} max_events={max_events} "
                        f" i_bob={i_bob} i_eob={i_eob}")

            if is_transition:
                if max_events == 0:
                    for i in range(i_bob+1, i_eob + 1):
                        n_events += 1
                        if TransitionId.isEvent(self.prl_reader.bufs[self.winner].sv_arr[i]):
                            n_L1Accepts +=1
                        if n_events == batch_size:
                            debug_print(f"    i={i} Configure/BeginRun n_events={n_events} batch_size={batch_size}")
                            break
                else:
                    for i in range(i_bob+1, i_eob + 1):
                        n_events += 1
                        if TransitionId.isEvent(self.prl_reader.bufs[self.winner].sv_arr[i]):
                            n_L1Accepts +=1
                        if n_events == batch_size:
                            break
                        if self.n_processed_events + n_events == max_events:
                            break
            else:
                if max_events == 0:
                    for i in range(i_bob+1, i_eob + 1):
                        n_events += 1
                        if TransitionId.isEvent(self.prl_reader.bufs[self.winner].sv_arr[i]):
                            n_L1Accepts +=1
                        if n_L1Accepts == batch_size:
                            debug_print(f"    i={i} Data n_L1={n_L1Accepts} n_events={n_events} batch_size={batch_size}")
                            break
                else:
                    for i in range(i_bob+1, i_eob + 1):
                        n_events += 1
                        if TransitionId.isEvent(self.prl_reader.bufs[self.winner].sv_arr[i]):
                            n_L1Accepts +=1
                        if n_L1Accepts == batch_size:
                            break
                        if self.n_processed_events + n_L1Accepts == max_events:
                            break

            i_eob = i
            limit_ts = self.prl_reader.bufs[self.winner].ts_arr[i_eob]
            debug_print(f"    Done apply batch i_eob={i_eob} winner={self.winner} limit_ts={limit_ts}")
            # Save timestamp and transition id of the last event in batch
            self.winner_last_sv = self.prl_reader.bufs[self.winner].sv_arr[i_eob]
            self.winner_last_ts = self.prl_reader.bufs[self.winner].ts_arr[i_eob]

        self.n_view_events  = n_events
        self.n_view_L1Accepts = n_L1Accepts
        self.n_processed_events += n_L1Accepts
        debug_print(f"Exit find_limit_ts limit_ts={limit_ts}")
        return limit_ts

    def find_intg_limit_ts(self, intg_stream_id, intg_delta_t, max_events):
        """ Find limit_ts for a single integrating event

        An integrating event accumulates over fast events in other streams.
        The winner is automatically set to the integrating stream and limit_ts
        is yielded only when the fast streams have events beyond the integrating
        event.
        """
        self.winner = intg_stream_id
        cdef uint64_t limit_ts = 0xFFFFFFFFFFFFFFFF
        cdef uint64_t buf_ts=0
        cdef int i=0, j=0

        # Index of the first and last event in the viewing window
        cdef int i_bob=0, i_eob=0, i_complete=0

        cdef int n_L1Accepts=0
        cdef int n_transitions=0
        cdef int is_split = 0

        # Locate an integrating event and check for max_events
        # We still need to loop ever the available events to skip
        # transitions.
        i_bob = self.prl_reader.bufs[self.winner].n_seen_events - 1
        i_eob = self.prl_reader.bufs[self.winner].n_ready_events - 1
        i_complete = i_bob
        limit_ts_complete = self.prl_reader.bufs[self.winner].ts_arr[i_complete]

        debug_print(f"Enter find_intg_limit_ts i_bob={i_bob} limit_ts_complete={limit_ts_complete}")
        cdef uint64_t ts_sec=0, ts_nsec=0, ts_sum=0
        for i in range(i_bob+1, i_eob + 1):
            if TransitionId.isEvent(self.prl_reader.bufs[self.winner].sv_arr[i]):
                # For correct math operation, we need to convert timestamp to seconds
                # and nanoseconds first, do the operation, and convert the result
                # back to the timestamp format.
                ts_sec = (self.prl_reader.bufs[self.winner].ts_arr[i] >> 32) & 0xffffffff
                ts_nsec = self.prl_reader.bufs[self.winner].ts_arr[i] & 0xffffffff
                ts_sum = ts_sec*1000000000 + ts_nsec + intg_delta_t
                limit_ts = (ts_sum/1000000000)<<32 | (ts_sum % 1000000000)
                # Check if other streams have timestamp up to this integrating event
                is_split = 0
                for j in range(self.prl_reader.nfiles):
                    if j == intg_stream_id:
                        continue
                    buf_ts = self.prl_reader.bufs[j].ts_arr[self.prl_reader.bufs[j].n_ready_events-1]
                    if buf_ts < limit_ts:
                        is_split = 1
                if not is_split:
                    n_L1Accepts += 1
                    i_complete = i
                    limit_ts_complete = limit_ts
            else:
                n_transitions += 1
                # TODO: Also check for split EndRun
                if self.prl_reader.bufs[self.winner].sv_arr[i] == TransitionId.EndRun:
                    i_complete = i
                    limit_ts_complete = self.prl_reader.bufs[self.winner].ts_arr[i_complete]

            if n_L1Accepts or is_split \
                    or (self.n_processed_events + n_L1Accepts == max_events and max_events > 0):
                break

        self.n_view_events  = n_L1Accepts + n_transitions
        self.n_view_L1Accepts = n_L1Accepts
        self.n_processed_events += n_L1Accepts
        # Save timestamp and transition id of the last event in batch
        self.winner_last_sv = self.prl_reader.bufs[self.winner].sv_arr[i_complete]
        self.winner_last_ts = self.prl_reader.bufs[self.winner].ts_arr[i_complete]

        debug_print(f"Exit find_intg_limit_ts limit_ts_complete={limit_ts_complete}")

        return limit_ts_complete

    def mark_endofbatch(self):
        """ Only for integrating detector. First, identify the stream with largest
        timestamp (fastest) and change the TransitionId of the last dgram in the
        integrating sub-batch to L1Accept_EndOfBatch.
        """
        # Find the stream with the largest timestamp
        cdef int eob_stream_id = -1
        cdef uint64_t eob_ts = 0
        cdef int i
        for i in range(self.prl_reader.nfiles):
            if self.prl_reader.bufs[i].ts_arr[self.i_en_blocks[i]] > eob_ts:
                eob_ts = self.prl_reader.bufs[i].ts_arr[self.i_en_blocks[i]]
                eob_stream_id = i

        # Update the transition
        cdef Dgram* d
        cdef uint8_t service
        cdef uint64_t new_env
        cdef uint64_t second_byte = 0xf0ffffff  # Safe if used in no-gil
        if self.block_sizes[eob_stream_id] > 0:
            d = <Dgram *>(self.prl_reader.bufs[eob_stream_id].chunk + self.prl_reader.bufs[eob_stream_id].st_offset_arr[self.i_en_blocks[eob_stream_id]])
            # From 8 bytes env, |0 |0 |0 |0 |x |- |- |- |  x is service byte
            service = (d.env>>24)&0xf
            new_env = d.env & second_byte | self.L1Accept_EndOfBatch << 24
            if service == self.L1Accept:
                memcpy(&(d.env), &new_env, sizeof(uint32_t))

    @cython.boundscheck(False)
    def build_batch_view(self, batch_size=1000, intg_stream_id=-1, intg_delta_t=0, max_events=0, ignore_transition=True):
        """
        Build a view (window) of the next batch of events from the buffers.

        This function selects events across all streams according to either
        normal mode (timestamp-ordered) or integrating mode (based on a target stream).
        It updates internal offsets and counters to mark events as seen.

        It returns True if at least one event was found and prepared, False otherwise.

        Args:
            batch_size (int): Number of events to group into one batch.
            intg_stream_id (int): Index of the integrating detector stream, or -1 if none.
            intg_delta_t (int): Integration time window (nanoseconds) for integrating detectors.
            max_events (int): Maximum number of events to process across the run.
            ignore_transition (bool): Whether to skip transition events.

        Returns:
            bool: True if a valid batch is prepared, False if no events found.
        """
        st_all = time.monotonic()

        debug_print("build_batch_view called")
        debug_print(f"    batch_size={batch_size} intg_stream_id={intg_stream_id} "
                    f"intg_delta_t={intg_delta_t} max_events={max_events} "
                    f"ignore_transition={ignore_transition}")

        cdef int i=0
        cdef uint64_t limit_ts

        # Locate the viewing window and update seen_offset for each buffer
        cdef Buffer* buf
        cdef uint64_t[:] i_st_nextblocks        = self.i_st_nextblocks
        cdef uint64_t[:] i_st_step_nextblocks   = self.i_st_step_nextblocks
        cdef uint64_t[:] i_st_blocks            = self.i_st_blocks
        cdef uint64_t[:] block_sizes            = self.block_sizes
        cdef uint64_t[:] i_en_blocks            = self.i_en_blocks
        cdef uint64_t[:] i_st_step_blocks       = self.i_st_step_blocks
        cdef uint64_t[:] step_block_sizes       = self.step_block_sizes
        cdef uint64_t[:] i_en_step_blocks       = self.i_en_step_blocks
        cdef uint64_t[:] i_st_blocks_firstbatch = self._i_st_blocks_firstbatch
        cdef uint64_t[:] i_st_step_blocks_firstbatch = self._i_st_step_blocks_firstbatch
        cdef uint64_t[:] cn_batch_bufs          = self._cn_batch_bufs
        cdef uint64_t[:] cn_batch_stepbufs      = self._cn_batch_stepbufs

        # Reset buffer index and size for both normal and step buffers.
        # They will get set to the boundary value if the current timestamp
        # does not exceed limiting timestamp.
        for i in range(self.prl_reader.nfiles):
            i_st_blocks[i] = 0
            i_en_blocks[i] = 0
            block_sizes[i] = 0
            i_st_step_blocks[i] = 0
            i_en_step_blocks[i] = 0
            step_block_sizes[i] = 0
            i_st_blocks_firstbatch[i] = 0
            i_st_step_blocks_firstbatch[i] = 0
            cn_batch_bufs[i] = 0
            cn_batch_stepbufs[i] = 0

        # Need to convert Python object to c++ data type for the nogil loop
        cdef unsigned endrun_id = TransitionId.EndRun

        # A batch is always complete for non-integrating detector runs. For integrating
        # runs, we continue to find the indices of the all the buffers until batch_size
        # is reached (find_intg_limit_ts only finds limit_ts for a single integrating
        # event).
        cdef int batch_complete_flag = 0
        cdef int cn_intg_events = 0

        while not batch_complete_flag:
            # With batch_size = 1 (used for reading Configure and BeginRun), this
            # automatically set ignore_transition flag to False so we can get any
            # type of event. We also use standard way to find limit_ts when asking
            # for Configure and BeginRun. We also check here if there's no event
            # to yield, and return False.
            if intg_stream_id == -1 or ignore_transition is False:
                limit_ts = self.find_limit_ts(batch_size, max_events, ignore_transition)
                if limit_ts == INVALID_TS:
                    debug_print(f"    invalid limit_ts ({limit_ts})")
                    return False
                batch_complete_flag = 1
            else:
                limit_ts = self.find_intg_limit_ts(intg_stream_id, intg_delta_t, max_events)
                if limit_ts == self.prl_reader.bufs[self.winner].ts_arr[self.prl_reader.bufs[self.winner].n_seen_events - 1] or limit_ts == INVALID_TS:
                    debug_print(f"EARLY EXIT: limit_ts == last_ts or invalid (limit_ts={limit_ts}")
                    return False
                cn_intg_events += 1
                if cn_intg_events == batch_size:
                    batch_complete_flag = 1
            debug_print(f"    batch loop: limit_ts={limit_ts} winner={self.winner}")

            # Reset timestamp and buffer for fake steps (will be calculated lazily)
            self._next_fake_ts = 0
            self._fakebuf_size = 0

            # Find the boundary of each buffer using limit_ts
            for i in prange(self.prl_reader.nfiles, nogil=True, num_threads=self.num_threads):
                buf = &(self.prl_reader.bufs[i])

                if buf.ts_arr[i_st_nextblocks[i]] > limit_ts:
                    continue

                i_en_blocks[i] = i_st_nextblocks[i]
                if i_en_blocks[i] < buf.n_ready_events:
                    if buf.ts_arr[i_en_blocks[i]] != limit_ts:
                        while buf.ts_arr[i_en_blocks[i] + 1] <= limit_ts \
                                and i_en_blocks[i] < buf.n_ready_events - 1:
                            i_en_blocks[i] += 1

                    i_st_blocks[i] = i_st_nextblocks[i]
                    block_sizes[i] = buf.en_offset_arr[i_en_blocks[i]] - buf.st_offset_arr[i_st_nextblocks[i]]
                    if cn_batch_bufs[i] == 0:
                        i_st_blocks_firstbatch[i] = i_st_blocks[i]

                    buf.seen_offset = buf.en_offset_arr[i_en_blocks[i]]
                    buf.n_seen_events =  i_en_blocks[i] + 1
                    if buf.sv_arr[i_en_blocks[i]] == endrun_id:
                        buf.found_endrun = 1
                    i_st_nextblocks[i] = i_en_blocks[i] + 1
                    if block_sizes[i] > 0:

                        cn_batch_bufs[i] += 1

                # FOR DEBUGGING - Calling print with gil can slow down performance.
                #with gil:
                #    debug_print(f"    data[{i}]: st={i_st_blocks[i]} en={i_en_blocks[i]} "
                #            f"block_size={block_sizes[i]} seen_offset={buf.seen_offset} "
                #            f"n_seen_events={buf.n_seen_events}")

                # Handle step buffers the same way
                buf = &(self.prl_reader.step_bufs[i])

                # Find boundary using limit_ts (omit check for exact match here because it's unlikely
                # for transition buffers.
                i_en_step_blocks[i] = i_st_step_nextblocks[i]
                if i_en_step_blocks[i] <  buf.n_ready_events \
                        and buf.ts_arr[i_en_step_blocks[i]] <= limit_ts:
                    while buf.ts_arr[i_en_step_blocks[i] + 1] <= limit_ts \
                            and i_en_step_blocks[i] < buf.n_ready_events - 1:
                        i_en_step_blocks[i] += 1

                    i_st_step_blocks[i] = i_st_step_nextblocks[i]
                    step_block_sizes[i] = buf.en_offset_arr[i_en_step_blocks[i]] - buf.st_offset_arr[i_st_step_nextblocks[i]]
                    if cn_batch_stepbufs[i] == 0:
                        i_st_step_blocks_firstbatch[i] = i_st_step_blocks[i]

                    buf.seen_offset = buf.en_offset_arr[i_en_step_blocks[i]]
                    buf.n_seen_events = i_en_step_blocks[i] + 1
                    i_st_step_nextblocks[i]  = i_en_step_blocks[i] + 1
                    if step_block_sizes[i] > 0:
                        cn_batch_stepbufs[i] += 1

                # FOR DEBUGGING - Calling print with gil can slow down performance.
                #with gil:
                #    debug_print(f"    step[{i}]: st={i_st_step_blocks[i]} en={i_en_step_blocks[i]} "
                #            f"block_size={step_block_sizes[i]} seen_offset={buf.seen_offset} "
                #            f"n_seen_events={buf.n_seen_events}")

            # end for i in ...

            # Mark EndOfBatch for integrating detector run
            if intg_stream_id > -1:
                self.mark_endofbatch()

        # end while not batch

        # Restore starting buffer indices in case there are more than one batch
        for i in range(self.prl_reader.nfiles):
            if cn_batch_bufs[i] > 1:
                buf = &(self.prl_reader.bufs[i])
                i_st_blocks[i] = i_st_blocks_firstbatch[i]
                block_sizes[i] = buf.en_offset_arr[i_en_blocks[i]] - buf.st_offset_arr[i_st_blocks[i]]
            if cn_batch_stepbufs[i] > 1:
                buf = &(self.prl_reader.step_bufs[i])
                i_st_step_blocks[i] = i_st_step_blocks_firstbatch[i]
                step_block_sizes[i] = buf.en_offset_arr[i_en_step_blocks[i]] - buf.st_offset_arr[i_st_step_blocks[i]]

        # First check: Did we actually gather any usable data?
        cdef bint all_empty = True
        for i in range(self.prl_reader.nfiles):
            if block_sizes[i] > 0 or step_block_sizes[i] > 0:
                all_empty = False
                break

        if all_empty:
            debug_print("    No buffers had non-zero block size — returning False.")
            return False

        cdef uint64_t expected_n_events = 0
        cdef uint64_t n_events

        for i in range(self.prl_reader.nfiles):
            n_events = self.prl_reader.step_bufs[i].n_seen_events
            if expected_n_events == 0 and n_events > 0:
                expected_n_events = n_events
                break

        # All streams must contribute to step buffers if any do
        if expected_n_events > 0:
            for i in range(self.prl_reader.nfiles):
                if self.prl_reader.step_bufs[i].n_seen_events != expected_n_events:
                    debug_print(f"    Mismatched step buffer in stream {i} ({self.prl_reader.step_bufs[i].n_seen_events}/{expected_n_events})")
                    return False

        en_all = time.monotonic()
        self.total_time += en_all - st_all
        debug_print(f"Success build_batch_view, limit_ts={limit_ts} found_endrun={self.found_endrun()} total_time={self.total_time:.6f} sec")
        return True


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
        if self.fakestep_flag == 1 and self.winner_last_sv in (TransitionId.L1Accept, TransitionId.L1Accept_EndOfBatch, TransitionId.SlowUpdate):
            # Only need to create once since fake transition set is the same for all streams.
            if self._fakebuf_size == 0:
                fake_config = DgramEdit(transition_id=TransitionId.Configure, ts=0)
                fake_transitions = [TransitionId.Disable,
                                    TransitionId.EndStep,
                                    TransitionId.BeginStep,
                                    TransitionId.Enable]
                fake_ts = self.get_next_fake_ts()
                for i_fake, fake_transition in enumerate(fake_transitions):
                    fake_dgram = DgramEdit(transition_id=fake_transition, config_dgramedit=fake_config, ts=fake_ts+i_fake)
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
        cdef uint64_t[:] block_sizes
        cdef uint64_t[:] i_st_blocks
        if step_buf:
            buf = &(self.prl_reader.step_bufs[i_buf])
            block_sizes = self.step_block_sizes
            i_st_blocks = self.i_st_step_blocks
        else:
            buf = &(self.prl_reader.bufs[i_buf])
            block_sizes = self.block_sizes
            i_st_blocks = self.i_st_blocks

        cdef char[:] view
        if block_sizes[i_buf] > 0:
            view = <char [:block_sizes[i_buf]]> (buf.chunk + buf.st_offset_arr[i_st_blocks[i_buf]])
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
            total_size += self.block_sizes[i]
        return total_size

    @property
    def view_size(self):
        """ Returns all events (L1Accept and transtions) in the viewing window."""
        return self.n_view_events

    @property
    def n_view_L1Accepts(self):
        """ Returns no. of L1Accept in the viewing widow."""
        return self.n_view_L1Accepts

    @property
    def n_processed_events(self):
        """ Returns the total number of processed events.
        If ignore_transition is False, this value also include no. of transtions
        """
        return self.n_processed_events

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
        cdef int eb_idx = eb_node_id - 1  # eb rank starts from 1 (0 is Smd0)
        cdef char* send_buf
        send_buf = self.send_bufs[eb_idx].chunk  # select the buffer for this eb
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
                smd_size = self.block_sizes[i]
                if smd_size > 0:
                    smd_buf = &(self.prl_reader.bufs[i])
                    memcpy(send_buf + offset, smd_buf.chunk + smd_buf.st_offset_arr[self.i_st_blocks[i]], smd_size)
                    offset += smd_size

            footer[i] = step_size + smd_size
            total_size += footer[i]

        # Copy footer
        footer_size = sizeof(unsigned) * (self.prl_reader.nfiles + 1)
        memcpy(send_buf + offset, footer, footer_size)
        total_size += footer_size
        view = <char [:total_size]> (send_buf)
        return view

    def repack_parallel(self, step_views, eb_node_id, only_steps=0, intg_stream_id=-1):
        """ Repack step and smd data in one consecutive chunk with footer at end.
        Memory copying is done is parallel.
        """
        cdef Py_buffer step_buf
        cdef char* ptr_step_bufs[1000]           # FIXME No easy way to fix this yet.
        cdef uint64_t* offsets = <uint64_t*>self.repack_offsets.data.as_voidptr
        cdef uint64_t* step_sizes = <uint64_t*>self.repack_step_sizes.data.as_voidptr
        cdef char[:] view
        cdef int c_only_steps = only_steps
        cdef int eb_idx = eb_node_id - 1         # eb rank starts from 1 (0 is Smd0)
        cdef char* send_buf
        send_buf = self.send_bufs[eb_idx].chunk  # select the buffer for this eb
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
                total_size += self.block_sizes[i]
                offset += self.block_sizes[i]

            # Move offset and total size to include fakestep transition set (if set)
            total_size += self._fakebuf_size
            offset += self._fakebuf_size

            # Stores the pointers to missing step buffers for parallel loop below
            PyObject_GetBuffer(step_views[i], &step_buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            ptr_step_bufs[i] = <char *>step_buf.buf
            PyBuffer_Release(&step_buf)

        assert total_size <= self.sendbufsize, f"Repacked data exceeds send buffer's size (total:{total_size} bufsize:{self.sendbufsize})."

        # Access raw C pointers so they can be used in nogil loop below
        cdef uint64_t* block_sizes = <uint64_t*>self.block_sizes.data.as_voidptr
        cdef uint64_t* i_st_blocks = <uint64_t*>self.i_st_blocks.data.as_voidptr
        cdef uint32_t* footer = <uint32_t*>self.repack_footer.data.as_voidptr

        # Copy step and smd buffers if exist
        for i in prange(self.prl_reader.nfiles, nogil=True, num_threads=self.num_threads):
            footer[i] = 0
            if step_sizes[i] > 0:
                memcpy(send_buf + offsets[i], ptr_step_bufs[i], step_sizes[i])
                offsets[i] += step_sizes[i]
                footer[i] += step_sizes[i]

            if c_only_steps == 0:
                if block_sizes[i] > 0:
                    memcpy(send_buf + offsets[i], self.prl_reader.bufs[i].chunk + self.prl_reader.bufs[i].st_offset_arr[i_st_blocks[i]], block_sizes[i])
                    offsets[i] += block_sizes[i]
                    footer[i] += block_sizes[i]

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
