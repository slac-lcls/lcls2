from psana.smdreader import SmdReader
from psana.eventbuilder import EventBuilder
import os, time
from psana import dgram
from psana.event import Event
from .run import RunSmallData

from psana import utils

logger = utils.Logger(myrank=0)


class BatchIterator(object):
    """Iterates over batches of events.

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
            self.eb = EventBuilder(views, configs, dsparms=dsparms, run=run)
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
            batch_dict, step_dict = self.eb.gen_bytearray_batch(
                self.run_smd.proxy_events, run_serial=True
            )
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
        self.smd0_n_events = int(os.environ.get("PS_SMD_N_EVENTS", 1000))
        if self.dsparms.max_events:
            if self.dsparms.max_events < self.smd0_n_events:
                self.smd0_n_events = self.dsparms.max_events

        # Sets the memory size for smalldata buffer for each stream file.
        self.chunksize = int(os.environ.get("PS_SMD_CHUNKSIZE", 0x10000000))

        self.smdr = SmdReader(smd_fds, self.chunksize, dsparms=dsparms)
        self.got_events = -1
        self._run = None

        # Collecting Smd0 performance using prometheus
        self.read_gauge = self.dsparms.prom_man.get_metric("psana_smd0_read")

    def get(self):
        """Reads new data and raise if unable too"""
        get_success = True
        if not self.smdr.is_complete():
            self._get()
            if not self.smdr.is_complete():
                get_success = False
        return get_success

    def _get(self):
        """Reads new data with retry and check for .inprogress"""
        st = time.monotonic()
        self.smdr.get(self.dsparms.smd_inprogress_converted)
        en = time.monotonic()
        read_rate = self.smdr.got / (1e6 * (en - st))
        logger.debug(
            f"READRATE SMD0 (0-) {read_rate:.2f} MB/s ({self.smdr.got/1e6:.2f}MB/ {en-st:.2f}s.)"
        )
        self.read_gauge.set(read_rate)

        if self.smdr.chunk_overflown > 0:
            msg = f"SmdReader found dgram ({self.smdr.chunk_overflown} MB) larger than chunksize ({self.chunksize/1e6} MB)"
            raise ValueError(msg)

    @property
    def processed_events(self):
        return self.smdr.n_processed_events

    def get_next_dgrams(self):
        """Returns list of dgrams as appeared in the current offset of the smd chunks.

        Currently used to retrieve Configure and BeginRun. This allows read with wait
        for these two types of dgram.
        """
        if (
            self.dsparms.max_events > 0
            and self.processed_events >= self.dsparms.max_events
        ):
            logger.debug(f"MESSAGE SMD0 max_events={self.dsparms.max_events} reached")
            return None

        dgrams = None
        if not self.smdr.is_complete():
            self._get()

        if self.smdr.is_complete():
            # Get chunks with only one dgram each. There's no need to set
            # integrating stream id here since Configure and BeginRun
            # must exist in this stream too. Note that by setting batch_size
            # to 1, we automatically ask for all types of event.
            self.smdr.find_view_offsets(batch_size=1, ignore_transition=False)

            # For configs, we need to copy data from smdreader's buffers
            # This prevents it from getting overwritten by other dgrams.
            bytearray_bufs = [bytearray(self.smdr.show(i)) for i in range(self.n_files)]

            if self.configs is None:
                dgrams = [
                    dgram.Dgram(view=ba_buf, offset=0) for ba_buf in bytearray_bufs
                ]
                self.configs = dgrams
                self.smdr.set_configs(self.configs)
            else:
                dgrams = [
                    dgram.Dgram(view=ba_buf, config=config, offset=0)
                    for ba_buf, config in zip(bytearray_bufs, self.configs)
                ]
        return dgrams

    def __iter__(self):
        return self

    def check_split_event(self, current_processed_events):
        # Return True
        #   - if there's no split integrating event or
        #   - when it has been handled correctly by reread once
        #   - EndRun is found
        #
        # How we check? If no. of viewed event is not increasing, this means there's a split
        # integrating event and we need to reread again.
        # Notes:
        #   - Reread in get() is performed for any streams with n_ready_events == n_seen_events
        #     or when force_reread flag is set (split integrating event).
        #   - We only try to reread one time for split events. If this is not successful,
        #     we abort with below message.
        check_pass = True
        if (
            self.processed_events <= current_processed_events
            and not self.smdr.found_endrun()
        ):
            # Also fail if we cannot get more data
            check_pass = self.get()
            if check_pass:
                self.smdr.find_view_offsets(
                    batch_size=self.smd0_n_events,
                    intg_stream_id=self.dsparms.intg_stream_id,
                    max_events=self.dsparms.max_events,
                )
                if self.processed_events <= current_processed_events:
                    print(
                        f"Exit: unable to fit one integrating event in the memory. Try increasing PS_SMD_CHUNKSIZE (current value: {self.chunksize}). Useful debug info: {self.processed_events=} {current_processed_events=}."
                    )
                    check_pass = False
            else:
                print(
                    f"Exit: unable to locate a new chunk. No data in one or more streams and no EndRun found."
                )
        return check_pass

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

        # Read new data if needed
        if not self.get():
            raise StopIteration

        # Locate viewing windows for each chunk
        current_processed_events = self.processed_events
        self.smdr.find_view_offsets(
            batch_size=self.smd0_n_events,
            intg_stream_id=intg_stream_id,
            max_events=self.dsparms.max_events,
        )

        # Check if reread is needed for split integrating event
        check_pass = self.check_split_event(current_processed_events)
        if not check_pass:
            raise StopIteration

        mmrv_bufs = [self.smdr.show(i) for i in range(self.n_files)]
        batch_iter = BatchIterator(mmrv_bufs, self.configs, self._run, self.dsparms)
        self.got_events = self.smdr.view_size
        return batch_iter

    def chunks(self):
        """Generates a tuple of smd and step dgrams"""
        is_done = False
        d_view, d_read = 0, 0
        cn_chunks = 0
        while not is_done:
            logger.debug(f"TIMELINE 1. STARTCHUNK {time.monotonic()}", level=2)
            st_view, en_view, st_read, en_read = 0, 0, 0, 0

            if self.smdr.is_complete():

                st_view = time.monotonic()

                # Gets the next batch of already read-in data.
                current_processed_events = self.processed_events
                self.smdr.find_view_offsets(
                    batch_size=self.smd0_n_events,
                    intg_stream_id=self.dsparms.intg_stream_id,
                    max_events=self.dsparms.max_events,
                )

                # Check if reread is needed for split integrating event
                check_pass = self.check_split_event(current_processed_events)
                if not check_pass:
                    break

                self.got_events = self.smdr.view_size

                if (
                    self.dsparms.max_events
                    and self.processed_events >= self.dsparms.max_events
                ):
                    logger.debug(
                        f"MESSAGE SMD0 max_events={self.dsparms.max_events} reached"
                    )
                    is_done = True

                en_view = time.monotonic()
                d_view += en_view - st_view
                logger.debug(f"TIMELINE 2. DONECREATEVIEW {time.monotonic()}", level=2)

                if self.got_events:
                    cn_chunks += 1
                    yield cn_chunks

            else:  # if self.smdr.is_complete()
                st_read = time.monotonic()
                self._get()
                en_read = time.monotonic()
                logger.debug(f"TIMELINE 3. DONEREAD {time.monotonic()}", level=2)
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
