import os
import time

from psana2 import dgram, utils
from psana2.eventbuilder import EventBuilder
from psana2.smdreader import SmdReader

from .run import RunSmallData


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
        self.logger = utils.get_logger(dsparms=self.dsparms, name=utils.get_class_name(self))

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

    def force_read(self):
        """
        Always performs a new read from the file descriptors.

        Overwrites any existing data in the buffer.
        Used when new data must be forced into the buffers,
        regardless of how much data is already present.
        """
        st = time.monotonic()
        self.smdr.force_read(self.dsparms.smd_inprogress_converted)
        en = time.monotonic()
        read_rate = self.smdr.got / (1e6 * (en - st))
        self.logger.debug(
            f"READRATE SMD0 (0-) {read_rate:.2f} MB/s ({self.smdr.got/1e6:.2f}MB/ {en-st:.2f}s.)"
        )
        self.read_gauge.set(read_rate)

        if self.smdr.chunk_overflown > 0:
            msg = f"SmdReader found dgram ({self.smdr.chunk_overflown} MB) larger than chunksize ({self.chunksize/1e6} MB)"
            raise ValueError(msg)
        return True

    @property
    def processed_events(self):
        return self.smdr.n_processed_events

    def get_next_dgrams(self):
        """
        Fetch Configure and BeginRun dgrams at the start of run.

        In live mode, will retry up to max_retries if the streams are incomplete.
        In non-live mode, attempts only once.

        Returns:
            list of dgrams or None if not available
        """
        if (
            self.dsparms.max_events > 0
            and self.processed_events >= self.dsparms.max_events
        ):
            self.logger.debug(f"Exit. - max_events={self.dsparms.max_events} reached")
            return None

        max_retries = getattr(self.dsparms, "max_retries", 0)
        if max_retries:
            retries = 0
        else:
            retries = -1
        success = False

        while retries <= max_retries:
            # Determine if we have a single event
            success = self.smdr.build_batch_view(batch_size=1, ignore_transition=False)
            if success:
                bytearray_bufs = [bytearray(self.smdr.show(i)) for i in range(self.n_files)]

                if self.configs is None:
                    dgrams = [dgram.Dgram(view=ba_buf, offset=0) for ba_buf in bytearray_bufs]
                    self.configs = dgrams
                    self.smdr.set_configs(self.configs)
                else:
                    dgrams = [
                        dgram.Dgram(view=ba_buf, config=config, offset=0)
                        for ba_buf, config in zip(bytearray_bufs, self.configs)
                    ]

                return dgrams

            # Check if no more data will ever arrive
            if self.smdr.found_endrun():
                self.logger.debug("EndRun found — giving up on fetching Configure/BeginRun.")
                return None

            # No data to yield, try to get more data
            self.force_read()
            # Didn't get complete streams yet
            if retries == max_retries:
                self.logger.info(
                    f"WARNING: Unable to fetch complete Configure/BeginRun dgrams after {retries} retries."
                )
                return None

            if retries > -1:
                self.logger.debug(f"Waiting for Configure/BeginRun... retry {retries+1}/{max_retries}")
                time.sleep(1)
            retries += 1

    def __iter__(self):
        return self

    def build_batch(self):
        """
        Build a batch (normal or integrating) using build_batch_view().
        Retries in live mode if no complete batch is found.

        Returns:
            bool or None: True if batch was built,
                        False if EndRun was found but no batch ready,
                        None if retries exhausted.
        """
        max_retries = getattr(self.dsparms, "max_retries", 0)
        retries = 0 if max_retries else -1
        success = False
        intg_stream_id = self.dsparms.intg_stream_id
        is_integrating = intg_stream_id >= 0

        while retries <= max_retries:
            success = self.smdr.build_batch_view(
                batch_size=self.smd0_n_events,
                intg_stream_id=intg_stream_id,
                intg_delta_t=self.dsparms.intg_delta_t,
                max_events=self.dsparms.max_events,
            )

            if success or self.smdr.found_endrun():
                break

            self.smdr.force_read(self.dsparms.smd_inprogress_converted)

            if retries == max_retries:
                return None

            if retries > -1:
                self.logger.debug(f"Waiting for data... retry {retries+1}/{max_retries}")
                time.sleep(1)
            retries += 1

        if not success:
            log_func = self.logger.error if is_integrating else self.logger.info
            log_func(f"Unable to build {'integrating' if is_integrating else 'normal'} batch after {retries} retries.")

        return success

    def chunks(self):
        """
        Generator that yields chunk numbers as we process smalldata batches.

        In both normal and integrating modes:
        - If there isn't enough data, it tries to reread more.
        - In live mode, it will retry up to max_retries before giving up.
        - `build_normal_batch()` and `build_integrating_batch()` both internally
        call build_batch_view() and handle retries.

        Yields:
            int: Chunk number starting from 1
        """
        is_done = False
        cn_chunks = 0

        while not is_done:
            # --- Build batch ---
            success = self.build_batch()

            if not success:
                is_done = True
                break

            self.got_events = self.smdr.view_size

            if (
                self.dsparms.max_events
                and self.processed_events >= self.dsparms.max_events
            ):
                self.logger.debug(f"Exit - max_events={self.dsparms.max_events} reached")
                is_done = True

            if self.got_events:
                cn_chunks += 1
                yield cn_chunks

    def __next__(self):
        """
        Returns a batch of events as an iterator object.
        Used by non-parallel (serial) mode.
        """
        if self.dsparms.max_events and self.processed_events >= self.dsparms.max_events:
            raise StopIteration

        success = self.build_batch()

        if not success:
            raise StopIteration

        mmrv_bufs = [self.smdr.show(i) for i in range(self.n_files)]
        batch_iter = BatchIterator(mmrv_bufs, self.configs, self._run, self.dsparms)
        self.got_events = self.smdr.view_size
        return batch_iter

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
