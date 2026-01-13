import os
import time

from psana import dgram, utils
from psana.eventbuilder import EventBuilder
from psana.smdreader import SmdReader
from psana.psexp.prometheus_manager import get_prom_manager
from psana.psexp.tools import get_smd_n_events

from .run import RunSmallData


class BatchIterator(object):
    """Iterates over batches of events.

    SmdReaderManager returns this object when a chunk is read.
    """

    def __init__(self, views, configs, dsparms):
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
            use_proxy_events = bool(dsparms.smd_callback or getattr(dsparms, "intg_det", ""))
            self.eb = EventBuilder(views,
                                   configs,
                                   filter_timestamps=dsparms.timestamps,
                                   intg_stream_id=dsparms.intg_stream_id,
                                   batch_size=dsparms.batch_size,
                                   use_proxy_events=use_proxy_events)
            self.run_smd = RunSmallData(self.eb, configs, dsparms)

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
        self.logger = utils.get_logger(name=utils.get_class_name(self))

        assert self.n_files > 0

        # Sets no. of events Smd0 sends to each EventBuilder core. This gets
        # overridden by max_events set by DataSource if max_events is smaller.
        self.smd0_n_events = get_smd_n_events()
        if self.dsparms.max_events:
            if self.dsparms.max_events < self.smd0_n_events:
                self.smd0_n_events = self.dsparms.max_events

        # Sets the memory size for smalldata buffer for each stream file.
        self.chunksize = int(os.environ.get("PS_SMD_CHUNKSIZE", 0x10000000))

        self.smdr = SmdReader(smd_fds, self.chunksize, dsparms=dsparms)
        self.got_events = -1

        # Collecting Smd0 performance using prometheus
        self.read_gauge = get_prom_manager().get_metric("psana_smd0_read")
        self._read_bytes = []
        self._read_times = []

    def check_transfer_complete(self):
        """
        Returns True only after we've seen .xtc2.inprogress files finish
        transferring (renamed to their final .xtc2 filenames).

        This is used in live-mode (normal filesystem, not ffb) to detect when
        the DAQ or rsync file transfers have completed — so psana can stop
        retrying for more data. If there were no .inprogress files to begin
        with (e.g. FFB writes directly to .xtc2), this returns False so we keep
        retrying until EndRun is seen.

        See https://github.com/monarin/psana-nersc/blob/master/psana2/write_then_move.sh
        to mimic a live run for testing this feature.

        Returns
        -------
        bool
            True once at least one .xtc2.inprogress file has been observed and
            every such file now has a finalized .xtc2 present.
        """
        self.logger.debug("Checking live transfer status...")

        # Only relevant for live-mode runs. Offline replays must keep reading
        # until EndRun or max_events is reached.
        if not getattr(self.dsparms, "live", False):
            return False

        smd_files = getattr(self.dsparms, "smd_files", [])
        if not smd_files:
            return False  # Nothing to monitor yet

        monitor_transfers = False
        for smd_file in smd_files:
            if not smd_file.endswith(".xtc2.inprogress"):
                continue

            monitor_transfers = True
            xtc2_file = os.path.splitext(smd_file)[0]
            if not os.path.isfile(xtc2_file):
                return False  # Still waiting for at least one file to finalize

        if not monitor_transfers:
            return False  # No .inprogress files -> keep retrying (FFB/live write)

        return True

    def force_read(self):
        """
        Always performs a new read from the file descriptors.

        Overwrites any existing data in the buffer.
        Used when new data must be forced into the buffers,
        regardless of how much data is already present.
        """
        st = time.monotonic()
        self.smdr.force_read()
        en = time.monotonic()
        elapsed = en - st
        bytes_read = self.smdr.got
        if bytes_read > 0 and elapsed > 0:
            read_rate = bytes_read / (1e6 * elapsed)
            self._read_bytes.append(bytes_read)
            self._read_times.append(elapsed)
            self.read_gauge.set(read_rate)

        if self.smdr.chunk_overflown > 0:
            msg = (
                "SmdReader found dgram "
                f"({self.smdr.chunk_overflown} MB) larger than chunksize "
                f"({self.chunksize/1e6} MB)"
            )
            raise ValueError(msg)
        return True

    def pop_read_stats(self):
        bytes_list = self._read_bytes
        times_list = self._read_times
        self._read_bytes = []
        self._read_times = []
        return bytes_list, times_list

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
                bytearray_bufs = [
                    bytearray(self.smdr.show(i)) for i in range(self.n_files)
                ]

                if self.configs is None:
                    dgrams = [
                        dgram.Dgram(view=ba_buf, offset=0)
                        for ba_buf in bytearray_bufs
                    ]
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
                self.logger.debug(
                    "EndRun found — giving up on fetching Configure/BeginRun."
                )
                return None

            # No data to yield, try to get more data
            self.force_read()

            # Stop waiting if file transfers are done (live mode only)
            if getattr(self.dsparms, "live", False) and self.check_transfer_complete():
                self.logger.debug(
                    "Live transfer complete — stopping Configure/BeginRun retries."
                )
                return None

            # Didn't get complete streams yet
            if retries == max_retries:
                self.logger.info(
                    "WARNING: Unable to fetch complete Configure/BeginRun dgrams "
                    f"after {retries} retries."
                )
                return None

            if retries > -1:
                self.logger.debug(
                    "Waiting for Configure/BeginRun... "
                    f"retry {retries+1}/{max_retries}"
                )
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

            self.force_read()

            # Stop waiting if all transfers complete (live mode only)
            if getattr(self.dsparms, "live", False) and self.check_transfer_complete():
                self.logger.debug(
                    "Live transfer complete — stopping batch read retries."
                )
                break

            if retries == max_retries:
                return None

            if retries > -1:
                self.logger.debug(
                    f"Waiting for data... retry {retries+1}/{max_retries}"
                )
                time.sleep(1)
            retries += 1

        if not success and not self.smdr.found_endrun():
            log_func = self.logger.error if is_integrating else self.logger.info
            log_func(
                "Unable to build "
                f"{'integrating' if is_integrating else 'normal'} batch "
                f"after {retries} retries."
            )

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
                self.logger.debug(
                    f"Exit - max_events={self.dsparms.max_events} reached"
                )
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

        mmrv_bufs = [
            self.smdr.show(i) for i in range(self.n_files)
        ]
        batch_iter = BatchIterator(mmrv_bufs, self.configs, self.dsparms)
        self.got_events = self.smdr.view_size
        return batch_iter

    @property
    def min_ts(self):
        return self.smdr.min_ts

    @property
    def max_ts(self):
        return self.smdr.max_ts
