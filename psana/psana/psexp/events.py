import time

from .event_manager import EventManager


class Events:
    """
    An iterator class for retrieving events from a run.

    Handles different run modes:
    - RunSerial: receives batches from a serial manager (smdr_man)
    - RunParallel: fetches SMD batches using get_smd()
    - RunSingleFile / RunShmem: reads directly from a DgramManager

    This class abstracts the complexity of batching, filtering empty events,
    and respecting termination signals, providing a uniform interface via `__next__()`.
    """
    def __init__(
        self,
        configs,
        dm,
        max_retries,
        use_smds,
        shared_state,
        get_smd=None,
        smdr_man=None,
        on_batch_end=None,
    ):
        self.configs = configs  # Configuration dgrams for event building
        self.dm = dm              # DgramManager for direct reading
        self.max_retries = max_retries  # Max retries for event fetching
        self.use_smds = use_smds  # Flag to indicate SMD usage
        self.shared_state = shared_state  # SimpleNamespace with shared state like terminate_flag
        self.get_smd = get_smd       # Callable to retrieve SMD batches (RunParallel)
        self.smdr_man = smdr_man     # Serial batch manager (RunSerial)
        self._on_batch_end = on_batch_end
        self._evt_man = iter([])     # Current EventManager instance
        self._batch_iter = iter([])  # Iterator over batches for RunSerial
        self._batch_event_count = 0
        self._batch_start_time = None

    def __iter__(self):
        return self

    def _is_valid_batch(self, batch_dict):
        return batch_dict and 0 in batch_dict and batch_dict[0]

    def _emit_batch_end(self):
        if not self._on_batch_end:
            return
        if not hasattr(self._evt_man, "get_bd_read_stats"):
            return
        if self._batch_start_time is None:
            return
        elapsed = time.monotonic() - self._batch_start_time
        self._on_batch_end(
            (self._evt_man.get_bd_read_stats(), self._batch_event_count, elapsed)
        )
        self._batch_event_count = 0
        self._batch_start_time = None

    def __next__(self):
        """
        Retrieve the next valid event, skipping empty ones.

        Raises:
            StopIteration: When the data source is exhausted or termination is requested.
        """
        if self.smdr_man:
            # RunSerial: iterate over batches, skipping empty ones
            cn = 0
            while True:
                if self.shared_state.terminate_flag.value:
                    raise StopIteration
                try:
                    dgrams = next(self._evt_man)
                    cn += 1
                    if not any(dgrams):
                        continue
                    self._batch_event_count += 1
                    return dgrams
                except StopIteration:
                    try:
                        self._emit_batch_end()
                        batch_dict, _ = next(self._batch_iter)
                        # Skip empty or malformed batches
                        if not self._is_valid_batch(batch_dict):
                            continue
                        self._evt_man = EventManager(
                            batch_dict[0][0],
                            self.configs,
                            self.dm,
                            self.max_retries,
                            self.use_smds,
                        )
                        self._batch_event_count = 0
                        self._batch_start_time = time.monotonic()
                    except StopIteration:
                        # Refill the batch iterator from the serial batch manager
                        self._batch_iter = next(self.smdr_man)

        elif self.get_smd:
            # RunParallel: fetch batch from get_smd() when needed
            while True:
                try:
                    dgrams = next(self._evt_man)
                    if not any(dgrams):
                        continue
                    self._batch_event_count += 1
                    return dgrams
                except StopIteration:
                    self._emit_batch_end()
                    smd_batch = self.get_smd()
                    if smd_batch == bytearray():
                        raise StopIteration

                    self._evt_man = EventManager(
                        smd_batch,
                        self.configs,
                        self.dm,
                        self.max_retries,
                        self.use_smds,
                    )
                    self._batch_event_count = 0
                    self._batch_start_time = time.monotonic()
        else:
            # RunSingleFile or RunShmem: read directly from the DgramManager
            while True:
                # Checks if users ask to exit
                if self.shared_state.terminate_flag.value:
                    raise StopIteration

                dgrams = next(self.dm)

                if not any(dgrams):
                    continue
                return dgrams
