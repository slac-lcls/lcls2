import time
from dataclasses import dataclass

from psana.event import Event
from .event_manager import EventManager


@dataclass
class BatchEnvelope:
    """One coherent EB-to-BD communication unit."""

    smd: object
    gpu: object = None


class Events:
    """
    An iterator class for retrieving events from a run.

    Handles different run modes:
    - RunSerial: receives batches from a serial manager (smdr_man)
    - RunParallel: consumes coherent BatchEnvelope objects
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
        batch_source=None,
        run=None,
        gpu_manager=None,
        smdr_man=None,
        on_batch_end=None,
    ):
        self.configs = configs  # Configuration dgrams for event building
        self.dm = dm              # DgramManager for direct reading
        self.max_retries = max_retries  # Max retries for event fetching
        self.use_smds = use_smds  # Flag to indicate SMD usage
        self.shared_state = shared_state  # SimpleNamespace with shared state like terminate_flag
        self._batch_source = (
            iter(batch_source) if batch_source is not None else None
        )
        self.run = run               # RunCtx for Event construction
        self.gpu_manager = gpu_manager
        self.smdr_man = smdr_man     # Serial batch manager (RunSerial)
        self._on_batch_end = on_batch_end
        self._evt_man = iter([])     # Current EventManager instance
        self._batch_iter = iter([])  # Iterator over batches for RunSerial
        self._batch_event_count = 0
        self._batch_start_time = None
        self._gpu_finished = False

    def __iter__(self):
        return self

    def _is_valid_batch(self, batch_dict):
        return batch_dict and 0 in batch_dict and batch_dict[0]

    def _emit_batch_end(self):
        if not self._on_batch_end:
            return
        if self._batch_start_time is None:
            return
        elapsed = time.monotonic() - self._batch_start_time
        read_stats = (0, 0.0)
        if self.gpu_manager is not None and hasattr(
            self.gpu_manager, "get_bd_read_stats"
        ):
            read_stats = self.gpu_manager.get_bd_read_stats()
        elif hasattr(self._evt_man, "get_bd_read_stats"):
            read_stats = self._evt_man.get_bd_read_stats()
        self._on_batch_end(
            (read_stats, self._batch_event_count, elapsed)
        )
        self._batch_event_count = 0
        self._batch_start_time = None

    def __next__(self):
        """
        Retrieve the next valid event, skipping empty ones.

        Raises:
            StopIteration: When the data source is exhausted or termination is requested.
        """
        if self._batch_source is not None:
            while True:
                terminate_flag = getattr(
                    self.shared_state, "terminate_flag", None
                )
                if terminate_flag is not None and terminate_flag.value:
                    raise StopIteration
                try:
                    item = next(self._evt_man)
                    if isinstance(item, Event):
                        self._batch_event_count += 1
                        return item
                    if not any(item):
                        continue
                    self._batch_event_count += 1
                    return Event(dgrams=item, run=self.run)
                except StopIteration:
                    self._emit_batch_end()
                    try:
                        envelope = next(self._batch_source)
                    except StopIteration:
                        if self.gpu_manager is not None and not self._gpu_finished:
                            self._gpu_finished = True
                            self._evt_man = iter(self.gpu_manager.finish())
                            continue
                        raise

                    if self.gpu_manager is None:
                        self._evt_man = EventManager(
                            envelope.smd,
                            self.configs,
                            self.dm,
                            self.max_retries,
                            self.use_smds,
                        )
                    else:
                        self._evt_man = iter(
                            self.gpu_manager.process_batch(
                                envelope.smd, envelope.gpu
                            )
                        )
                    self._batch_event_count = 0
                    self._batch_start_time = time.monotonic()

        elif self.smdr_man:
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
