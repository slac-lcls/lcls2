from . import TransitionId
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
    def __init__(self, ds, run, get_smd=None, smdr_man=None):
        self.ds = ds                 # Reference to the DataSource
        self.run = run               # The current Run object
        self.get_smd = get_smd       # Callable to retrieve SMD batches (RunParallel)
        self.smdr_man = smdr_man     # Serial batch manager (RunSerial)
        self._evt_man = iter([])     # Current EventManager instance
        self._batch_iter = iter([])  # Iterator over batches for RunSerial

    def __iter__(self):
        return self

    def _is_valid_batch(self, batch_dict):
        return batch_dict and 0 in batch_dict and batch_dict[0]

    def __next__(self):
        """
        Retrieve the next valid event, skipping empty ones.

        Raises:
            StopIteration: When the data source is exhausted or termination is requested.
        """
        if self.smdr_man:
            # RunSerial: iterate over batches, skipping empty ones
            while True:
                if self.ds.dsparms.terminate_flag:
                    raise StopIteration
                try:
                    evt = next(self._evt_man)
                    if not any(evt._dgrams):
                        continue
                    self.smdr_man.last_seen_event = evt
                    return evt
                except StopIteration:
                    try:
                        batch_dict, _ = next(self._batch_iter)
                        # Skip empty or malformed batches
                        if not self._is_valid_batch(batch_dict):
                            continue
                        self._evt_man = EventManager(
                            batch_dict[0][0],
                            self.ds,
                            self.run,
                        )
                    except StopIteration:
                        # Refill the batch iterator from the serial batch manager
                        self._batch_iter = next(self.smdr_man)

        elif self.get_smd:
            # RunParallel: fetch batch from get_smd() when needed
            while True:
                try:
                    evt = next(self._evt_man)
                    if not any(evt._dgrams):
                        continue
                    return evt
                except StopIteration:
                    smd_batch = self.get_smd()
                    if smd_batch == bytearray():
                        raise StopIteration

                    self._evt_man = EventManager(
                        smd_batch,
                        self.ds,
                        self.run,
                    )
        else:
            # RunSingleFile or RunShmem: read directly from the DgramManager
            while True:
                # Checks if users ask to exit
                if self.ds.dsparms.terminate_flag:
                    raise StopIteration

                evt = next(self.ds.dm)

                # Update environment store with non-event transitions
                if not TransitionId.isEvent(evt.service()):
                    self.run.esm.update_by_event(evt)

                if not any(evt._dgrams):
                    continue
                return evt
