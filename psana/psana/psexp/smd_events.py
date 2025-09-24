from .event_manager import EventManager

class SmdEvents:
    def __init__(self, ds, run, get_smd=None, smdr_man=None):
        self.ds = ds
        self.run = run
        self.get_smd = get_smd
        self.smdr_man = smdr_man
        self._evt_man = iter([])
        self._batch_iter = iter([])

    def __iter__(self):
        return self

    def _is_valid_batch(self, batch_dict):
        return batch_dict and 0 in batch_dict and batch_dict[0]

    def __next__(self):
        if self.smdr_man:
            # RunSerial: iterate using smdr_man
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
                        if not self._is_valid_batch(batch_dict):
                            continue
                        self._evt_man = EventManager(
                            batch_dict[0][0],
                            self.ds,
                            self.run,
                            smd=True,
                        )
                    except StopIteration:
                        self._batch_iter = next(self.smdr_man)

        elif self.get_smd:
            # RunParallel: use get_smd to fetch batches
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
                        smd=True,
                    )
        else:
            raise RuntimeError("SmdEvents must be constructed with get_smd or smdr_man")
