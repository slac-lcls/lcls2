from .event_manager import EventManager

class SmdEvents:
    def __init__(self, ds, run, get_smd):
        self.ds = ds
        self.run = run
        self.get_smd = get_smd
        self._evt_man = iter([])

    def __iter__(self):
        return self

    def __next__(self):
        try:
            evt = next(self._evt_man)
            if not any(evt._dgrams):
                return self.__next__()
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
            return self.__next__()
