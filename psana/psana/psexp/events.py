from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.event_manager import EventManager, TransitionId

class Events:
    def __init__(self, run, get_smd=0):
        self.run = run
        self.get_smd = get_smd # RunParallel
        if not self.get_smd:
            self._smdr_man = SmdReaderManager(run) # RunSerial
        self._evt_man = iter([])
        self._batch_iter = iter([])
        self.flag_empty_smd_batch = False

    def __iter__(self):
        return self

    def _get_evt_and_update_store(self):
        evt = next(self._evt_man)
        if evt._dgrams[0].service() != TransitionId.L1Accept:
            self.run.esm.update_by_event(evt)
        return evt

    def __next__(self):
        if self.flag_empty_smd_batch:
            raise StopIteration

        if not self.get_smd:
            # RunSerial - get smd chuk from SmdReader iterator
            try:
                return self._get_evt_and_update_store()
            except StopIteration: 
                try:
                    batch_dict = next(self._batch_iter)
                except StopIteration: 
                    self._batch_iter = next(self._smdr_man) 
                    batch_dict = next(self._batch_iter)

                self._evt_man = EventManager(batch_dict[0][0], self.run.configs, \
                        self.run.dm, filter_fn=self.run.filter_callback)
                return self._get_evt_and_update_store()
        else: 
            # RunParallel - get smd chunk from a callback (MPI receive)
            try:
                return self._get_evt_and_update_store()
            except StopIteration: 
                smd_batch = self.get_smd()
                if smd_batch == bytearray():
                    self.flag_empty_smd_batch = True
                    raise StopIteration

                else:
                    self._evt_man = EventManager(smd_batch, self.run.configs, \
                            self.run.dm, filter_fn=self.run.filter_callback)
                    return self._get_evt_and_update_store()



