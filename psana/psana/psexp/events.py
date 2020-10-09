from psana.psexp import EventManager, TransitionId
import logging

class Events:
    def __init__(self, run, get_smd=0, dm=None):
        self.run     = run
        self.get_smd = get_smd              # RunParallel
        self.dm      = dm                   # RunSingleFile, RunShmem
        if self.get_smd==0 and self.dm is None:
            self._smdr_man = run.smdr_man   # RunSerial
        self._evt_man               = iter([])
        self._batch_iter            = iter([])
        self.flag_empty_smd_batch   = False
        self.c_read = self.run.prom_man.get_metric('psana_bd_read')

    def __iter__(self):
        return self

    def _get_evt_and_update_store(self):
        evt = next(self._evt_man)
        if evt.service() != TransitionId.L1Accept:
            self.run.esm.update_by_event(evt)
        return evt

    def __next__(self):
        if self.flag_empty_smd_batch:
            raise StopIteration

        if self.get_smd:
            # RunParallel - get smd chunk from a callback (MPI receive)
            try:
                return self._get_evt_and_update_store()
            except StopIteration: 
                smd_batch = self.get_smd()

                if smd_batch == bytearray():
                    self.flag_empty_smd_batch = True
                    raise StopIteration
                else:
                    self.c_read.labels('batches','None').inc()
                    self._evt_man = EventManager(smd_batch, 
                            self.run.configs, 
                            self.run.dm, 
                            filter_fn           = self.run.filter_callback,
                            prometheus_counter  = self.c_read)
                    return self._get_evt_and_update_store()
        else: 
            if self.dm:
                # RunSingleFile or RunShmem - get event from DgramManager
                evt = next(self.dm)
                if evt.service() != TransitionId.L1Accept:
                    self.run.esm.update_by_event(evt)
                return evt
            else:
                # RunSerial - get smd chunk from SmdReader iterator
                try:
                    return self._get_evt_and_update_store()
                except StopIteration: 
                    try:
                        batch_dict, _ = next(self._batch_iter)
                    except StopIteration: 
                        self._batch_iter = next(self._smdr_man)
                        batch_dict, _ = next(self._batch_iter)

                    self._evt_man = EventManager(batch_dict[0][0], 
                            self.run.configs, 
                            self.run.dm, 
                            filter_fn=self.run.filter_callback,
                            prometheus_counter  = self.c_read)
                    return self._get_evt_and_update_store()



