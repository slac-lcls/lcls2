from . import TransitionId
from .event_manager import EventManager

import types

class Events:
    def __init__(self, ds, run, get_smd=None, smdr_man=None):
        self.ds             = ds
        self.run            = run
        self.get_smd        = get_smd          # RunParallel
        self.smdr_man       = smdr_man         # RunSerial
        self._evt_man       = iter([])
        self._batch_iter    = iter([])
        self.st_yield       = 0
        self.en_yield       = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.smdr_man:
            # RunSerial

            # Checks if users ask to exit
            if self.ds.dsparms.terminate_flag: raise StopIteration

            try:
                evt = next(self._evt_man)
                if not any(evt._dgrams): return self.__next__() # FIXME: MONA find better way to handle empty event.
                self.smdr_man.last_seen_event = evt
                return evt
            except StopIteration:
                try:
                    batch_dict, _ = next(self._batch_iter)
                    self._evt_man = EventManager(batch_dict[0][0], 
                            self.run.configs, 
                            self.ds.dm, 
                            self.run.esm,
                            filter_fn           = self.ds.dsparms.filter,
                            prom_man            = self.ds.dsparms.prom_man,
                            max_retries         = self.ds.dsparms.max_retries,
                            use_smds            = self.ds.dsparms.use_smds,
                            )
                    return self.__next__()
                except StopIteration:
                    self._batch_iter = next(self.smdr_man)
                    return self.__next__()

        elif self.get_smd:
            # RunParallel
            try:
                evt = next(self._evt_man)
                if not any(evt._dgrams): return self.__next__()
                return evt
            except StopIteration: 
                smd_batch = self.get_smd()
                if smd_batch == bytearray():
                    raise StopIteration

                self._evt_man = EventManager(smd_batch, 
                        self.run.configs, 
                        self.ds.dm, 
                        self.run.esm,
                        filter_fn           = self.ds.dsparms.filter,
                        prom_man            = self.ds.dsparms.prom_man,
                        max_retries         = self.ds.dsparms.max_retries,
                        use_smds            = self.ds.dsparms.use_smds,
                        )
                evt = next(self._evt_man)
                if not any(evt._dgrams): return self.__next__()
                
                return evt
        else: 
            # RunSingleFile or RunShmem - get event from DgramManager
            
            # Checks if users ask to exit
            if self.ds.dsparms.terminate_flag: raise StopIteration
            
            evt = next(self.ds.dm)

            # TODO: MONA Update EnvStore here instead of inside DgramManager.
            # To mirror withe RunSerial/RunParallel, consider moving update
            # into DgramManager.
            if evt.service() != TransitionId.L1Accept:
                self.run.esm.update_by_event(evt)

            if not any(evt._dgrams): return self.__next__()
            return evt



