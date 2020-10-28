from psana.psexp import EventManager, TransitionId
import logging
import types

class Events:
    """
    Needs prom_man, configs, dm, filter_callback
    """
    def __init__(self, configs, dm, prom_man, filter_callback=None, get_smd=None, smdr_man=None):
        self.dm             = dm                   
        self.configs        = configs
        self.prom_man       = prom_man
        self.filter_callback= filter_callback
        self.get_smd        = get_smd          # RunParallel
        self.smdr_man       = smdr_man         # RunSerial
        self._evt_man               = iter([])
        self._batch_iter            = iter([])
        self.flag_empty_smd_batch   = False
        self.c_read = self.prom_man.get_metric('psana_bd_read')

    def __iter__(self):
        return self

    def __next__(self):
        if self.flag_empty_smd_batch:
            raise StopIteration

        if self.smdr_man:
            # RunSerial
            try:
                return next(self._evt_man)
            except StopIteration:
                try:
                    batch_dict, _ = next(self._batch_iter)
                except StopIteration:
                    self._batch_iter = next(self.smdr_man)
                    batch_dict, _ = next(self._batch_iter)

                self._evt_man = EventManager(batch_dict[0][0], 
                        self.configs, 
                        self.dm, 
                        filter_fn           = self.filter_callback,
                        prometheus_counter  = self.c_read)
                return next(self._evt_man)
        elif self.get_smd:
            # RunParallel
            try:
                return next(self._evt_man) 
            except StopIteration: 
                smd_batch = self.get_smd()
                if smd_batch == bytearray():
                    self.flag_empty_smd_batch = True
                    raise StopIteration

                self.c_read.labels('batches','None').inc()
                self._evt_man = EventManager(smd_batch, 
                        self.configs, 
                        self.dm, 
                        filter_fn           = self.filter_callback,
                        prometheus_counter  = self.c_read)
                evt = next(self._evt_man)
                return evt
        else: 
            # RunSingleFile or RunShmem - get event from DgramManager
            return next(self.dm) 



