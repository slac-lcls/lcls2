from psana.psexp import EventManager, TransitionId
import types

class Events:
    """
    Needs prom_man, configs, dm, filter_callback
    """
    def __init__(self, configs, dm, dsparms, filter_callback=None, get_smd=None, smdr_man=None):
        self.dm             = dm                   
        self.configs        = configs
        self.dsparms        = dsparms
        self.prom_man       = dsparms.prom_man
        self.max_retries    = dsparms.max_retries
        self.filter_callback= filter_callback
        self.get_smd        = get_smd          # RunParallel
        self.smdr_man       = smdr_man         # RunSerial
        self._evt_man       = iter([])
        self._batch_iter    = iter([])
        self.c_read         = self.prom_man.get_metric('psana_bd_read')
        self.st_yield       = 0
        self.en_yield       = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.smdr_man:
            # RunSerial
            try:
                evt = next(self._evt_man)
                if not any(evt._dgrams): return self.__next__() # FIXME: MONA find better way to handle empty event.
                self.smdr_man.last_seen_event = evt
                return evt
            except StopIteration:
                try:
                    batch_dict, _ = next(self._batch_iter)
                except StopIteration:
                    self._batch_iter = next(self.smdr_man)
                    batch_dict, _ = next(self._batch_iter)

                self._evt_man = EventManager(batch_dict[0][0], 
                        self.configs, 
                        self.dm, 
                        self.dsparms.esm,
                        filter_fn           = self.filter_callback,
                        prometheus_counter  = self.c_read,
                        max_retries         = self.max_retries,
                        use_smds            = self.dsparms.use_smds,
                        )
                #print(f'debug events get next event')
                evt = next(self._evt_man)
                #epics_store = self.dsparms.esm.stores['epics']
                #chunk_id = epics_store.values([evt], '_NEW_CHUNK_ID_S00')
                #print(f'debug events got next event evt={[d.service() for d in evt._dgrams]} chunk_id={chunk_id}')
                if not any(evt._dgrams): return self.__next__()
                self.smdr_man.last_seen_event = evt
                return evt 

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
                        self.configs, 
                        self.dm, 
                        self.dsparms.esm,
                        filter_fn           = self.filter_callback,
                        prometheus_counter  = self.c_read,
                        max_retries         = self.max_retries,
                        use_smds            = self.dsparms.use_smds,
                        )
                evt = next(self._evt_man)
                if not any(evt._dgrams): return self.__next__()
                
                return evt
        else: 
            # RunSingleFile or RunShmem - get event from DgramManager
            evt = next(self.dm)

            # TODO: MONA Update EnvStore here instead of inside DgramManager.
            # To mirror withe RunSerial/RunParallel, consider moving update
            # into DgramManager.
            if evt.service() != TransitionId.L1Accept:
                self.dsparms.esm.update_by_event(evt)

            if not any(evt._dgrams): return self.__next__()
            return evt



