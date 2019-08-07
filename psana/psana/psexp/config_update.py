from psana.psexp.event_manager import EventManager

class ConfigUpdate(object):
    
    def __init__(self, run, eb_man=0, update_dgram=0, smd_batch=0):
        if eb_man:
            assert update_dgram

        self.run = run
        self.eb_man = eb_man
        self.update_dgram = update_dgram
        self.smd_batch = smd_batch

    def events(self):
        # Event generator that yields list of events from 
        # an smd_chunk: data from smd0 that needs to be event built (serial mode) or
        # an smd_batch: prebuilt smd data (parallel mode)
        ev_man = EventManager(self.run.configs, self.run.dm, \
                filter_fn=self.run.filter_callback)
        
        if self.eb_man:

            for batch_dict in self.eb_man.batches(limit_ts=self.update_dgram.seq.timestamp()):
                batch, _ = batch_dict[0]
                for evt in ev_man.events(batch):
                    if evt._dgrams[0].seq.service() != 12: continue
                    yield evt

        else:
            for evt in ev_man.events(self.smd_batch):
                if evt._dgrams[0].seq.service() != 12: continue
                yield evt 

