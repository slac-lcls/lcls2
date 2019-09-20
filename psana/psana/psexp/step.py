from psana.psexp.event_manager import EventManager

class Step(object):
    
    def __init__(self, run, eb_man=0, limit_ts=-1, smd_batch=None):
        self.run = run
        self.eb_man = eb_man
        self.limit_ts = limit_ts
        self.smd_batch = smd_batch
    
    def events(self):
        # Event generator that yields list of events from 
        # an smd_chunk: data from smd0 that needs to be event built (serial mode) or
        # an smd_batch: prebuilt smd data (parallel mode)
        ev_man = EventManager(self.run.configs, self.run.dm, \
                filter_fn=self.run.filter_callback)
        
        if self.eb_man:

            for batch_dict in self.eb_man.batches(limit_ts=self.limit_ts):
                batch, _ = batch_dict[0]
                for evt in ev_man.events(batch):
                    if evt._dgrams[0].seq.service() != 12: continue
                    yield evt

        else:
            for evt in ev_man.events(self.smd_batch):
                if evt._dgrams[0].seq.service() != 12: continue
                yield evt 


