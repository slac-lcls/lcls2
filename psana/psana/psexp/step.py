from psana.psexp import TransitionId

class Step(object):
    
    def __init__(self, step_evt, events, esm):
        self.evt    = step_evt
        self._events= events
        self.esm    = esm
    
    def events(self):
        for evt in self._events:
            if evt.service() != TransitionId.L1Accept:
                self.esm.update_by_event(evt)
                if evt.service() == TransitionId.EndStep: return
                continue
            yield evt
        

            
