from psana.psexp.event_manager import TransitionId

class Step(object):
    
    def __init__(self, step_evt, events):
        self._events = events
        self.evt = step_evt
    
    def events(self):
        for evt in self._events:
            if evt.service() == TransitionId.EndStep: return
            if evt.service() == TransitionId.L1Accept: yield evt

            
