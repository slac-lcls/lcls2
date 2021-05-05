from psana.psexp import TransitionId

class Step(object):
    
    def __init__(self, step_evt, evt_iter):
        self.evt        = step_evt
        self.evt_iter   = evt_iter
    
    def events(self):
        for evt in self.evt_iter:
            if evt.service() != TransitionId.L1Accept:
                if evt.service() == TransitionId.EndStep: 
                    return
                continue
            yield evt
        

            
