from psana.psexp import TransitionId

class Step(object):
    
    def __init__(self, step_evt, evt_iter, esm):
        self.evt        = step_evt
        self.evt_iter   = evt_iter
        self.esm        = esm
        self.esm.update_by_event(step_evt)
    
    def events(self):
        for evt in self.evt_iter:
            if evt.service() != TransitionId.L1Accept:
                self.esm.update_by_event(evt)
                if evt.service() == TransitionId.EndStep: 
                    return
                continue
            yield evt
        

            
