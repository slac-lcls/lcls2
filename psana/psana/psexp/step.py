from psana.psexp import TransitionId

class Step(object):
    
    def __init__(self, step_evt, evt_iter, proxy_events=None):
        self.evt        = step_evt
        self.evt_iter   = evt_iter

        #RunSmallData can pass proxy_events so that when Step goes
        #through events, it can add all non L1Accept transitions to the list.
        self.proxy_events = proxy_events
    
    def events(self):
        for evt in self.evt_iter:
            if evt.service() != TransitionId.L1Accept: 
                if evt.service() == TransitionId.EndStep: return
                if self.proxy_events is not None:
                    self.proxy_events.append(evt._proxy_evt)
                continue
            yield evt
        

            
