from psana.psexp import TransitionId


class Step(object):

    def __init__(self, step_evt, evt_iter, proxy_events=None, esm=None):
        self.evt = step_evt
        self.evt_iter = evt_iter
        self.esm = esm

        # RunSmallData can pass proxy_events so that when Step goes
        # through events, it can add all non L1Accept transitions to the list.
        self.proxy_events = proxy_events

    def events(self):
        for evt in self.evt_iter:
            if not TransitionId.isEvent(evt.service()):
                if evt.service() == TransitionId.EndStep:
                    return
                if self.proxy_events is not None:
                    self.proxy_events.append(evt._proxy_evt)
                if self.esm is not None:
                    self.esm.update_by_event(evt)
                continue
            yield evt
