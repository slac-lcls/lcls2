from psana.psexp import TransitionId
import time
from psana.psexp.prometheus_manager import get_prom_manager


class Step(object):

    def __init__(self, step_evt, evt_iter, proxy_events=None, esm=None):
        self.evt = step_evt
        self.evt_iter = evt_iter
        self.esm = esm

        # RunSmallData can pass proxy_events so that when Step goes
        # through events, it can add all non L1Accept transitions to the list.
        self.proxy_events = proxy_events
        self.ana_t_gauge = get_prom_manager().get_metric("psana_bd_ana_rate")

    def events(self):
        st = time.time()
        for i, evt in enumerate(self.evt_iter):
            if not TransitionId.isEvent(evt.service()):
                if evt.service() == TransitionId.EndStep:
                    return
                if self.proxy_events is not None:
                    self.proxy_events.append(evt._proxy_evt)
                if self.esm is not None:
                    self.esm.update_by_event(evt)
                continue
            yield evt

            if i % 1000 == 0:
                en = time.time()
                ana_rate = 1000 / (en - st)
                self.ana_t_gauge.set(ana_rate)
                st = time.time()
