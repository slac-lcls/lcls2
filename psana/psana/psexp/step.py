from psana.psexp import TransitionId
import time
from psana.psexp.prometheus_manager import get_prom_manager
from psana import utils
from psana.dgramedit import DgramEdit
from psana.event import Event


class Step(object):

    def __init__(self, step_evt, evt_iter, run_ctx, proxy_events=None, esm=None, run=None):
        self.evt = step_evt
        self._evt_iter = evt_iter
        self._run_ctx = run_ctx
        self.esm = esm
        self.run = run  # For RunDrp to access dm and curr_dgramedit

        # RunSmallData can pass proxy_events so that when Step goes
        # through events, it can add all non L1Accept transitions to the list.
        self.proxy_events = proxy_events
        self.ana_t_gauge = get_prom_manager().get_metric("psana_bd_ana_rate")

    def events(self):
        st = time.time()
        for i, item in enumerate(self._evt_iter):
            proxy_evt = None
            if self.proxy_events is not None:
                dgrams, proxy_evt = item
            else:
                dgrams = item
            svc = utils.first_service(dgrams)
            evt = Event(dgrams=dgrams, run=self._run_ctx, proxy_evt=proxy_evt)
            if self.run is not None:
                bufsize = self.run.dm.pebble_bufsize if TransitionId.isEvent(svc) else self.run.dm.transition_bufsize

                self.run.curr_dgramedit = DgramEdit(
                    dgrams[0],
                    config_dgramedit=self.run.config_dgramedit,
                    bufsize=bufsize,
                )

            # Handle non-L1 transitions
            if not TransitionId.isEvent(svc):
                if self.esm is not None:
                    self.esm.update_by_event(evt)
                if self.run is not None:
                    self.run.curr_dgramedit.save(self.run.dm.shm_res_mv)
                if self.proxy_events is not None:
                    self.proxy_events.append(evt._proxy_evt)
                if svc == TransitionId.EndStep:
                    return
                continue

            # L1Accept: yield first and for RunDrp, update current dgramedit after
            yield evt
            if self.run is not None:
                self.run.curr_dgramedit.save(self.run.dm.shm_res_mv)

            if i % 1000 == 0:
                en = time.time()
                ana_rate = 1000 / (en - st)
                self.ana_t_gauge.set(ana_rate)
                st = time.time()
