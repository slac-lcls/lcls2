"""Define `Step`, which iterates over the events of one scan step."""
from psana.psexp import TransitionId
import time
from psana.psexp.prometheus_manager import get_prom_manager
from psana import utils
from psana.dgramedit import DgramEdit
from psana.event import Event


class Step(object):

    """One step of a run: the BeginStep event (`evt`) and an iterator over the step's events.

    Parameters
    ----------
    step_evt : Event
        The step transition event, stored as `evt`.
    evt_iter : iterator
        Iterator of dgram lists, or of (dgrams, proxy event) tuples when `proxy_events` is given;
        shared with the run that created the step.
    run_ctx : RunCtx or None
        Passed as `run` to every `Event` created.
    proxy_events : list, optional
        List to which the proxy events of non-L1Accept transitions are appended.
    esm : EnvStoreManager, optional
        Updated with every non-L1Accept transition.
    run : RunDrp, optional
        When given, each dgram is wrapped in a `DgramEdit` stored as `run.curr_dgramedit` and saved
        to `run.dm.shm_res_mv`.
    callback_run_state : CallbackRunState, optional
        Its step state is cleared at EndStep.
    """
    def __init__(
        self,
        step_evt,
        evt_iter,
        run_ctx,
        proxy_events=None,
        esm=None,
        run=None,
        callback_run_state=None,
    ):
        self.evt = step_evt
        self._evt_iter = evt_iter
        self._run_ctx = run_ctx
        self.esm = esm
        self.run = run  # For RunDrp to access dm and curr_dgramedit
        self.callback_run_state = callback_run_state

        # RunSmallData can pass proxy_events so that when Step goes
        # through events, it can add all non L1Accept transitions to the list.
        self.proxy_events = proxy_events
        self.ana_t_gauge = get_prom_manager().get_metric("psana_bd_ana_rate")

    def events(self):
        """Yield the L1Accept events of this step as `Event` objects, stopping at EndStep.

        Other transitions are not yielded: they update `esm`, are saved to DRP shared memory and have their proxy event appended to `proxy_events` (each only when set), and EndStep also clears the step state in `callback_run_state` and ends the iteration. With `run` set, each L1Accept is saved to shared memory when the caller resumes the iteration.

        Notes
        -----
        After an L1Accept whose loop index is a multiple of 1000, the "psana_bd_ana_rate" gauge is set to 1000 / seconds since the last update.
        """
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
                    if self.callback_run_state is not None:
                        self.callback_run_state.in_step = False
                        self.callback_run_state.current_step_evt = None
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
