import inspect
import time

from psana2 import dgram
from psana2.detector.detector_cache import DetectorCacheManager
from psana2.detector.detector_impl import MissingDet
from psana2.dgramedit import DgramEdit
from psana2.event import Event
from psana2.utils import Logger

from . import TransitionId
from .envstore_manager import EnvStoreManager
from .events import Events
from .step import Step
from .tools import RunHelper


def _is_hidden_attr(obj, name):
    if name.startswith("_"):
        return True
    else:
        attr = getattr(obj, name)
        if hasattr(attr, "_hidden"):
            return attr._hidden
        else:
            return False


def _enumerate_attrs(obj):
    state = []
    found = []

    def mygetattr(obj):
        children = [
            attr
            for attr in dir(obj)
            if not _is_hidden_attr(obj, attr) and attr != "dtype"
        ]
        for child in children:
            childobj = getattr(obj, child)

            if (
                len(
                    [
                        attr
                        for attr in dir(childobj)
                        if not _is_hidden_attr(childobj, attr)
                    ]
                )
                == 0
            ):
                found.append(".".join(state + [child]))
            elif type(childobj) is property:
                found.append(".".join(state + [child]))
            elif not inspect.isclass(childobj) and callable(childobj):
                found.append(".".join(state + [child]))
            else:
                state.append(child)
                mygetattr(childobj)
                state.pop()

    mygetattr(obj)
    return found


class StepEvent(object):
    """Use by AMI to access"""

    def __init__(self, env_store):
        self.env_store = env_store

    def dgrams(self, evt):
        return self.env_store.get_step_dgrams_of_event(evt)

    def docstring(self, evt) -> str:
        env_values = self.env_store.values([evt], "step_docstring")
        return env_values[0]

    def value(self, evt) -> float:
        env_values = self.env_store.values([evt], "step_value")
        return env_values[0]


class Run(object):
    def __init__(self, ds):
        self.ds = ds
        self.expt, self.runnum, self.timestamp = ds._get_runinfo()
        self.dsparms = ds.dsparms
        self.configs = None
        self.smd_dm = None
        self.nfiles = 0
        self.scan = False
        self.smd_fds = None

        self.logger = getattr(ds, "logger", None)
        if self.logger is None:
            self.logger = Logger(name="Run")

        if hasattr(ds, "dm"):
            ds.dm.set_run(self)
        if hasattr(ds, "smdr_man"):
            ds.smdr_man.set_run(self)

        RunHelper(self)

    @property
    def dm(self):
        return self.ds.dm

    def run(self):
        """Returns integer representaion of run no.
        default: (when no run is given) is set to -1"""
        return self.runnum

    def _check_empty_calibconst(self, det_name):
        # Some detectors do not have calibration constants - set default value to None
        if not hasattr(self.dsparms, "calibconst") or self.dsparms.calibconst is None:
            self.dsparms.calibconst = {det_name: None}
        elif det_name not in self.dsparms.calibconst:
            self.dsparms.calibconst[det_name] = None

    def _get_valid_env_var_name(self, det_name):
        # Check against detector names
        if self.esm.env_from_variable(det_name) is not None:
            return det_name

        # Check against epics names (return detector name not epics name)
        if det_name in self.esm.stores["epics"].epics_inv_mapper:
            return self.esm.stores["epics"].epics_inv_mapper[det_name]

        return None

    def Detector(self, name, accept_missing=False, **kwargs):

        mapped_env_var_name = self._get_valid_env_var_name(name)

        if name not in self.dsparms.configinfo_dict and mapped_env_var_name is None:
            if not accept_missing:
                err_msg = f"No available detector class matched with {name}. If this is a new detector/version, make sure to add new class in detector folder."
                raise KeyError(err_msg)
            else:
                return MissingDet()

        if mapped_env_var_name is not None:
            name = mapped_env_var_name

        class Container:
            def __init__(self):
                pass

        det = Container()
        # instantiate the detector xface for this detector/drp_class
        # (e.g. (xppcspad,raw) or (xppcspad,fex)
        # make them attributes of the container
        flag_found = False
        for (det_name, drp_class_name), drp_class in self.dsparms.det_classes[
            "normal"
        ].items():
            if det_name == name:
                # Detectors with cfgscan also owns an EnvStore
                env_store = None
                var_name = None
                if det_name in self.esm.stores:
                    env_store = self.esm.stores[det_name]
                    setattr(det, "step", StepEvent(env_store))

                self._check_empty_calibconst(det_name)

                # set and add propertiees for det.<drp_class_name>.<drp_class> level, i.e. det.raw.archon_raw_1_0_0
                iface = drp_class(
                    det_name,
                    drp_class_name,
                    self.dsparms.configinfo_dict[det_name],
                    self.dsparms.calibconst[det_name],
                    env_store,
                    var_name,
                    **kwargs,
                )
                setattr(det, drp_class_name, iface)

                # Optionally load cached _calibc_ attributes
                if getattr(self.ds, "use_calib_cache", False) and det_name in getattr(self.ds, "cached_detectors", []):
                    max_retries = getattr(self.ds, "fetch_calib_cache_max_retries", 3)
                    for attempt in range(max_retries):
                        try:
                            if DetectorCacheManager.load(det_name, iface, logger=self.logger):
                                break
                        except Exception:
                            pass
                        self.logger.debug(f"Retrying cache load for {det_name}.{drp_class_name} (attempt {attempt + 1}/{max_retries})")
                        time.sleep(1.0)  # optional delay between retries
                    else:
                        self.logger.error(f"Failed to load cache for {det_name}.{drp_class_name} after {max_retries} attempts")

                # add properties for det.raw level
                setattr(det, "_configs", self.configs)
                setattr(det, "calibconst", self.dsparms.calibconst[det_name])
                setattr(det, "_dettype", self.dsparms.det_info_table[det_name][0])
                setattr(det, "_detid", self.dsparms.det_info_table[det_name][1])
                setattr(det, "_det_name", det_name)
                # setattr(det, "_kwargs", kwargs)

                flag_found = True

        # If no detector found, EnvStore variable is assumed to have been passed in.
        # Environment values are identified by variable names (e.g. 'XPP:VARS:FLOAT:02').
        # From d.epics[0].raw.HX2:DVD:GCC:01:PMON = 41.0
        # (d.env_name[0].alg.variable),
        # EnvStoreManager searches all the stores assuming that the variable name is
        # unique and returns env_name ('epics' or 'scan') and algorithm.
        if not flag_found:
            found = self.esm.env_from_variable(
                name
            )  # assumes that variable name is unique
            if found is not None:
                env_name, alg = found
                det_name = env_name
                var_name = name
                drp_class_name = alg
                det_class_table = self.dsparms.det_classes[det_name]
                drp_class = det_class_table[(det_name, drp_class_name)]

                self._check_empty_calibconst(det_name)

                det = drp_class(
                    det_name,
                    drp_class_name,
                    self.dsparms.configinfo_dict[det_name],
                    self.dsparms.calibconst[det_name],
                    self.esm.stores[env_name],
                    var_name,
                )
                setattr(det, "_det_name", det_name)

        return det

    @property
    def scannames(self):
        return set([x[0] for x in self.scaninfo.keys()])

    @property
    def detnames(self):
        return set([x[0] for x in self.dsparms.det_classes["normal"].keys()])

    @property
    def detinfo(self):
        """
        Returns a mapping of detector interface attributes, guarding against
        infinite recursion during attribute enumeration.
        """
        info = {}
        for (detname, det_xface_name), _ in self.dsparms.det_classes["normal"].items():
            try:
                xface_obj = getattr(self.Detector(detname), det_xface_name)
                info[(detname, det_xface_name)] = _enumerate_attrs(xface_obj)
            except RecursionError:
                msg = (
                f"<error: RecursionError while walking {detname}.{det_xface_name}> "
                f"(Consider reviewing custom attributes for missing '_' prefix)"
            )
                info[(detname, det_xface_name)] = msg
                self.logger.warning(msg)
            except Exception as e:
                info[(detname, det_xface_name)] = f"<error: {e}>"
        return info

    @property
    def epicsinfo(self):
        # EnvStore a key-value pair of detectorname (e.g. AT1K0_PressureSetPt)
        # and epics name (AT1K0:GAS:TRANS_SP_RBV) in epics_mapper.
        return self.esm.stores["epics"].get_info()

    @property
    def scaninfo(self):
        return self.esm.stores["scan"].get_info()

    @property
    def xtcinfo(self):
        return self.dsparms.xtc_info

    def analyze(self, event_fn=None, det=None):
        for event in self.events():
            if event_fn is not None:
                event_fn(event, det)

    def step(self, evt):
        step_dgrams = self.esm.stores["scan"].get_step_dgrams_of_event(evt)
        return Event(dgrams=step_dgrams, run=self)

    def _setup_envstore(self):
        assert hasattr(self, "configs")
        assert hasattr(self, "_evt")  # BeginRun
        self.esm = EnvStoreManager(self.configs, run=self)
        # Special case for BeginRun - normally EnvStore gets updated
        # by EventManager but we get BeginRun w/o EventManager
        # so we need to update the EnvStore here.
        self.esm.update_by_event(self._evt)


class RunShmem(Run):
    """Yields list of events from a shared memory client (no event building routine)."""

    def __init__(self, ds, run_evt):
        super(RunShmem, self).__init__(ds)
        self._evt = run_evt
        self.beginruns = run_evt._dgrams
        self.configs = ds._configs
        super()._setup_envstore()
        self._evt_iter = Events(ds, self)

    def events(self):

        for evt in self._evt_iter:
            if not TransitionId.isEvent(evt.service()):
                if evt.service() == TransitionId.EndRun:
                    return
                continue
            yield evt

    def steps(self):
        for evt in self._evt_iter:
            if evt.service() == TransitionId.EndRun:
                return
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, self._evt_iter)


class RunDrp(Run):
    """Yields list of events from drp python (no event building routine)."""

    def __init__(self, ds, run_evt):
        super(RunDrp, self).__init__(ds)
        self._evt = run_evt
        self._ds = ds
        self.beginruns = run_evt._dgrams
        self.configs = ds._configs
        super()._setup_envstore()
        self._evt_iter = Events(self._ds, self)
        self._edtbl_config = False

    def events(self):
        for evt in self._evt_iter:
            if TransitionId.isEvent(evt.service()):
                buffer_size = self._ds.dm.pebble_bufsize
            else:
                buffer_size = self._ds.dm.transition_bufsize

            self._ds.curr_dgramedit = DgramEdit(
                evt._dgrams[0],
                config_dgramedit=self._ds.config_dgramedit,
                bufsize=buffer_size,
            )

            if not TransitionId.isEvent(evt.service()):
                if evt.service() == TransitionId.EndRun:
                    self._ds.curr_dgramedit.save(self._ds.dm.shm_res_mv)
                    return
                self._ds.curr_dgramedit.save(self._ds.dm.shm_res_mv)
                continue
            yield evt
            self._ds.curr_dgramedit.save(self._ds.dm.shm_res_mv)

    def steps(self):
        for evt in self._evt_iter:
            if TransitionId.isEvent(evt.service()):
                buffer_size = self._ds.dm.pebble_bufsize
            else:
                buffer_size = self._ds.dm.transition_bufsize

            self._ds.curr_dgramedit = DgramEdit(
                evt._dgrams[0],
                config_dgramedit=self._ds.config_dgramedit,
                bufsize=buffer_size,
            )

            if evt.service() == TransitionId.EndRun:
                self._ds.curr_dgramedit.save(self._ds.dm.shm_res_mv)
                return
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, self._evt_iter)
            self._ds.curr_dgramedit.save(self._ds.dm.shm_res_mv)


class RunSingleFile(Run):
    """Yields list of events from a single bigdata file."""

    def __init__(self, ds, run_evt):
        super(RunSingleFile, self).__init__(ds)
        self._evt = run_evt
        self.beginruns = run_evt._dgrams
        self.configs = ds._configs
        super()._setup_envstore()
        self._evt_iter = Events(ds, self)

    def events(self):
        for evt in self._evt_iter:
            if not TransitionId.isEvent(evt.service()):
                if evt.service() == TransitionId.EndRun:
                    return
                continue
            yield evt

    def steps(self):
        for evt in self._evt_iter:
            if evt.service() == TransitionId.EndRun:
                return
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, self._evt_iter)


class RunSerial(Run):
    """Yields list of events from multiple smd/bigdata files using single core."""

    def __init__(self, ds, run_evt):
        super(RunSerial, self).__init__(ds)
        self._evt = run_evt
        self.beginruns = run_evt._dgrams
        self.configs = ds._configs
        super()._setup_envstore()
        self._evt_iter = Events(ds, self, smdr_man=ds.smdr_man)

    def events(self):
        for evt in self._evt_iter:
            if not TransitionId.isEvent(evt.service()):
                if evt.service() == TransitionId.EndRun:
                    return
                continue
            yield evt

    def steps(self):
        for evt in self._evt_iter:
            if evt.service() == TransitionId.EndRun:
                return
            if evt.service() == TransitionId.BeginStep:
                yield Step(evt, self._evt_iter)


class RunLegion(Run):
    def __init__(self, ds, run_evt):
        self.dsparms = ds.dsparms
        self.ana_t_gauge = self.dsparms.prom_man.get_metric("psana_bd_ana_rate")
        RunHelper(self)
        self._evt = run_evt
        self.beginruns = run_evt._dgrams
        self.smdr_man = ds.smdr_man
        self.configs = ds._configs
        super()._setup_envstore()


class RunSmallData(Run):
    """Yields list of smalldata events

    This class is created by SmdReaderManager and used exclusively by EventBuilder.
    There's no DataSource class associated with it a "bigdata" run is passed in
    so that we can propagate some values (w/o the ds). This class makes step and
    event generator available to user in smalldata callback.
    """

    def __init__(self, bd_run, eb):
        self._evt = bd_run._evt
        configs = [dgram.Dgram(view=cfg) for cfg in bd_run.configs]
        self.expt, self.runnum, self.timestamp, self.dsparms = (
            bd_run.expt,
            bd_run.runnum,
            bd_run.timestamp,
            bd_run.dsparms,
        )
        self.eb = eb

        # Setup EnvStore - this is also done differently from other Run types.
        # since RunSmallData doesn't user EventManager (updates EnvStore), it'll
        # need to take care of updating transition events in the step and
        # event generators.
        self.esm = EnvStoreManager(configs, run=self)
        self.esm.update_by_event(self._evt)

        # Converts EventBuilder generator to an iterator for steps() call. This is
        # done so that we can pass it to Step (not sure why). Note that for
        # events() call, we stil use the generator.
        self._evt_iter = iter(self.eb.events())

        # SmdDataSource and BatchIterator share this list. SmdDataSource automatically
        # adds transitions to this list (skip yield and so hidden from smd_callback).
        # BatchIterator adds user-selected L1Accept to the list (default is add all).
        self.proxy_events = []

    def steps(self):
        for evt in self._evt_iter:
            if not TransitionId.isEvent(evt.service()):
                self.esm.update_by_event(evt)
                self.proxy_events.append(evt._proxy_evt)
                if evt.service() == TransitionId.EndRun:
                    return
                if evt.service() == TransitionId.BeginStep:
                    yield Step(
                        evt,
                        self._evt_iter,
                        proxy_events=self.proxy_events,
                        esm=self.esm,
                    )

    def events(self):
        for evt in self.eb.events():
            if not TransitionId.isEvent(evt.service()):
                self.esm.update_by_event(evt)
                self.proxy_events.append(evt._proxy_evt)
                if evt.service() == TransitionId.EndRun:
                    return
                continue
            yield evt
