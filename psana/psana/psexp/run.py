import inspect
import time
import os
import gc
import weakref
from contextlib import contextmanager

from psana import dgram
from psana.detector.detector_cache import DetectorCacheManager
from psana.detector.detector_impl import MissingDet
from psana.dgramedit import DgramEdit
from psana.event import Event
from psana.utils import Logger, WeakDict
from psana.psexp.prometheus_manager import get_prom_manager
from psana.pscalib.app.calib_prefetch import calib_utils
import psana.pscalib.calib.MDBWebUtils as wu
from psana import utils

from . import TransitionId
from .envstore_manager import EnvStoreManager
from .events import Events
from .smd_events import SmdEvents
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

        # Calibration constant structures
        # Held by _calib_const, _cc_weak holds weak references to the sub-dicts
        # dsparms.calibconst holds a weak reference to _cc_weak
        self._calib_const = None
        """Holds calibration constants for all detectors."""
        self._cc_weak = utils.WeakDict({})
        """Holds weakrefs to each detectors constants. Passed via weakref in dsparms."""

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
            self.dsparms.calibconst = {det_name: WeakDict({})}
        elif det_name not in self.dsparms.calibconst:
            self.dsparms.calibconst[det_name] = WeakDict({})

    def _get_valid_env_var_name(self, det_name):
        # Check against detector names
        if self.esm.env_from_variable(det_name) is not None:
            return det_name

        # Check against epics names (return detector name not epics name)
        if det_name in self.esm.stores["epics"].epics_inv_mapper:
            return self.esm.stores["epics"].epics_inv_mapper[det_name]

        return None

    def _clear_calibconst(self):
        """Explicitly clear calibration constants and force garbage collection.

        Helps to prevent leakage of calibration constants between runs.
        """
        if self._calib_const is not None:
            self._calib_const = None
            gc.collect()
            self._calib_const = utils.WeakDict({})
            self._cc_weak = utils.WeakDict({})

    def _create_weak_calibconst(self):
        """Setup weak references to calibration constants.

        Pass along only weak references to large calibration data structures
        since references are held in many places downstream making garbage
        collection unreliable if any one object leaks/has circular references.
        """
        if self._calib_const is not None:
            for key in self._calib_const:
                self._cc_weak[key] = weakref.WeakValueDictionary(self._calib_const[key])
        # Only use dsparms.calibconst to pass to runs, detectors etc.
        self.dsparms.calibconst = weakref.proxy(self._cc_weak)

    def _setup_run_calibconst(self):
        """
        Initialize the `dsparms.calibconst` dictionary with calibration constants.

        If `use_calib_cache` is enabled, attempts to load calibration constants from
        shared memory (e.g., `/dev/shm`) using `ensure_valid_calibconst`. Otherwise,
        fetches calibration constants directly from the database for each detector.

        This function supports skipping selected detectors via `skip_calib_load`.

        Args:
            None

        Returns:
            None
        """
        expt, runnum = self.expt, self.runnum
        self.logger.debug(f"_setup_run_calibconst called {expt=} {runnum=}")

        self._clear_calibconst()

        if self.ds.use_calib_cache:
            self.logger.debug("using calibration constant from shared memory, if exists")
            calib_const, existing_info, max_retry = (None, None, 0)
            latest_info = {k: v.uniqueid for k, v in self.dsparms.configinfo_dict.items()}
            while not calib_const and max_retry < self.ds.fetch_calib_cache_max_retries and existing_info != latest_info:
                try:
                    loaded_data = calib_utils.try_load_data_from_file(self.logger)
                except Exception as e:
                    self.logger.warning(f"failed to retrieve calib_const: {e}")
                if loaded_data is not None:
                    self._calib_const = utils.make_weak_refable(loaded_data.get("calib_const"))
                    existing_info = loaded_data.get("det_info")
                if existing_info != latest_info:
                    self._calib_const = utils.WeakDict({})
                    self.logger.warning("det_info in pickle file does not matched with the latest run")
                    self.logger.warning(f"{existing_info=}")
                    self.logger.warning(f"{latest_info=}")
                elif self._calib_const:
                    self.logger.debug(f"received calib_const for {','.join(self._calib_const.keys())}")
                    break
                max_retry += 1
                self.logger.debug(f"try to read calib_const from shared memory (retry: {max_retry}/{self.fetch_calib_cache_max_retries})")
                time.sleep(1)
        else:
            if self._calib_const is None:
                self._calib_const = utils.WeakDict({})
            for det_name, configinfo in self.dsparms.configinfo_dict.items():
                if expt:
                    if expt == "xpptut15":
                        det_uniqueid = "cspad_detnum1234"
                    else:
                        det_uniqueid = configinfo.uniqueid
                    if hasattr(self.ds, "skip_calib_load") and (det_name in self.ds.skip_calib_load or self.ds.skip_calib_load=="all"):
                        self._calib_const[det_name] = utils.WeakDict({})
                        continue
                    st = time.monotonic()
                    calib_const = wu.calib_constants_all_types(
                        det_uniqueid, exp=expt, run=runnum, dbsuffix=self.ds.dbsuffix
                    )
                    en = time.monotonic()
                    self.logger.debug(f"received calibconst for {det_name} in {en-st:.4f}s.")
                    self._calib_const[det_name] = utils.make_weak_refable(calib_const)
                else:
                    self.logger.info(
                        "Warning: cannot access calibration constant (exp is None)"
                    )
                    self._calib_const[det_name] = utils.WeakDict({})
        self._create_weak_calibconst()


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
        self._smd_iter = None
        self._ts_table = None

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

    @contextmanager
    def build_table(self):
        """
        Context manager for building timestamp-offset table in RunSerial.
        Useful in both serial and MPI modes to isolate this setup step.
        Returns True if table was successfully built.
        """
        if self._smd_iter is None:
            self._smd_iter = SmdEvents(self.ds, self, smdr_man=self.ds.smdr_man)

        self._ts_table = {}
        for evt in self._smd_iter:
            if evt.service() != TransitionId.L1Accept:
                continue

            ts = evt.timestamp
            self._ts_table[ts] = {
                i: (d.smdinfo[0].offsetAlg.intOffset, d.smdinfo[0].offsetAlg.intDgramSize)
                for i, d in enumerate(evt._dgrams)
                if d is not None and hasattr(d, 'smdinfo')
            }
        yield bool(self._ts_table)

    def event(self, ts):
        if self._ts_table is None:
            raise RuntimeError("build_table() must be called before event(ts)")

        offsets = self._ts_table.get(ts)
        if offsets is None:
            raise ValueError(f"Timestamp {ts} not found in offset table.")

        dgrams = [None] * len(self.configs)
        for i, (offset, size) in offsets.items():
            buf = os.pread(self.ds.dm.fds[i], size, offset)
            dgrams[i] = dgram.Dgram(config=self.ds.dm.configs[i], view=buf)

        return Event(dgrams=dgrams, run=self)


class RunLegion(Run):
    def __init__(self, ds, run_evt):
        self.dsparms = ds.dsparms
        self.ana_t_gauge = get_prom_manager().get_metric("psana_bd_ana_rate")
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
