import inspect
import time
import os
import gc
from contextlib import contextmanager
from multiprocessing import Value
from types import SimpleNamespace

from psana import dgram
from psana.detector.detector_cache import DetectorCacheManager
from psana.detector.detector_impl import MissingDet
from psana.dgramedit import DgramEdit
from psana.event import Event
from psana.pscalib.app.calib_prefetch import calib_utils
import psana.pscalib.calib.MDBWebUtils as wu
from psana import utils

from . import TransitionId
from .envstore_manager import EnvStoreManager
from .events import Events
from .smd_events import SmdEvents
from .step import Step
from .tools import RunHelper
from .run_ctx import RunCtx


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
    def __init__(self, expt, runnum, timestamp, dsparms, dm, smdr_man, begingrun_dgrams):
        self.expt, self.runnum, self.timestamp = (expt, runnum, timestamp)
        self.dsparms = dsparms
        self.dm = dm
        self.smdr_man = smdr_man
        self._run_ctx = RunCtx(expt, runnum, timestamp, dsparms.intg_det)
        self._evt = Event(dgrams=begingrun_dgrams, run=self._run_ctx)
        self.configs = None
        self.smd_dm = None
        self.nfiles = 0
        self.scan = False
        self.smd_fds = None

        self.shared_state = SimpleNamespace(
            terminate_flag=Value('b', False)
        )

        # Calibration constant structures
        self._calib_const = None
        """Holds calibration constants for all detectors."""
        self.dsparms.calibconst = {}

        self.logger = utils.get_logger(name=utils.get_class_name(self))

        self.build_detinfo_dict()

        RunHelper(self)

    def run(self):
        """Returns integer representaion of run no.
        default: (when no run is given) is set to -1"""
        return self.runnum

    def events(self):
        for dgrams in self._evt_iter:
            if self._handle_transition(dgrams):
                # EndRun handling here ends the stream
                if utils.first_service(dgrams) == TransitionId.EndRun:
                    return
                continue  # swallow non-L1 transitions in events() stream
            # L1Accept: construct Event at the Run level
            yield Event(dgrams=dgrams, run=self._run_ctx)

    def steps(self):
        for dgrams in self._evt_iter:
            svc = utils.first_service(dgrams)
            if TransitionId.isEvent(svc):
                # steps() only yields on BeginStep transitions; ignore L1
                continue
            # Update envstore for every transition
            self._update_envstore_from_dgrams(dgrams)

            if svc == TransitionId.EndRun:
                return
            if svc == TransitionId.BeginStep:
                yield Step(
                    Event(dgrams=dgrams, run=self._run_ctx),
                    self._evt_iter,
                    self._run_ctx,
                    esm=self.esm,
                )

    def terminate(self):
        """Sets terminate flag

        The Events iterators of all Run types check this flag
        to see if they need to stop (raise StopIterator for
        RunSerial, RunShmem, & RunSinglefile or skip events
        for RunParallel (see BigDataNode implementation).

        Note that mpi_ds implements this differently by adding
        Isend prior to setting this flag.
        """
        flag = self.shared_state.terminate_flag
        flag.value = True

    def _check_empty_calibconst(self, det_name):
        # Some detectors do not have calibration constants - set default value to None
        if self._calib_const is None:
            self._calib_const = {}
        if det_name not in self._calib_const:
            self._calib_const[det_name] = {}
        self.dsparms.calibconst = self._calib_const

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
            self.dsparms.calibconst = {}
            gc.collect()
        self._calib_const = {}
        self.dsparms.calibconst = self._calib_const

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

        if self.dsparms.use_calib_cache:
            self.logger.debug("using calibration constant from shared memory, if exists")
            calib_const, existing_info, max_retry = (None, None, 0)
            latest_info = self.get_filtered_detinfo()

            while not calib_const and max_retry < self.dsparms.fetch_calib_cache_max_retries:
                try:
                    loaded_data = calib_utils.try_load_data_from_file(self.logger)
                except Exception as e:
                    self.logger.warning(f"failed to retrieve calib_const: {e}")
                    loaded_data = None

                if loaded_data is not None:
                    self._calib_const = loaded_data.get("calib_const") or {}
                    existing_info = loaded_data.get("det_info")

                if self._calib_const:
                    if existing_info is None:
                        self.logger.warning("calib_const loaded but existing_info is missing")
                    elif not latest_info.items() <= existing_info.items():
                        self.logger.warning("det_info in pickle file is incomplete for this run")
                        self.logger.warning(f"{existing_info=}")
                        self.logger.warning(f"{latest_info=}")
                        self._calib_const = {}
                    else:
                        self.logger.debug(f"received calib_const for {','.join(self._calib_const.keys())}")
                        break

                max_retry += 1
                self.logger.debug(f"try to read calib_const from shared memory (retry: {max_retry}/{self.dsparms.fetch_calib_cache_max_retries})")
                time.sleep(1)
        else:
            if self._calib_const is None:
                self._calib_const = {}
            total_time = 0.0
            max_time = 0.0
            max_det = None
            fetched = 0
            for det_name, configinfo in self.dsparms.configinfo_dict.items():
                if expt:
                    if expt == "xpptut15":
                        det_uniqueid = "cspad_detnum1234"
                    else:
                        det_uniqueid = configinfo.uniqueid
                    if hasattr(self.dsparms, "skip_calib_load") and (det_name in self.dsparms.skip_calib_load or self.dsparms.skip_calib_load=="all"):
                        self._calib_const[det_name] = {}
                        continue
                    st = time.monotonic()
                    calib_const = wu.calib_constants_all_types(
                        det_uniqueid, exp=expt, run=runnum, dbsuffix=self.dsparms.dbsuffix
                    )
                    en = time.monotonic()
                    det_time = en - st
                    total_time += det_time
                    fetched += 1
                    if det_time > max_time:
                        max_time = det_time
                        max_det = det_name
                    self._calib_const[det_name] = calib_const or {}
                else:
                    self.logger.info(
                        "Warning: cannot access calibration constant (exp is None)"
                    )
                    self._calib_const[det_name] = {}
            if fetched:
                max_label = max_det or "unknown"
                self.logger.debug(
                    "received calibconst for %d detectors total=%.4fs max=%s(%.4fs)",
                    fetched,
                    total_time,
                    max_label,
                    max_time,
                )
        self.dsparms.calibconst = self._calib_const


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
                if getattr(self.dsparms, "use_calib_cache", False) and det_name in getattr(self.dsparms, "cached_detectors", []):
                    max_retries = getattr(self.dsparms, "fetch_calib_cache_max_retries", 3)
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

                shared_cache = getattr(self, "_shared_geo_cache", None)
                if shared_cache is not None:
                    setattr(iface, "_shared_geo_cache", shared_cache)

                shared_calibc_cache = getattr(self, "_shared_calibc_cache", None)
                if shared_calibc_cache is not None:
                    setattr(iface, "_shared_calibc_cache", shared_calibc_cache)

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

    def get_filtered_detinfo(self):
        """
        Returns detector info filtered by detector names present in detinfo.

        Returns:
            dict[str, str]: A dictionary mapping detector names to unique IDs,
                            filtered to only include detectors present in self.detinfo.
        """
        return {
            k: v.uniqueid
            for k, v in self.dsparms.configinfo_dict.items()
            if k in {dk[0] for dk in self.detinfo.keys()}
        }

    def build_detinfo_dict(self):
        """
        Creates a dictionary of detector interfaces and their enumerated attributes,
        using minimal instantiation with config=None, calib=None, env=None.

        Returns:
            dict[tuple[str, str], Any]: Mapping of (detname, xface_name) -> attribute summary
        """
        info = {}
        for (detname, xface_name), drp_class in self.dsparms.det_classes["normal"].items():
            try:
                # Minimal instantiation with None inputs to avoid side effects
                config = self.dsparms.configinfo_dict.get(detname, None)
                calib, env_store, var_name = None, None, None
                iface = drp_class(
                    detname,
                    xface_name,
                    config,
                    calib,
                    env_store,
                    var_name,
                )

                info[(detname, xface_name)] = _enumerate_attrs(iface)

            except RecursionError:
                msg = (
                    f"<error: RecursionError while walking {detname}.{xface_name}> "
                    f"(Consider reviewing custom attributes for missing '_' prefix)"
                )
                self.logger.warning(msg)
                info[(detname, xface_name)] = msg

            except Exception as e:
                info[(detname, xface_name)] = f"<error: {e}>"
                self.logger.debug(f"[detinfo-lite] Skipped {detname}.{xface_name}: {e}", exc_info=True)
        self.detinfo = info

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
        return Event(dgrams=step_dgrams, run=self._run_ctx)

    def _setup_envstore(self):
        assert hasattr(self, "configs")
        assert hasattr(self, "_evt")  # BeginRun
        self.esm = EnvStoreManager(self.configs)
        # Special case for BeginRun - normally EnvStore gets updated
        # by EventManager but we get BeginRun w/o EventManager
        # so we need to update the EnvStore here.
        self.esm.update_by_event(self._evt)

    def _update_envstore_from_dgrams(self, dgrams):
        """
        Update EnvStore with a transition carried by `dgrams`.
        Wraps the list in a transient Event to reuse existing EnvStoreManager API.
        """
        # EnvStoreManager currently expects an Event; keep that API.
        evt = Event(dgrams=dgrams, run=self._run_ctx)
        self.esm.update_by_event(evt)

    def _handle_transition(self, dgrams):
        """Common transition handling for non-L1Accept dgrams."""
        svc = utils.first_service(dgrams)
        if TransitionId.isEvent(svc):
            return False  # caller should treat it as an L1 event
        # Update env on every non-L1 transition (BeginRun/EndRun/BeginStep/EndStep/etc.)
        self._update_envstore_from_dgrams(dgrams)
        return True


class RunShmem(Run):
    """Yields list of events from a shared memory client (no event building routine)."""
    def __init__(self, expt, runnum, timestamp, dsparms, dm, smdr_man, configs, begingrun_dgrams, **kwargs):
        super(RunShmem, self).__init__(expt, runnum, timestamp, dsparms, dm, smdr_man, begingrun_dgrams)
        self.configs = configs
        super()._setup_envstore()
        self._evt_iter = Events(configs,
                                dm,
                                self.dsparms.max_retries,
                                self.dsparms.use_smds,
                                self.shared_state,
                                smdr_man=smdr_man)
        self.supervisor = kwargs.get('shmem_supervisor', -1)
        self._pub_socket = kwargs.get('shmem_pub_socket', None)
        self._sub_socket = kwargs.get('shmem_sub_socket', None)
        self._setup_run_calibconst()

    def _setup_run_calibconst(self):
        st = time.monotonic()
        if self.supervisor:
            super()._setup_run_calibconst()
            if self.supervisor == 1:
                self._pub_socket.send(self._calib_const)
        else:
            self._clear_calibconst()
            self._calib_const = self._sub_socket.recv()
            self.dsparms.calibconst = self._calib_const
        self.logger.debug(f"Exit _setup_run_calibconst total time: {time.monotonic()-st:.4f}s.")


class RunDrp(Run):
    """Yields list of events from drp python (no event building routine)."""
    def __init__(self, expt, runnum, timestamp, dsparms, dm, smdr_man, configs, begingrun_dgrams, **kwargs):
        super(RunDrp, self).__init__(expt, runnum, timestamp, dsparms, dm, smdr_man, begingrun_dgrams)
        self.configs = configs
        super()._setup_envstore()

        # DgramEdit for editing events
        self.config_dgramedit = DgramEdit(configs[-1], bufsize=dm.transition_bufsize)
        self.curr_dgramedit = self.config_dgramedit
        self._ed_detectors = []
        self._ed_detectors_handles = {}
        self._edtbl_config = True  # allow editing configuration before event iteration
        self._primed = False  # run priming not yet done
        self._evt_iter = Events(configs,
                                dm,
                                self.dsparms.max_retries,
                                self.dsparms.use_smds,
                                self.shared_state,
                                smdr_man=smdr_man)
        # For distributing calibconst
        self._is_supervisor = kwargs.get('drp_is_supervisor', False)
        self._is_publisher = kwargs.get('drp_is_publisher', False)
        self._tcp_pub_socket = kwargs.get('drp_tcp_pub_socket', None)
        self._ipc_pub_socket = kwargs.get('drp_ipc_pub_socket', None)
        self._setup_run_calibconst()

    def _setup_run_calibconst(self):
        if self._is_supervisor:
            if self._is_publisher:
                super()._setup_run_calibconst()
                self._tcp_pub_socket.send(self.dsparms.calibconst)
                # This is for drp_python
                self._ipc_pub_socket.send(self.dsparms.calibconst)
            else:
                self._calib_const = self._ipc_sub_socket.recv()
                self.dsparms.calibconst = self._calib_const
        else:
            if self._is_publisher:
                self._calib_const = self._tcp_sub_socket.recv()
                self.dsparms.calibconst = self._calib_const
                self._ipc_pub_socket.send(self.dsparms.calibconst)
            else:
                self._calib_const = self._ipc_sub_socket.recv()
                self.dsparms.calibconst = self._calib_const

    def add_detector(
        self, detdef, algdef, datadef, nodeId=None, namesId=None, segment=None
    ):
        if self._edtbl_config:
            return self.curr_dgramedit.Detector(
                detdef, algdef, datadef, nodeId, namesId, segment
            )
        else:
            raise RuntimeError(
                "[Python - Worker {self.worker_num}] Cannot edit the configuration "
                "after starting iteration over events"
            )

    def add_data(self, data):
        return self.curr_dgramedit.adddata(data)

    def remove_data(self, det_name, alg):
        if not self._edtbl_config:
            return self.curr_dgramedit.removedata(det_name, alg)
        else:
            raise RuntimeError(
                "[Python - Worker {self.worker_num}] Cannot remove data from events "
                "before starting iteration over events"
            )

    # --- internal: run priming (CONFIG + BeginRun saved once) ---
    def _prime_run_once(self):
        if self._primed:
            return
        # Save CONFIG first (allows pre-run edits to be captured)
        self.curr_dgramedit.save(self.dm.shm_res_mv)
        # Freeze config-time edits now
        self._edtbl_config = False
        # Save BeginRun transition
        self.curr_dgramedit = DgramEdit(
            self.beginruns[0],
            config_dgramedit=self.config_dgramedit,
            bufsize=self.dm.transition_bufsize,
        )
        self.curr_dgramedit.save(self.dm.shm_res_mv)
        self._primed = True

    def events(self):
        self._prime_run_once()

        for dgrams in self._evt_iter:
            svc = utils.first_service(dgrams)
            bufsize = self.dm.pebble_bufsize if TransitionId.isEvent(svc) else self.dm.transition_bufsize

            self.curr_dgramedit = DgramEdit(
                dgrams[0],
                config_dgramedit=self.config_dgramedit,
                bufsize=bufsize,
            )

            # Transitions: not yielded here; save immediately
            if self._handle_transition(dgrams):
                self.curr_dgramedit.save(self.dm.shm_res_mv)
                if svc == TransitionId.EndRun:
                    return
                continue

            # L1Accept: yield first so user may edit, then save
            evt = Event(dgrams=dgrams, run=self._run_ctx)
            yield evt
            self.curr_dgramedit.save(self.dm.shm_res_mv)

    def steps(self):
        self._prime_run_once()

        for dgrams in self._evt_iter:
            svc = utils.first_service(dgrams)
            if TransitionId.isEvent(svc):
                # steps() ignores L1s entirely
                continue

            self._update_envstore_from_dgrams(dgrams)
            bufsize = self.dm.pebble_bufsize if TransitionId.isEvent(svc) else self.dm.transition_bufsize

            self.curr_dgramedit = DgramEdit(
                dgrams[0],
                config_dgramedit=self.config_dgramedit,
                bufsize=bufsize,
            )

            if svc == TransitionId.BeginStep:
                # Yield a Step (user may edit), then save
                step_evt = Event(dgrams=dgrams, run=self._run_ctx)
                yield Step(step_evt, self._evt_iter, self._run_ctx, esm=self.esm, run=self)
                self.curr_dgramedit.save(self.dm.shm_res_mv)
                continue

            # Other transitions in steps(): save immediately
            self.curr_dgramedit.save(self.dm.shm_res_mv)
            if svc == TransitionId.EndRun:
                return

class RunSingleFile(Run):
    """Yields list of events from a single bigdata file."""

    def __init__(self, expt, runnum, timestamp, dsparms, dm, smdr_man, configs, begingrun_dgrams):
        super(RunSingleFile, self).__init__(expt, runnum, timestamp, dsparms, dm, smdr_man, begingrun_dgrams)
        self.configs = configs
        super()._setup_envstore()
        self._evt_iter = Events(configs,
                                dm,
                                self.dsparms.max_retries,
                                self.dsparms.use_smds,
                                self.shared_state,
                                smdr_man=smdr_man)
        self._setup_run_calibconst()


class RunSerial(Run):
    """Yields list of events from multiple smd/bigdata files using single core."""

    def __init__(self, expt, runnum, timestamp, dsparms, dm, smdr_man, configs, begingrun_dgrams):
        super(RunSerial, self).__init__(expt, runnum, timestamp, dsparms, dm, smdr_man, begingrun_dgrams)
        self.configs = configs
        super()._setup_envstore()
        self._evt_iter = Events(configs,
                                dm,
                                self.dsparms.max_retries,
                                self.dsparms.use_smds,
                                self.shared_state,
                                smdr_man=smdr_man)
        self._smd_iter = None
        self._ts_table = None
        self._setup_run_calibconst()

    @contextmanager
    def build_table(self):
        """
        Build timestamp→(fd offset, size) table by scanning SMD.
        Updates EnvStore for any non-L1 transitions encountered during the scan.
        Yields True if any L1 timestamps were captured.
        """
        if self._smd_iter is None:
            # Now yields dgram lists
            self._smd_iter = SmdEvents(self.configs,
                                       self.dm,
                                       self.dsparms.max_retries,
                                       self.dsparms.use_smds,
                                       self.shared_state,
                                       smdr_man=self.smdr_man)

        self._ts_table = {}
        try:
            for dgrams in self._smd_iter:
                svc = utils.first_service(dgrams)
                if svc != TransitionId.L1Accept:
                    # Keep EnvStore in sync for transitions observed during the scan
                    self._handle_transition(dgrams)
                    if svc == TransitionId.EndRun:
                        break
                    continue

                ts = utils.first_timestamp(dgrams)
                self._ts_table[ts] = {
                    i: (d.smdinfo[0].offsetAlg.intOffset, d.smdinfo[0].offsetAlg.intDgramSize)
                    for i, d in enumerate(dgrams)
                    if d is not None and hasattr(d, 'smdinfo')
                }
            yield bool(self._ts_table)
        finally:
            # Optional: if this iterator is one-shot, drop the reference
            # to avoid keeping large buffers alive
            # self._smd_iter = None
            pass

    def event(self, ts):
        if self._ts_table is None:
            raise RuntimeError("build_table() must be called before event(ts)")

        offsets = self._ts_table.get(ts)
        if offsets is None:
            raise ValueError(f"Timestamp {ts} not found in offset table.")

        dgrams = [None] * len(self.configs)
        for i, (offset, size) in offsets.items():
            buf = os.pread(self.dm.fds[i], size, offset)
            dgrams[i] = dgram.Dgram(config=self.dm.configs[i], view=buf)

        return Event(dgrams=dgrams, run=self._run_ctx)


class RunSmallData(Run):
    """Yields list of smalldata events

    This class is created by SmdReaderManager and used exclusively by EventBuilder.
    There's no DataSource class associated with it a "bigdata" run is passed in
    so that we can propagate some values (w/o the ds). This class makes step and
    event generator available to user in smalldata callback.
    """

    def __init__(self, eb, configs, dsparms):
        self.eb = eb
        self.dsparms = dsparms
        if not hasattr(self.dsparms, "calibconst") or self.dsparms.calibconst is None:
            self.dsparms.calibconst = {}
        self._calib_const = self.dsparms.calibconst
        self._run_ctx = None  # No RunCtx for smalldata

        # Converts EventBuilder generator to an iterator for steps() call. This is
        # done so that we can pass it to Step (not sure why). Note that for
        # events() call, we stil use the generator.
        self._evt_iter = iter(self.eb.events())

        # SmdDataSource and BatchIterator share this list. SmdDataSource automatically
        # adds transitions to this list (skip yield and so hidden from smd_callback).
        # BatchIterator adds user-selected L1Accept to the list (default is add all).
        self.proxy_events = []

        self.esm = EnvStoreManager(configs)

    def steps(self):
        for (dgrams, proxy_evt)  in self._evt_iter:
            svc = utils.first_service(dgrams)
            if not TransitionId.isEvent(svc):
                self._update_envstore_from_dgrams(dgrams)
                self.proxy_events.append(proxy_evt)
                if svc == TransitionId.EndRun:
                    return
                if svc == TransitionId.BeginStep:
                    yield Step(
                        Event(dgrams=dgrams, run=self._run_ctx),
                        self._evt_iter,
                        self._run_ctx,
                        proxy_events=self.proxy_events,
                        esm=self.esm,
                    )

    def events(self):
        for (dgrams, proxy_evt) in self.eb.events():
            svc = utils.first_service(dgrams)
            if not TransitionId.isEvent(svc):
                self._update_envstore_from_dgrams(dgrams)
                self.proxy_events.append(proxy_evt)
                if svc == TransitionId.EndRun:
                    return
                continue
            yield Event(dgrams=dgrams, run=self._run_ctx, proxy_evt=proxy_evt)
