import os
import sys
import time
import resource
try:
    import psutil
except Exception:
    psutil = None
import math

import numpy as np
from contextlib import contextmanager

from psana import dgram
from psana import utils
from psana.dgrammanager import DgramManager
from psana.event import Event
from psana.psexp import TransitionId
from psana.psexp.ds_base import DataSourceBase
from psana.psexp.node import (
    BigDataNode,
    EventBuilderNode,
    MarchingBigDataNode,
    MarchingEventBuilderNode,
    Smd0,
)
from psana.psexp.run import Run
from psana.psexp.calib_xtc import load_calib_xtc_from_buffer
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.step import Step
from psana.psexp.tools import mode, get_smd_n_events
from psana.psexp.mpi_shmem import MPISharedMemory
from psana.detector.shared_geo_cache import SharedGeoCache
from psana.detector.shared_calibc_cache import SharedCalibcCache
from psana.smalldata import SmallData
from psana.psexp.prometheus_manager import get_prom_manager
from psana.psexp.calib_xtc import CalibXtcConverter

if mode == "mpi":
    from mpi4py import MPI


class InvalidEventBuilderCores(Exception):
    pass


nodetype = None


class RunParallel(Run):
    """Yields list of events from multiple smd/bigdata files using > 3 cores."""
    def __init__(self, expt, runnum, timestamp, dsparms, dm, smdr_man, configs, begingrun_dgrams, comms=None):
        super(RunParallel, self).__init__(expt, runnum, timestamp, dsparms, dm, smdr_man, begingrun_dgrams)
        self.configs = configs
        self.comms = comms

        self.logger = utils.get_logger(name=utils.get_class_name(self))

        super()._setup_envstore()

        self._init_marching_shared_buffers()

        marching_enabled = (
            self.dsparms.marching_read
            and getattr(self.comms, "marching_enabled", False)
            and getattr(self.comms, "march_shared_mem", None) is not None
        )

        if nodetype == "smd0":
            self.smd0 = Smd0(comms, smdr_man, configs)
        elif nodetype == "eb":
            if marching_enabled:
                self.eb_node = MarchingEventBuilderNode(comms, configs, dsparms)
            else:
                self.eb_node = EventBuilderNode(comms, configs, dsparms)
        elif nodetype == "bd":
            if marching_enabled:
                self.bd_node = MarchingBigDataNode(
                    comms, configs, dm, dsparms, self.shared_state
                )
            else:
                self.bd_node = BigDataNode(
                    comms, configs, dm, dsparms, self.shared_state
                )
            self.ana_t_gauge = get_prom_manager().get_metric("psana_bd_ana_rate")

        self._setup_run_calibconst()

    def _setup_run_calibconst(self):
        if nodetype == "smd0":
            super()._setup_run_calibconst()
            self.build_xtc_buffer(self.get_filtered_detinfo())
            if getattr(self, "_calib_xtc_buffer", None) is None:
                has_entries = bool(self._calib_const)
                has_nonempty = any(self._calib_const.values()) if has_entries else False
                if has_nonempty:
                    raise RuntimeError("Failed to build calibration xtc buffer on smd0")
                if self.logger:
                    self.logger.info("calibconst empty on smd0; skipping shared calibration buffer")
                self._calib_xtc_buffer = b""
        else:
            self._clear_calibconst()

        self._distribute_calib_xtc()
        self._setup_jungfrau_shared_calib()
        self._setup_jungfrau_shared_geometry_cache()

    def build_xtc_buffer(self, det_info):
        if not self._calib_const:
            self._calib_xtc_buffer = None
            return
        det_info = det_info or {}
        try:
            t0 = time.perf_counter()
            converter = CalibXtcConverter(det_info)
            buffer_view, config_size, data_size = converter.convert_to_buffer(self._calib_const)
            total_size = config_size + data_size
            self._calib_xtc_buffer = buffer_view
            elapsed = time.perf_counter() - t0
            size_gib = total_size / (1024 ** 3)
            self.logger.debug(
                "Built calibration xtc buffer from pickle size=%.3f GiB time=%.3fs",
                size_gib,
                elapsed,
            )
        except Exception as exc:
            self.logger.warning(f"Failed to build calibration xtc buffer: {exc}")

    def _distribute_calib_xtc(self):
        node_comm = self.comms.get_node_comm()
        leader_comm = self.comms.get_node_leader_comm()
        if node_comm is None:
            raise RuntimeError("Node communicator unavailable for shared calibration distribution")

        is_leader = self.comms.is_node_leader()
        leader_buffer = None
        total_bytes = None
        recv_start = None
        recv_end = None
        populate_start = None
        populate_end = None

        if leader_comm != MPI.COMM_NULL:
            size_arr = np.array([0], dtype=np.int64) if is_leader else np.empty(1, dtype=np.int64)
            if is_leader and nodetype == "smd0":
                leader_buffer = np.frombuffer(self._calib_xtc_buffer, dtype=np.uint8)
                size_arr[0] = leader_buffer.size

            leader_comm.Bcast(size_arr, root=0)
            total_bytes = int(size_arr[0])

        total_bytes = node_comm.bcast(total_bytes, root=0)
        if total_bytes <= 0:
            self._calib_const = {}
            self.dsparms.calibconst = self._calib_const
            return

        if leader_comm != MPI.COMM_NULL and is_leader:
            if nodetype != "smd0":
                leader_buffer = np.empty(total_bytes, dtype=np.uint8)
            elif leader_buffer.size != total_bytes:
                leader_buffer = leader_buffer[:total_bytes]
            recv_start = time.perf_counter()
            leader_comm.Bcast(leader_buffer, root=0)
            recv_end = time.perf_counter()

        win = MPI.Win.Allocate_shared(total_bytes if is_leader else 0, 1, comm=node_comm)
        buf, _ = win.Shared_query(0)
        shared_array = np.ndarray(buffer=buf, dtype=np.uint8, shape=(total_bytes,))
        if is_leader:
            if leader_buffer is None:
                raise RuntimeError("Leader buffer missing during shared memory population")
            populate_start = time.perf_counter()
            shared_array[:] = leader_buffer
            populate_end = time.perf_counter()
        node_comm.Barrier()

        if self.logger and is_leader and recv_start is not None and populate_end is not None:
            recv_time = (recv_end - recv_start) if recv_end is not None else 0.0
            populate_time = populate_end - populate_start
            total_time = populate_end - recv_start
            size_gib = total_bytes / (1024 ** 3)
            self.logger.debug(
                "node leader rank %d shared calib size=%.3f GiB "
                "recv=%.3fs populate=%.3fs total=%.3fs",
                self.comms.psana_comm.Get_rank(),
                size_gib,
                recv_time,
                populate_time,
                total_time,
            )

        shared_view = memoryview(shared_array)
        calib_const, owner = load_calib_xtc_from_buffer(shared_view)
        self._calib_const = calib_const
        self._calib_xtc_buffer = owner
        self._calib_xtc_shared = shared_array
        self._calib_xtc_win = win
        self.dsparms.calibconst = self._calib_const

    def _setup_jungfrau_shared_calib(self):
        flag = os.environ.get("PS_JF_SHARE_DERIVED", "1").strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return
        if mode != "mpi" or self.comms is None:
            return
        node_comm = self.comms.get_node_comm()
        if node_comm in (None, MPI.COMM_NULL):
            return

        if getattr(self, "_jungfrau_shared_mem", None) is None:
            self._jungfrau_shared_mem = MPISharedMemory(shm_comm=node_comm)
        shared_mem = self._jungfrau_shared_mem

        try:
            import psana.detector.UtilsJungfrau as uj
        except Exception:
            self.logger.debug("Failed to import UtilsJungfrau for shared calib setup", exc_info=True)
            return

        for det_name, drp_class_name, drp_class, configinfo, calibconst in self._iter_jungfrau_raw():

            try:
                t_build_start = time.perf_counter()
                iface = drp_class(
                    det_name,
                    drp_class_name,
                    configinfo,
                    calibconst,
                    None,
                    None,
                )
                shared = uj.build_shared_jungfrau_calib(
                    iface,
                    shared_mem,
                    runnum=self.runnum,
                )
                if shared:
                    mask_shared = uj.build_shared_jungfrau_mask(
                        iface,
                        shared_mem,
                        runnum=self.runnum,
                    )
                    if mask_shared:
                        shared.update(mask_shared)
                t_build_end = time.perf_counter()
            except Exception as exc:
                self.logger.debug(
                    "Failed to build shared Jungfrau calib for %s: %s",
                    det_name,
                    exc,
                    exc_info=True,
                )
                shared = None

            if shared:
                setattr(iface, "_jf_shared", shared)
                if shared_mem.is_leader:
                    self.logger.debug(
                        "Jungfrau shared calib ready det=%s total=%.3fs",
                        det_name,
                        t_build_end - t_build_start,
                    )

    def _setup_jungfrau_shared_geometry_cache(self):
        flag = os.environ.get("PS_GEO_SHARE", "1").strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return
        if mode != "mpi" or self.comms is None:
            return
        node_comm = self.comms.get_node_comm()
        if node_comm in (None, MPI.COMM_NULL):
            return

        if getattr(self, "_geo_shared_mem", None) is None:
            self._geo_shared_mem = MPISharedMemory(shm_comm=node_comm)
        shared_mem = self._geo_shared_mem
        cache = SharedGeoCache(shared_mem=shared_mem, logger=self.logger)
        self._shared_geo_cache = cache
        calibc_cache = SharedCalibcCache(shared_mem=shared_mem, logger=self.logger)
        self._shared_calibc_cache = calibc_cache
        rank = self.comms.psana_comm.Get_rank() if self.comms is not None else -1
        t_setup_start = time.perf_counter()

        for det_name, drp_class_name, drp_class, configinfo, calibconst in self._iter_jungfrau_raw(area_only=True):

            try:
                iface = drp_class(
                    det_name,
                    drp_class_name,
                    configinfo,
                    calibconst,
                    None,
                    None,
                )
            except Exception as exc:
                self.logger.debug(
                    "Failed to init area detector for shared geometry %s: %s",
                    det_name,
                    exc,
                    exc_info=True,
                )
                continue

            setattr(iface, "_shared_geo_cache", cache)
            setattr(iface, "_shared_calibc_cache", calibc_cache)
            try:
                t0 = time.perf_counter()
                iface._pixel_coords()
                t_coords = time.perf_counter() - t0
                t0 = time.perf_counter()
                iface._pixel_coord_indexes()
                t_indexes = time.perf_counter() - t0
                t_cached = 0.0
                cache_payload = None
                if shared_mem.is_leader:
                    cc = iface._calibconstants()
                    if cc is not None:
                        t0 = time.perf_counter()
                        cache_payload = cc.seed_cached_pixel_coord_indexes_shared(
                            segnums=iface._segment_numbers
                        )
                        if cache_payload is not None:
                            t_cached = time.perf_counter() - t0
                meta_specs = None
                if cache_payload is not None:
                    meta_specs = (cache_payload["meta"], cache_payload["specs"])
                shm_comm = getattr(shared_mem, "shm_comm", None)
                if shm_comm is not None:
                    meta_specs = shm_comm.bcast(meta_specs, root=0)
                if meta_specs is not None:
                    meta, specs = meta_specs
                    if meta is not None and specs is not None:
                        key = calibc_cache.make_key(*meta)
                        shared_arrays = {}
                        for name, (shape, dtype_str) in specs.items():
                            arr, _ = calibc_cache.get_or_allocate(
                                key, name, shape, np.dtype(dtype_str)
                            )
                            shared_arrays[name] = arr
                        if shared_mem.is_leader and cache_payload is not None:
                            for name, arr in cache_payload["arrays"].items():
                                shared_arrays[name][:] = arr
                        calibc_cache.barrier()
                self.logger.debug(
                    f"shared geo det={det_name} "
                    f"pixel_coords={t_coords:.6f}s "
                    f"pixel_indexes={t_indexes:.6f}s "
                    f"cached_indexes={t_cached:.6f}s"
                )
            except Exception as exc:
                self.logger.debug(
                    "Failed to seed shared geometry for %s: %s",
                    det_name,
                    exc,
                    exc_info=True,
                )
            if hasattr(shared_mem, "barrier"):
                shared_mem.barrier()

        t_setup_end = time.perf_counter()
        self.logger.debug(
            f"shared geo setup total={t_setup_end - t_setup_start:.6f}s"
        )

    def _iter_jungfrau_raw(self, area_only=False):
        det_classes = self.dsparms.det_classes.get("normal", {})
        targets = []
        for (det_name, drp_class_name), drp_class in det_classes.items():
            if drp_class_name != "raw":
                continue
            mod_name = getattr(drp_class, "__module__", "")
            class_name = getattr(drp_class, "__name__", "").lower()
            if "jungfrau" not in mod_name and "jungfrau" not in class_name:
                continue
            if area_only:
                try:
                    from psana.detector.areadetector import AreaDetector
                    is_area = issubclass(drp_class, AreaDetector)
                except Exception:
                    is_area = False
                if not is_area:
                    continue
            configinfo = self.dsparms.configinfo_dict.get(det_name)
            if configinfo is None:
                continue
            calibconst = self.dsparms.calibconst.get(det_name)
            if calibconst is None:
                continue
            targets.append((det_name, drp_class_name, drp_class, configinfo, calibconst))
        targets.sort(key=lambda item: (item[0], item[1]))
        return targets

    def _init_marching_shared_buffers(self):
        if not (
            self.dsparms.marching_read
            and getattr(self.comms, "marching_enabled", False)
            and getattr(self.comms, "march_shm_comm", None) not in (None, MPI.COMM_NULL)
        ):
            return
        if getattr(self.comms, "march_shared_mem", None) is not None:
            return
        shared_mem = MPISharedMemory(shm_comm=self.comms.march_shm_comm)
        params = self._marching_params()
        slot_shape = (params["n_slots"], params["max_events"], len(self.configs))
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_bd_offsets", slot_shape, np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_bd_sizes", slot_shape, np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_smd_offsets", slot_shape, np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_smd_sizes", slot_shape, np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_services", slot_shape, np.int32)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_cutoff_flags", slot_shape, np.int8)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_new_chunk_ids", slot_shape, np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_slot_events", (params["n_slots"],), np.int32)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_slot_chunk_ids", (params["n_slots"],), np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_slot_states", (params["n_slots"],), np.int32)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_slot_next_evt", (params["n_slots"],), np.int64)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_slot_consumers_done", (params["n_slots"],), np.int32)
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_chunk_sizes", (params["n_slots"],), np.int32)
        self._alloc_marching_array(
            shared_mem,
            f"{params['prefix']}_chunk_bytes",
            (params["n_slots"], params["max_chunk_bytes"]),
            np.uint8,
        )
        self._alloc_marching_array(shared_mem, f"{params['prefix']}_shutdown", (1,), np.int32)
        n_consumers = max(getattr(self.comms, "march_shm_size", 1) - 1, 1)
        self._alloc_marching_array(
            shared_mem,
            f"{params['prefix']}_slot_reader_counts",
            (params["n_slots"], n_consumers),
            np.int32,
        )
        self._alloc_marching_array(
            shared_mem,
            f"{params['prefix']}_slot_start_times",
            (params["n_slots"],),
            np.float64,
        )
        params["n_consumers"] = n_consumers
        self.comms.march_shared_mem = shared_mem
        self.comms.march_params = params

    def _alloc_marching_array(self, shared_mem, name, shape, dtype):
        if not shared_mem.has_array(name):
            shared_mem.allocate_array(name, shape, dtype)

    def _marching_params(self):
        prefix = os.environ.get("PS_MARCH_PREFIX", "march")
        env_slots = os.environ.get("PS_MARCH_SLOTS")
        march_nodes = getattr(self.comms, "n_smd_nodes", 1)
        if march_nodes <= 0:
            march_nodes = 1
        if env_slots is None:
            requested_slots = march_nodes
        else:
            try:
                requested_slots = int(env_slots)
            except ValueError:
                self.logger.warning(
                    "Invalid PS_MARCH_SLOTS=%r; defaulting to %d", env_slots, march_nodes
                )
                requested_slots = march_nodes
        requested_slots = max(requested_slots, 1)
        n_slots = min(requested_slots, march_nodes)
        if env_slots is not None and n_slots < requested_slots:
            self.logger.warning(
                "PS_MARCH_SLOTS=%d exceeds available marching EB nodes (%d); using %d slots instead",
                requested_slots,
                march_nodes,
                n_slots,
            )
        smd_n_events = get_smd_n_events()
        try:
            events_scale = float(os.environ.get("PS_MARCH_EVENTS_SCALE", "1.2"))
        except ValueError:
            events_scale = 1.2
        if events_scale <= 1.0:
            events_scale = 1.2
        max_events = max(int(math.ceil(smd_n_events * events_scale)), 1)
        max_chunk_bytes = int(os.environ.get("PS_MARCH_MAX_CHUNK_MB", "32"))
        max_chunk_bytes = max_chunk_bytes * 1024 * 1024
        if nodetype == "eb":
            self.logger.debug(
                "Marching EB params: n_slots=%d max_events=%d max_chunk_bytes=%d",
                n_slots,
                max_events,
                max_chunk_bytes,
            )
        return {
            "prefix": prefix,
            "n_slots": n_slots,
            "max_events": max_events,
            "max_chunk_bytes": max_chunk_bytes,
        }

    def events(self):
        evt_iter = self.start()
        st = time.time()
        try:
            ana_interval = int(os.environ.get("PS_BD_ANA_INTERVAL", "1000"))
        except ValueError:
            ana_interval = 1000
        if ana_interval <= 0:
            ana_interval = 1000
        for i, dgrams in enumerate(evt_iter):
            if self._handle_transition(dgrams):
                continue  # swallow non-L1 transitions in events() stream
            # L1Accept: construct Event at the Run level
            yield Event(dgrams=dgrams, run=self._run_ctx)
            if i % ana_interval == 0:
                en = time.time()
                interval = en - st
                ana_rate = ana_interval / interval if interval > 0 else 0.0
                rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_cur_mb = -1.0
                if psutil is not None:
                    try:
                        rss_cur_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
                    except Exception:
                        rss_cur_mb = -1.0
                self.logger.debug(
                    "bd analysis stats rate_hz=%.2f interval_s=%.2f events=%d rss_kb=%d rss_cur_mb=%.2f",
                    ana_rate,
                    interval,
                    ana_interval,
                    rss_kb,
                    rss_cur_mb,
                )
                self.ana_t_gauge.set(ana_rate)
                st = time.time()

    def steps(self):
        evt_iter = self.start()
        for dgrams in evt_iter:
            svc = utils.first_service(dgrams)
            if TransitionId.isEvent(svc):
                # steps() only yields on BeginStep transitions; ignore L1
                continue
            # Update envstore for every transition
            self._update_envstore_from_dgrams(dgrams)

            if svc == TransitionId.BeginStep:
                yield Step(
                    Event(dgrams=dgrams, run=self._run_ctx),
                    evt_iter,
                    self._run_ctx,
                    esm=self.esm,
                )

    def start(self):
        """Request data for this run"""
        if nodetype == "smd0":
            self.smd0.start()
        elif nodetype == "eb":
            self.eb_node.start()
        elif nodetype == "bd":
            yield from self.bd_node.start()
        elif nodetype == "srv":
            return

    @contextmanager
    def build_table(self):
        """
        Context manager for building timestamp-offset table.
        Returns True only on BigDataNode if the table was successfully built.

        Requires PS_EB_NODES=1 for broadcast mode.
        """
        if self.dsparms.marching_read:
            raise RuntimeError("build_table() is not supported in marching mode")
        if os.environ.get("PS_EB_NODES", "1") != "1":
            raise RuntimeError("build_table() currently supports only PS_EB_NODES=1")

        success = False
        if nodetype == "smd0":
            self.smd0.start()
        elif nodetype == "eb":
            self.eb_node.start_broadcast()
        elif nodetype == "bd":
            self._ts_table = self.bd_node.start_smdonly()
            success = bool(self._ts_table)
        yield success

    def event(self, ts):
        offsets = self._ts_table.get(ts)
        if offsets is None:
            raise ValueError(f"Timestamp {ts} not found in offset table.")

        dgrams = [None] * len(self.configs)
        for i, (offset, size) in offsets.items():
            buf = os.pread(self.dm.fds[i], size, offset)
            dgrams[i] = dgram.Dgram(config=self.dm.configs[i], view=buf)

        return Event(dgrams=dgrams, run=self)

    def terminate(self):
        self.comms.terminate()
        super().terminate()


def safe_mpi_abort(msg):
    print(msg)
    sys.stdout.flush()  # make sure error is printed
    MPI.COMM_WORLD.Abort()


class MPIDataSource(DataSourceBase):
    def __init__(self, comms, *args, **kwargs):
        # Check if an I/O-friendly numpy file storing timestamps is given by the user
        if "timestamps" in kwargs:
            if isinstance(kwargs["timestamps"], str):
                kwargs["mpi_ts"] = 1

        # Initialize base class
        super(MPIDataSource, self).__init__(**kwargs)
        self.smd_fds = None

        # Set up the MPI communication
        self.comms = comms
        comm = self.comms.psana_comm  # todo could be better
        rank = comm.Get_rank()
        size = comm.Get_size()
        global nodetype
        nodetype = self.comms.node_type()

        # prepare comms for running SmallData
        PS_SRV_NODES = int(os.environ.get("PS_SRV_NODES", 0))
        if PS_SRV_NODES > 0:
            self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        else:
            self.smalldata_obj = None

        # check if no. of ranks is enough
        nsmds = int(os.environ.get("PS_EB_NODES", 1))  # No. of smd cores
        if not (size > (nsmds + 1)):
            msg = f"""ERROR Too few MPI processes. MPI size must be more than
                   no. of all workers.
                  \n\tTotal psana size:{size}
                  \n\tPS_EB_NODES:     {nsmds}"""
            safe_mpi_abort(msg)

        # Load timestamp files on EventBuilder Node
        if "mpi_ts" in kwargs and nodetype == "eb":
            self.dsparms.timestamps = self.get_filter_timestamps(self.timestamps)

        # setup runnum list
        if nodetype == "smd0":
            super()._setup_runnum_list()
        else:
            self.runnum_list = None
            self.xtc_path = None
        self.runnum_list = comm.bcast(self.runnum_list, root=0)
        self.xtc_path = comm.bcast(self.xtc_path, root=0)
        self.runnum_list_index = 0

        self._start_prometheus_client(mpi_rank=rank)
        self._setup_run()

    def __del__(self):
        if nodetype == "smd0":
            super()._close_opened_smd_files()
        self._end_prometheus_client()

    def _get_configs(self):
        """Creates and broadcasts configs
        only called by _setup_run()
        """
        if nodetype == "smd0":
            super()._close_opened_smd_files()
            self.smd_fds = np.array(
                [os.open(smd_file, os.O_RDONLY) for smd_file in self.smd_files],
                dtype=np.int32,
            )
            self.logger.debug(f"smd0 opened smd_fds: {self.smd_fds}")
            self.smdr_man = SmdReaderManager(self.smd_fds, self.dsparms)
            configs = self.smdr_man.get_next_dgrams()
            nbytes = np.array(
                [memoryview(config).shape[0] for config in configs], dtype="i"
            )
        else:
            self.smdr_man = None  # Only smd0 uses SmdReaderManager
            configs = None
            nbytes = np.empty(len(self.smd_files), dtype="i")

        self.comms.psana_comm.Bcast(
            nbytes, root=0
        )  # no. of bytes is required for mpich
        if nodetype != "smd0":
            configs = [np.empty(nbyte, dtype="b") for nbyte in nbytes]

        for i in range(len(configs)):
            self.comms.psana_comm.Bcast([configs[i], nbytes[i], MPI.BYTE], root=0)

        if nodetype != "smd0":
            configs = [dgram.Dgram(view=config, offset=0) for config in configs]
        return configs

    def _setup_run(self):
        if self.runnum_list_index == len(self.runnum_list):
            return False

        runnum = self.runnum_list[self.runnum_list_index]
        self.runnum_list_index += 1

        if nodetype == "smd0":
            super()._setup_run_files(runnum)
            super()._apply_detector_selection()
        else:
            self.xtc_files = None
            self.smd_files = None
            self.dsparms.use_smds = None
        self.xtc_files = self.comms.psana_comm.bcast(self.xtc_files, root=0)
        self.smd_files = self.comms.psana_comm.bcast(self.smd_files, root=0)
        self.dsparms.use_smds = self.comms.psana_comm.bcast(
            self.dsparms.use_smds, root=0
        )

        configs = self._get_configs()
        self.dm = DgramManager(
            self.xtc_files, configs=configs, config_consumers=[self.dsparms]
        )
        return True

    def _setup_beginruns(self):
        """Determines if there is a next run as
        1) New run found in the same smalldata files
        2) New run found in the new smalldata files
        """
        while True:
            if nodetype == "smd0":
                dgrams = self.smdr_man.get_next_dgrams()
                nbytes = np.zeros(len(self.smd_files), dtype="i")
                if dgrams is not None:
                    nbytes = np.array(
                        [memoryview(d).shape[0] for d in dgrams], dtype="i"
                    )
            else:
                dgrams = None
                nbytes = np.empty(len(self.smd_files), dtype="i")

            self.comms.psana_comm.Bcast(nbytes, root=0)

            if np.sum(nbytes) == 0:
                return False

            if nodetype != "smd0":
                dgrams = [np.empty(nbyte, dtype="b") for nbyte in nbytes]

            for i in range(len(dgrams)):
                self.comms.psana_comm.Bcast([dgrams[i], nbytes[i], MPI.BYTE], root=0)

            if nodetype != "smd0":
                dgrams = [
                    dgram.Dgram(view=d, config=config, offset=0)
                    for d, config in zip(dgrams, self._configs)
                ]

            if dgrams[0].service() == TransitionId.BeginRun:
                self.beginruns = dgrams
                return True
        # end while True

    def _start_run(self):
        if self._setup_beginruns():  # try to get next run from current files
            return True
        elif self._setup_run():  # try to get next run from next files
            if self._setup_beginruns():
                return True

    def runs(self):
        while self._start_run():
            # Pull (expt, runnum, ts) from the BeginRun dgrams
            expt, runnum, ts = self._get_runinfo()
            run = RunParallel(
                expt,                 # experiment string
                runnum,               # run number (int)
                ts,                   # begin-run timestamp
                self.dsparms,         # shared parameters / config tables
                self.dm,              # DgramManager
                self.smdr_man,        # SmdReaderManager (may be None for non-SMD modes)
                self._configs,        # configs for this run
                self.beginruns,       # beginrun dgrams
                comms=self.comms,     # MPI communications
            )
            yield run

    def is_mpi(self):
        return True
