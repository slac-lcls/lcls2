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
from psana.psexp.node import BigDataNode, EventBuilderNode, Smd0
from psana.psexp.run import Run
from psana.psexp.calib_xtc import load_calib_xtc_from_buffer
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.step import Step
from psana.psexp.tools import mode
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
    def __init__(
        self,
        expt,
        runnum,
        timestamp,
        dsparms,
        dm,
        smdr_man,
        configs,
        begingrun_dgrams,
        comms=None,
    ):
        super(RunParallel, self).__init__(expt, runnum, timestamp, dsparms, dm, smdr_man, begingrun_dgrams)
        self.configs = configs
        self.comms = comms
        self._shared_mem_closed = False

        self.logger = utils.get_logger(name=utils.get_class_name(self))

        super()._setup_envstore()

        if nodetype == "smd0":
            self.smd0 = Smd0(comms, smdr_man, configs)
        elif nodetype == "eb":
            self.eb_node = EventBuilderNode(comms, configs, dsparms)
        elif nodetype == "bd":
            self.bd_node = BigDataNode(comms, configs, dm, dsparms, self.shared_state)
            self.ana_t_gauge = get_prom_manager().get_metric("psana_bd_ana_rate")

        self._setup_run_calibconst()

    def _close_shared_memory(self):
        if self._shared_mem_closed:
            return
        self._shared_mem_closed = True

        # Release calibration xtc shared window if present.
        win = getattr(self, "_calib_xtc_win", None)
        if win is not None:
            try:
                win.Free()
            except Exception:
                self.logger.debug("Failed to free calib xtc shared window", exc_info=True)
            self._calib_xtc_win = None
        self._calib_xtc_shared = None
        self._calib_xtc_buffer = None

        # Release MPISharedMemory helpers.
        for attr in ("_jungfrau_shared_mem", "_geo_shared_mem"):
            helper = getattr(self, attr, None)
            if helper is not None:
                try:
                    helper.close()
                except Exception:
                    self.logger.debug("Failed to close shared memory helper %s", attr, exc_info=True)
                setattr(self, attr, None)

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

        from psana.gpu.gpu_mpi import log_gpu_mem
        _r = self.comms.psana_comm.Get_rank() if self.comms is not None else None
        log_gpu_mem('before _distribute_calib_xtc', rank=_r)
        self._distribute_calib_xtc()
        log_gpu_mem('after  _distribute_calib_xtc', rank=_r)
        self._setup_jungfrau_shared_calib()
        log_gpu_mem('after  _setup_jungfrau_shared_calib', rank=_r)
        self._setup_jungfrau_shared_caches()
        log_gpu_mem('after  _setup_jungfrau_shared_caches', rank=_r)
        # _pixel_coord_indexes(all_segs=True) was seeded in
        # _setup_jungfrau_shared_caches() above, so the call here is a
        # cheap cache read with no shmem barrier — safe for all N_BD_PER_GPU.
        self._setup_gpu_geometry()
        log_gpu_mem('after  _setup_gpu_geometry', rank=_r)

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
        shmem_comm = self.comms.get_shmem_comm()
        leader_comm = self.comms.get_node_leader_comm()
        if node_comm is None:
            raise RuntimeError("Node communicator unavailable for shared calibration distribution")
        if shmem_comm is None:
            raise RuntimeError("Shared-memory communicator unavailable for shared calibration distribution")

        is_leader = self.comms.is_node_leader()
        is_shmem_leader = shmem_comm != MPI.COMM_NULL and shmem_comm.Get_rank() == 0
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

        if node_comm != MPI.COMM_NULL:
            if leader_buffer is None:
                leader_buffer = np.empty(total_bytes, dtype=np.uint8)
            node_comm.Bcast(leader_buffer, root=0)

        win = MPI.Win.Allocate_shared(total_bytes if is_shmem_leader else 0, 1, comm=shmem_comm)
        buf, _ = win.Shared_query(0)
        shared_array = np.ndarray(buffer=buf, dtype=np.uint8, shape=(total_bytes,))
        if is_shmem_leader:
            if leader_buffer is None:
                raise RuntimeError("Leader buffer missing during shared memory population")
            populate_start = time.perf_counter()
            shared_array[:] = leader_buffer
            populate_end = time.perf_counter()
        shmem_comm.Barrier()

        if self.logger and is_shmem_leader and populate_end is not None:
            recv_time = (recv_end - recv_start) if recv_start is not None and recv_end is not None else 0.0
            populate_time = populate_end - populate_start
            total_time = populate_end - recv_start if recv_start is not None else populate_time
            size_gib = total_bytes / (1024 ** 3)
            self.logger.debug(
                "shmem leader rank %d shared calib size=%.3f GiB "
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
        shmem_comm = self.comms.get_shmem_comm()
        if shmem_comm in (None, MPI.COMM_NULL):
            return

        if getattr(self, "_jungfrau_shared_mem", None) is None:
            self._jungfrau_shared_mem = MPISharedMemory(shm_comm=shmem_comm)
        shared_mem = self._jungfrau_shared_mem

        try:
            import psana.detector.UtilsJungfrau as uj
        except Exception:
            self.logger.debug("Failed to import UtilsJungfrau for shared calib setup", exc_info=True)
            return

        for det_name, drp_class_name, drp_class, configinfo, calibconst in self._iter_jungfrau_raw(area_only=False):

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

    def _setup_jungfrau_shared_caches(self):
        flag = os.environ.get("PS_GEO_SHARE", "1").strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return
        if mode != "mpi" or self.comms is None:
            return
        shmem_comm = self.comms.get_shmem_comm()
        if shmem_comm in (None, MPI.COMM_NULL):
            return

        if getattr(self, "_geo_shared_mem", None) is None:
            self._geo_shared_mem = MPISharedMemory(shm_comm=shmem_comm)
        shared_mem = self._geo_shared_mem
        # SharedGeoCache: shared memory for geometry/pixel index arrays.
        cache = SharedGeoCache(shared_mem=shared_mem, logger=self.logger)
        self._shared_geo_cache = cache
        # SharedCalibcCache: shared memory for CalibConstants-derived image mapping arrays.
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
                # Also seed the all_segs=True variant used by _setup_gpu_geometry().
                # Without this, _pixel_coord_indexes(all_segs=True) would be a
                # cache miss and trigger a new shmem collective.  That collective
                # deadlocks when smd0/EB are in the same NUMA group as BD ranks
                # (N_BD_PER_GPU > 1, numa_size=3) because smd0/EB exit via
                # try/except before completing the barrier.  Seeding here while
                # all ranks are already participating in the same collective
                # makes the later call a cheap cache read with no barrier.
                iface._pixel_coord_indexes(all_segs=True)
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
        if self.logger and shared_mem.is_leader:
            self.logger.info(
                "[MPI-role] rank=%d host=%s numa_id=%d numa_size=%d",
                self.comms.psana_comm.Get_rank() if self.comms is not None else -1,
                getattr(self.comms, "hostname", "unknown"),
                getattr(self.comms, "numa_id", -1),
                getattr(self.comms, "numa_size", -1),
            )

    def _setup_gpu_geometry(self):
        """Pre-compute GPU scatter indices for GPU BD ranks.

        Called from _setup_run_calibconst() immediately after
        _setup_jungfrau_shared_caches(), while all ranks are still
        synchronising in RunParallel.__init__().

        At this point _setup_jungfrau_shared_caches() has already computed
        pixel coordinate indices and stored them in the SharedGeoCache
        shared-memory region.  We attach that cache to a fresh detector
        interface and call _pixel_coord_indexes(all_segs=True) to read the
        pre-computed arrays without triggering any additional collective.

        The resulting numpy arrays are stored as
            self._gpu_geometry_arrays = {det_name: (ix_all, iy_all)}
        and passed to GpuEvents(prebuilt_geometry=...) so that
        _setup_detectors() calls gpu_detector.setup_geometry_from_arrays()
        instead of setup_geometry(det), avoiding a shmem collective during
        the event loop.

        Non-BD ranks and runs without gpu_det set are a no-op.
        """
        # Only needed when gpu_det is set — skip entirely for CPU-only runs.
        # _pixel_coord_indexes(all_segs=True) is slow (~seconds for large
        # detectors) and adds unnecessary overhead to CPU production jobs.
        if not getattr(self.dsparms, "gpu_det", None):
            return

        # _pixel_coord_indexes(all_segs=True) was seeded in
        # _setup_jungfrau_shared_caches() above — this call is a cache hit
        # (no shmem barrier).  Only GPU BD ranks store the result.
        cache = getattr(self, "_shared_geo_cache", None)
        if cache is None:
            return

        is_gpu_bd = (
            nodetype == "bd"
            and bool(getattr(self.dsparms, "gpu_det", None))
        )
        gpu_det_names = self.dsparms.gpu_det if is_gpu_bd else []
        if isinstance(gpu_det_names, str):
            gpu_det_names = [gpu_det_names]
        gpu_det_set = set(gpu_det_names)

        calibc_cache = getattr(self, "_shared_calibc_cache", None)
        if is_gpu_bd:
            self._gpu_geometry_arrays = {}

        for det_name, drp_class_name, drp_class, configinfo, calibconst in \
                self._iter_area_detector_raw(area_only=True):
            try:
                iface = drp_class(det_name, drp_class_name,
                                   configinfo, calibconst, None, None)
                setattr(iface, "_shared_geo_cache", cache)
                if calibc_cache is not None:
                    setattr(iface, "_shared_calibc_cache", calibc_cache)
                # All ranks call this to satisfy the shmem barrier; mirrors
                # exactly what _setup_jungfrau_shared_caches() does.
                ix_all, iy_all = iface._pixel_coord_indexes(all_segs=True)
                if is_gpu_bd and det_name in gpu_det_set:
                    if ix_all is not None and iy_all is not None:
                        self._gpu_geometry_arrays[det_name] = (
                            np.asarray(ix_all), np.asarray(iy_all)
                        )
                        self.logger.debug(
                            "GPU geometry pre-built for %s: shape=%s",
                            det_name, ix_all.shape,
                        )
            except Exception as exc:
                self.logger.debug(
                    "Failed to pre-build GPU geometry for %s: %s",
                    det_name, exc, exc_info=True,
                )

    def _iter_area_detector_raw(self, area_only=True):
        """Iterate over raw-interface detector classes for area detectors.

        Previously named _iter_jungfrau_raw and filtered to Jungfrau only.
        Generalised to cover all area detectors so that gpu_det= works for
        ePix, CSPAD, and any other area detector — not just Jungfrau.

        When area_only=True (default) only AreaDetector subclasses are
        returned, which is the correct behaviour for geometry / GPU setup.
        When area_only=False all raw-interface detectors are returned
        (retained for callers that previously passed area_only=False).
        """
        det_classes = self.dsparms.det_classes.get("normal", {})
        targets = []
        for (det_name, drp_class_name), drp_class in det_classes.items():
            if drp_class_name != "raw":
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

    def _iter_jungfrau_raw(self, area_only=False):
        """Iterate over Jungfrau raw-interface detector classes only.

        Used by _setup_shared_calib which calls build_shared_jungfrau_calib —
        a function that hardcodes the Jungfrau 3-gain pedestal layout.  Passing
        a non-Jungfrau detector (ePix, CSPAD, etc.) would produce a shape
        mismatch in the gain-mode broadcast.

        For GPU geometry setup (_setup_gpu_geometry) use _iter_area_detector_raw
        instead, which covers all area detectors.
        """
        det_classes = self.dsparms.det_classes.get("normal", {})
        targets = []
        for (det_name, drp_class_name), drp_class in det_classes.items():
            if drp_class_name != "raw":
                continue
            mod_name   = getattr(drp_class, "__module__", "")
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

    def events(self):
        # GPU BD rank: delegate to the MPI-aware GPU event loop which drives
        # BigDataNode.start_gpu() instead of the CPU Events iterator.
        if getattr(self.dsparms, 'gpu_det', None) and nodetype == 'bd':
            yield from self._gpu_events_mpi()
            return

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

    def _gpu_events_mpi(self):
        """GPU BD rank event loop driven by MPI batches from EB.

        Uses GpuEvents (the same class that RunSerial uses for the
        single-process GPU path) with a thin _MpiGpuBatchSource adapter that
        translates the MPI batch pairs from self.start() / start_gpu() into
        the smdr_man interface that GpuEvents._next_batch() expects.

        This reuses all of GpuEvents' logic:
          - GPU detector setup from dsparms tables (populated by Configure dgrams)
          - EventPool, KvikioGpuReader, DetectorRouter creation
          - Transition handling, max_events, EndRun, EventPool flush
          - GpuEventContext creation via _yield_ready / _flush_event_pool

        Notes
        -----
        * Calibration constants are already distributed to all BD ranks by
          RunParallel._setup_run_calibconst() → _distribute_calib_xtc().
          self.Detector(det_name) therefore returns calibconst without
          opening additional DataSource instances.
        * GPU pinning (init_gpu_rank) was already called in
          MPIDataSource.__init__() before this method can run.
        * self.start() dispatches to bd_node.start_gpu() (GPU BD path in
          RunParallel.start()), which handles terminate_flag and per-batch
          EB load-accounting stats via set_batch_stats().
        """
        from psana.gpu.gpu_events import GpuEvents
        from psana.gpu.gpu_mpi import gpu_error_handler

        class _MpiGpuBatchSource:
            """Adapts self.start() (smd_batch, gpubat1_bytes) pairs into the
            smdr_man interface expected by GpuEvents._next_batch().

            smdr_man.__next__() must return a batch iterator.  Each batch
            iterator must support next_with_gpu() returning the 3-tuple
            (batch_dict, gpu_batch_dict, step_dict) that GpuEvents._events()
            consumes.  When start_gen is exhausted, __next__() raises
            StopIteration, which GpuEvents._events() catches and uses to
            flush the EventPool and terminate.
            """

            class _OneBatch:
                """Single-use iterator for one (smd_batch, gpubat1_bytes) pair."""
                def __init__(self, smd_batch, gpubat1_bytes):
                    self._smd       = smd_batch
                    self._gpu       = gpubat1_bytes
                    self._yielded   = False

                def next_with_gpu(self):
                    if self._yielded:
                        raise StopIteration
                    self._yielded = True
                    batch_dict     = {0: (self._smd, [])}
                    gpu_batch_dict = {0: (self._gpu, [])} if self._gpu else {}
                    return batch_dict, gpu_batch_dict, {}

            def __init__(self, start_gen, bd_node):
                self._gen      = start_gen
                self._bd_node  = bd_node
                self._t_start  = None   # wall time at start of current batch
                self._n_events = 0      # events yielded by current batch

            def __next__(self):
                # Report stats for the batch that just finished.
                # This is called at the START of each new batch request,
                # i.e. immediately after GpuEvents has consumed all events
                # from the previous batch — matching the timing of the CPU
                # path's on_batch_end() callback inside Events.
                if self._t_start is not None:
                    proc_ns = int((time.monotonic() - self._t_start) * 1e9)
                    self._bd_node.set_batch_stats(
                        read_bytes=0,       # GDS bytes tracked separately
                        read_time_ns=0,
                        proc_events=self._n_events,
                        proc_time_ns=proc_ns,
                    )
                smd_batch, gpubat1_bytes = next(self._gen)   # StopIteration propagates
                self._t_start  = time.monotonic()
                self._n_events = 0
                return _MpiGpuBatchSource._OneBatch(smd_batch, gpubat1_bytes)

        comm = self.comms.psana_comm

        with gpu_error_handler(comm):
            # Compact CuPy pool at the start of each GPU event loop.
            # When multiple batch-size runs execute sequentially (benchmark
            # sweep), data_gpu slot buffers of different sizes from previous
            # runs accumulate as idle pool fragments.  Returning them to CUDA
            # here prevents OOM at larger batch sizes.
            try:
                import cupy as _cp
                import gc as _gc
                _gc.collect()
                _cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

            # NUMA load-balance sync: ensure all BD ranks reach this point
            # before any of them sends the first EB request.
            #
            # On sdfampere nodes (8 NUMA nodes, NPS=4) the two BD ranks land
            # on different NUMA domains.  The rank on NUMA 0 exits _setup_run()
            # up to ~30–40 s faster, sends its initial look-ahead request to EB
            # first, and wins the round-robin for every subsequent batch —
            # starving the slower BD rank for the entire run.
            #
            # bd_comm contains BOTH EB and BD ranks (rank 0 = EB, ranks 1..N =
            # BD workers), so bd_comm.Barrier() would deadlock because EB ranks
            # are not inside this code path.  Instead we use a manual centralized
            # barrier that only BD ranks (bd_rank > 0) participate in:
            #   • BD rank 1 waits to hear from every other BD rank, then signals
            #     all-clear to each of them.
            # This is equivalent to MPI_Barrier restricted to the BD sub-group.
            try:
                bd_comm    = self.comms.bd_comm
                my_bd_rank = self.comms.bd_rank          # 0=EB, 1..N=BD
                n_bd       = bd_comm.Get_size() - 1  # always 1 EB at rank 0
                if n_bd > 1 and my_bd_rank > 0:
                    from mpi4py import MPI as _MPI
                    _buf = bytearray(1)
                    if my_bd_rank == 1:
                        # Coordinator: collect a ready-byte from every peer.
                        for peer in range(2, n_bd + 1):
                            bd_comm.Recv([_buf, _MPI.BYTE], source=peer)
                        # Then release all peers simultaneously.
                        for peer in range(2, n_bd + 1):
                            bd_comm.Send([b'\x01', _MPI.BYTE], dest=peer)
                    else:
                        # Worker: signal coordinator and wait for release.
                        bd_comm.Send([b'\x00', _MPI.BYTE], dest=1)
                        bd_comm.Recv([_buf, _MPI.BYTE], source=1)
            except Exception:
                pass   # non-fatal: if sync fails, continue without it

            # self.start() → RunParallel.start() → bd_node.start_gpu()
            # The _MpiGpuBatchSource adapter makes start_gpu() look like a
            # smdr_man so GpuEvents can drive its own batch loop.
            adapter = _MpiGpuBatchSource(self.start(), self.bd_node)
            gpu_ev = GpuEvents(
                self.configs,
                self.dm,
                self.dsparms.max_retries,
                self.dsparms.use_smds,
                self.shared_state,
                self.dsparms,
                self,
                smdr_man=adapter,
                # Pixel coordinate arrays were pre-computed in
                # _setup_gpu_geometry() during RunParallel.__init__() while
                # all ranks were still synchronising — safe to call
                # _pixel_coord_indexes() there.  Pass them here so
                # _setup_detectors() uses setup_geometry_from_arrays()
                # instead of the lazy setup_geometry(det) call that would
                # trigger a shmem collective during the event loop.
                prebuilt_geometry=getattr(self, '_gpu_geometry_arrays', None),
                setup_geometry=not bool(
                    getattr(self, '_gpu_geometry_arrays', None)
                ),
            )
            # Share peds_gpu/gmask_gpu between BD ranks on the same physical
            # GPU via CUDA IPC.  When N_BD_PER_GPU > 1, multiple BD ranks
            # share one A100; this saves ~400 MB per follower rank.
            _phys_gpu = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0')
                            .split(',')[0])
            try:
                from psana.gpu.gpu_mpi import share_calib_between_gpu_peers
                share_calib_between_gpu_peers(
                    gpu_ev.gpu_detectors,
                    self.comms.bd_comm,
                    _phys_gpu,
                )
            except Exception as _share_exc:
                # Non-fatal: e.g. only one BD rank per GPU, or IPC unavailable.
                self.logger.debug(
                    'share_calib_between_gpu_peers skipped: %s', _share_exc
                )

            # Rate logging mirrors the CPU events() loop: report throughput
            # every ana_interval events using the same self.ana_t_gauge gauge.
            try:
                _ana_interval = int(os.environ.get("PS_BD_ANA_INTERVAL", "1000"))
            except ValueError:
                _ana_interval = 1000
            if _ana_interval <= 0:
                _ana_interval = 1000
            _ana_i = 0
            _ana_st = time.time()

            try:
                # GpuEvents._events() flushes the EventPool internally
                # (on StopIteration from _next_batch or on EndRun/stop_after).
                for _gpu_ctx in gpu_ev:
                    # Track batch event count for set_batch_stats().
                    adapter._n_events += 1
                    yield _gpu_ctx
                    # Rate logging — same logic as CPU events() loop.
                    _ana_i += 1
                    if _ana_i % _ana_interval == 0:
                        _ana_en = time.time()
                        _interval = _ana_en - _ana_st
                        _ana_rate = _ana_interval / _interval if _interval > 0 else 0.0
                        _rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        self.logger.debug(
                            "bd gpu analysis stats rate_hz=%.2f interval_s=%.2f "
                            "events=%d rss_kb=%d",
                            _ana_rate, _interval, _ana_interval, _rss_kb,
                        )
                        self.ana_t_gauge.set(_ana_rate)
                        _ana_st = time.time()
            finally:
                # Flush the last batch's proc_events to EB before draining.
                # The _MpiGpuBatchSource.__next__() reports stats for the
                # *previous* batch at the start of each new request; the final
                # batch's stats are never reported because __next__() is not
                # called again after the last event.  Report them here so EB
                # receives accurate accounting for every batch.
                if adapter._t_start is not None and adapter._n_events > 0:
                    try:
                        proc_ns = int((time.monotonic() - adapter._t_start) * 1e9)
                        self.bd_node.set_batch_stats(
                            read_bytes=0,
                            read_time_ns=0,
                            proc_events=adapter._n_events,
                            proc_time_ns=proc_ns,
                        )
                    except Exception:
                        pass

                # Drain any remaining EB batches so EB can terminate cleanly.
                #
                # GpuEvents._events() may break early when max_events is
                # reached or EndRun is seen, closing the adapter generator
                # before EB has sent its termination signal (empty batch).
                # Without this drain, EB blocks waiting for a BD request that
                # never arrives — a deadlock.
                #
                # Draining keeps sending requests to EB until EB sends the
                # empty-batch termination signal, then both sides exit cleanly.
                try:
                    for _ in adapter._gen:
                        pass
                except Exception:
                    pass

    def steps(self):
        # GPU BD ranks handle BeginStep internally via GpuEvents._dispatch_transition().
        # start() yields (smd_batch, gpubat1_bytes) tuples for GPU BD ranks, not dgrams,
        # so utils.first_service() below would crash.  BeginStep triggers a calibconst
        # refresh on the GPU via GPUDetector.beginstep() — there is nothing for user
        # step-iteration code to do on a GPU BD rank.
        if getattr(self.dsparms, 'gpu_det', None) and nodetype == 'bd':
            return

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

    def close_shared_memory(self):
        self._close_shared_memory()

    def __del__(self):
        try:
            self.close_shared_memory()
        except Exception:
            pass

    def start(self):
        """Request data for this run"""
        if nodetype == "smd0":
            self.smd0.start()
        elif nodetype == "eb":
            self.eb_node.start()
        elif nodetype == "bd":
            if getattr(self.dsparms, 'gpu_det', None):
                # GPU BD rank: yield (smd_batch, gpubat1_bytes) pairs.
                # _gpu_events_mpi() calls self.start() to drive this path,
                # keeping start() as the single dispatch point for all BD work.
                yield from self.bd_node.start_gpu()
            else:
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

        # Non-BD / non-GPU ranks (smd0, EB) must not allocate GPU memory.
        # _distribute_calib_xtc() and _setup_jungfrau_shared_caches() import
        # CuPy and allocate ~1.8 GB on the default CUDA device for ALL ranks
        # when CUDA_VISIBLE_DEVICES is unset.  Slurm sets CUDA_VISIBLE_DEVICES
        # to the allocated GPU list (e.g. '0,1') for ALL tasks, so os.environ
        # .setdefault() has no effect — we must force-overwrite to '' (empty =
        # no devices visible) for CPU-role ranks before any CuPy import.
        # _setup_jungfrau_shared_calib() / _setup_jungfrau_shared_caches()
        # already guard against GPU failures with try/except, so disabling
        # the GPU for smd0/EB causes graceful fallback, not a crash.
        if getattr(self.dsparms, 'gpu_det', None) and nodetype not in ('bd',):
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        # GPU BD ranks: pin each rank to the correct GPU device BEFORE any
        # CuPy import.  We use the BD-local rank (bd_rank - 1, 0-indexed within
        # the BD worker pool) rather than SLURM_LOCALID so that GPU pinning
        # works correctly both when using:
        #   (a) srun -n N python3 script.py  — SLURM_LOCALID is reliable
        #   (b) srun -n 1 bash -c "mpirun -n N ..."  — all ranks share
        #       SLURM_LOCALID=0, so SLURM_LOCALID is useless; bd_rank is not.
        #
        # bd_rank=0 is always the EB in bd_comm; BD workers start at bd_rank=1.
        # bd_local_rank = bd_rank - 1  gives a 0-indexed BD worker index.
        # n_gpus is read from SLURM_GPUS_ON_NODE (set by --gres=gpu:a100:N).
        #
        # This is a no-op when gpu_det is not set (standard CPU-only jobs).
        if getattr(self.dsparms, 'gpu_det', None) and nodetype == 'bd':
            from psana.gpu.gpu_mpi import init_gpu_rank
            bd_local_rank = self.comms.bd_rank - 1   # 0-indexed BD worker
            n_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', 1))
            init_gpu_rank(local_rank=bd_local_rank, n_gpus=n_gpus)

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
        # Sync file lists onto dsparms so that GPU setup code on all ranks
        # (including non-smd0) can discover GPU stream IDs via auto-discovery.
        self.dsparms.smd_files = self.smd_files or []
        self.dsparms.xtc_files = self.xtc_files or []

        configs = self._get_configs()
        self.dm = DgramManager(
            self.xtc_files, configs=configs, config_consumers=[self.dsparms]
        )

        # Set gpu_stream_ids on ALL ranks (including EB) so that
        # EventBuilderManager enables GPU splitting for the right streams.
        # det_stream_segments_table was just populated by DgramManager on all
        # ranks; its keys are the stream IDs that carry the GPU detector data.
        if getattr(self.dsparms, 'gpu_det', None):
            gpu_det_names = self.dsparms.gpu_det
            if isinstance(gpu_det_names, str):
                gpu_det_names = [gpu_det_names]
            seg_table = getattr(self.dsparms, 'det_stream_segments_table', {})
            ids_table  = getattr(self.dsparms, 'det_stream_ids_table', {})
            all_gpu_ids = set()
            for name in gpu_det_names:
                # Mirror GpuEvents._setup_detectors(): prefer det_stream_ids_table,
                # fall back to det_stream_segments_table keys.
                stream_ids = ids_table.get(name) or list(seg_table.get(name, {}).keys())
                all_gpu_ids.update(stream_ids)
            if all_gpu_ids:
                self.dsparms.gpu_stream_ids = sorted(all_gpu_ids)

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
