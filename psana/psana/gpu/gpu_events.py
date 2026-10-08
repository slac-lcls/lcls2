import logging
import math
import sys
from contextlib import closing, nullcontext
from dataclasses import dataclass, field


_log = logging.getLogger(__name__)

from psana import dgram, utils
from psana.event import EventEnvelope
from psana.gpu.context import GpuEventState
from psana.gpu.gpu_batch import GPU_DESC_FLAG_VALID, GpuBatchView, GpuSubbatchView
from psana.gpu.gpu_budget import GpuMemoryPressureError
from psana.gpu.gpu_input import GpuDetectorBinding
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpudgram import GpuStreamConfigTable, GpuXtcBatchPool
from psana.gpu.gpu_stream import EventPool
from psana.psexp import TransitionId
from psana.psexp.event_manager import EventManager
from psana.psexp.packet_footer import PacketFooter


class _GpuOnlyDgram:
    """Minimal L1Accept metadata for an event whose streams all went GPU-only.

    Exclusive gpu_det routing removes its SMD dgrams from the CPU batch. When
    every selected stream is exclusive, EventManager therefore returns an
    all-None dgram list. EventEnvelope still needs timestamp/service metadata
    so Run.events() can preserve the normal API without causing a redundant
    CPU BigData read. hybrid_det streams remain present on the CPU path.
    """

    def __init__(self, timestamp):
        self._timestamp = int(timestamp)
        self._env = int(TransitionId.L1Accept) << 24

    def timestamp(self):
        return self._timestamp

    def env(self):
        return self._env


def _iter_step_events(batch_bytes, configs):
    if not batch_bytes or len(batch_bytes) < 12:
        return

    batch_pf = PacketFooter(view=batch_bytes)
    event_offset = 0
    for event_index in range(batch_pf.n_packets):
        event_size = batch_pf.get_size(event_index)
        event_view = memoryview(batch_bytes)[event_offset : event_offset + event_size]
        event_offset += event_size

        event_pf = PacketFooter(view=event_view)
        event_footer_nbytes = memoryview(event_pf.footer).nbytes
        dgram_offset = 0
        dgrams = [None] * len(configs)
        for i_stream in range(event_pf.n_packets):
            dgram_size = event_pf.get_size(i_stream)
            if dgram_size:
                dgrams[i_stream] = dgram.Dgram(
                    config=configs[i_stream],
                    view=event_view,
                    offset=dgram_offset,
                )
            dgram_offset += dgram_size

        if dgram_offset + event_footer_nbytes != event_size:
            raise RuntimeError(f"Malformed step event {event_index}: dgrams={dgram_offset} footer={event_footer_nbytes} event_size={event_size}")

        service = 0
        for dg in dgrams:
            if dg is not None:
                service = dg.service()
                break
        yield service, dgrams


def _fmt_mib(n: int) -> str:
    """Format byte count as MiB string for logging."""
    return f"{n / 1024**2:.1f} MiB"


@dataclass
class _GpuMemStats:
    """Snapshot of GPU and pinned-host memory broken down by owner.

    All values are bytes.  Recorded by GpuEventManager.log_memory() and used
    to update per-category high-water marks.

    GPU categories (device VRAM):
        raw_input    KvikioGpuReader per-slot input buffers
        xtc_config   run-scoped flattened Configure tables
        xtc_slots    per-slot dgram, ShapesData, and field-locator tables
        cupy_pool    CuPy memory-pool total committed bytes
        device_used  bytes in use according to CUDA (total - free)
        device_total total device memory

    Pinned-host category:
        pinned       framework-owned host staging
    """

    input_bytes: dict = field(default_factory=dict)
    # aggregate GPU
    raw_input: int = 0
    xtc_config: int = 0
    xtc_slots: int = 0
    committed: int = 0
    held: int = 0
    retained: int = 0
    failed: int = 0
    borrowed: int = 0
    allocations: tuple = ()
    cupy_used: int = 0
    cupy_pool: int = 0
    device_used: int = 0
    device_total: int = 0
    # pinned host
    pinned: int = 0
    # label for logging
    label: str = ""

    # _mb is now the module-level _fmt_mib; kept as alias for log() callers
    _mb = staticmethod(lambda n: _fmt_mib(n))

    def log(self):
        """Emit a structured INFO log summarising the snapshot."""
        _log.info("GPU ownership [%s] committed=%s held=%s retained=%s failed=%s borrowed=%s pool_used=%s",
                  self.label, self._mb(self.committed), self._mb(self.held),
                  self._mb(self.retained), self._mb(self.failed),
                  self._mb(self.borrowed), self._mb(self.cupy_used))
        _log.info(
            "GPU mem [%s] raw_input=%s  xtc_config=%s  xtc_slots=%s  "
            "cupy_pool=%s  device_used=%s / %s  pinned=%s",
            self.label,
            self._mb(self.raw_input),
            self._mb(self.xtc_config),
            self._mb(self.xtc_slots),
            self._mb(self.cupy_pool),
            self._mb(self.device_used),
            self._mb(self.device_total),
            self._mb(self.pinned),
        )


class GpuEventManager:
    """Run-scoped GPU event processor.

    The serial path still drives this object through its iterator compatibility
    interface. The MPI path supplies coherent SMD/GPU batches directly through
    ``process_batch``; the manager never owns MPI communication.
    """

    def __init__(
        self,
        configs,
        dm,
        max_retries,
        use_smds,
        shared_state,
        dsparms,
        run,
        smdr_man=None,
        n_bd_per_gpu=1,
        placement=None,
    ):
        self.configs = configs
        self.dm = dm
        self.max_retries = max_retries
        self.use_smds = use_smds
        self.shared_state = shared_state
        self.dsparms = dsparms
        self.run = run
        self.smdr_man = smdr_man
        # Device placement discovered before the event loop: which physical
        # GPU this rank holds, who shares it, and who owns shared constants.
        # When absent (serial runs, or an un-migrated caller) fall back to the
        # supplied peer count, which is what n_bd_per_gpu meant.
        self._placement = placement
        if placement is not None:
            self._n_bd_per_gpu = max(1, int(placement.n_device_peers))
        else:
            self._n_bd_per_gpu = max(1, int(n_bd_per_gpu or 1))

        self._batch_iter = iter([])
        self._iter = None
        self._has_gpu_batch_iter = False   # cached in beginrun; avoids per-batch hasattr
        self._n_events = 0
        self._done = False
        self._closed = False
        self._closing = False

        self.gpu_det_names = list(dsparms.gpu_detector_names)
        self.gpu_detector_bindings = {}
        self.input_preparers = {}
        self._gpu_task = getattr(dsparms, "gpu_fn", None)
        self._task_constants = None
        self._output_d2h = None
        if self._gpu_task is not None:
            from .gpu_d2h import PublicationD2H, DEFAULT_PINNED_BYTES
            self._output_d2h = PublicationD2H(getattr(dsparms, 'gpu_d2h_pinned_bytes', DEFAULT_PINNED_BYTES))
        self._step_generation = 0
        self.event_pool = None
        self.gpu_reader = None
        self.gpu_xtc_configs = None
        self.gpu_xtc_parser = None
        # At most one KvikIO read is pre-issued ahead of the CPU event loop.
        # Keep explicit ownership so generator close/early termination can
        # drain it before gpu_reader.close() releases its buffers.
        self._pending_gpu_read = None

        self._setup_gpu_pipeline()

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        if self._iter is None:
            self._iter = self._events()
        return next(self._iter)

    def _snapshot_memory(self, label: str) -> _GpuMemStats:
        """Collect a _GpuMemStats snapshot from all pipeline components."""
        s = _GpuMemStats(label=label)
        for name, preparer in getattr(self, "input_preparers", {}).items():
            s.input_bytes[name] = preparer.memory_bytes()["total"]
            s.pinned += preparer.pinned_bytes()
        if self.event_pool is not None and hasattr(self.event_pool, 'pinned_bytes'):
            s.pinned += self.event_pool.pinned_bytes()
        if self.gpu_reader is not None and hasattr(self.gpu_reader, "memory_bytes"):
            s.raw_input = self.gpu_reader.memory_bytes()["raw_input_slots"]
        if self.gpu_xtc_parser is not None:
            parser_memory = self.gpu_xtc_parser.memory_bytes()
            s.xtc_config = parser_memory["config"]
            s.xtc_slots = parser_memory["batch_slots"]
        output_d2h = getattr(self, '_output_d2h', None)
        if output_d2h is not None:
            s.pinned += output_d2h.pinned_bytes
        budget = getattr(self, '_gpu_budget', None)
        if budget is not None:
            from .gpu_allocation import backing_capacity
            s.committed, s.held = budget.committed(), budget._held
            s.allocations = budget.allocation_snapshot()
            s.retained = max(0, s.committed - s.raw_input - s.xtc_config - s.xtc_slots - sum(s.input_bytes.values()))
            s.failed = sum(backing_capacity(a) for _, arrays, _ in budget._failed_allocations for a in arrays)
        # Query CuPy pool and CUDA device info only when a GPU is active.
        # These calls fail on CPU-only nodes and are skipped silently.
        cupy_mod = sys.modules.get("cupy")
        if cupy_mod is not None:
            try:
                # Probe the runtime before constructing/accessing CuPy's
                # device-local memory pool.  On CPU-only hosts, touching the
                # pool first can leave a partially initialized object whose
                # destructor raises an unraisable CUDA driver exception.
                if cupy_mod.cuda.runtime.getDeviceCount() <= 0:
                    return s
                s.cupy_used = cupy_mod.get_default_memory_pool().used_bytes()
                s.cupy_pool = cupy_mod.get_default_memory_pool().total_bytes()
                free, total = cupy_mod.cuda.Device().mem_info
                s.device_used = total - free
                s.device_total = total
            except Exception:
                pass
        return s

    def log_memory(self, label: str = ""):
        """Snapshot memory usage, update high-water marks, and log.

        Emits one INFO log line per detector plus a summary line.
        Call after GPU setup, after the first batch, and at EndRun.

        High-water marks track the peak value seen for each category
        across all calls within this run.
        """
        s = self._snapshot_memory(label)
        s.log()
        hw = self._high_water
        for category in ('committed', 'held', 'retained', 'failed', 'borrowed', 'cupy_used'):
            hw[category] = max(hw.get(category, 0), getattr(s, category))
        hw["raw_input"] = max(hw.get("raw_input", 0), s.raw_input)
        hw["xtc_config"] = max(hw.get("xtc_config", 0), s.xtc_config)
        hw["xtc_slots"] = max(hw.get("xtc_slots", 0), s.xtc_slots)
        hw["cupy_pool"] = max(hw.get("cupy_pool", 0), s.cupy_pool)
        hw["device_used"] = max(hw.get("device_used", 0), s.device_used)
        hw["pinned"] = max(hw.get("pinned", 0), s.pinned)

    def log_high_water(self):
        """Log the peak memory values seen since the last reset."""
        hw = self._high_water
        _log.info(
            "GPU mem high-water raw_input=%s xtc_config=%s xtc_slots=%s "
            "cupy_pool=%s device_used=%s pinned=%s",
            _fmt_mib(hw.get("raw_input", 0)),
            _fmt_mib(hw.get("xtc_config", 0)),
            _fmt_mib(hw.get("xtc_slots", 0)),
            _fmt_mib(hw.get("cupy_pool", 0)),
            _fmt_mib(hw.get("device_used", 0)),
            _fmt_mib(hw.get("pinned", 0)),
        )

    def _resize_budget_for_shared(self, shared_bytes):
        """Account for the shared intersection before it is allocated.

        Called by SharedRequestedConstants once the intersection is known and
        before anything is allocated. Shared constants are device overhead
        counted once, so every rank's share is computed net of them while the
        owner -- which actually holds the allocation -- gets those bytes added
        back.

        Doing this after refresh() meant the owner allocated against
        usable/peers: a 12 GiB intersection on a 40 GiB four-peer device fits
        the documented accounting (7 + 12 = 19 GiB) but was charged against
        10 GiB, so the shared copy was refused, the group degraded, and the
        private copy was refused by the same limit.
        """
        placement = self._placement
        shared = int(shared_bytes or 0)
        if placement is None:
            return
        # shared == 0 is a RESET, not a no-op: the fallback calls it after
        # releasing the shared blocks so private copies are checked against
        # usable/peers again. Returning early here left followers holding
        # (usable - shared)/peers while each uploaded its own full copy, so a
        # group that used to degrade cleanly aborted instead -- measured on a
        # 40 GiB four-peer device for a 9-10 GiB intersection, which fits
        # 10 GiB but not 7.
        placement.shared_bytes = shared
        if float(getattr(self.dsparms, "gpu_memory_budget_gb", 0) or 0):
            # An explicit budget is the user's ceiling and discover_peers has
            # already validated it against the group; do not move it.
            return
        from psana.gpu.gpu_placement import per_rank_limit
        limit = per_rank_limit(placement)
        if placement.is_owner:
            # _OwnedBlock charges the shared bytes to the owner, and
            # per_rank_limit has subtracted the same amount from every rank's
            # share. Without this the owner pays twice.
            limit += shared
        self._gpu_budget.set_limit(limit)

    def _setup_gpu_pipeline(self):
        """Initialize this BD's run-scoped GPU resources and processing pipeline."""
        # Budget must exist before constructing input resources.
        from psana.gpu.gpu_budget import _GpuBudget

        budget_gb = float(getattr(self.dsparms, "gpu_memory_budget_gb", 0) or 0)
        placement = self._placement
        if budget_gb > 0:
            # discover_peers has already validated this against the group
            # total and folded it into usable_bytes; re-checking it here would
            # be dead code.
            self._gpu_budget = _GpuBudget(limit_bytes=int(budget_gb * 1024**3))
        elif placement is not None:
            # Divide the discovered device capacity among its true peers,
            # counting any shared constants once rather than per rank.
            from psana.gpu.gpu_placement import per_rank_limit
            self._gpu_budget = _GpuBudget(limit_bytes=per_rank_limit(placement))
        else:
            # Serial, or a caller that supplied only a peer count.
            self._gpu_budget = _GpuBudget.auto(n_bd_ranks=self._n_bd_per_gpu)

        ids_table = getattr(self.dsparms, "det_stream_ids_table", {})
        segments_table = getattr(
            self.dsparms, "det_stream_segments_table", {}
        )
        streams_by_detector = {
            name: sorted(ids_table.get(name) or segments_table.get(name, {}).keys())
            for name in self.gpu_det_names
        }
        missing = [name for name, stream_ids in streams_by_detector.items()
                   if not stream_ids]
        if missing:
            raise RuntimeError(
                f"GPU detectors did not resolve to any stream ids: {missing}"
            )

        all_gpu_stream_ids = {
            stream_id
            for stream_ids in streams_by_detector.values()
            for stream_id in stream_ids
        }
        requested_stream_ids = getattr(self.dsparms, "gpu_stream_ids", None)
        if requested_stream_ids is None:
            raise RuntimeError("GPU stream routing was not resolved from Configure")
        if set(requested_stream_ids) != all_gpu_stream_ids:
            raise RuntimeError(
                "GPU stream routing must include every stream for each "
                f"GPU detector selection: expected {sorted(all_gpu_stream_ids)}, got "
                f"{sorted(requested_stream_ids)}"
            )

        from psana.gpu.gpu_mpi import log_gpu_mem

        try:
            from mpi4py import MPI

            _rank = MPI.COMM_WORLD.Get_rank()
        except Exception:
            _rank = None

        # Compile all stream Configures once, then resolve the exact field
        # consumed by each detector segment.  This replaces the legacy first-L1
        # CPU probe and its inferred raw offset/panel stride.
        self.gpu_xtc_configs = GpuStreamConfigTable.from_configs(self.configs)
        xtc_field_handles = []

        log_gpu_mem("_setup_gpu_pipeline entry", rank=_rank)
        for det_name in self.gpu_det_names:
            stream_segments = dict(segments_table.get(det_name, {}))
            gpu_stream_ids = streams_by_detector[det_name]
            configured_segment_ids = sorted({
                segment_id
                for stream_id in gpu_stream_ids
                for segment_id in stream_segments.get(stream_id, ())
            })
            canonical_segment_ids = configured_segment_ids
            routed_stream_segments = {
                stream_id: tuple(stream_segments.get(stream_id, ()))
                for stream_id in gpu_stream_ids
            }
            field_handles_by_name = (
                self.gpu_xtc_configs.detector_field_handles(
                    det_name,
                    stream_segments=routed_stream_segments,
                )
            )
            # Configure may describe event algorithms for which psana has no
            # CPU detector class.  Keep those fields available through the
            # detector-independent parser interface; only Configure payloads
            # themselves are outside the event-field contract.
            field_handles_by_name = {
                key: handles
                for key, handles in field_handles_by_name.items()
                if key[0] != "config"
            }
            for handles in field_handles_by_name.values():
                xtc_field_handles.extend(handles.values())

            detector_binding = GpuDetectorBinding(
                det_name,
                canonical_segment_ids=canonical_segment_ids,
                field_handles_by_segment={},
                field_handles_by_name=field_handles_by_name,
            )
            self.gpu_detector_bindings[det_name] = detector_binding

        if not self.dsparms.batch_size:
            self.dsparms.batch_size = 1

        pool_depth = getattr(self.dsparms, "n_gpu_streams", 2)
        self.event_pool = EventPool(n=pool_depth, budget=self._gpu_budget)

        # Eagerly locate every event field exposed by configured GPU detectors.
        # This makes arbitrary field access a budgeted part of each parser slot.
        # Deduplication preserves canonical detector order across detectors.
        xtc_field_handles = tuple(dict.fromkeys(xtc_field_handles))
        self.gpu_xtc_parser = GpuXtcBatchPool(
            self.gpu_xtc_configs,
            field_handles=xtc_field_handles,
            n_slots=pool_depth + (len(self.configs) + 1 if self.dsparms.gpu_bulk_read else 0),
            budget=self._gpu_budget,
        )
        if self._gpu_task is not None:
            from .gpu_task import prepare_task_inputs, RequestedConstants
            self._gpu_task.validate_detectors(self.gpu_det_names)
            self.input_preparers = prepare_task_inputs(
                self._gpu_task, self.gpu_xtc_configs, self.gpu_detector_bindings,
                n_slots=pool_depth, budget=self._gpu_budget)
            for preparer in self.input_preparers.values():
                preparer.configure_gather(self.gpu_xtc_parser.handle_indices)
            placement = self._placement
            if placement is not None and placement.can_share:
                # One device copy for every BD rank on this GPU. Peers that
                # declared different selectors share the intersection and
                # privately upload the remainder.
                from .gpu_shared_constants import SharedRequestedConstants
                self._task_constants = SharedRequestedConstants(
                    self._gpu_task.calibconst, self._gpu_budget, placement,
                    sizing=self._resize_budget_for_shared)
            else:
                self._task_constants = RequestedConstants(
                    self._gpu_task.calibconst, self._gpu_budget)
            self._task_constants.refresh(getattr(self.dsparms, 'calibconst', {}))
        self._setup_input_io()

        # Report which I/O path kvikio will use for this run.
        # GDS (compat_mode=False) reads NVMe → GPU VRAM directly (fast).
        # CPU-fallback (compat_mode=True) reads NVMe → CPU DRAM → GPU VRAM
        # via cudaMemcpy (slower; common on Lustre/GPFS filesystems like S3DF).
        _path = self.gpu_reader.io_path
        if self.gpu_reader._compat_mode:
            _log.warning(
                "GpuEventManager: kvikio I/O path = %s "
                "(NVMe → CPU DRAM → GPU VRAM via cudaMemcpy). "
                "True GDS is not available — likely Lustre/GPFS filesystem "
                "or cuFile driver not loaded.  GDS would give NVMe → GPU VRAM "
                "directly, bypassing CPU DRAM entirely.",
                _path,
            )
        else:
            _log.info("GpuEventManager: kvikio I/O path = %s (NVMe → GPU VRAM direct)", _path)

        # Phase-0 accounting: high-water marks reset each run.
        self._high_water: dict = {}
        self._first_batch_logged = False

        # Log framework-owned input allocations.
        try:
            self.log_memory("after_setup")
        except Exception:
            pass

        # Phase-3: per-subbatch byte budget for byte-bounded splitting.
        # Computed once after all GPU detectors are set up.
        self._subbatch_budget_bytes = self._compute_subbatch_budget()

    def _setup_input_io(self):
        """Create BD-owned input I/O after the shared budget and slots exist.

        The reader serves all selected GPU streams. File resolution state is
        local to this BD and run, independent of individual detector adapters.
        """
        self.gpu_reader = KvikioGpuReader(
            n_slots=(getattr(self.dsparms, 'n_gpu_streams', 2)
                     * max(1, self.dsparms.batch_size) * max(1, len(self.configs))
                     + len(self.configs) if self.dsparms.gpu_bulk_read
                     else getattr(self.dsparms, 'n_gpu_streams', 2)),
            budget=self._gpu_budget,
            bulk_read=self.dsparms.gpu_bulk_read,
        )
        if self.dsparms.gpu_bulk_read:
            from psana.gpu.gpu_file_epochs import GpuFileEpochs
            self._gpu_file_epochs = GpuFileEpochs(self.dm)
            from psana.gpu.gpu_input_group import InputGroupPool
            self._group_inputs = InputGroupPool(self.gpu_reader)

    # ------------------------------------------------------------------
    # Phase 3: byte-bounded subbatch helpers
    # ------------------------------------------------------------------

    def _compute_subbatch_budget(self) -> int:
        """Per-execution target after charged fixed storage and 10% headroom."""
        self._admission_margin = self._gpu_budget.limit() // 10
        self._admission_capacity = max(
            0, self._gpu_budget.available() - self._admission_margin
        )
        depth = max(1, getattr(self.dsparms, 'n_gpu_streams', 2))
        return self._admission_capacity // depth

    def _event_memory(self, gpu_view):
        """Actual descriptor presence and dense allocation cost for each event."""
        from .gpu_admission import AdmissionEvent
        if isinstance(gpu_view, GpuSubbatchView):
            parent = gpu_view._parent
            indices = range(gpu_view._start, gpu_view._end)
        else:
            parent = gpu_view
            indices = range(parent.header.n_events)
        events = []
        for i in indices:
            streams = tuple((int(d['stream_id']), int(d['bd_size']))
                            for d in parent.desc_rows_for_event(i)
                            if int(d['flags']) & GPU_DESC_FLAG_VALID)
            present = {stream for stream, _ in streams}
            task = getattr(self, '_gpu_task', None)
            prepared_bytes = sum(p.estimate_subbatch_bytes(1)
                                 for p in getattr(self, "input_preparers", {}).values()
                                 if (bool(present) if task is not None
                                     else p.binding.has_sources(present)))
            if task is not None and present:
                from .gpu_task_batch import metadata_bytes
                prepared_bytes += metadata_bytes(task, self.gpu_detector_bindings, 1)
            events.append(AdmissionEvent(streams, prepared_bytes))
        return events

    def _split_subbatches(self, gpu_view) -> list:
        """Admit complete events; fail before I/O when a minimum event cannot fit."""
        if getattr(self, '_group_inputs', None) is not None:
            from .gpu_group_schedule import GroupReadSchedule
            self._group_schedule = GroupReadSchedule(
                tuple(gpu_view.iter_read_descs(self.dm)), self._gpu_read_files,
                self._event_memory(gpu_view), batch_id=self._input_batch_id,
                capacity=self._admission_capacity,
                parser_bytes=self.gpu_xtc_parser.estimate_batch_bytes(1) + 24,
                depth=self.event_pool.depth,
                target_bytes=getattr(self.dsparms, "gpu_bulk_target_bytes", 1 << 20))
            return [GpuSubbatchView(gpu_view, a, b)
                    for a, b in self._group_schedule.execution_ranges]
        from .gpu_admission import plan_admission
        parser = getattr(self, 'gpu_xtc_parser', None)
        per_dgram = parser.estimate_batch_bytes(1) if parser is not None else 0
        plan = plan_admission(
            self._event_memory(gpu_view),
            getattr(self, '_admission_capacity', self._subbatch_budget_bytes),
            parser_bytes_per_dgram=per_dgram,
            max_inflight=max(1, getattr(getattr(self, 'dsparms', None), 'n_gpu_streams', 1)),
        )
        self._last_admission_plan = plan
        return [GpuSubbatchView(gpu_view, start, end)
                for start, end in plan.execution_ranges]

    def _input_allocation_requirements(self, read_view, slot):
        n_dgrams = sum(1 for _ in read_view.iter_read_descs(self.dm))
        if not n_dgrams:
            return []
        requirements = self.gpu_reader.allocation_requirements(read_view.total_read_bytes, slot)
        if self.gpu_xtc_parser is not None:
            requirements += self.gpu_xtc_parser.allocation_requirements(n_dgrams)
        return requirements

    def _reserve_gpu_subbatch(self, subbatch, slot):
        """Hold reader/parser growth before any read is submitted."""
        from .gpu_budget import allocation_growth_bytes
        requirements = self._input_allocation_requirements(
            subbatch, slot)
        events = self._event_memory(subbatch)
        requirements += self._task_input_requirements(events, slot)
        return self._gpu_budget.hold(allocation_growth_bytes(requirements),
                                     margin=getattr(self, '_admission_margin', 0))

    def _task_input_requirements(self, events, slot):
        """Conservative pre-selection bound, including every aligned input row."""
        task = getattr(self, '_gpu_task', None)
        selected_count = sum(bool(e.streams) for e in events)
        requirements = []
        for preparer in getattr(self, 'input_preparers', {}).values():
            count = (selected_count if task is not None else
                     sum(preparer.binding.has_sources({s for s, _ in e.streams})
                         for e in events))
            requirements += preparer.allocation_requirements(count, slot)
        if task is not None:
            import cupy as cp
            from .gpu_allocation import allocation_requirement
            from .gpu_task_batch import metadata_bytes
            requirements.append(allocation_requirement(
                cp, metadata_bytes(task, self.gpu_detector_bindings, selected_count), None))
        return requirements


    def _close_gpu_reservation(self):
        hold = getattr(self, '_gpu_read_reservation', None)
        if hold is not None:
            hold.close()
            self._gpu_read_reservation = None

    def _trim_gpu_caches(self):
        """Called only after execution leases drain; input pins remain authoritative."""
        if self.event_pool.active_count:
            raise RuntimeError('cannot trim input buffers with active executions')
        if getattr(self, '_group_inputs', None) is not None:
            self._group_inputs.drain_idle()
        self.gpu_reader.trim_free_buffers()
        if self.gpu_xtc_parser is not None:
            self.gpu_xtc_parser.trim_free_buffers()
        for preparer in getattr(self, "input_preparers", {}).values():
            preparer.trim_slot_buffers()

    def _next_batch(self):
        if self.smdr_man is None:
            raise StopIteration

        while True:
            if self.shared_state.terminate_flag.value:
                raise StopIteration

            try:
                if self._has_gpu_batch_iter:
                    return self._batch_iter.next_with_gpu()
                batch_dict, step_dict = next(self._batch_iter)
                return batch_dict, {}, step_dict
            except StopIteration:
                self._batch_iter = next(self.smdr_man)
                self._has_gpu_batch_iter = hasattr(self._batch_iter, "next_with_gpu")

    def _dispatch_transition(self, service, dgrams):
        self.run._handle_transition(dgrams)
        if service == TransitionId.BeginStep:
            self._step_generation = getattr(self, '_step_generation', 0) + 1
        constants = getattr(self, '_task_constants', None)
        if service == TransitionId.BeginStep and constants is not None:
            # _handle_steps already drained execution and input consumers.
            # Resolve after the host transition; BeginStep does not fetch the DB.
            if constants.refresh(getattr(self.dsparms, 'calibconst', {}),
                                 before_upload=self._trim_gpu_caches):
                self._subbatch_budget_bytes = self._compute_subbatch_budget()

    def _handle_steps(self, step_dict):
        end_run_seen = False
        if not step_dict:
            return end_run_seen

        pending_transitions = []
        for step_batch, _ in step_dict.values():
            for service, dgrams in _iter_step_events(step_batch, self.configs):
                if service == 0:
                    # A GPU-only L1 has no dgrams in the CPU/SMD packet.
                    continue
                if TransitionId.isEvent(service):
                    continue
                pending_transitions.append((service, dgrams))

        needs_drain = any(service in (TransitionId.BeginStep, TransitionId.EndRun) for service, _ in pending_transitions)
        if needs_drain:
            yield from self._flush_event_pool()
            # Group input leases transfer completion to deferred windows. An
            # empty execution pool alone does not finish those consumers.
            inputs = getattr(self, '_group_inputs', None)
            if inputs is not None:
                inputs.drain_idle()

        for service, dgrams in pending_transitions:
            if service == TransitionId.EndRun:
                end_run_seen = True
                try:
                    self.log_memory("end_run")
                    self.log_high_water()
                except Exception:
                    pass
            self._dispatch_transition(service, dgrams)

        return end_run_seen

    def _attach_gpu(self, envelope, gpu_results, leases=None,
                    pending_d2h=None, cached_cpu_results=None,
                    event_dgrams=None, input_lease=None,
                    device_released=False):
        state = GpuEventState(
            gpu_results=gpu_results,
            detector_names=self.gpu_det_names,
            leases=leases,
            pending_d2h=pending_d2h,
            cached_cpu_results=cached_cpu_results,
            detector_bindings=getattr(self, "gpu_detector_bindings", {}),
            event_dgrams=event_dgrams,
            input_lease=input_lease,
            device_released=device_released,
        )
        return EventEnvelope(dgrams=envelope.dgrams, gpu_state=state)

    def _submit_gpu(self, subbatch, gpu_read, event_envelopes):
        """Submit one input execution with its completion and owner leases."""
        hold = getattr(self, '_gpu_read_reservation', None)
        try:
            with hold if hold is not None else nullcontext():
                if getattr(self, '_group_inputs', None) is not None:
                    record = self._submit_group_gpu(subbatch, gpu_read, event_envelopes)
                else:
                    record = self._submit_per_dgram_gpu(subbatch, gpu_read, event_envelopes)
        finally:
            self._close_gpu_reservation()
        output_d2h = getattr(self, '_output_d2h', None)
        if output_d2h is not None:
            try:
                output_d2h.enqueue(record)
            except BaseException:
                # Also protect the MPI process_batch path, which has no serial
                # iterator close-on-error wrapper. Failed drains quarantine the
                # occupied EventPool and every copy/producer owner for retry.
                for _ in self.event_pool.flush():
                    pass
                raise
        return record

    def _submit_per_dgram_gpu(self, subbatch, gpu_read, event_envelopes):
        return self.event_pool.submit(
            subbatch, gpu_read, event_envelopes, getattr(self, "input_preparers", {}),
            xtc_parser=self.gpu_xtc_parser,
            batch_id=getattr(self, "_input_batch_id", 0), **self._task_submission())

    def _task_submission(self):
        task = getattr(self, '_gpu_task', None)
        if task is None:
            return {}
        return dict(task=task, detector_bindings=self.gpu_detector_bindings,
                    task_constants=self._task_constants,
                    run=self.run.runnum, step_generation=self._step_generation)

    def _submit_group_gpu(self, subbatch, pending, event_envelopes):
        inputs = self._group_inputs
        inputs.parse_groups(pending, self.gpu_xtc_parser, self.event_pool.next_stream)
        groups = self._group_schedule.groups_for(subbatch._start, subbatch._end)
        windows, uses, transferred = [], {}, []
        try:
            for group in groups:
                key = (self._input_batch_id, group.group_id)
                window = inputs.window(key)
                windows.append(window)
                for d in group.dgrams:
                    if subbatch._start <= d.batch_event_index < subbatch._end:
                        use = inputs.take_use(key, d.batch_event_index)
                        transferred.append(use)
                        uses.setdefault(window, use)
            return self.event_pool.submit(
                subbatch, None, event_envelopes, getattr(self, "input_preparers", {}),
                batch_id=self._input_batch_id, input_windows=windows, input_uses=uses,
                **self._task_submission())
        finally:
            error = None
            for use in transferred:
                try:
                    use.wait_until_safe_to_reuse()
                except BaseException as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error

    def _issue_group_reads(self, subbatch, slot_id):
        from .gpu_budget import allocation_growth_bytes
        inputs = self._group_inputs
        groups = self._group_schedule.new_groups(subbatch._start, subbatch._end)
        slots = inputs.plan_slots(groups)
        if slots is None:
            raise GpuMemoryPressureError('input groups await stream credit or raw slots')
        requirements = []
        for group, slot in zip(groups, slots):
            requirements += self.gpu_reader.allocation_requirements(group.size, slot)
        n_dgrams = sum(len(group.dgrams) for group in groups)
        if n_dgrams:
            requirements += self.gpu_xtc_parser.allocation_requirements(n_dgrams, groups=True)
        events = self._event_memory(subbatch)
        requirements += self._task_input_requirements(events, slot_id)
        hold = self._gpu_budget.hold(allocation_growth_bytes(requirements),
                                     margin=self._admission_margin)
        self._gpu_read_reservation = hold
        pending = []
        self._pending_gpu_read = pending
        with hold:
            for group, slot in zip(groups, slots):
                key = inputs.issue(self._input_batch_id, group, slot_id=slot)
                if key is None:
                    raise RuntimeError('reserved input group became unavailable')
                pending.append(key)
                self._group_schedule.issued.add(group.group_id)
        return pending

    def _yield_ready(self, ready, device_released=False):
        if ready is None:
            return
        # Log after the first batch: slot buffers have grown to their
        # initial sizes so this shows the steady-state allocation.
        if not self._first_batch_logged and ready.event_envelopes:
            self._first_batch_logged = True
            self.log_memory("first_batch")
        for envelope in ready.event_envelopes:
            ts = utils.first_timestamp(envelope.dgrams)
            gpu_results = ready.gpu_results_by_ts.get(ts, {})
            if device_released:
                # Preserve the result-key API but never retain a stale view
                # into a slot which the replacement H2D may now overwrite.
                gpu_results = {key: None for key in gpu_results}
            yield self._attach_gpu(
                envelope,
                gpu_results,
                leases={} if device_released else ready.leases_by_ts.get(ts, {}),
                pending_d2h=ready.pending_d2h_by_ts.pop(ts, {}),
                cached_cpu_results=ready.cached_cpu_results_by_ts.get(ts, {}),
                event_dgrams=(None if device_released else getattr(
                    ready, "input_dgrams_by_ts", {}
                ).get(ts)),
                input_lease=(None if device_released else getattr(
                    ready, "input_leases_by_ts", {}
                ).get(ts)),
                device_released=device_released,
            )

    def _issue_gpu_read(self, subbatch, slot_id):
        """Issue and own the single read allowed ahead of CPU processing."""
        if (getattr(self, '_pending_gpu_read', None) is not None
                or getattr(self, '_gpu_read_reservation', None) is not None):
            raise RuntimeError("a pre-issued GPU read is already outstanding")
        if getattr(self, '_group_inputs', None) is not None:
            return self._issue_group_reads(subbatch, slot_id)
        hold = self._reserve_gpu_subbatch(subbatch, slot_id)
        try:
            with hold:
                pending = self.gpu_reader.issue_batch(subbatch, self.dm, slot_id=slot_id)
        except BaseException:
            hold.close()
            raise
        self._gpu_read_reservation = hold
        self._pending_gpu_read = pending
        return pending

    def _wait_gpu_read(self, pending):
        """Complete a read and relinquish its controller-side ownership."""
        if getattr(self, '_pending_gpu_read', None) is not pending:
            raise RuntimeError("attempted to wait for an unowned GPU read")
        try:
            if getattr(self, '_group_inputs', None) is not None:
                for key in pending:
                    self._group_inputs.read(key)
                return pending
            return self.gpu_reader.wait_batch(pending)
        except BaseException:
            self._close_gpu_reservation()
            raise
        finally:
            self._pending_gpu_read = None

    def _drain_pending_gpu_read(self):
        """Finish a pre-issued read before the reader and buffers are closed."""
        pending = getattr(self, '_pending_gpu_read', None)
        try:
            if pending is not None:
                if getattr(self, '_group_inputs', None) is not None:
                    for key in pending:
                        self._group_inputs.read(key)
                else:
                    self.gpu_reader.wait_batch(pending)
        finally:
            self._pending_gpu_read = None
            self._close_gpu_reservation()

    def _retire_issue_and_yield(self, subbatch):
        """Expose parsed inputs before draining consumers and reusing a slot."""
        ready = self.event_pool.begin_retire_next()
        try:
            yield from self._yield_ready(ready)
        finally:
            self.event_pool.finish_retire_next()
        ready = None
        slot = self.event_pool.next_slot_id
        try:
            return self._issue_gpu_read(subbatch, slot)
        except GpuMemoryPressureError:
            # Admission failed before I/O. Reduce overlap, then relinquish only
            # unowned cached storage before retrying the same complete events.
            yield from self._flush_event_pool()
            self._trim_gpu_caches()
            return self._issue_gpu_read(subbatch, self.event_pool.next_slot_id)

    def _flush_event_pool(self):
        # Explicitly unwind the current slot on generator close; do not rely
        # on garbage collection of the pool's suspended registration window.
        with closing(self.event_pool.flush()) as slots:
            for slot_data in slots:
                yield from self._yield_ready(slot_data)

    def _process_batch(self, batch_dict, gpu_batch_dict, step_dict):
        n_events = self._n_events
        try:
            while True:
                self._input_batch_id = getattr(self, "_input_batch_id", 0) + 1
                gpu_views = [GpuBatchView(packet, validate=True)
                             for packet, _ in gpu_batch_dict.values()]
                if getattr(getattr(self, "dsparms", None), "gpu_bulk_read", False):
                    if len(gpu_views) > 1:
                        raise ValueError("bulk reads require one coherent GPUBAT1 packet")
                    transitions = [
                        transition
                        for packet, _ in step_dict.values()
                        for transition in _iter_step_events(packet, self.configs)
                        if transition[0] and not TransitionId.isEvent(transition[0])
                    ]
                    self._gpu_read_files = self._gpu_file_epochs.resolve(
                        [d for view in gpu_views for d in view.iter_read_descs(self.dm)],
                        transitions,
                    )
                end_run_seen = yield from self._handle_steps(step_dict)

                # ── Phase 3: GPU path — split batch into subbatches ──────────
                # Parse every GPU batch from this EB communication and split
                # each into byte-bounded GpuSubbatchViews.  Issue the FIRST
                # subbatch's reads now (before the CPU EventManager loop) so
                # GDS/PCIe I/O overlaps with CPU SMD deserialization.
                all_subbatches = []
                first_pending  = None   # (subbatch_0, PendingBatch)

                for gpu_view in gpu_views:
                    if not gpu_view.has_work:
                        continue
                    all_subbatches.extend(self._split_subbatches(gpu_view))

                if all_subbatches:
                    first_pending = (
                        all_subbatches[0],
                        (yield from self._retire_issue_and_yield(
                            all_subbatches[0]
                        )),
                    )

                # ── CPU path ─────────────────────────────────────────────────
                # EventManager loop runs while subbatch 0 reads are in-flight.
                stop_after = False
                event_envelopes = []
                for smd_batch, _ in batch_dict.values():
                    if not smd_batch:
                        continue
                    event_manager = EventManager(
                        smd_batch,
                        self.configs,
                        self.dm,
                        self.max_retries,
                        self.use_smds,
                    )
                    for envelope in event_manager:
                        dgrams = envelope.dgrams
                        # Exclusive GPU streams are absent from the CPU batch.
                        # An all-exclusive dataset therefore has no CPU dgram
                        # from which the envelope could obtain service/time.
                        # Synthesize that metadata below from GPUBAT1 instead.
                        if not any(dgrams):
                            continue
                        if not TransitionId.isEvent(utils.first_service(dgrams)):
                            continue
                        event_envelopes.append(envelope)
                    if event_manager.exit_id:
                        raise RuntimeError(f"EventManager exit {event_manager.exit_id}")

                # ── Submit subbatches ─────────────────────────────────────────
                if all_subbatches:
                    # Build a timestamp → envelope lookup, filling GPU-only
                    # events from GPUBAT1 timestamps without reading BigData
                    # through the CPU path.
                    ts_to_envelope = {
                        utils.first_timestamp(envelope.dgrams): envelope
                        for envelope in event_envelopes
                    }
                    selected_envelopes = []
                    for subbatch in all_subbatches:
                        for timestamp in subbatch.timestamps:
                            if (
                                self.dsparms.max_events > 0
                                and n_events >= self.dsparms.max_events
                            ):
                                stop_after = True
                                break
                            timestamp = int(timestamp)
                            envelope = ts_to_envelope.get(timestamp)
                            if envelope is None:
                                dgrams = [None] * len(self.configs)
                                dgrams[0] = _GpuOnlyDgram(timestamp)
                                envelope = EventEnvelope(dgrams=dgrams)
                            selected_envelopes.append(envelope)
                            n_events += 1
                        if stop_after:
                            break
                    event_envelopes = selected_envelopes
                    ts_to_envelope = {
                        utils.first_timestamp(envelope.dgrams): envelope
                        for envelope in event_envelopes
                    }

                    for i, subbatch in enumerate(all_subbatches):
                        # Event envelopes whose timestamps appear in this subbatch.
                        sb_ts  = subbatch.timestamps
                        sb_envelopes = [
                            ts_to_envelope[ts]
                            for ts in sb_ts
                            if ts in ts_to_envelope
                        ]

                        if i == 0 and first_pending is not None:
                            # Subbatch 0: reads were already issued before the
                            # CPU loop.  Just wait for them to complete.
                            _, pending_0 = first_pending
                            gpu_read = self._wait_gpu_read(pending_0)
                            self._submit_gpu(subbatch, gpu_read, sb_envelopes)
                            first_pending = pending_0 = gpu_read = None
                        else:
                            pending = yield from self._retire_issue_and_yield(
                                subbatch
                            )
                            gpu_read = self._wait_gpu_read(pending)
                            self._submit_gpu(subbatch, gpu_read, sb_envelopes)
                            pending = gpu_read = None
                else:
                    # No GPU batch — yield CPU-only events directly.
                    for envelope in event_envelopes:
                        if (
                            self.dsparms.max_events > 0
                            and n_events >= self.dsparms.max_events
                        ):
                            stop_after = True
                            break
                        n_events += 1
                        yield self._attach_gpu(envelope, {})

                if stop_after or end_run_seen:
                    yield from self._flush_event_pool()
                    self._done = True
                return
        finally:
            self._n_events = n_events

    def process_batch(self, smd_batch, gpu_batch=None):
        """Process one coherent EB-to-BD batch and yield EventEnvelopes."""
        if self._done:
            return
        if self.gpu_reader is not None:
            self.gpu_reader.reset_io_stats()
        batch_dict = {0: (smd_batch, [])}
        gpu_batch_dict = {0: (gpu_batch, [])} if gpu_batch else {}
        # MPI transition history is embedded in the SMD packet itself.
        step_dict = {0: (smd_batch, [])}
        yield from self._process_batch(batch_dict, gpu_batch_dict, step_dict)

    def get_bd_read_stats(self):
        """Return bytes and seconds spent in GPU big-data reads this batch."""
        if self.gpu_reader is None:
            return 0, 0.0
        stats = self.gpu_reader.io_stats()
        return int(stats["total_bytes"]), stats["total_ns"] / 1e9

    def finish(self):
        """Drain in-flight work and close GPU reader resources once."""
        if self._closed:
            return
        try:
            yield from self._flush_event_pool()
        finally:
            self.close()

    def _close_resources(self):
        """Called after all executions have retired successfully."""
        self._drain_pending_gpu_read()
        if getattr(self, '_group_inputs', None) is not None:
            self._group_inputs.close()
        parser = getattr(self, "gpu_xtc_parser", None)
        if parser is not None:
            parser.close()
        budget = getattr(self, '_gpu_budget', None)
        if budget is not None:
            budget.drain_failed_allocations()
        if self.gpu_reader is not None:
            self.gpu_reader.close()
        constants = getattr(self, '_task_constants', None)
        if constants is not None:
            constants.close()
        output_d2h = getattr(self, '_output_d2h', None)
        if output_d2h is not None:
            output_d2h.close()
        self._closed = True

    def close(self):
        """Discard remaining deliveries while safely retiring their slots."""
        if self._closed or getattr(self, '_closing', False):
            return
        self._closing = True
        self._done = True
        try:
            iterator = getattr(self, '_iter', None)
            # Unwind a suspended serial producer before starting a new drain.
            # Calls from its own finally must not close a running generator.
            if iterator is not None and not iterator.gi_running:
                iterator.close()
            pool = getattr(self, 'event_pool', None)
            if getattr(pool, '_retiring', None) is not None:
                pool.finish_retire_next()
            for _ in self._flush_event_pool():
                pass
            self._close_resources()
        finally:
            # A failed join leaves owners intact and allows another close.
            self._closing = False

    def _events(self):
        try:
            while not self._done:
                try:
                    batch_dict, gpu_batch_dict, step_dict = self._next_batch()
                except StopIteration:
                    break
                yield from self._process_batch(
                    batch_dict, gpu_batch_dict, step_dict
                )
            yield from self.finish()
        finally:
            self.close()
