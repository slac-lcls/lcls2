from psana import dgram
from psana.event import Event
from psana.gpu.context import GpuEventContext
from psana.gpu.detector_router import DetectorRouter
from psana.gpu.gpu_batch import GpuBatchView
from psana.gpu.gpu_calib import (
    GPUDetector,
    _compute_calib_constants_cpu,
    build_stream_seg_map,
    optimal_kernel_batch_size,
    prep_calib_constants,
)
from psana.gpu.gpu_kernel_registry import default_registry
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_stream import EventPool
from psana.psexp import TransitionId
from psana.psexp.event_manager import EventManager
from psana.psexp.packet_footer import PacketFooter


def _apply_full_routing(gpu_results, evt, gpu_detectors, router):
    if not router or not hasattr(router, "has_full_routing"):
        return gpu_results

    for det_name, det_info in gpu_detectors.items():
        if not router.has_full_routing(det_name):
            continue
        calib_key = f"{det_name}.calib"
        gpu_calib = gpu_results.get(calib_key)
        if gpu_calib is None:
            continue

        det = det_info[0]
        cpu_calib = router.compute_cpu_calib(det_name, det, evt)
        combined = router.assemble_full_calib(det_name, gpu_calib, cpu_calib)
        if combined is not None:
            gpu_results[calib_key] = combined

    return gpu_results


def _iter_step_events(batch_bytes, configs):
    if not batch_bytes or len(batch_bytes) < 12:
        return

    batch_pf = PacketFooter(view=batch_bytes)
    event_offset = 0
    for event_index in range(batch_pf.n_packets):
        event_size = batch_pf.get_size(event_index)
        event_view = memoryview(batch_bytes)[event_offset:event_offset + event_size]
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
            raise RuntimeError(
                f"Malformed step event {event_index}: "
                f"dgrams={dgram_offset} footer={event_footer_nbytes} "
                f"event_size={event_size}"
            )

        service = 0
        for dg in dgrams:
            if dg is not None:
                service = dg.service()
                break
        yield service, dgrams


class GpuEvents:
    """GPU-aware event iterator for the existing serial psana read path.

    This mirrors Events, but consumes the GPU split batch produced by the
    existing SmdReaderManager/BatchIterator/EventBuilder stack.  It does not
    create DsParms, SmdReaderManager, DgramManager, or EventBuilderManager.
    """

    is_gpu_events = True

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
        registry=None,
        setup_geometry=True,
        prebuilt_geometry=None,
    ):
        self.configs = configs
        self.dm = dm
        self.max_retries = max_retries
        self.use_smds = use_smds
        self.shared_state = shared_state
        self.dsparms = dsparms
        self.run = run
        self.smdr_man         = smdr_man
        self.registry         = registry
        self._setup_geometry  = setup_geometry
        self._prebuilt_geometry = prebuilt_geometry  # {det_name: (ix_all, iy_all)}

        self._batch_iter = iter([])
        self._iter = None

        self.gpu_det_names = self._normalize_gpu_det(dsparms.gpu_det)
        self.gpu_detectors = {}
        self.cpu_dets = {}
        self.router = DetectorRouter()
        self.event_pool = None
        self.gpu_reader = None

        self._setup_detectors()

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = self._events()
        return next(self._iter)

    @staticmethod
    def _normalize_gpu_det(gpu_det):
        if gpu_det is None:
            return []
        if isinstance(gpu_det, str):
            return [gpu_det]
        return list(gpu_det)

    def _setup_detectors(self):
        reg = self.registry if self.registry is not None else default_registry()
        all_gpu_stream_ids = set()
        opt_batch_sizes = []
        requested_stream_ids = getattr(self.dsparms, "gpu_stream_ids", None)
        requested_stream_ids = (
            set(requested_stream_ids) if requested_stream_ids is not None else None
        )

        from psana.gpu.gpu_mpi import log_gpu_mem
        try:
            from mpi4py import MPI
            _rank = MPI.COMM_WORLD.Get_rank()
        except Exception:
            _rank = None

        log_gpu_mem('_setup_detectors entry', rank=_rank)
        for det_name in self.gpu_det_names:
            det = self.run.Detector(det_name)
            peds_gpu, gmask_gpu = prep_calib_constants(det)
            log_gpu_mem(f'after prep_calib_constants ({det_name})', rank=_rank)
            det_shape = det.calibconst["pedestals"][0].shape[1:]

            stream_segments = dict(
                getattr(self.dsparms, "det_stream_segments_table", {}).get(
                    det_name, {}
                )
            )
            det_stream_ids = sorted(
                getattr(self.dsparms, "det_stream_ids_table", {}).get(
                    det_name, stream_segments.keys()
                )
            )
            if requested_stream_ids is None:
                gpu_stream_ids = det_stream_ids
            else:
                gpu_stream_ids = [
                    stream_id
                    for stream_id in det_stream_ids
                    if stream_id in requested_stream_ids
                ]
            if not gpu_stream_ids:
                raise RuntimeError(
                    f"gpu_det={det_name!r} did not resolve to any stream ids"
                )

            # Configure identifies which physical segments belong to each
            # stream, but its dictionary order is not necessarily the order
            # of ShapesData children in L1Accept.  The fixed-stride GPU gather
            # preserves L1 child order, so discover that order from the first
            # detector event in each routed bigdata stream.
            xtc_files = getattr(self.dm, "xtc_files", None)
            if xtc_files is None:
                xtc_files = getattr(self.dsparms, "xtc_files", [])
            stream_files = {
                stream_id: xtc_files[stream_id]
                for stream_id in gpu_stream_ids
                if stream_id < len(xtc_files)
            }
            stream_seg_map = build_stream_seg_map(stream_files, det_name)

            for stream_id in gpu_stream_ids:
                segment_ids = stream_seg_map.get(stream_id)
                if not segment_ids:
                    raise RuntimeError(
                        f"gpu_det={det_name!r} could not determine L1Accept "
                        f"segment order for stream {stream_id}"
                    )
                configured = set(stream_segments.get(stream_id, []))
                if configured and set(segment_ids) != configured:
                    raise RuntimeError(
                        f"gpu_det={det_name!r} stream {stream_id} segment "
                        f"mismatch: Configure={sorted(configured)} "
                        f"L1Accept={segment_ids}"
                    )
            cpu_stream_seg_map = {
                stream_id: sorted(segment_ids)
                for stream_id, segment_ids in stream_segments.items()
                if stream_id not in gpu_stream_ids
            }
            all_gpu_stream_ids.update(gpu_stream_ids)

            kernel = reg.get(det_name, "calib")
            gpu_detector = GPUDetector(
                det_shape=det_shape,
                peds_gpu=peds_gpu,
                gmask_gpu=gmask_gpu,
                stream_seg_map=stream_seg_map or None,
                calib_kernel=kernel,
                n_slots=getattr(self.dsparms, 'n_gpu_streams', 4),  # one calib_gpu per slot
            )
            if kernel is not None:
                kernel.setup(det, gpu_detector)
            if self._prebuilt_geometry and det_name in self._prebuilt_geometry:
                ix_all, iy_all = self._prebuilt_geometry[det_name]
                gpu_detector.setup_geometry_from_arrays(ix_all, iy_all)
            elif self._setup_geometry:
                gpu_detector.setup_geometry(det)
            log_gpu_mem(f'after setup_geometry ({det_name})', rank=_rank)

            opt_batch_sizes.append(optimal_kernel_batch_size(det_shape))
            self.gpu_detectors[det_name] = (det, gpu_detector, cpu_stream_seg_map)
            self.router.register_gpu(det_name)

            gpu_seg_ids = []
            for stream_id in sorted(stream_seg_map):
                gpu_seg_ids.extend(stream_seg_map[stream_id])
            cpu_seg_ids = []
            for stream_id in sorted(cpu_stream_seg_map):
                cpu_seg_ids.extend(cpu_stream_seg_map[stream_id])

            self.router.setup_full_routing(
                det_name=det_name,
                gpu_seg_ids=gpu_seg_ids,
                cpu_seg_ids=cpu_seg_ids,
                calibconst_n_segs=det_shape[0],
                nrows=det_shape[1],
                ncols=det_shape[2],
                gpu_det_obj=gpu_detector,
            )

        if all_gpu_stream_ids:
            self.dsparms.gpu_stream_ids = sorted(all_gpu_stream_ids)

        if not self.dsparms.batch_size:
            self.dsparms.batch_size = min(opt_batch_sizes) if opt_batch_sizes else 1

        gpu_det_set = set(self.gpu_det_names)
        for det_name in self.run.detnames:
            if det_name in gpu_det_set:
                continue
            self.cpu_dets[det_name] = self.run.Detector(det_name)
            self.router.register_cpu(det_name)

        n_streams = getattr(self.dsparms, "n_gpu_streams", 4)
        self.event_pool = EventPool(n=n_streams)
        # KvikioGpuReader: pre-allocate one data_gpu buffer per slot (Option D)
        self.gpu_reader = KvikioGpuReader(n_slots=n_streams)

        # Report which I/O path kvikio will use for this run.
        # GDS (compat_mode=False) reads NVMe → GPU VRAM directly (fast).
        # CPU-fallback (compat_mode=True) reads NVMe → CPU DRAM → GPU VRAM
        # via cudaMemcpy (slower; common on Lustre/GPFS filesystems like S3DF).
        import logging as _log
        _logger = _log.getLogger(__name__)
        _path = self.gpu_reader.io_path
        if self.gpu_reader._compat_mode:
            _logger.warning(
                'GpuEvents: kvikio I/O path = %s '
                '(NVMe → CPU DRAM → GPU VRAM via cudaMemcpy). '
                'True GDS is not available — likely Lustre/GPFS filesystem '
                'or cuFile driver not loaded.  GDS would give NVMe → GPU VRAM '
                'directly, bypassing CPU DRAM entirely.',
                _path,
            )
        else:
            _logger.info('GpuEvents: kvikio I/O path = %s (NVMe → GPU VRAM direct)', _path)

    def _next_batch(self):
        if self.smdr_man is None:
            raise StopIteration

        while True:
            if self.shared_state.terminate_flag.value:
                raise StopIteration

            try:
                if hasattr(self._batch_iter, "next_with_gpu"):
                    return self._batch_iter.next_with_gpu()
                batch_dict, step_dict = next(self._batch_iter)
                return batch_dict, {}, step_dict
            except StopIteration:
                self._batch_iter = next(self.smdr_man)

    def free_calib_bufs(self):
        """Release pre-allocated calib_gpu slot buffers for all GPU detectors.

        Delegates to GPUDetector.free_calib_bufs() for each detector.
        See GPUDetector.free_calib_bufs() for usage guidance.
        """
        for det_info in self.gpu_detectors.values():
            det_info[1].free_calib_bufs()

    def _dispatch_transition(self, service, dgrams):
        if service == TransitionId.BeginStep:
            for det_info in self.gpu_detectors.values():
                det, gpu_detector = det_info[0], det_info[1]
                peds, gmask = _compute_calib_constants_cpu(det)
                gpu_detector.beginstep(peds, gmask)

        self.run._handle_transition(dgrams)

    def _handle_steps(self, step_dict):
        end_run_seen = False
        if not step_dict:
            return end_run_seen

        pending_transitions = []
        for step_batch, _ in step_dict.values():
            for service, dgrams in _iter_step_events(step_batch, self.configs):
                pending_transitions.append((service, dgrams))

        needs_drain = any(
            service in (TransitionId.BeginStep, TransitionId.EndRun)
            for service, _ in pending_transitions
        )
        if needs_drain:
            yield from self._flush_event_pool()

        for service, dgrams in pending_transitions:
            if service == TransitionId.EndRun:
                end_run_seen = True
            self._dispatch_transition(service, dgrams)

        return end_run_seen

    def _make_context(self, evt, gpu_results):
        gpu_results = _apply_full_routing(
            gpu_results,
            evt,
            self.gpu_detectors,
            self.router,
        )
        return GpuEventContext(
            evt=evt,
            gpu_results=gpu_results,
            cpu_dets=self.cpu_dets,
            stream=None,
            router=self.router,
        )

    def _yield_ready(self, ready):
        if ready is None:
            return
        gpu_results_by_ts, cpu_evts = ready
        for evt in cpu_evts:
            yield self._make_context(
                evt,
                gpu_results_by_ts.get(evt.timestamp, {}),
            )

    def _flush_event_pool(self):
        for gpu_results_by_ts, cpu_evts in self.event_pool.flush():
            for evt in cpu_evts:
                yield self._make_context(
                    evt,
                    gpu_results_by_ts.get(evt.timestamp, {}),
                )

    def _events(self):
        n_events = 0
        try:
            while True:
                try:
                    batch_dict, gpu_batch_dict, step_dict = self._next_batch()
                except StopIteration:
                    yield from self._flush_event_pool()
                    return

                end_run_seen = yield from self._handle_steps(step_dict)

                gpu_pending = None
                for gpu_batch, _ in gpu_batch_dict.values():
                    gpu_view = GpuBatchView(gpu_batch, validate=True)
                    if gpu_view.has_work:
                        # Retire the batch occupying the next slot before
                        # KvikIO overwrites that slot's raw input buffer.  Its
                        # calibrated output aliases the same logical slot, so
                        # yield it before submitting the replacement batch.
                        ready = self.event_pool.retire_next()
                        slot_id = self.event_pool.next_slot_id
                        gpu_pending = (
                            gpu_view,
                            self.gpu_reader.issue_batch(
                                gpu_view, self.dm, slot_id=slot_id
                            ),
                        )
                        yield from self._yield_ready(ready)

                stop_after = False
                cpu_evts = []
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
                    for dgrams in event_manager:
                        evt = Event(dgrams=dgrams, run=self.run._run_ctx)
                        if not TransitionId.isEvent(evt.service()):
                            continue
                        cpu_evts.append(evt)
                        n_events += 1
                        if (
                            self.dsparms.max_events > 0
                            and n_events >= self.dsparms.max_events
                        ):
                            stop_after = True
                            break
                    if event_manager.exit_id:
                        raise RuntimeError(
                            f"EventManager exit {event_manager.exit_id}"
                        )
                    if stop_after:
                        break

                if gpu_pending is not None:
                    gpu_view, pending = gpu_pending
                    gpu_read = self.gpu_reader.wait_batch(pending)
                    self.event_pool.submit(
                        gpu_view,
                        gpu_read,
                        cpu_evts,
                        self.gpu_detectors,
                    )
                else:
                    for evt in cpu_evts:
                        yield self._make_context(evt, {})

                if stop_after or end_run_seen:
                    yield from self._flush_event_pool()
                    break
        finally:
            if self.gpu_reader is not None:
                self.gpu_reader.close()
