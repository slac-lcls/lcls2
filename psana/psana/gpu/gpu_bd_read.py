import argparse
import glob
import os
import numpy as np

from psana.psexp.ds_base import DsParms
from psana.psexp.eventbuilder_manager import EventBuilderManager
from psana.psexp.packet_footer import PacketFooter
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.event_manager import EventManager
from psana.dgrammanager import DgramManager
from psana.event import Event
from psana.psexp import TransitionId
from psana.gpu.gpu_batch import GpuBatchView
from psana.gpu.gpu_compare import (
    collect_no_split_reference,
    compare_split_event,
    digest_bytes,
)
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpudgramlite import (
    extract_dgram_info,
    INFO_TIMESTAMP,
    INFO_SERVICE,
    INFO_EXTENT,
    INFO_PAYLOAD_SIZE,
    INFO_READ_SIZE,
    INFO_STREAM_ID,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build SMD chunks in the byte layout SMD0 sends to EventBuilder."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "SMD xtc2 glob(s) or file path(s)."
        ),
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop after this many L1Accept events. 0 means all events.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of L1Accept events per GPU batch.  "
            "Transitions are always handled on CPU and bundled into the "
            "same batch as the following L1Accepts.  Default 1."
        ),
    )
    parser.add_argument(
        "--compare-nosplit",
        action="store_true",
        help="Build no-GPU-split reference events for each SMD chunk.",
    )
    parser.add_argument(
        "--test-calib",
        action="store_true",
        help=(
            "Load calibration constants and run GPUDetector.process_batch() "
            "on each GPU batch.  Prints per-event calib stats (min/max/mean)."
        ),
    )
    parser.add_argument(
        "--det-name",
        default="jungfrau",
        help="Detector name passed to run.Detector() when --test-calib is set.",
    )
    return parser.parse_args()


def _open_calib_detector(smd_files, det_name):
    """Open the SMD files with a standard psana DataSource and return the
    named Detector object (needed for calibration constants)."""
    import psana
    ds  = psana.DataSource(files=list(smd_files))
    run = next(ds.runs())
    return run.Detector(det_name)


def _resolve_input_files(inputs):
    if not inputs:
        raise SystemExit("No SMD file glob or path was provided")

    files = []
    for item in inputs:
        files.extend(sorted(glob.glob(item)))

    seen = set()
    unique_files = []
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        unique_files.append(path)

    if not unique_files:
        raise SystemExit(f"No files found matching input(s): {' '.join(inputs)}")

    return unique_files


def _create_default_dsparms(batch_size=1) -> DsParms:
    # NOTE: Update this helper if DsParms adds/removes required fields.
    return DsParms(
        batch_size=batch_size,
        max_events=0,
        max_retries=0,
        live=False,
        timestamps=np.empty(0, dtype=np.uint64),
        intg_det="",
        intg_delta_t=0,
        use_calib_cache=False,
        cached_detectors=[],
        fetch_calib_cache_max_retries=60,
        skip_calib_load=[],
        dbsuffix="",
    )


def _open_fd(path: str) -> int:
    return os.open(path, os.O_RDONLY)


def _build_eb_ready_chunk(smd_manager, eb_node_id=1):
    empty_step_views = [bytearray() for _ in range(smd_manager.n_files)]
    return bytearray(
        smd_manager.smdr.repack_parallel(
            empty_step_views,
            eb_node_id,
            intg_stream_id=smd_manager.dsparms.intg_stream_id,
        )
    )


def smd_to_xtc_file(smd_file):
    xtc_dir = os.path.dirname(os.path.dirname(smd_file))
    xtc_name = os.path.basename(smd_file).split(".smd")[0] + ".xtc2"
    xtc_file = os.path.join(xtc_dir, xtc_name)
    if not os.path.exists(xtc_file):
        raise FileNotFoundError(xtc_file)
    return xtc_file


def main():
    args = _parse_args()
    smd_files = _resolve_input_files(args.inputs)
    xtc_files = [smd_to_xtc_file(path) for path in smd_files]

    dsparms = _create_default_dsparms(batch_size=args.batch_size)
    dsparms.update_smd_state(list(smd_files), [False] * len(smd_files))
    smd_fds = np.array(
        [_open_fd(path) for path in smd_files],
        dtype=np.int32,
    )

    gpu_detector = None
    from psana.psexp import TransitionId as _TID   # needed for beginstep check

    _compute_calib = None   # set below when --test-calib is active
    _calib_det     = None

    if args.test_calib:
        from psana.gpu.gpu_calib import (GPUDetector, prep_calib_constants,
                                          build_stream_seg_map,
                                          _compute_calib_constants_cpu)
        _compute_calib = _compute_calib_constants_cpu
        _calib_det = _open_calib_detector(smd_files, args.det_name)
        det        = _calib_det   # alias for readability below
        peds_gpu, gmask_gpu = prep_calib_constants(det)
        peds_shape = det.calibconst["pedestals"][0].shape  # (3, n_segs, nrows, ncols)
        det_shape  = peds_shape[1:]                         # (n_segs, nrows, ncols)

        # Build correct per-stream segment mapping from the bigdata files.
        gpu_stream_ids_str = os.environ.get("PS_TEST_GPU_STREAM_IDS", "")
        stream_seg_map = {}
        if gpu_stream_ids_str:
            gpu_ids = [int(x) for x in gpu_stream_ids_str.split(",")]
            stream_bd_files = {i: xtc_files[i] for i in gpu_ids
                               if i < len(xtc_files)}
            stream_seg_map = build_stream_seg_map(stream_bd_files, args.det_name)
            print(f"stream_seg_map: {stream_seg_map}")

        gpu_detector = GPUDetector(
            det_shape=det_shape,
            peds_gpu=peds_gpu,
            gmask_gpu=gmask_gpu,
            stream_seg_map=stream_seg_map or None,
        )
        print(
            f"GPUDetector ready: det_shape={det_shape}  "
            f"peds {peds_gpu.shape}  gmask {gmask_gpu.shape}"
        )

    gpu_reader = None
    try:
        smd_manager = SmdReaderManager(smd_fds, dsparms)
        # Read Configure and BeginRun
        if smd_manager.get_next_dgrams() is None:
            raise RuntimeError("missing Configure")
        if smd_manager.get_next_dgrams() is None:
            raise RuntimeError("missing BeginRun")

        # Create DgramManager for bigdata reads
        bd_dm = DgramManager(xtc_files, configs=smd_manager.configs)

        # GPU reader
        gpu_reader = KvikioGpuReader()

        # Smd0 builds smd chunks in a loop until it finds an EndRun or runs out of events.
        n_events = 0
        stop = False
        for chunk_id in smd_manager.chunks():
            smd_chunk = _build_eb_ready_chunk(smd_manager)
            pf = PacketFooter(view=smd_chunk)
            msg = (
                f"chunk={chunk_id} eb_ready_bytes={memoryview(smd_chunk).nbytes} "
                f"stream_packets={pf.n_packets} view_events={smd_manager.got_events}"
            )

            if args.compare_nosplit:
                ref_by_ts = collect_no_split_reference(
                    smd_chunk,
                    smd_manager.configs,
                    dsparms,
                    xtc_files,
                    max_events=args.max_events,
                )
                msg += f" ref_events={len(ref_by_ts)}"

            # EventBuilder builds smd events by parsing the chunk header and walking the step buffers.
            eb_manager = EventBuilderManager(smd_chunk, smd_manager.configs, dsparms)
            n_batches = 0
            for batch_dict, gpu_batch_dict, step_dict in eb_manager.batches_with_gpu():
                n_batches += 1
                split_by_ts = {}

                # -------------------------------------------------------
                # BeginStep hook: refresh calibration constants in-place.
                # -------------------------------------------------------
                if step_dict and gpu_detector is not None and _compute_calib is not None:
                    for step_batch, _ in step_dict.values():
                        if len(step_batch) >= 12:
                            env = int.from_bytes(memoryview(step_batch)[8:12], 'little')
                            if (env >> 24) & 0xFF == _TID.BeginStep:
                                peds_new, gmask_new = _compute_calib(_calib_det)
                                gpu_detector.beginstep(peds_new, gmask_new)
                        break

                # -------------------------------------------------------
                # Step 8: double buffering — issue GPU reads immediately
                # (non-blocking) so they overlap with the CPU EventManager
                # path below, which reads bigdata for the non-GPU streams.
                # -------------------------------------------------------
                gpu_pending = None
                for gpu_batch, _ in gpu_batch_dict.values():
                    gpu_view = GpuBatchView(gpu_batch, validate=True)
                    if gpu_view.has_work:
                        gpu_pending = (gpu_view,
                                       gpu_reader.issue_batch(gpu_view, bd_dm))

                # -------------------------------------------------------
                # CPU path — EventManager reads bigdata for CPU streams
                # while GPU-stream GDS reads are in-flight above.
                # -------------------------------------------------------
                cpu_events = []   # collect for deferred compare_nosplit
                for smd_batch, _ in batch_dict.values():
                    if not smd_batch:
                        continue

                    evt_manager = EventManager(
                        smd_batch,
                        smd_manager.configs,
                        bd_dm,
                        dsparms.max_retries,
                        [False] * len(smd_manager.configs),
                    )
                    for dgrams in evt_manager:
                        evt = Event(dgrams=dgrams)
                        if not TransitionId.isEvent(evt.service()):
                            continue

                        per_stream = split_by_ts.setdefault(evt.timestamp, {})
                        for stream_id, dg in enumerate(dgrams):
                            if dg is None:
                                per_stream.setdefault(stream_id, (0, None))
                                continue
                            dg_bytes = bytearray(dg)
                            per_stream[stream_id] = (len(dg_bytes), digest_bytes(dg_bytes))

                        cpu_events.append((evt, dgrams))
                        n_events += 1
                        if args.max_events > 0 and n_events >= args.max_events:
                            stop = True
                            break

                    if evt_manager.exit_id:
                        raise RuntimeError(f"EventManager failed with exit_id={evt_manager.exit_id}")
                    if stop:
                        break

                # -------------------------------------------------------
                # Wait for GPU reads (likely already done) and compute.
                # -------------------------------------------------------
                if gpu_pending is not None:
                    gpu_view, pending = gpu_pending
                    gpu_read = gpu_reader.wait_batch(
                        pending, compute_digest=args.compare_nosplit
                    )

                    info_gpu = extract_dgram_info(
                        gpu_read.data_gpu,
                        gpu_read.desc_table_gpu,
                        len(gpu_read.read_descs),
                    )
                    info_cpu = info_gpu.get()
                    for desc, row in zip(gpu_read.read_descs, info_cpu):
                        gpu_ts         = int(row[INFO_TIMESTAMP])
                        gpu_stream_id  = int(row[INFO_STREAM_ID])
                        gpu_service    = int(row[INFO_SERVICE])
                        gpu_extent     = int(row[INFO_EXTENT])
                        gpu_payload_sz = int(row[INFO_PAYLOAD_SIZE])
                        gpu_read_size  = int(row[INFO_READ_SIZE])
                        print(
                            f"gpu_dgram: event={desc.batch_event_index} "
                            f"stream={gpu_stream_id} timestamp={gpu_ts} "
                            f"service={gpu_service} extent={gpu_extent} "
                            f"payload_size={gpu_payload_sz} read_size={gpu_read_size}"
                        )
                        if args.compare_nosplit:
                            split_value = gpu_read.by_timestamp[desc.timestamp][desc.stream_id]
                            split_by_ts.setdefault(gpu_ts, {})[gpu_stream_id] = split_value

                    if gpu_detector is not None:
                        for evt_ctx in gpu_detector.process_batch(gpu_view, gpu_read):
                            calib_np = evt_ctx.calib_gpu.get()
                            print(
                                f"calib: ts={evt_ctx.timestamp} "
                                f"shape={calib_np.shape} "
                                f"min={calib_np.min():.2f} "
                                f"max={calib_np.max():.2f} "
                                f"mean={calib_np.mean():.2f}"
                            )

                # -------------------------------------------------------
                # Per-event output and compare (after both paths complete).
                # -------------------------------------------------------
                for evt, dgrams in cpu_events:
                    none_streams = [i for i, dg in enumerate(dgrams) if dg is None]
                    print(
                        f"bd_event: svc:{evt.service()} timestamp:{evt.timestamp} "
                        f"n_dgrams:{len(dgrams)} n_none:{len(none_streams)} "
                        f"none_streams:{none_streams}"
                    )
                    if args.compare_nosplit:
                        n_compare_streams = compare_split_event(
                            ref_by_ts, split_by_ts, evt.timestamp
                        )
                        print(f"compare_ok timestamp={evt.timestamp} streams={n_compare_streams}")

                if stop:
                    break

            msg += f" eb_batches={n_batches} bd_events={n_events}"
            print(msg)
            if stop:
                break

    finally:
        for fd in smd_fds:
            os.close(int(fd))
        if "bd_dm" in locals():
            bd_dm.close()
        if gpu_reader is not None:
            gpu_reader.close()

if __name__ == "__main__":
    main()
