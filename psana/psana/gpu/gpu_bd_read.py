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
    compare_jungfrau_raw,
    compare_split_event,
    digest_bytes,
)
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_raw_offset_cache import GpuRawOffsetCache
from psana.gpu.gpu_jungfrau import (
    assemble_jungfrau_raw,
    build_jungfrau_raw_loc_table,
    prepare_jungfrau_raw_layout,
    JF_LOC_DIM0,
    JF_LOC_DIM1,
    JF_LOC_DIM2,
    JF_LOC_DTYPE_SIZE,
    JF_LOC_NAMES_ID_VALUE,
    JF_LOC_RAW_DEVICE_OFFSET,
    JF_LOC_RAW_NBYTES,
    JF_LOC_SEGMENT,
    JF_LOC_STATUS,
    JF_LOC_STATUS_FOUND,
    JF_LOC_STREAM_ID,
    JF_LOC_TIMESTAMP,
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
        help="Stop after this many events. 0 means all events.",
    )
    parser.add_argument(
        "--compare-nosplit",
        action="store_true",
        help="Build no-GPU-split reference events for each SMD chunk.",
    )
    return parser.parse_args()


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


def _create_default_dsparms() -> DsParms:
    # NOTE: Update this helper if DsParms adds/removes required fields.
    return DsParms(
        batch_size=1,
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


def _load_jungfrau_config(gpu_reader, bd_dm):
    table = bd_dm.gpu_config_tables.get("jungfrau")
    if table is None or not table:
        return None, None

    layout = prepare_jungfrau_raw_layout(table, cp=gpu_reader.cp)
    return table, layout


def _bootstrap_raw_offset_cache(raw_offset_cache, read_descs):
    if raw_offset_cache is None:
        return

    before = raw_offset_cache.n_rows
    cached_streams = []
    for desc in read_descs:
        if raw_offset_cache.is_stream_cached(desc.stream_id):
            continue
        raw_offset_cache.ensure_stream_cached(
            desc.stream_id,
            desc.offset,
            desc.size,
        )
        cached_streams.append(desc.stream_id)

    if raw_offset_cache.n_rows != before:
        streams = ",".join(str(stream_id) for stream_id in cached_streams)
        print(
            "gpu_raw_offset_cache: "
            f"cached_streams={streams} "
            f"rows={raw_offset_cache.n_rows}/"
            f"{raw_offset_cache.expected_n_rows} "
            f"ready={raw_offset_cache.ready}"
        )


def main():
    args = _parse_args()
    smd_files = _resolve_input_files(args.inputs)
    xtc_files = [smd_to_xtc_file(path) for path in smd_files]

    dsparms = _create_default_dsparms()
    dsparms.update_smd_state(list(smd_files), [False] * len(smd_files))
    smd_fds = np.array(
        [_open_fd(path) for path in smd_files],
        dtype=np.int32,
    )
    gpu_reader = None
    raw_offset_cache = None
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
        gpu_reader = KvikioGpuReader(keep_device_buffers=True)
        # This is the jungfrau raw array mapping per segment
        jungfrau_config, jungfrau_layout = _load_jungfrau_config(
            gpu_reader,
            bd_dm,
        )
        if jungfrau_config is not None:
            print(
                f"gpu_config: jungfrau rows={jungfrau_config.n_rows} "
                f"cols={len(jungfrau_config.rows.dtype.names)} "
                f"raw_shape={jungfrau_layout.raw_shape}"
            )
            raw_offset_cache = GpuRawOffsetCache(
                jungfrau_config,
                xtc_files=xtc_files,
                configs=smd_manager.configs,
            )

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
                ref_jungfrau_raw_by_ts = (
                    {} if jungfrau_layout is not None else None
                )
                ref_by_ts = collect_no_split_reference(
                    smd_chunk,
                    smd_manager.configs,
                    dsparms,
                    xtc_files,
                    max_events=args.max_events,
                    jungfrau_raw_by_ts=ref_jungfrau_raw_by_ts,
                    jungfrau_segments=(
                        None
                        if jungfrau_layout is None
                        else jungfrau_layout.segments.tolist()
                    ),
                )
                msg += f" ref_events={len(ref_by_ts)}"

            # EventBuilder builds smd events by parsing the chunk header and walking the step buffers.
            eb_manager = EventBuilderManager(smd_chunk, smd_manager.configs, dsparms)
            n_batches = 0
            for batch_dict, gpu_batch_dict, _ in eb_manager.batches_with_gpu():
                n_batches += 1

                # gpu_batch_dict is kept separate. Later this is where GPU BD scheduling starts.
                split_by_ts = {}  # for comparing with no split reference
                gpu_jungfrau_raw_by_ts = {}
                for gpu_batch, _ in gpu_batch_dict.values():
                    gpu_view = GpuBatchView(gpu_batch, validate=True)
                    if gpu_view.has_work:
                        read_descs = tuple(gpu_view.iter_read_descs(bd_dm))
                        _bootstrap_raw_offset_cache(raw_offset_cache, read_descs)
                        gpu_read = gpu_reader.read_batch(gpu_view, bd_dm)
                        for desc in gpu_read.read_descs:
                            # Keep existing split/no-split comparison format:
                            # split_by_ts[timestamp][stream_id] = (dgram_size, digest)
                            split_value = gpu_read.by_timestamp[desc.timestamp][desc.stream_id]
                            split_by_ts.setdefault(desc.timestamp, {})[desc.stream_id] = split_value
                        if raw_offset_cache is not None:
                            jf_loc_cpu = build_jungfrau_raw_loc_table(
                                gpu_read.desc_table,
                                raw_offset_cache,
                            )
                            jf_loc_gpu = gpu_reader.cp.asarray(jf_loc_cpu)
                            raw_gpu = assemble_jungfrau_raw(
                                gpu_read.data_gpu,
                                jf_loc_gpu,
                                jungfrau_layout,
                                gpu_view.header.n_events,
                            )
                            if args.compare_nosplit:
                                raw_cpu_from_gpu = raw_gpu.get()
                                for event in gpu_view.iter_events():
                                    gpu_jungfrau_raw_by_ts[event.timestamp] = (
                                        raw_cpu_from_gpu[event.batch_event_index].copy()
                                    )
                            for row in jf_loc_cpu:
                                if int(row[JF_LOC_STATUS]) != JF_LOC_STATUS_FOUND:
                                    continue
                                print(
                                    "gpu_jungfrau: "
                                    f"timestamp={int(row[JF_LOC_TIMESTAMP])} "
                                    f"stream={int(row[JF_LOC_STREAM_ID])} "
                                    f"names_id={int(row[JF_LOC_NAMES_ID_VALUE])} "
                                    f"segment={int(row[JF_LOC_SEGMENT])} "
                                    f"raw_device_offset={int(row[JF_LOC_RAW_DEVICE_OFFSET])} "
                                    f"raw_nbytes={int(row[JF_LOC_RAW_NBYTES])} "
                                    f"shape=({int(row[JF_LOC_DIM0])},"
                                    f"{int(row[JF_LOC_DIM1])},"
                                    f"{int(row[JF_LOC_DIM2])}) "
                                    f"dtype_size={int(row[JF_LOC_DTYPE_SIZE])}"
                                )

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

                        # Collect data for comparison with non-split dgrams
                        per_stream = split_by_ts.setdefault(evt.timestamp, {})
                        for stream_id, dg in enumerate(dgrams):
                            if dg is None:
                                per_stream.setdefault(stream_id, (0, None))
                                continue
                            dg_bytes = bytearray(dg)
                            per_stream[stream_id] = (len(dg_bytes), digest_bytes(dg_bytes))

                        if args.compare_nosplit:
                            n_compare_streams = compare_split_event(ref_by_ts, split_by_ts, evt.timestamp)
                            print(f"compare_ok timestamp={evt.timestamp} streams={n_compare_streams}")
                            if ref_jungfrau_raw_by_ts is not None:
                                raw_shape = compare_jungfrau_raw(
                                    ref_jungfrau_raw_by_ts,
                                    gpu_jungfrau_raw_by_ts,
                                    evt.timestamp,
                                )
                                print(
                                    f"gpu_raw_compare_ok timestamp={evt.timestamp} "
                                    f"shape={raw_shape}"
                                )

                        n_events += 1
                        none_streams = [i for i, dg in enumerate(dgrams) if dg is None]
                        n_none = len(none_streams)
                        print(
                            f"bd_event: svc:{evt.service()} timestamp:{evt.timestamp} "
                            f"n_dgrams:{len(dgrams)} n_none:{n_none} none_streams:{none_streams}"
                        )
                        if args.max_events > 0 and n_events >= args.max_events:
                            stop = True
                            break

                    if evt_manager.exit_id:
                        raise RuntimeError(f"EventManager failed with exit_id={evt_manager.exit_id}")

                    if stop:
                        break
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
        if raw_offset_cache is not None:
            raw_offset_cache.close()

if __name__ == "__main__":
    main()
