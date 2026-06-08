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
        "--direct",
        action="store_true",
        help="Try O_DIRECT when opening SMD files.",
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


def _open_fd(path: str, use_direct: bool = False) -> int:
    """Open a file descriptor, optionally trying O_DIRECT first."""
    flags = os.O_RDONLY
    o_direct = getattr(os, "O_DIRECT", 0)
    if use_direct and o_direct:
        try:
            return os.open(path, flags | o_direct)
        except OSError:
            pass
    return os.open(path, flags)


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

    dsparms = _create_default_dsparms()
    dsparms.update_smd_state(list(smd_files), [False] * len(smd_files))
    smd_fds = np.array(
        [_open_fd(path, use_direct=args.direct) for path in smd_files],
        dtype=np.int32,
    )
    try:
        smd_manager = SmdReaderManager(smd_fds, dsparms)
        # Read Configure and BeginRun
        if smd_manager.get_next_dgrams() is None:
            raise RuntimeError("missing Configure")
        if smd_manager.get_next_dgrams() is None:
            raise RuntimeError("missing BeginRun")

        # Create DgramManager for bigdata reads
        bd_dm = DgramManager(xtc_files, configs=smd_manager.configs)

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

            # EventBuilder builds smd events by parsing the chunk header and walking the step buffers.
            eb_manager = EventBuilderManager(smd_chunk, smd_manager.configs, dsparms)
            n_batches = 0
            for batch_dict, gpu_batch_dict, _ in eb_manager.batches_with_gpu():
                n_batches += 1

                # gpu_batch_dict is kept separate. Later this is where GPU BD scheduling starts.
                for gpu_batch, _ in gpu_batch_dict.values():
                    if gpu_batch:
                        print(f"gpu_batch_bytes={len(gpu_batch)}")

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

if __name__ == "__main__":
    main()
