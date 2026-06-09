import hashlib
import os

from psana.dgrammanager import DgramManager
from psana.event import Event
from psana.psexp import TransitionId
from psana.psexp.event_manager import EventManager
from psana.psexp.eventbuilder_manager import EventBuilderManager


def digest_bytes(view):
    return hashlib.blake2b(memoryview(view), digest_size=16).hexdigest()


def compare_split_event(ref_by_ts, split_by_ts, timestamp, max_mismatches=5):
    ref_event = ref_by_ts.get(timestamp)
    split_event = split_by_ts.get(timestamp)

    if ref_event is None:
        raise RuntimeError(f"Missing reference event for timestamp {timestamp}")

    if split_event is None:
        raise RuntimeError(f"Missing split event for timestamp {timestamp}")

    mismatches = []
    for stream_id, ref_value in ref_event.items():
        split_value = split_event.get(stream_id)
        if split_value != ref_value:
            mismatches.append((stream_id, ref_value, split_value))

    if mismatches:
        detail = ", ".join(
            f"stream={stream_id} ref={ref_value} split={split_value}"
            for stream_id, ref_value, split_value in mismatches[:max_mismatches]
        )
        raise RuntimeError(
            f"Split/no-split mismatch timestamp={timestamp}: {detail}"
        )

    return len(ref_event)


def collect_no_split_reference(smd_chunk, configs, dsparms, xtc_files, max_events=0):
    """
    Build no-GPU-split BigData reference events for one EB-ready SMD chunk.

    Returns:
        dict[timestamp][stream_id] = (dgram_size, digest)
    """
    old_gpu_stream_ids = os.environ.pop("PS_TEST_GPU_STREAM_IDS", None)

    try:
        ref_bd_dm = DgramManager(xtc_files, configs=configs)
        ref_by_ts = {}
        n_ref_events = 0

        ref_eb_manager = EventBuilderManager(smd_chunk, configs, dsparms)
        for batch_dict, _ in ref_eb_manager.batches():
            for smd_batch, _ in batch_dict.values():
                if not smd_batch:
                    continue

                ref_evt_manager = EventManager(
                    smd_batch,
                    configs,
                    ref_bd_dm,
                    dsparms.max_retries,
                    [False] * len(configs),
                )

                for dgrams in ref_evt_manager:
                    evt = Event(dgrams=dgrams)
                    if not TransitionId.isEvent(evt.service()):
                        continue

                    per_stream = {}
                    for stream_id, dg in enumerate(dgrams):
                        if dg is None:
                            per_stream[stream_id] = (0, None)
                            continue

                        dg_bytes = bytearray(dg)
                        per_stream[stream_id] = (
                            len(dg_bytes),
                            digest_bytes(dg_bytes),
                        )

                    ref_by_ts[evt.timestamp] = per_stream
                    n_ref_events += 1

                    if max_events and n_ref_events >= max_events:
                        return ref_by_ts

                if ref_evt_manager.exit_id:
                    raise RuntimeError(
                        f"Reference EventManager failed with exit_id={ref_evt_manager.exit_id}"
                    )

        return ref_by_ts

    finally:
        if "ref_bd_dm" in locals():
            ref_bd_dm.close()

        if old_gpu_stream_ids is not None:
            os.environ["PS_TEST_GPU_STREAM_IDS"] = old_gpu_stream_ids
