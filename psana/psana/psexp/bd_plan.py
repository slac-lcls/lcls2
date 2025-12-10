import json
import struct
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np

from psana import dgram
from psana.psexp.transitionid import TransitionId
from psana.psexp.packet_footer import PacketFooter


_PLAN_MAGIC = b"BDPL"
_PLAN_STRUCT = struct.Struct("<4sII")  # magic, version, plan_size
_PLAN_VERSION = 1


def _default_use_smds(use_smds, n_files):
    if use_smds:
        return np.array(use_smds, dtype=bool)
    return np.zeros(n_files, dtype=bool)


def compute_offset_tables(
    view,
    configs,
    use_smds,
    bd_chunksize,
    chunk_id_cb: Optional[Callable[[int], int]] = None,
):
    """
    Parse the smd chunk and build offset/size arrays along with chunk metadata.
    Returns a SimpleNamespace so callers can pick whichever arrays they need.
    """
    smd_chunk_pf = PacketFooter(view=view)
    n_events = smd_chunk_pf.n_packets
    n_smd_files = len(configs)
    dtype = np.int64

    bd_offset_array = np.zeros((n_events, n_smd_files), dtype=dtype)
    bd_size_array = np.zeros((n_events, n_smd_files), dtype=dtype)
    smd_offset_array = np.zeros((n_events, n_smd_files), dtype=dtype)
    smd_size_array = np.zeros((n_events, n_smd_files), dtype=dtype)
    new_chunk_id_array = np.zeros((n_events, n_smd_files), dtype=dtype)
    cutoff_flag_array = np.ones((n_events, n_smd_files), dtype=dtype)
    service_array = np.zeros((n_events, n_smd_files), dtype=dtype)

    smd_aux_sizes = np.zeros(n_smd_files, dtype=dtype)
    current_bd_offsets = np.zeros(n_smd_files, dtype=dtype)
    current_bd_chunk_sizes = np.zeros(n_smd_files, dtype=dtype)
    use_smds_arr = _default_use_smds(use_smds, n_smd_files)
    chunkinfo = {}

    offset = 0
    i_evt = 0
    i_smd = 0
    i_first_L1 = -1

    mv = memoryview(view)
    footer_nbytes = memoryview(smd_chunk_pf.footer).nbytes
    chunk_id_cb = chunk_id_cb or (lambda _i: 0)

    while offset < mv.nbytes - footer_nbytes:
        if i_smd == 0:
            smd_evt_size = smd_chunk_pf.get_size(i_evt)
            smd_evt_pf = PacketFooter(view=mv[offset : offset + smd_evt_size])
            smd_aux_sizes[:] = [
                smd_evt_pf.get_size(i) for i in range(smd_evt_pf.n_packets)
            ]

        if smd_aux_sizes[i_smd] == 0:
            cutoff_flag_array[i_evt, i_smd] = 0
        else:
            d = dgram.Dgram(config=configs[i_smd], view=view, offset=offset)

            smd_offset_array[i_evt, i_smd] = offset
            smd_size_array[i_evt, i_smd] = d._size
            service_array[i_evt, i_smd] = d.service()

            if TransitionId.isEvent(d.service()) and not use_smds_arr[i_smd]:
                if i_first_L1 == -1:
                    i_first_L1 = i_evt

                bd_offset_array[i_evt, i_smd] = d.smdinfo[0].offsetAlg.intOffset
                bd_size_array[i_evt, i_smd] = d.smdinfo[0].offsetAlg.intDgramSize

                if (
                    current_bd_offsets[i_smd] == bd_offset_array[i_evt, i_smd]
                    and i_evt != i_first_L1
                    and current_bd_chunk_sizes[i_smd] + bd_size_array[i_evt, i_smd]
                    < bd_chunksize
                ):
                    cutoff_flag_array[i_evt, i_smd] = 0
                    current_bd_chunk_sizes[i_smd] += bd_size_array[i_evt, i_smd]
                else:
                    current_bd_chunk_sizes[i_smd] = bd_size_array[i_evt, i_smd]

                current_bd_offsets[i_smd] = (
                    bd_offset_array[i_evt, i_smd] + bd_size_array[i_evt, i_smd]
                )

            elif d.service() == TransitionId.Enable and hasattr(d, "chunkinfo"):
                _chunk_ids = [
                    getattr(d.chunkinfo[seg_id].chunkinfo, "chunkid")
                    for seg_id in d.chunkinfo
                ]
                _chunk_filenames = [
                    getattr(d.chunkinfo[seg_id].chunkinfo, "filename")
                    for seg_id in d.chunkinfo
                ]
                if _chunk_ids:
                    new_chunk_id = _chunk_ids[0]
                    new_filename = _chunk_filenames[0]
                    current_chunk_id = chunk_id_cb(i_smd)
                    if new_chunk_id > current_chunk_id:
                        new_chunk_id_array[i_evt, i_smd] = new_chunk_id
                        chunkinfo[(i_smd, new_chunk_id)] = new_filename

        offset += smd_aux_sizes[i_smd]
        i_smd += 1
        if i_smd == n_smd_files:
            offset += PacketFooter.n_bytes * (n_smd_files + 1)
            i_smd = 0
            i_evt += 1

    cutoff_indices = []
    for i_smd in range(n_smd_files):
        cutoff_indices.append(np.where(cutoff_flag_array[:, i_smd] == 1)[0])

    return SimpleNamespace(
        bd_offset_array=bd_offset_array,
        bd_size_array=bd_size_array,
        smd_offset_array=smd_offset_array,
        smd_size_array=smd_size_array,
        new_chunk_id_array=new_chunk_id_array,
        cutoff_flag_array=cutoff_flag_array,
        cutoff_indices=cutoff_indices,
        service_array=service_array,
        chunkinfo=chunkinfo,
        n_events=n_events,
        n_smd_files=n_smd_files,
    )


def build_chunk_descriptors(
    bd_offset_array,
    bd_size_array,
    cutoff_indices: List[np.ndarray],
    n_events: int,
    file_names: Optional[List[str]] = None,
):
    descriptors = []
    n_files = bd_offset_array.shape[1]
    for i_smd in range(n_files):
        indices = cutoff_indices[i_smd]
        if indices.size == 0:
            continue

        fname = None
        if file_names and i_smd < len(file_names):
            fname = file_names[i_smd]

        for chunk_idx, start_evt in enumerate(indices):
            if chunk_idx + 1 < indices.shape[0]:
                end_evt = int(indices[chunk_idx + 1])
            else:
                end_evt = int(n_events)

            start_evt = int(start_evt)
            if end_evt <= start_evt:
                continue

            total_bytes = int(np.sum(bd_size_array[start_evt:end_evt, i_smd]))
            if total_bytes == 0:
                continue

            entry = {
                "file_index": i_smd,
                "chunk_index": chunk_idx,
                "start_evt": start_evt,
                "end_evt": end_evt,
                "n_offsets": int(end_evt - start_evt),
                "start_offset": int(bd_offset_array[start_evt, i_smd]),
                "total_bytes": total_bytes,
            }
            if fname:
                entry["file_name"] = fname
            descriptors.append(entry)
    return descriptors


def compute_bd_plan(
    view,
    configs,
    use_smds,
    bd_chunksize,
    file_names: Optional[List[str]] = None,
    chunk_id_cb: Optional[Callable[[int], int]] = None,
):
    offsets = compute_offset_tables(
        view, configs, use_smds, bd_chunksize, chunk_id_cb=chunk_id_cb
    )
    descriptors = build_chunk_descriptors(
        offsets.bd_offset_array,
        offsets.bd_size_array,
        offsets.cutoff_indices,
        offsets.n_events,
        file_names=file_names,
    )
    if not descriptors:
        return None, offsets

    plan = {
        "version": _PLAN_VERSION,
        "n_events": offsets.n_events,
        "n_smd_files": offsets.n_smd_files,
        "chunks": descriptors,
    }
    return plan, offsets


def encode_plan(plan_dict) -> bytes:
    return json.dumps(plan_dict).encode("utf-8")


def wrap_plan_with_batch(plan_bytes: bytes, batch: bytearray):
    header = _PLAN_STRUCT.pack(_PLAN_MAGIC, _PLAN_VERSION, len(plan_bytes))
    wrapped = bytearray()
    wrapped += header
    wrapped += plan_bytes
    wrapped += batch
    return wrapped


def extract_plan(buffer: bytearray) -> Tuple[Optional[dict], object]:
    if len(buffer) < _PLAN_STRUCT.size:
        return None, buffer

    header = buffer[: _PLAN_STRUCT.size]
    magic, version, plan_size = _PLAN_STRUCT.unpack(header)
    if magic != _PLAN_MAGIC or version != _PLAN_VERSION:
        return None, buffer

    total_size = _PLAN_STRUCT.size + plan_size
    if len(buffer) < total_size:
        return None, buffer

    plan_bytes = bytes(buffer[_PLAN_STRUCT.size:total_size])
    try:
        plan = json.loads(plan_bytes.decode("utf-8"))
    except json.JSONDecodeError:
        return None, buffer

    payload_view = memoryview(buffer)[total_size:]
    return plan, payload_view


def plan_from_event_manager(evtman) -> Optional[dict]:
    descriptors = build_chunk_descriptors(
        evtman.bd_offset_array,
        evtman.bd_size_array,
        evtman.cutoff_indices,
        evtman.n_events,
        getattr(evtman.dm, "xtc_files", None),
    )
    if not descriptors:
        return None
    return {
        "version": _PLAN_VERSION,
        "n_events": evtman.n_events,
        "n_smd_files": evtman.n_smd_files,
        "chunks": descriptors,
    }


def partition_plan(chunks: List[dict], n_slices: int) -> List[List[dict]]:
    """
    Partition the chunk list into n_slices contiguous ranges.
    Returns a list where entry i contains the slice assigned to rank i+1.
    """
    if n_slices <= 0:
        return [chunks]
    total = len(chunks)
    base = total // n_slices
    remainder = total % n_slices
    slices = []
    start = 0
    for i in range(n_slices):
        count = base + (1 if i < remainder else 0)
        end = start + count
        slices.append(chunks[start:end])
        start = end
    return slices
