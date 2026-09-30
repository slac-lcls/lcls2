import struct
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


# GPU batch ABI v1.
#
# Layout:
#   [GpuBatchHeader][GpuEventTable][GpuDescTable]
#
# All table fields are little-endian uint64. Keep this table CPU/GPU friendly:
# fixed-width rows, no Python objects, no PacketFooter, no embedded fd values.
GPU_BATCH_MAGIC = int.from_bytes(b"GPUBAT1\0", "little")
GPU_BATCH_VERSION = 1

GPU_HEADER_FMT = "<11Q"
GPU_EVENT_FMT = "<5Q"
GPU_DESC_FMT = "<7Q"

GPU_HEADER_NBYTES = struct.calcsize(GPU_HEADER_FMT)
GPU_EVENT_NBYTES = struct.calcsize(GPU_EVENT_FMT)
GPU_DESC_NBYTES = struct.calcsize(GPU_DESC_FMT)

GPU_DESC_FLAG_VALID = 1

# Header fields:
#   magic
#   version
#   header_nbytes
#   event_entry_nbytes
#   desc_entry_nbytes
#   n_events
#   n_desc
#   gpu_stream_mask
#   event_table_offset
#   desc_table_offset
#   total_nbytes

# Event row fields:
#   batch_event_index
#   timestamp
#   first_desc
#   n_desc
#   present_gpu_mask

GPU_EVENT_DTYPE = np.dtype(
    [
        ("batch_event_index", "<u8"),
        ("timestamp", "<u8"),
        ("first_desc", "<u8"),
        ("n_desc", "<u8"),
        ("present_gpu_mask", "<u8"),
    ]
)

# Desc row fields:
#   batch_event_index
#   stream_id
#   bd_offset
#   bd_size
#   smd_size
#   flags
#   reserved

GPU_DESC_DTYPE = np.dtype(
    [
        ("batch_event_index", "<u8"),
        ("stream_id", "<u8"),
        ("bd_offset", "<u8"),
        ("bd_size", "<u8"),
        ("smd_size", "<u8"),
        ("flags", "<u8"),
        ("reserved", "<u8"),
    ]
)


class GpuBatchFormatError(ValueError):
    pass


@dataclass(frozen=True)
class GpuBatchHeader:
    magic: int
    version: int
    header_nbytes: int
    event_entry_nbytes: int
    desc_entry_nbytes: int
    n_events: int
    n_desc: int
    gpu_stream_mask: int
    event_table_offset: int
    desc_table_offset: int
    total_nbytes: int


@dataclass(frozen=True)
class GpuBatchEvent:
    batch_event_index: int
    timestamp: int
    first_desc: int
    n_desc: int
    present_gpu_mask: int


@dataclass(frozen=True)
class GpuReadDesc:
    batch_event_index: int
    timestamp: int
    stream_id: int
    fd: int
    offset: int
    size: int
    smd_size: int
    flags: int


class GpuBatchView:
    def __init__(self, view, validate=True):
        self._view = memoryview(view).cast("B")
        self.header = self._parse_header()
        if validate:
            self._validate_header()

        self.events = np.frombuffer(
            self._view,
            dtype=GPU_EVENT_DTYPE,
            count=self.header.n_events,
            offset=self.header.event_table_offset,
        )
        self.descs = np.frombuffer(
            self._view,
            dtype=GPU_DESC_DTYPE,
            count=self.header.n_desc,
            offset=self.header.desc_table_offset,
        )

        if validate:
            self._validate_tables()

    def _parse_header(self):
        if len(self._view) < GPU_HEADER_NBYTES:
            raise GpuBatchFormatError(
                f"gpu_batch too small for header: {len(self._view)} bytes"
            )

        values = struct.unpack_from(GPU_HEADER_FMT, self._view, 0)
        return GpuBatchHeader(*map(int, values))

    def _validate_header(self):
        h = self.header

        if h.magic != GPU_BATCH_MAGIC:
            raise GpuBatchFormatError(f"bad gpu batch magic: 0x{h.magic:x}")

        if h.version != GPU_BATCH_VERSION:
            raise GpuBatchFormatError(f"unsupported gpu batch version: {h.version}")

        if h.header_nbytes != GPU_HEADER_NBYTES:
            raise GpuBatchFormatError(
                f"header size mismatch: {h.header_nbytes} != {GPU_HEADER_NBYTES}"
            )

        if h.event_entry_nbytes != GPU_EVENT_NBYTES:
            raise GpuBatchFormatError(
                f"event entry size mismatch: {h.event_entry_nbytes} != {GPU_EVENT_NBYTES}"
            )

        if h.desc_entry_nbytes != GPU_DESC_NBYTES:
            raise GpuBatchFormatError(
                f"desc entry size mismatch: {h.desc_entry_nbytes} != {GPU_DESC_NBYTES}"
            )

        if h.total_nbytes != len(self._view):
            raise GpuBatchFormatError(
                f"total size mismatch: header={h.total_nbytes} actual={len(self._view)}"
            )

        expected_event_offset = GPU_HEADER_NBYTES
        expected_desc_offset = h.event_table_offset + h.n_events * h.event_entry_nbytes
        expected_total = h.desc_table_offset + h.n_desc * h.desc_entry_nbytes

        if h.event_table_offset != expected_event_offset:
            raise GpuBatchFormatError(
                f"event table offset mismatch: {h.event_table_offset} != {expected_event_offset}"
            )

        if h.desc_table_offset != expected_desc_offset:
            raise GpuBatchFormatError(
                f"desc table offset mismatch: {h.desc_table_offset} != {expected_desc_offset}"
            )

        if h.total_nbytes != expected_total:
            raise GpuBatchFormatError(
                f"computed total mismatch: {h.total_nbytes} != {expected_total}"
            )

    def _validate_tables(self):
        h = self.header

        for i_evt, event_row in enumerate(self.events):
            batch_event_index = int(event_row["batch_event_index"])
            first_desc = int(event_row["first_desc"])
            n_desc = int(event_row["n_desc"])

            if batch_event_index != i_evt:
                raise GpuBatchFormatError(
                    f"event row {i_evt} has batch_event_index={batch_event_index}"
                )

            if first_desc + n_desc > h.n_desc:
                raise GpuBatchFormatError(
                    f"event {i_evt} desc range outside table: "
                    f"first_desc={first_desc} n_desc={n_desc} n_desc_total={h.n_desc}"
                )

            for desc_row in self.descs[first_desc:first_desc + n_desc]:
                desc_event_index = int(desc_row["batch_event_index"])
                stream_id = int(desc_row["stream_id"])
                flags = int(desc_row["flags"])

                if desc_event_index != batch_event_index:
                    raise GpuBatchFormatError(
                        f"desc event mismatch: desc={desc_event_index} "
                        f"event={batch_event_index}"
                    )

                if flags & GPU_DESC_FLAG_VALID:
                    if stream_id >= 64:
                        raise GpuBatchFormatError(
                            f"stream_id too large for mask: {stream_id}"
                        )

                    if not (h.gpu_stream_mask & (1 << stream_id)):
                        raise GpuBatchFormatError(
                            f"desc stream {stream_id} not present in gpu_stream_mask"
                        )

    @property
    def has_work(self):
        return self.header.n_desc > 0

    @property
    def total_read_bytes(self):
        total = 0
        for row in self.descs:
            if int(row["flags"]) & GPU_DESC_FLAG_VALID:
                total += int(row["bd_size"])
        return total

    def iter_events(self) -> Iterator[GpuBatchEvent]:
        for row in self.events:
            yield GpuBatchEvent(
                batch_event_index=int(row["batch_event_index"]),
                timestamp=int(row["timestamp"]),
                first_desc=int(row["first_desc"]),
                n_desc=int(row["n_desc"]),
                present_gpu_mask=int(row["present_gpu_mask"]),
            )

    def desc_rows_for_event(self, event_index):
        event_row = self.events[event_index]
        first_desc = int(event_row["first_desc"])
        n_desc = int(event_row["n_desc"])
        return self.descs[first_desc:first_desc + n_desc]

    def iter_read_descs(
        self,
        bd_dm,
        event_index: Optional[int] = None,
    ) -> Iterator[GpuReadDesc]:
        if event_index is None:
            event_indices = range(self.header.n_events)
        else:
            event_indices = [event_index]

        for i_evt in event_indices:
            event_row = self.events[i_evt]
            timestamp = int(event_row["timestamp"])

            for desc_row in self.desc_rows_for_event(i_evt):
                flags = int(desc_row["flags"])
                if not (flags & GPU_DESC_FLAG_VALID):
                    continue

                stream_id = int(desc_row["stream_id"])
                if stream_id >= len(bd_dm.fds):
                    raise GpuBatchFormatError(
                        f"stream_id {stream_id} outside bd_dm.fds length {len(bd_dm.fds)}"
                    )

                yield GpuReadDesc(
                    batch_event_index=int(desc_row["batch_event_index"]),
                    timestamp=timestamp,
                    stream_id=stream_id,
                    fd=int(bd_dm.fds[stream_id]),
                    offset=int(desc_row["bd_offset"]),
                    size=int(desc_row["bd_size"]),
                    smd_size=int(desc_row["smd_size"]),
                    flags=flags,
                )
