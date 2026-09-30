"""CPU-only, adjacent-range planning for one admitted GPU input window.

The caller resolves file/chunk identities and limits this input to one batch
and one transition-safe window. This module neither opens files nor decides
event selection, residency, or CUDA lifetimes. GPUBAT1 is not modified.
"""

from dataclasses import dataclass, replace
from operator import index
from typing import Iterable, Optional, Tuple


_U64_MAX = (1 << 64) - 1


def _uint64(name, value):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        value = index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if not 0 <= value <= _U64_MAX:
        raise ValueError(f"{name} must fit uint64: {value}")
    return value


@dataclass(frozen=True, order=True)
class ResolvedFile:
    """Caller-resolved file identity; no path lookup or normalization is done.

    The caller must supply the same canonical path/chunk pair for the same
    physical file, and distinct identities for different files or chunks.
    """

    path: str
    chunk_id: int

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise TypeError("path must be a string")
        if not self.path:
            raise ValueError("path must not be empty")
        object.__setattr__(self, "chunk_id", _uint64("chunk_id", self.chunk_id))


@dataclass(frozen=True)
class ResolvedDgram:
    """One logical stream dgram, after file resolution on the BD.

    Missing dgrams have no row. A supplied zero-size row is retained but
    causes no read. Multiple detector consumers must share one stream row,
    rather than supplying duplicate (event, stream) rows.
    """

    batch_event_index: int
    timestamp: int
    stream_id: int
    file: ResolvedFile
    file_offset: int
    size: int
    smd_size: int = 0

    def __post_init__(self):
        if not isinstance(self.file, ResolvedFile):
            raise TypeError("file must be a ResolvedFile")
        for name in (
            "batch_event_index", "timestamp", "stream_id",
            "file_offset", "size", "smd_size",
        ):
            object.__setattr__(self, name, _uint64(name, getattr(self, name)))
        if self.stream_id >= 64:
            raise ValueError("stream_id must fit the GPUBAT1 64-stream mask")
        if self.size > _U64_MAX - self.file_offset:
            raise ValueError("file_offset + size exceeds uint64")


@dataclass(frozen=True)
class ReadRange:
    """One physical read into the plan's single input-window allocation."""

    file: ResolvedFile
    file_offset: int
    size: int
    device_offset: int


@dataclass(frozen=True)
class LogicalDgram:
    """Original descriptor and its location in the physical read plan.

    Tuple position in ReadPlan.logical_dgrams is the parser-row index.
    Zero-size dgrams have range_index=None and device_offset=0.
    """

    source: ResolvedDgram
    range_index: Optional[int]
    device_offset: int


@dataclass(frozen=True)
class ReadPlan:
    """Immutable result; offsets are relative to this input window's base.

    capacity_bytes is the required allocation, not the caller's admission
    limit. Adjacent-only planning adds no gaps or padding, so fetched_bytes,
    useful_bytes, and capacity_bytes are equal. No parser/output bytes are
    included: their admission is the caller's responsibility.
    """

    batch_id: int
    input_window_id: int
    physical_ranges: Tuple[ReadRange, ...]
    logical_dgrams: Tuple[LogicalDgram, ...]
    useful_bytes: int
    fetched_bytes: int
    capacity_bytes: int

    @property
    def n_reads(self):
        """Number of psana-level pread submissions, not KvikIO subtasks."""
        return len(self.physical_ranges)


def build_read_plan(
    descriptors: Iterable[ResolvedDgram],
    *,
    capacity_bytes: int,
    max_read_bytes: Optional[int] = None,
    batch_id: int = 0,
    input_window_id: int = 0,
) -> ReadPlan:
    """Merge adjacent dgrams per resolved file, preserving logical row order.

    Physical ranges are ordered by (path, chunk_id, file_offset). Dgrams are
    indivisible; a range cap can separate adjacent dgrams but cannot split one.
    The capacity limit covers the entire raw-input window, not each request.
    Oversized windows/dgrams, duplicate event/stream identities, inconsistent
    event timestamps, and overlapping nonempty file spans raise ValueError.

    Event indices may be sparse and rows interleaved or physically unordered.
    The caller must not mix EB batches or transition fences in one invocation.
    This routine does not add flags, create missing rows, or mutate inputs.
    """
    capacity_bytes = _uint64("capacity_bytes", capacity_bytes)
    batch_id = _uint64("batch_id", batch_id)
    input_window_id = _uint64("input_window_id", input_window_id)
    if max_read_bytes is None:
        max_read_bytes = _U64_MAX
    else:
        max_read_bytes = _uint64("max_read_bytes", max_read_bytes)
        if max_read_bytes == 0:
            raise ValueError("max_read_bytes must be positive")

    descriptors = tuple(descriptors)
    seen = set()
    timestamps = {}
    useful_bytes = 0
    for desc in descriptors:
        if not isinstance(desc, ResolvedDgram):
            raise TypeError("descriptors must contain ResolvedDgram records")
        key = (desc.batch_event_index, desc.stream_id)
        if key in seen:
            raise ValueError(f"duplicate event/stream descriptor: {key}")
        seen.add(key)
        previous_ts = timestamps.setdefault(desc.batch_event_index, desc.timestamp)
        if previous_ts != desc.timestamp:
            raise ValueError(f"inconsistent timestamp for event {desc.batch_event_index}")
        if desc.size > max_read_bytes:
            raise ValueError(
                f"dgram event={desc.batch_event_index} stream={desc.stream_id} "
                f"size={desc.size} exceeds max_read_bytes={max_read_bytes}"
            )
        useful_bytes += desc.size
        if useful_bytes > _U64_MAX:
            raise ValueError("total input size exceeds uint64")
    if useful_bytes > capacity_bytes:
        raise ValueError(
            f"input window requires {useful_bytes} bytes; capacity_bytes={capacity_bytes}"
        )

    ordered = sorted(
        (i for i, desc in enumerate(descriptors) if desc.size),
        key=lambda i: (descriptors[i].file, descriptors[i].file_offset),
    )
    ranges = []
    logical = [LogicalDgram(desc, None, 0) for desc in descriptors]
    cursor = 0
    for i in ordered:
        desc = descriptors[i]
        previous = ranges[-1] if ranges else None
        adjacent = False
        if previous is not None and previous.file == desc.file:
            end = previous.file_offset + previous.size
            if desc.file_offset < end:
                raise ValueError(
                    f"overlapping dgrams in {desc.file}: offset={desc.file_offset} "
                    f"precedes previous end={end}"
                )
            adjacent = desc.file_offset == end
        if adjacent and previous.size + desc.size <= max_read_bytes:
            ranges[-1] = replace(previous, size=previous.size + desc.size)
        else:
            ranges.append(ReadRange(desc.file, desc.file_offset, desc.size, cursor))
        read_range = ranges[-1]
        logical[i] = LogicalDgram(
            desc, len(ranges) - 1,
            read_range.device_offset + desc.file_offset - read_range.file_offset,
        )
        cursor += desc.size

    return ReadPlan(
        batch_id, input_window_id, tuple(ranges), tuple(logical),
        useful_bytes, cursor, cursor,
    )
