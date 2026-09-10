"""Contracts between the GPU XTC input path and detector consumers."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from psana.gpu.gpudgram.config import GpuFieldHandle


@dataclass(frozen=True)
class GpuStreamDgramView:
    """One event's parsed dgram in a physical input stream.

    The view does not own another copy of the XTC bytes. ``batch`` remains the
    owner of ``data_gpu`` and all parser-produced device tables.
    """

    stream_id: int
    dgram_index: int
    batch: object

    @property
    def data_gpu(self):
        return self.batch.data_gpu

    def locate(self, handle, *, stream=None):
        """Return the batch locator table for a field in this stream."""
        if not isinstance(handle, GpuFieldHandle):
            raise TypeError("handle must be a GpuFieldHandle")
        if int(handle.stream_id) != self.stream_id:
            raise ValueError(
                f"field handle belongs to stream {handle.stream_id}, "
                f"not stream {self.stream_id}"
            )
        return self.batch.locate(handle, stream=stream)


class GpuEventDgrams(Mapping):
    """Event-scoped mapping ``dgrams[stream_id]`` over a parsed GPU batch."""

    def __init__(self, event, batch):
        if batch is None:
            raise ValueError("batch is required")
        stream_ids = getattr(batch, "stream_ids_by_dgram", None)
        if stream_ids is None:
            raise ValueError(
                "GpuEventDgrams requires batch.stream_ids_by_dgram"
            )
        stream_ids = np.asarray(stream_ids)
        n_dgrams = int(getattr(batch, "n_dgrams", stream_ids.size))
        if stream_ids.shape != (n_dgrams,):
            raise ValueError(
                "batch.stream_ids_by_dgram shape does not match n_dgrams"
            )

        first = int(event.first_desc)
        count = int(event.n_desc)
        end = first + count
        if first < 0 or count < 0 or end > n_dgrams:
            raise ValueError(
                f"event dgram range [{first}, {end}) is outside batch with "
                f"{n_dgrams} dgrams"
            )

        dgrams = {}
        for dgram_index in range(first, end):
            stream_id = int(stream_ids[dgram_index])
            if stream_id in dgrams:
                raise RuntimeError(
                    f"event {int(event.timestamp)} has duplicate GPU dgrams "
                    f"for stream {stream_id}"
                )
            dgrams[stream_id] = GpuStreamDgramView(
                stream_id=stream_id,
                dgram_index=dgram_index,
                batch=batch,
            )

        self.event = event
        self.batch = batch
        self._dgrams = MappingProxyType(dgrams)

    @classmethod
    def from_batch(cls, gpu_view, batch):
        """Build the event views once for all consumers of a parsed batch."""
        return tuple(cls(event, batch) for event in gpu_view.iter_events())

    @property
    def timestamp(self):
        return int(self.event.timestamp)

    @property
    def batch_event_index(self):
        return int(self.event.batch_event_index)

    @property
    def dgrams(self):
        """Read-only stream-indexed mapping, matching ``evt.gpu.dgrams``."""
        return self._dgrams

    def __getitem__(self, stream_id):
        return self._dgrams[int(stream_id)]

    def __iter__(self):
        return iter(self._dgrams)

    def __len__(self):
        return len(self._dgrams)


class GpuDetectorBinding:
    """Run-scoped detector segments bound to Configure field handles.

    This object establishes detector membership and canonical segment order.
    It does not impose a payload shape or calibration policy.
    """

    def __init__(
        self,
        det_name,
        *,
        canonical_segment_ids,
        field_handles_by_segment,
    ):
        self.det_name = str(det_name)
        canonical = tuple(int(segment) for segment in canonical_segment_ids)
        if len(set(canonical)) != len(canonical):
            raise ValueError("canonical_segment_ids contains duplicates")

        handles = {
            int(segment): handle
            for segment, handle in field_handles_by_segment.items()
        }
        if set(handles) != set(canonical):
            raise ValueError(
                "field_handles_by_segment must identify every canonical "
                f"segment exactly once: handles={sorted(handles)}, "
                f"canonical={sorted(canonical)}"
            )
        for segment, handle in handles.items():
            if not isinstance(handle, GpuFieldHandle):
                raise TypeError(
                    f"field handle for segment {segment} must be a "
                    "GpuFieldHandle"
                )

        rows = {segment: row for row, segment in enumerate(canonical)}
        sources_by_stream = {}
        for segment in canonical:
            handle = handles[segment]
            sources_by_stream.setdefault(int(handle.stream_id), []).append(
                (rows[segment], segment, handle)
            )

        self._canonical_segment_ids = canonical
        self._field_handles_by_segment = MappingProxyType(handles)
        self._canonical_segment_rows = MappingProxyType(rows)
        self._sources_by_stream = MappingProxyType({
            stream_id: tuple(sources)
            for stream_id, sources in sources_by_stream.items()
        })

    @property
    def canonical_segment_ids(self):
        return self._canonical_segment_ids

    @property
    def field_handles_by_segment(self):
        return self._field_handles_by_segment

    @property
    def canonical_segment_rows(self):
        return self._canonical_segment_rows

    @property
    def sources_by_stream(self):
        return self._sources_by_stream

    @property
    def stream_ids(self):
        return tuple(self._sources_by_stream)

    def has_sources(self, event_dgrams):
        return any(stream_id in event_dgrams for stream_id in self.stream_ids)

    def iter_sources(self, event_dgrams):
        """Yield present sources in canonical segment order.

        Missing stream dgrams are omitted. The current calibration adapter
        preserves its existing behavior by leaving those output rows zero.
        """
        for segment in self._canonical_segment_ids:
            handle = self._field_handles_by_segment[segment]
            dgram = event_dgrams.get(handle.stream_id)
            if dgram is None:
                continue
            yield (
                dgram,
                self._canonical_segment_rows[segment],
                segment,
                handle,
            )


__all__ = [
    "GpuDetectorBinding",
    "GpuEventDgrams",
    "GpuStreamDgramView",
]
