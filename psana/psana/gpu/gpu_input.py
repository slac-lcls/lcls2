"""Contracts between the GPU XTC input path and detector consumers."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from psana.gpu.gpudgram.config import GpuFieldHandle
from psana.gpu.gpudgram.batch import (
    LOC_DIM0,
    LOC_MAX_RANK,
    LOC_NBYTES,
    LOC_OFFSET,
    LOC_RANK,
    LOC_STATUS,
    LOC_TYPE,
)
from psana.gpu.gpudgram.parser import STATUS_FOUND, STATUS_NAMES, STATUS_NOT_PRESENT


_XTC_DTYPES = {
    0: np.dtype("u1"),
    1: np.dtype("u2"),
    2: np.dtype("u4"),
    3: np.dtype("u8"),
    4: np.dtype("i1"),
    5: np.dtype("i2"),
    6: np.dtype("i4"),
    7: np.dtype("i8"),
    8: np.dtype("f4"),
    9: np.dtype("f8"),
    # CHARSTR stays byte-addressable on the GPU. Decoding is a CPU policy.
    10: np.dtype("u1"),
    11: np.dtype("u4"),
    12: np.dtype("u4"),
}


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
    """Run-scoped detector segments bound to Configure fields.

    This object establishes detector membership and canonical segment order.
    It does not impose a payload shape or calibration policy.
    """

    def __init__(
        self,
        det_name,
        *,
        canonical_segment_ids,
        field_handles_by_segment,
        field_handles_by_name=None,
    ):
        self.det_name = str(det_name)
        canonical = tuple(int(segment) for segment in canonical_segment_ids)
        if len(set(canonical)) != len(canonical):
            raise ValueError("canonical_segment_ids contains duplicates")

        handles = {
            int(segment): handle
            for segment, handle in (field_handles_by_segment or {}).items()
        }
        if handles and set(handles) != set(canonical):
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
            handle = handles.get(segment)
            if handle is None:
                continue
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

        named_fields = {}
        for key, field_handles in (field_handles_by_name or {}).items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError(
                    "field_handles_by_name keys must be (alg_name, field_name)"
                )
            alg_name, field_name = (str(key[0]), str(key[1]))
            normalized = {
                int(segment): handle
                for segment, handle in field_handles.items()
            }
            unknown = set(normalized) - set(canonical)
            if unknown:
                raise ValueError(
                    f"field {(alg_name, field_name)!r} has unknown detector "
                    f"segments: {sorted(unknown)}"
                )
            for segment, handle in normalized.items():
                if not isinstance(handle, GpuFieldHandle):
                    raise TypeError(
                        f"field handle for {(alg_name, field_name)!r} segment "
                        f"{segment} must be a GpuFieldHandle"
                    )
            named_fields[(alg_name, field_name)] = GpuDetectorFieldBinding(
                self,
                alg_name=alg_name,
                field_name=field_name,
                field_handles_by_segment=normalized,
            )
        self._fields = MappingProxyType(named_fields)

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
    def fields(self):
        return self._fields

    @property
    def stream_ids(self):
        stream_ids = list(self._sources_by_stream)
        for field in self._fields.values():
            for stream_id in field.stream_ids:
                if stream_id not in stream_ids:
                    stream_ids.append(stream_id)
        return tuple(stream_ids)

    def has_sources(self, event_dgrams):
        return any(stream_id in event_dgrams for stream_id in self.stream_ids)

    def field(self, alg_name, field_name):
        key = (str(alg_name), str(field_name))
        try:
            return self._fields[key]
        except KeyError:
            available = [f"{alg}.{name}" for alg, name in self._fields]
            raise KeyError(
                f"{self.det_name}.{key[0]}.{key[1]} is not configured; "
                f"available fields: {available}"
            ) from None

    def iter_sources(self, event_dgrams):
        """Yield present sources in canonical segment order.

        Missing stream dgrams are omitted. The current calibration adapter
        preserves its existing behavior by leaving those output rows zero.
        """
        for segment in self._canonical_segment_ids:
            handle = self._field_handles_by_segment.get(segment)
            if handle is None:
                continue
            dgram = event_dgrams.get(handle.stream_id)
            if dgram is None:
                continue
            yield (
                dgram,
                self._canonical_segment_rows[segment],
                segment,
                handle,
            )


class GpuDetectorFieldBinding:
    """One named detector field bound to its configured physical segments."""

    def __init__(
        self,
        detector,
        *,
        alg_name,
        field_name,
        field_handles_by_segment,
    ):
        self.detector = detector
        self.alg_name = str(alg_name)
        self.field_name = str(field_name)
        self._field_handles_by_segment = MappingProxyType(
            dict(field_handles_by_segment)
        )

    @property
    def det_name(self):
        return self.detector.det_name

    @property
    def field_handles_by_segment(self):
        return self._field_handles_by_segment

    @property
    def segment_ids(self):
        return tuple(
            segment
            for segment in self.detector.canonical_segment_ids
            if segment in self._field_handles_by_segment
        )

    @property
    def stream_ids(self):
        return tuple(dict.fromkeys(
            int(self._field_handles_by_segment[segment].stream_id)
            for segment in self.segment_ids
        ))

    def iter_sources(self, event_dgrams, *, segment_ids=None):
        selected = None if segment_ids is None else {
            int(segment) for segment in segment_ids
        }
        for segment in self.segment_ids:
            if selected is not None and segment not in selected:
                continue
            handle = self._field_handles_by_segment[segment]
            dgram = event_dgrams.get(handle.stream_id)
            if dgram is not None:
                yield dgram, segment, handle


class InputSlotLease:
    """Protect one parsed input slot across any number of field consumers."""

    __slots__ = ("result_ready", "_consumer_done")

    def __init__(self, result_ready):
        self.result_ready = result_ready
        self._consumer_done = []

    def register_consumer_done(self, event):
        self._consumer_done.append(event)

    def wait_until_safe_to_reuse(self):
        for event in self._consumer_done:
            event.synchronize()


class GpuFieldData(Mapping):
    """Segment-preserving values for one detector algorithm field."""

    def __init__(self, field, values):
        self.field = field
        self._values = MappingProxyType(dict(values))

    @property
    def segment_ids(self):
        return tuple(self._values)

    def only(self):
        """Return the sole segment value, rejecting an ambiguous selection."""
        if len(self._values) != 1:
            raise ValueError(
                f"{self.field.det_name}.{self.field.alg_name}."
                f"{self.field.field_name} has {len(self._values)} segments; "
                "index it by segment id"
            )
        return next(iter(self._values.values()))

    def __getitem__(self, segment_id):
        return self._values[int(segment_id)]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return (
            f"GpuFieldData(field={self.field.det_name}."
            f"{self.field.alg_name}.{self.field.field_name}, "
            f"segments={list(self._values)})"
        )


class _GpuFieldViewContext:
    __slots__ = ("_result", "_stream", "_exited")

    def __init__(self, result, stream):
        self._result = result
        self._stream = stream
        self._exited = False

    def __enter__(self):
        import cupy as cp

        stream = self._stream or cp.cuda.Stream.null
        ready = self._result._lease.result_ready
        if ready is not None:
            stream.wait_event(ready)
        return self._result._slot_views()

    def __exit__(self, *_):
        import cupy as cp

        stream = self._stream or cp.cuda.Stream.null
        done = cp.cuda.Event(disable_timing=True)
        stream.record(done)
        self._result._lease.register_consumer_done(done)
        self._exited = True

    def __del__(self):
        if not self._exited:
            try:
                import cupy as cp

                done = cp.cuda.Event(disable_timing=True)
                cp.cuda.Stream.null.record(done)
                self._result._lease.register_consumer_done(done)
            except Exception:
                pass


class GpuFieldResult:
    """Event-scoped access to one parsed detector field.

    Values are mappings keyed by physical segment id. This keeps arbitrary
    detector layouts, scalars, and ragged segment shapes representable without
    imposing the dense layout required by the calibration adapter.
    """

    __slots__ = (
        "binding",
        "event_dgrams",
        "_lease",
        "_segment_ids",
        "_device_released",
        "_views",
        "_cpu_cache",
    )

    def __init__(
        self,
        binding,
        event_dgrams,
        lease,
        *,
        segment_ids=None,
        device_released=False,
    ):
        self.binding = binding
        self.event_dgrams = event_dgrams
        self._lease = lease
        self._segment_ids = (
            binding.segment_ids
            if segment_ids is None
            else tuple(int(segment) for segment in segment_ids)
        )
        self._device_released = bool(device_released)
        self._views = None
        self._cpu_cache = None

    @property
    def segment_ids(self):
        return self._segment_ids

    def _require_device_storage(self, accessor):
        if self._device_released or self.event_dgrams is None:
            raise RuntimeError(
                f"{accessor} is unavailable because the parsed GPU input "
                "slot has already been released"
            )

    @staticmethod
    def _locator_row(dgram, handle):
        locators = dgram.locate(handle)
        if locators.ready is not None:
            locators.ready.synchronize()
        row_gpu = locators.rows_gpu[dgram.dgram_index]
        return np.asarray(row_gpu.get() if hasattr(row_gpu, "get") else row_gpu)

    def _slot_views(self):
        self._require_device_storage("GPU field access")
        if self._views is not None:
            return self._views

        values = {}
        for dgram, segment, handle in self.binding.iter_sources(
            self.event_dgrams,
            segment_ids=self._segment_ids,
        ):
            row = self._locator_row(dgram, handle)
            status = int(row[LOC_STATUS])
            if status == STATUS_NOT_PRESENT:
                continue
            if status != STATUS_FOUND:
                raise RuntimeError(
                    f"GPU locator for {self.binding.det_name}."
                    f"{self.binding.alg_name}.{self.binding.field_name} "
                    f"segment {segment} has status "
                    f"{STATUS_NAMES.get(status, status)!r}"
                )
            if int(row[LOC_TYPE]) != handle.type:
                raise RuntimeError("GPU locator type disagrees with Configure")
            rank = int(row[LOC_RANK])
            if rank != handle.rank or not 0 <= rank <= LOC_MAX_RANK:
                raise RuntimeError("GPU locator rank disagrees with Configure")
            shape = tuple(int(v) for v in row[LOC_DIM0:LOC_DIM0 + rank])
            dtype = _XTC_DTYPES.get(handle.type)
            if dtype is None or dtype.itemsize != handle.element_size:
                raise TypeError(
                    f"unsupported XTC field type={handle.type}, "
                    f"element_size={handle.element_size}"
                )
            offset = int(row[LOC_OFFSET])
            nbytes = int(row[LOC_NBYTES])
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if rank == 0:
                expected = dtype.itemsize
            if nbytes != expected:
                raise RuntimeError(
                    f"GPU locator byte count {nbytes} does not match "
                    f"shape {shape} and dtype {dtype}"
                )
            if offset < 0 or offset + nbytes > int(dgram.data_gpu.nbytes):
                raise RuntimeError("GPU locator is outside the input buffer")
            values[segment] = (
                dgram.data_gpu[offset:offset + nbytes]
                .view(dtype)
                .reshape(shape)
            )

        self._views = GpuFieldData(self.binding, values)
        return self._views

    @property
    def on_gpu(self):
        """Return independent device copies keyed by physical segment id."""
        import cupy as cp

        self._require_device_storage("on_gpu")
        stream = cp.cuda.Stream.null
        if self._lease.result_ready is not None:
            stream.wait_event(self._lease.result_ready)
        with stream:
            copied = {
                segment: value.copy()
                for segment, value in self._slot_views().items()
            }
        done = cp.cuda.Event(disable_timing=True)
        stream.record(done)
        self._lease.register_consumer_done(done)
        return GpuFieldData(self.binding, copied)

    def on_gpu_view(self, stream=None):
        """Return a context manager for zero-copy segment views."""
        self._require_device_storage("on_gpu_view")
        if self._lease is None:
            raise RuntimeError("on_gpu_view requires an input SlotLease")
        return _GpuFieldViewContext(self, stream)

    @property
    def on_cpu(self):
        """Return independent NumPy values keyed by physical segment id."""
        if self._cpu_cache is None:
            self._require_device_storage("on_cpu")
            if self._lease.result_ready is not None:
                self._lease.result_ready.synchronize()
            self._cpu_cache = GpuFieldData(
                self.binding,
                {
                    segment: value.get()
                    for segment, value in self._slot_views().items()
                },
            )
        return self._cpu_cache


class GpuDetectorEvent:
    """One detector's Configure-backed fields for one GPU event."""

    __slots__ = (
        "binding",
        "event_dgrams",
        "_lease",
        "_device_released",
        "_cache",
    )

    def __init__(self, binding, event_dgrams, lease, *, device_released=False):
        self.binding = binding
        self.event_dgrams = event_dgrams
        self._lease = lease
        self._device_released = bool(device_released)
        self._cache = {}

    @property
    def det_name(self):
        return self.binding.det_name

    @property
    def fields(self):
        return tuple(self.binding.fields)

    def field(self, alg, name, *, segment=None):
        field = self.binding.field(alg, name)
        segment_ids = None
        if segment is not None:
            segment = int(segment)
            if segment not in field.field_handles_by_segment:
                raise KeyError(
                    f"{field.det_name}.{field.alg_name}.{field.field_name} "
                    f"is not configured for segment {segment}"
                )
            segment_ids = (segment,)
        key = (field.alg_name, field.field_name, segment)
        if key not in self._cache:
            self._cache[key] = GpuFieldResult(
                field,
                self.event_dgrams,
                self._lease,
                segment_ids=segment_ids,
                device_released=self._device_released,
            )
        return self._cache[key]


__all__ = [
    "GpuDetectorBinding",
    "GpuDetectorEvent",
    "GpuDetectorFieldBinding",
    "GpuEventDgrams",
    "GpuFieldData",
    "GpuFieldResult",
    "GpuStreamDgramView",
    "InputSlotLease",
]
