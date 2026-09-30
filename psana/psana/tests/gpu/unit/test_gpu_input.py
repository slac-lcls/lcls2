"""CPU-only tests for GPU input-to-detector ownership contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from psana.gpu.gpu_input import (
    GpuDetectorBinding,
    GpuEventDgrams,
    InputSlotLease,
    GpuStreamDgramView,
)
from psana.gpu.gpudgram.batch import (
    LOC_DIM0,
    LOC_NBYTES,
    LOC_NCOLS,
    LOC_OFFSET,
    LOC_RANK,
    LOC_STATUS,
    LOC_TYPE,
)
from psana.gpu.gpudgram.config import GpuFieldHandle
from psana.gpu.gpudgram.parser import STATUS_FOUND, STATUS_NOT_PRESENT


def _handle(
    stream_id,
    names_id,
    field_index=0,
    *,
    type_id=1,
    element_size=2,
    rank=2,
    shape_index=0,
):
    return GpuFieldHandle(
        stream_id=stream_id,
        names_id=names_id,
        config_names_index=names_id,
        config_field_index=field_index,
        field_index=field_index,
        type=type_id,
        element_size=element_size,
        rank=rank,
        shape_index=shape_index,
    )


class _Batch:
    def __init__(self, stream_ids):
        self.stream_ids_by_dgram = np.asarray(stream_ids, dtype=np.uint64)
        self.n_dgrams = len(stream_ids)
        self.data_gpu = object()
        self.locate_calls = []

    def locate(self, handle, *, stream=None):
        self.locate_calls.append((handle, stream))
        return (handle, stream)


def _event(timestamp, first_desc, n_desc, batch_event_index=0):
    return SimpleNamespace(
        timestamp=timestamp,
        first_desc=first_desc,
        n_desc=n_desc,
        batch_event_index=batch_event_index,
    )


def test_event_dgrams_map_streams_to_dense_dgram_indices():
    batch = _Batch([3, 8, 2, 5])
    event = GpuEventDgrams(_event(101, 1, 2, batch_event_index=7), batch)

    assert event.timestamp == 101
    assert event.batch_event_index == 7
    assert tuple(event) == (8, 2)
    assert event.dgrams[8].dgram_index == 1
    assert event[2].dgram_index == 2
    assert event[8].batch is batch
    assert event[8].data_gpu is batch.data_gpu
    with pytest.raises(TypeError):
        event.dgrams[4] = object()


def test_event_dgrams_from_batch_builds_each_mapping_once():
    batch = _Batch([1, 4, 2])
    events = [
        _event(10, 0, 2, batch_event_index=3),
        _event(11, 2, 1, batch_event_index=4),
    ]
    gpu_view = SimpleNamespace(iter_events=lambda: iter(events))

    mapped = GpuEventDgrams.from_batch(gpu_view, batch)

    assert [item.timestamp for item in mapped] == [10, 11]
    assert tuple(mapped[0]) == (1, 4)
    assert tuple(mapped[1]) == (2,)


def test_event_dgrams_reject_duplicate_stream_and_bad_range():
    with pytest.raises(RuntimeError, match="duplicate GPU dgrams for stream 3"):
        GpuEventDgrams(_event(22, 0, 2), _Batch([3, 3]))

    with pytest.raises(ValueError, match="outside batch"):
        GpuEventDgrams(_event(22, 1, 2), _Batch([3, 4]))


def test_stream_dgram_view_locates_only_its_stream():
    batch = _Batch([6])
    view = GpuStreamDgramView(stream_id=6, dgram_index=0, batch=batch)
    handle = _handle(6, 10)

    assert view.locate(handle, stream="consumer") == (handle, "consumer")
    assert batch.locate_calls == [(handle, "consumer")]
    with pytest.raises(ValueError, match="belongs to stream 7"):
        view.locate(_handle(7, 11))


def test_detector_binding_preserves_canonical_order_across_streams():
    handles = {
        9: _handle(1, 10),
        4: _handle(0, 11),
        7: _handle(1, 12),
    }
    binding = GpuDetectorBinding(
        "det",
        canonical_segment_ids=[9, 4, 7],
        field_handles_by_segment=handles,
    )
    event = GpuEventDgrams(_event(33, 0, 2), _Batch([0, 1]))

    sources = list(binding.iter_sources(event))

    assert binding.det_name == "det"
    assert binding.canonical_segment_ids == (9, 4, 7)
    assert binding.canonical_segment_rows == {9: 0, 4: 1, 7: 2}
    assert binding.stream_ids == (1, 0)
    assert [source[1] for source in sources] == [0, 1, 2]
    assert [source[2] for source in sources] == [9, 4, 7]
    assert [source[0].stream_id for source in sources] == [1, 0, 1]
    assert binding.has_sources(event)


def test_detector_binding_omits_missing_streams_without_reordering():
    binding = GpuDetectorBinding(
        "det",
        canonical_segment_ids=[5, 2, 8],
        field_handles_by_segment={
            5: _handle(4, 10),
            2: _handle(1, 11),
            8: _handle(4, 12),
        },
    )
    event = GpuEventDgrams(_event(44, 0, 1), _Batch([4]))

    sources = list(binding.iter_sources(event))

    assert [source[1] for source in sources] == [0, 2]
    assert [source[2] for source in sources] == [5, 8]


def test_detector_binding_validates_segment_contract():
    with pytest.raises(ValueError, match="contains duplicates"):
        GpuDetectorBinding(
            "det",
            canonical_segment_ids=[1, 1],
            field_handles_by_segment={1: _handle(0, 10)},
        )

    with pytest.raises(ValueError, match="identify every canonical segment"):
        GpuDetectorBinding(
            "det",
            canonical_segment_ids=[1, 2],
            field_handles_by_segment={1: _handle(0, 10)},
        )

    with pytest.raises(TypeError, match="GpuFieldHandle"):
        GpuDetectorBinding(
            "det",
            canonical_segment_ids=[1],
            field_handles_by_segment={1: object()},
        )


def test_detector_binding_resolves_multiple_named_fields():
    raw = {9: _handle(1, 10), 4: _handle(0, 11)}
    counter = {
        9: _handle(
            1, 10, 1, type_id=3, element_size=8, rank=0, shape_index=-1
        ),
        4: _handle(
            0, 11, 1, type_id=3, element_size=8, rank=0, shape_index=-1
        ),
    }
    binding = GpuDetectorBinding(
        "det",
        canonical_segment_ids=[9, 4],
        field_handles_by_segment=raw,
        field_handles_by_name={
            ("raw", "image"): raw,
            ("raw", "counter"): counter,
        },
    )

    assert tuple(binding.fields) == (("raw", "image"), ("raw", "counter"))
    assert binding.field("raw", "counter").segment_ids == (9, 4)
    assert binding.field("raw", "counter").stream_ids == (1, 0)
    with pytest.raises(KeyError, match="available fields"):
        binding.field("raw", "missing")


def test_field_only_detector_binding_does_not_require_calibration_adapter():
    image = {3: _handle(2, 10)}
    binding = GpuDetectorBinding(
        "camera",
        canonical_segment_ids=[3],
        field_handles_by_segment={},
        field_handles_by_name={("raw", "image"): image},
    )
    event = GpuEventDgrams(_event(45, 0, 1), _Batch([2]))

    assert binding.field_handles_by_segment == {}
    assert binding.stream_ids == (2,)
    assert binding.has_sources(event)
    assert list(binding.iter_sources(event)) == []
    assert binding.field("raw", "image").segment_ids == (3,)


class _Done:
    def __init__(self):
        self.sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1


class _DeviceArray:
    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def nbytes(self):
        return self.array.nbytes

    def __getitem__(self, item):
        return _DeviceArray(self.array[item])

    def view(self, dtype):
        return _DeviceArray(self.array.view(dtype))

    def reshape(self, shape):
        return _DeviceArray(self.array.reshape(shape))

    def copy(self):
        return _DeviceArray(self.array.copy())

    def get(self):
        return self.array.copy()


class _LocatorRows:
    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=np.uint64)

    def __getitem__(self, item):
        return _DeviceArray(self.rows[item])


class _FieldBatch(_Batch):
    def __init__(self, stream_ids, data, rows_by_handle):
        super().__init__(stream_ids)
        self.data_gpu = _DeviceArray(np.asarray(data, dtype=np.uint8))
        self.rows_by_handle = rows_by_handle

    def locate(self, handle, *, stream=None):
        return SimpleNamespace(
            rows_gpu=_LocatorRows(self.rows_by_handle[handle]),
            ready=_Done(),
        )


def _locator_rows(n_dgrams, dgram_index, handle, offset, shape):
    rows = np.zeros((n_dgrams, LOC_NCOLS), dtype=np.uint64)
    rows[:, LOC_STATUS] = STATUS_NOT_PRESENT
    row = rows[dgram_index]
    row[LOC_TYPE] = handle.type
    row[LOC_RANK] = handle.rank
    row[LOC_DIM0:LOC_DIM0 + handle.rank] = shape
    row[LOC_OFFSET] = offset
    row[LOC_NBYTES] = int(np.prod(shape, dtype=np.int64)) * handle.element_size
    if handle.rank == 0:
        row[LOC_NBYTES] = handle.element_size
    row[LOC_STATUS] = STATUS_FOUND
    return rows


def test_event_detector_field_access_preserves_segment_shape_and_dtype():
    from psana.gpu.context import GpuEventState

    seg9 = _handle(1, 10)
    seg4 = _handle(0, 11)
    values9 = np.arange(6, dtype=np.uint16).reshape(2, 3)
    values4 = np.arange(4, dtype=np.uint16).reshape(2, 2) + 20
    data = np.zeros(64, dtype=np.uint8)
    data[8:20] = values9.view(np.uint8).reshape(-1)
    data[32:40] = values4.view(np.uint8).reshape(-1)
    rows = {
        seg9: _locator_rows(2, 1, seg9, 8, values9.shape),
        seg4: _locator_rows(2, 0, seg4, 32, values4.shape),
    }
    batch = _FieldBatch([0, 1], data, rows)
    event_dgrams = GpuEventDgrams(_event(55, 0, 2), batch)
    binding = GpuDetectorBinding(
        "camera",
        canonical_segment_ids=[9, 4],
        field_handles_by_segment={9: seg9, 4: seg4},
        field_handles_by_name={
            ("raw", "image"): {9: seg9, 4: seg4},
        },
    )
    state = GpuEventState(
        {},
        detector_bindings={"camera": binding},
        event_dgrams=event_dgrams,
        input_lease=InputSlotLease(_Done()),
    )

    result = state.detector("camera").field("raw", "image")
    host = result.on_cpu

    assert result.segment_ids == (9, 4)
    assert host.segment_ids == (9, 4)
    np.testing.assert_array_equal(host[9], values9)
    np.testing.assert_array_equal(host[4], values4)
    selected = state.detector("camera").field(
        "raw", "image", segment=4
    ).on_cpu
    np.testing.assert_array_equal(selected.only(), values4)


def test_two_detector_bindings_can_read_the_same_event_stream():
    from psana.gpu.context import GpuEventState

    camera_handle = _handle(4, 10)
    timing_handle = _handle(
        4, 11, type_id=3, element_size=8, rank=0, shape_index=-1
    )
    camera_value = np.arange(4, dtype=np.uint16).reshape(2, 2)
    timing_value = np.asarray(123456, dtype=np.uint64)
    data = np.zeros(64, dtype=np.uint8)
    data[8:16] = camera_value.view(np.uint8).reshape(-1)
    data[32:40] = np.frombuffer(timing_value.tobytes(), dtype=np.uint8)
    rows = {
        camera_handle: _locator_rows(
            1, 0, camera_handle, 8, camera_value.shape
        ),
        timing_handle: _locator_rows(1, 0, timing_handle, 32, ()),
    }
    event_dgrams = GpuEventDgrams(
        _event(56, 0, 1), _FieldBatch([4], data, rows)
    )
    camera = GpuDetectorBinding(
        "camera",
        canonical_segment_ids=[0],
        field_handles_by_segment={},
        field_handles_by_name={("raw", "image"): {0: camera_handle}},
    )
    timing = GpuDetectorBinding(
        "timing",
        canonical_segment_ids=[0],
        field_handles_by_segment={},
        field_handles_by_name={("raw", "counter"): {0: timing_handle}},
    )
    state = GpuEventState(
        {},
        detector_bindings={"camera": camera, "timing": timing},
        event_dgrams=event_dgrams,
        input_lease=InputSlotLease(_Done()),
    )

    np.testing.assert_array_equal(
        state.detector("camera").field("raw", "image").on_cpu.only(),
        camera_value,
    )
    assert (
        state.detector("timing").field("raw", "counter").on_cpu.only()
        == timing_value
    )


def test_input_slot_lease_waits_for_every_field_consumer():
    ready = _Done()
    first = _Done()
    second = _Done()
    lease = InputSlotLease(ready)

    lease.register_consumer_done(first)
    lease.register_consumer_done(second)
    lease.wait_until_safe_to_reuse()

    assert first.sync_calls == 1
    assert second.sync_calls == 1
