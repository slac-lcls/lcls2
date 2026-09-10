"""CPU-only tests for GPU input-to-detector ownership contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from psana.gpu.gpu_input import (
    GpuDetectorBinding,
    GpuEventDgrams,
    GpuStreamDgramView,
)
from psana.gpu.gpudgram.config import GpuFieldHandle


def _handle(stream_id, names_id, field_index=0):
    return GpuFieldHandle(
        stream_id=stream_id,
        names_id=names_id,
        config_names_index=names_id,
        config_field_index=field_index,
        field_index=field_index,
        type=1,
        element_size=2,
        rank=2,
        shape_index=0,
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
