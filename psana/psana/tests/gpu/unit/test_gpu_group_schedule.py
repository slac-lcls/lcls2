from types import SimpleNamespace as NS

import pytest

from psana.gpu.gpu_admission import AdmissionEvent
from psana.gpu.gpu_batch import GpuReadDesc
from psana.gpu.gpu_group_schedule import GroupReadSchedule
from psana.gpu.gpu_read_plan import ResolvedFile


def schedule(n=10, fee_size=10, capacity=100000, fence_at=None, empty=False):
    descriptors, epochs, events = [], {}, []
    for i in range(n):
        sizes = [(0, fee_size), (1, 2048), (2, 2048)]
        if empty:
            sizes.append((3, 0))
        events.append(AdmissionEvent(tuple(sizes), 200))
        for stream, size in sizes:
            descriptors.append(GpuReadDesc(i, 100+i, stream, i*size, size, 0, 1))
            epochs[i, stream] = NS(file=ResolvedFile(f'/s{stream}', 0),
                                   fence=int(fence_at is not None and i >= fence_at))
    return GroupReadSchedule(descriptors, epochs, events, batch_id=4, capacity=capacity,
                             parser_bytes=4, depth=2, target_bytes=48)


def test_small_boundaries_split_execution_without_serializing_large_streams():
    s = schedule()
    assert s.execution_ranges == ((0, 4), (4, 8), (8, 10))
    first = s.new_groups(0, 4)
    assert [(g.stream_id, g.first_event) for g in first[:5]] == [
        (0, 0), (1, 0), (2, 0), (1, 1), (2, 1)]
    assert first[0].event_stop == 4
    s.issued.update(g.group_id for g in first)
    assert not s.new_groups(0, 4)
    assert s.new_groups(4, 8)[0].stream_id == 0


def test_budget_can_split_inside_one_small_group_without_rereading_it():
    s = schedule(capacity=14000)
    assert s.execution_ranges[:2] == ((0, 1), (1, 2))
    first = s.new_groups(0, 1)
    s.issued.update(g.group_id for g in first)
    assert all(g.stream_id != 0 for g in s.new_groups(1, 2))
    assert any(g.stream_id == 0 for g in s.groups_for(1, 2))


def test_transition_fence_is_an_execution_boundary_for_next_small_group():
    s = schedule(fence_at=2)
    assert s.execution_ranges[0] == (0, 2)
    assert [(g.first_event, g.event_stop) for g in s.plan.groups if g.small] == [
        (0, 2), (2, 6), (6, 10)]


def test_empty_descriptors_are_preserved_without_physical_read_bytes():
    s = schedule(empty=True)
    for event in range(10):
        assert len(s.groups_by_event[event]) == 4
    assert len([g for g in s.plan.groups if not g.size]) == 10


def test_minimum_progress_rejected_before_io():
    with pytest.raises(ValueError, match='one-group-per-stream'):
        schedule(capacity=100)
