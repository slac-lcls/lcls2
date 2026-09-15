"""Pure CPU acceptance for physical read ranges and logical dgram identity."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import random
import subprocess
import sys

import pytest

from psana.gpu import gpu_read_plan
from psana.gpu.gpu_read_plan import ResolvedDgram, ResolvedFile, build_read_plan


FAST = ResolvedFile("/data/fast-c000.xtc2", 0)
JF = ResolvedFile("/data/jungfrau-c000.xtc2", 0)
U64_MAX = (1 << 64) - 1


def dgram(event, offset, size, *, stream=0, file=FAST):
    return ResolvedDgram(event, 100 + event, stream, file, offset, size, 80)


def example():
    return (
        dgram(0, 4096, 256),
        dgram(1, 4352, 256),
        dgram(1, 8192, 1024, stream=1, file=JF),
        dgram(2, 4608, 256),
        dgram(3, 4864, 256),
        dgram(3, 9216, 1024, stream=1, file=JF),
    )


def assert_bytes_match(plan, files):
    """Simulate range reads, then compare each logical view to direct reads."""
    destination = bytearray(plan.capacity_bytes)
    for r in plan.physical_ranges:
        payload = files[r.file][r.file_offset:r.file_offset + r.size]
        assert len(payload) == r.size
        destination[r.device_offset:r.device_offset + r.size] = payload
    for row in plan.logical_dgrams:
        desc = row.source
        expected = files[desc.file][desc.file_offset:desc.file_offset + desc.size]
        assert destination[row.device_offset:row.device_offset + desc.size] == expected
        if desc.size:
            r = plan.physical_ranges[row.range_index]
            assert r.file == desc.file
            assert r.file_offset <= desc.file_offset
            assert desc.file_offset + desc.size <= r.file_offset + r.size
        else:
            assert row.range_index is None
    assert sum(r.size for r in plan.physical_ranges) == plan.fetched_bytes
    assert plan.fetched_bytes == plan.useful_bytes == plan.capacity_bytes


def test_fast_jungfrau_example_keeps_six_logical_rows_and_reads_two_ranges():
    descriptors = example()
    plan = build_read_plan(
        iter(descriptors), capacity_bytes=4096, batch_id=7, input_window_id=2,
    )
    assert (plan.batch_id, plan.input_window_id) == (7, 2)
    assert [(r.file, r.file_offset, r.size, r.device_offset)
            for r in plan.physical_ranges] == [
        (FAST, 4096, 1024, 0), (JF, 8192, 2048, 1024),
    ]
    assert plan.n_reads == 2
    assert plan.capacity_bytes == 3072
    assert tuple(row.source for row in plan.logical_dgrams) == descriptors
    assert [row.range_index for row in plan.logical_dgrams] == [0, 0, 1, 0, 0, 1]
    assert [row.device_offset for row in plan.logical_dgrams] == [0, 256, 1024, 512, 768, 2048]
    assert_bytes_match(plan, {
        FAST: bytes(i % 251 for i in range(5120)),
        JF: bytes((i + 19) % 251 for i in range(10240)),
    })


@pytest.mark.parametrize("cap,counts", [(1024, [1024, 1024, 1024]), (2048, [1024, 2048])])
def test_read_cap_splits_ranges_without_splitting_dgrams(cap, counts):
    plan = build_read_plan(example(), capacity_bytes=3072, max_read_bytes=cap)
    assert [r.size for r in plan.physical_ranges] == counts
    assert [r.device_offset for r in plan.logical_dgrams] == [0, 256, 1024, 512, 768, 2048]


def test_gaps_are_not_fetched_and_trailing_empty_row_does_not_size_buffer():
    descriptors = [dgram(0, 100, 4), dgram(2, 111, 3), dgram(3, 114, 0)]
    plan = build_read_plan(descriptors, capacity_bytes=7)
    assert [(r.file_offset, r.size, r.device_offset) for r in plan.physical_ranges] == [
        (100, 4, 0), (111, 3, 4),
    ]
    assert plan.capacity_bytes == 7
    assert plan.logical_dgrams[-1].device_offset == 0
    assert_bytes_match(plan, {FAST: bytes(range(128))})


def test_file_and_chunk_identity_fence_adjacent_offsets():
    next_chunk = ResolvedFile(FAST.path, 1)
    other_file = ResolvedFile("/data/other.xtc2", 0)
    descriptors = [
        dgram(0, 0, 4), dgram(1, 4, 4, file=next_chunk),
        dgram(2, 8, 4, file=other_file),
    ]
    plan = build_read_plan(descriptors, capacity_bytes=12)
    assert plan.n_reads == 3
    assert [r.file for r in plan.physical_ranges] == [FAST, next_chunk, other_file]


def test_physical_order_does_not_reorder_logical_events():
    descriptors = tuple(reversed(example()))
    plan = build_read_plan(descriptors, capacity_bytes=3072)
    forward = build_read_plan(example(), capacity_bytes=3072)
    assert plan.physical_ranges == forward.physical_ranges
    assert plan.logical_dgrams == tuple(reversed(forward.logical_dgrams))


@pytest.mark.parametrize("seed", range(8))
def test_mixed_sparse_streams_match_individual_reads(seed):
    rng = random.Random(seed)
    descriptors = []
    files = {}
    for stream, file in enumerate((FAST, JF)):
        offset = 7
        for event in range(30):
            if rng.random() < 0.25:
                continue
            size = rng.randrange(0, 17)
            descriptors.append(dgram(event, offset, size, stream=stream, file=file))
            offset += size + rng.choice((0, 0, 3))
        files[file] = bytes(rng.randrange(256) for _ in range(offset))
    rng.shuffle(descriptors)
    plan = build_read_plan(descriptors, capacity_bytes=2048, max_read_bytes=32)
    assert all(r.size <= 32 for r in plan.physical_ranges)
    assert tuple(row.source for row in plan.logical_dgrams) == tuple(descriptors)
    assert_bytes_match(plan, files)


@pytest.mark.parametrize("descriptors", [[], [dgram(0, 0, 0), dgram(2, 99, 0)]])
def test_empty_inputs_require_no_allocation_or_reads(descriptors):
    plan = build_read_plan(descriptors, capacity_bytes=0)
    assert plan.n_reads == plan.capacity_bytes == plan.useful_bytes == 0
    assert len(plan.logical_dgrams) == len(descriptors)


@pytest.mark.parametrize("overrides,match", [
    ({"capacity_bytes": 3071}, "requires 3072"),
    ({"max_read_bytes": 1023}, "exceeds max_read_bytes"),
    ({"max_read_bytes": 0}, "must be positive"),
    ({"capacity_bytes": -1}, "uint64"),
    ({"batch_id": U64_MAX + 1}, "uint64"),
    ({"input_window_id": -1}, "uint64"),
])
def test_invalid_admission_limits(overrides, match):
    kwargs = dict(capacity_bytes=3072)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        build_read_plan(example(), **kwargs)


@pytest.mark.parametrize("offset,size", [(0, 8), (3, 8), (4, 2)])
def test_overlapping_file_spans_are_rejected(offset, size):
    with pytest.raises(ValueError, match="overlapping"):
        build_read_plan([dgram(0, 0, 8), dgram(1, offset, size)], capacity_bytes=32)


def test_overlap_is_checked_even_when_read_cap_splits_previous_range():
    with pytest.raises(ValueError, match="overlapping"):
        build_read_plan(
            [dgram(0, 0, 4), dgram(1, 4, 4), dgram(2, 6, 4)],
            capacity_bytes=12, max_read_bytes=4,
        )


def test_duplicate_event_stream_descriptor_is_rejected():
    desc = dgram(0, 0, 4)
    with pytest.raises(ValueError, match="duplicate event/stream"):
        build_read_plan([desc, desc], capacity_bytes=8)


def test_event_identity_requires_consistent_timestamp_across_streams():
    fast, jf = example()[1:3]
    with pytest.raises(ValueError, match="inconsistent timestamp"):
        build_read_plan([fast, replace(jf, timestamp=999)], capacity_bytes=2048)


@pytest.mark.parametrize("field,value", [
    ("file_offset", -1), ("size", -1), ("size", U64_MAX + 1),
    ("timestamp", -1), ("stream_id", 64), ("batch_event_index", -1),
    ("smd_size", -1),
])
def test_invalid_descriptor_integers(field, value):
    with pytest.raises(ValueError):
        replace(dgram(0, 0, 4), **{field: value})


@pytest.mark.parametrize("value", [True, 1.5, "4"])
def test_integer_fields_do_not_silently_coerce(value):
    with pytest.raises(TypeError):
        replace(dgram(0, 0, 4), size=value)
    with pytest.raises(TypeError):
        build_read_plan([], capacity_bytes=value)


def test_file_and_total_address_overflow_are_rejected_without_allocating():
    with pytest.raises(ValueError, match="file_offset \\+ size"):
        dgram(0, U64_MAX, 1)
    with pytest.raises(ValueError, match="total input size"):
        build_read_plan(
            [dgram(0, 0, U64_MAX), dgram(1, 0, 1, stream=1, file=JF)],
            capacity_bytes=U64_MAX,
        )
    plan = build_read_plan([dgram(0, U64_MAX - 4, 4)], capacity_bytes=4)
    assert plan.physical_ranges[0].file_offset == U64_MAX - 4


@pytest.mark.parametrize("factory,error", [
    (lambda: ResolvedFile("", 0), ValueError),
    (lambda: ResolvedFile(123, 0), TypeError),
    (lambda: ResolvedFile("/data/a", -1), ValueError),
    (lambda: replace(dgram(0, 0, 4), file="/data/a"), TypeError),
    (lambda: build_read_plan([object()], capacity_bytes=4), TypeError),
])
def test_invalid_record_types(factory, error):
    with pytest.raises(error):
        factory()


def test_plan_and_descriptors_are_immutable():
    descriptors = list(example())
    plan = build_read_plan(descriptors, capacity_bytes=3072)
    descriptors.clear()
    assert len(plan.logical_dgrams) == 6
    with pytest.raises(FrozenInstanceError):
        plan.logical_dgrams[0].source.size = 1
    with pytest.raises(FrozenInstanceError):
        plan.physical_ranges[0].device_offset = 99


def test_planner_can_run_with_only_the_python_standard_library():
    # Load the module directly: psana package initialization needs native libs,
    # but the planner itself must also work without site-packages or CUDA/MPI.
    script = """
import runpy, sys
m = runpy.run_path(sys.argv[1])
f = m['ResolvedFile']('/data/a', 0)
d = m['ResolvedDgram'](0, 100, 0, f, 0, 4)
p = m['build_read_plan']([d], capacity_bytes=4)
assert p.n_reads == 1
"""
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(Path(gpu_read_plan.__file__))],
        check=True, capture_output=True, text=True,
    )
