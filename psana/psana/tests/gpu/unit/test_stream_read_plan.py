"""Stage 1 request coverage, ordering, fences and bounded look-ahead."""
import importlib.util
from pathlib import Path
import random
import sys

import pytest
from psana.gpu.gpu_read_plan import ResolvedDgram, ResolvedFile

# Allow a source-only planner check against an existing native psana install.
path = Path(__file__).resolve().parents[3] / 'gpu/gpu_stream_read_plan.py'
spec = importlib.util.spec_from_file_location('psana.gpu._stream_read_plan_under_test', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
plan = module.build_stream_read_plan


def d(event, stream, offset, size, chunk=0):
    return ResolvedDgram(event, 1000+event, stream,
                         ResolvedFile(f'/stream{stream}-c{chunk}', chunk), offset, size)


def test_small_bulk_and_large_event_rounds_preserve_every_byte():
    rows = [d(e, s, e*n, n) for e in range(100)
            for s, n in ((0, 9), (5, 110), (6, 120))]
    random.Random(3).shuffle(rows)
    # Use a smaller target than either large datagram for this miniature fixture.
    p = plan(rows, n_events=100, small_target_bytes=100, input_capacity_bytes=400)
    assert sum(g.stream_id == 0 for g in p.groups) == 10
    assert all(len(g.dgrams) == 1 for g in p.groups if g.stream_id != 0)
    assert [(g.first_event, g.stream_id) for g in p.groups[:5]] == [(0,0),(0,5),(0,6),(1,5),(1,6)]
    actual = [r for g in p.groups for r in g.dgrams]
    assert set(actual) == set(rows) and len(actual) == len(rows)
    assert sum(g.size for g in p.groups) == p.useful_bytes == sum(r.size for r in rows)
    for g in p.groups:
        source = bytes((i % 251 for i in range(g.file_offset + g.size)))
        fetched = source[g.file_offset:g.file_offset+g.size]
        for r in g.dgrams:
            local = r.file_offset-g.file_offset
            assert fetched[local:local+r.size] == source[r.file_offset:r.file_offset+r.size]
    small = [g for g in p.groups if g.stream_id == 0]
    assert small[0].after_group is None
    assert [g.after_group for g in small[1:]] == [g.group_id for g in small[:-1]]
    assert all(g.after_group is None for g in p.groups if not g.small)


def test_100_feespec_and_five_large_streams():
    rows = [d(e, s, e*n, n) for e in range(100)
            for s, n in ((0, 9736), (5, 6291972), (6, 7340630),
                         (7, 5243314), (8, 7340630), (9, 7340630))]
    p = plan(rows, n_events=100, input_capacity_bytes=64*2**20)
    assert len(p.groups) == 501
    assert len(p.groups[0].dgrams) == 100
    assert [g.stream_id for g in p.groups[:11]] == [0,5,6,7,8,9,5,6,7,8,9]
    assert p.one_group_per_stream_bytes == 973600 + sum(r.size for r in rows[:6] if r.stream_id)


def test_gaps_transitions_chunks_and_partial_tail():
    rows = [d(0,0,0,10), d(1,0,10,10), d(2,0,20,10),
            d(3,0,40,10), d(4,0,0,10,chunk=1), d(5,0,10,0,chunk=1)]
    p = plan(rows, n_events=6, small_target_bytes=100, input_capacity_bytes=100,
             fence_by_event={0:0,1:0,2:1,3:1,4:1,5:1})
    assert [len(g.dgrams) for g in p.groups] == [2,1,1,1]
    assert p.empty_dgrams == (rows[-1],)
    assert p.useful_bytes == 50
    assert [g.file_offset for g in p.groups] == [0,20,40,0]


def test_sparse_detector_retains_explicit_planned_uses():
    rows = [d(0,0,0,10), d(99,0,10,10), d(99,5,0,100)]
    p = plan(rows, n_events=100, small_target_bytes=50, input_capacity_bytes=120)
    assert [x.batch_event_index for x in p.groups[0].dgrams] == [0,99]
    assert p.groups[0].event_stop == 100
    assert p.groups[0].after_group is None  # all planned uses still need ownership


def test_batches_never_coalesce_with_each_other():
    left = plan([d(0,0,0,10)], n_events=1, batch_id=4, input_capacity_bytes=10)
    right = plan([d(0,0,10,10)], n_events=1, batch_id=5, input_capacity_bytes=10)
    assert left.batch_id != right.batch_id
    assert left.groups[0].size == right.groups[0].size == 10
    # Group identity is (batch_id, group_id). Runtime must carry its outstanding
    # small-stream credit across EB batches; a new batch is not a release event.


def test_exact_threshold_is_single_and_oversize_is_not_split():
    rows = [d(0,0,0,100), d(1,0,100,101), d(2,0,201,1)]
    p = plan(rows, n_events=3, small_target_bytes=100, input_capacity_bytes=101)
    assert [g.size for g in p.groups] == [100,101,1]
    assert [g.small for g in p.groups] == [False,False,True]


@pytest.mark.parametrize('rows,kwargs', [
    ([d(0,0,0,10)]*2, {}),
    ([d(0,0,0,10),d(1,0,5,10)], {}),
    ([d(2,0,0,10)], {}),
    ([d(0,0,0,10)], {'small_target_bytes':0}),
    ([d(0,0,0,10),d(0,1,0,10)], {'input_capacity_bytes':19}),
    ([d(0,0,0,10),d(1,0,10,10)], {'fence_by_event':{0:1,1:0}}),
])
def test_invalid_or_unfunded_plan_fails_before_io(rows, kwargs):
    options = dict(n_events=2, input_capacity_bytes=100)
    options.update(kwargs)
    with pytest.raises(ValueError):
        plan(rows, **options)


def test_empty_batch_needs_no_input_capacity():
    p = plan([], n_events=0, input_capacity_bytes=0)
    assert p.groups == () and p.useful_bytes == p.one_group_per_stream_bytes == 0


def test_incomplete_transition_metadata_reports_event():
    with pytest.raises(ValueError, match='missing transition fence for event 1'):
        plan([d(0,0,0,10), d(1,0,10,10)], n_events=2,
             input_capacity_bytes=100, fence_by_event={0:0})
