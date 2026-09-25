"""Direct group submission matches the independent CPU reference plan."""
from dataclasses import replace
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_read_plan import ResolvedDgram, ResolvedFile, build_read_plan
from psana.gpu.gpu_stream_read_plan import StreamReadGroup
from test_gpu_bulk_read import io  # noqa: F401


def group(sizes=(4, 8), events=(3, 1), *, offset=5, path='/data', fence=0):
    rows = []
    for size, event in zip(sizes, events):
        rows.append(ResolvedDgram(event, 100 + event, 2, ResolvedFile(path, 0), offset, size, 80))
        offset += size
    return StreamReadGroup(0, 2, fence, rows[0].file, rows[0].file_offset,
                           sum(sizes), tuple(rows), True)


def test_direct_group_matches_reference_rows_ranges_and_bytes(io, monkeypatch):
    io.files = {'/data': bytes(range(256))}
    reader = KvikioGpuReader(n_slots=2)
    rng = random.Random(260925)
    groups = [group((0,), (7,)), group()]
    for _ in range(100):
        n = rng.randrange(1, 10)
        groups.append(group(tuple(rng.randrange(1, 12) for _ in range(n)),
                            tuple(rng.sample(range(100), n)), offset=rng.randrange(20)))
    for g in groups:
        reference = build_read_plan(g.dgrams, capacity_bytes=g.size)
        with monkeypatch.context() as patch:
            def no_legacy(*args, **kwargs):
                raise AssertionError('direct group invoked legacy replanning')
            patch.setattr(reader, 'issue_batch', no_legacy)
            direct = reader.issue_group(g, slot_id=1)
        assert direct.plan.physical_ranges == reference.physical_ranges
        assert direct.plan.logical_dgrams == reference.logical_dgrams
        expected = [(d.batch_event_index, d.stream_id, d.timestamp, d.file_offset,
                     d.size, row.device_offset)
                    for d, row in zip(g.dgrams, reference.logical_dgrams)]
        np.testing.assert_array_equal(direct.desc_table, expected)
        assert [(r, n) for r, n, _ in direct.futures] == [(r, r.size) for r in reference.physical_ranges]
        b = reader.wait_batch(direct)
        assert bytes(b.data_gpu) == io.files['/data'][g.file_offset:g.file_offset + g.size]
    reader.close()


@pytest.mark.parametrize('bad', ['duplicate', 'gap', 'overlap', 'file', 'stream',
                                'empty', 'mixed_zero', 'short_size', 'long_size',
                                'negative_size', 'overflow_size', 'bool_offset', 'unresolved'])
def test_invalid_group_rejected_before_allocation_or_submission(io, bad):
    g = group()
    first, second = g.dgrams
    if bad == 'duplicate':
        g = replace(g, dgrams=(first, replace(second, batch_event_index=first.batch_event_index)))
    elif bad in ('gap', 'overlap'):
        g = replace(g, dgrams=(first, replace(second, file_offset=second.file_offset + (1 if bad == 'gap' else -1))))
    elif bad == 'file':
        g = replace(g, file=ResolvedFile('/other', 0))
    elif bad == 'stream':
        g = replace(g, stream_id=3)
    elif bad == 'empty':
        g = replace(g, dgrams=(), size=0)
    elif bad == 'mixed_zero':
        g = replace(g, size=first.size, dgrams=(first, replace(second, size=0)))
    elif bad in ('short_size', 'long_size', 'negative_size', 'overflow_size'):
        g = replace(g, size={'short_size': g.size-1, 'long_size': g.size+1,
                             'negative_size': -1, 'overflow_size': 1 << 64}[bad])
    elif bad == 'bool_offset':
        g = replace(g, file_offset=True)
    else:
        g = replace(g, dgrams=(NS(**vars(first)), second))
    reader = KvikioGpuReader()
    with pytest.raises((ValueError, TypeError)):
        reader.issue_group(g, slot_id=0)
    assert not io.calls and not reader._pending and not reader._files
    assert reader._slot_bufs == [None, None]
    reader.close()


def test_group_calls_keep_transition_and_file_fences_and_old_handle(io):
    io.files = {'/data': bytes(range(64)), '/next': bytes(range(64))}
    reader = KvikioGpuReader(n_slots=3)
    groups = [group((4,), (0,), offset=0), group((4,), (1,), offset=4, fence=1),
              group((4,), (2,), offset=0, path='/next', fence=2)]
    pending = [reader.issue_group(g, slot_id=i) for i, g in enumerate(groups)]
    assert reader.io_stats()['total_requests'] == 3  # adjacent cross-fence ranges stay separate
    reader.wait_batch(pending[2])
    reader.wait_batch(pending[1])
    assert not io.handles[0].closed
    reader.wait_batch(pending[0])
    assert io.handles[0].closed
    reader.close()


def test_direct_group_preserves_busy_slots_input_holds_and_full_replacement_cost(io):
    io.files = {'/data': bytes(range(64))}
    reader = KvikioGpuReader(n_slots=1, budget=_GpuBudget(10))
    g = group((4,), (0,))
    pending = reader.issue_group(g, slot_id=0)
    with pytest.raises(RuntimeError, match='pending I/O'):
        reader.issue_group(g, slot_id=0)
    read = reader.wait_batch(pending)
    release = read.retain_input()
    with pytest.raises(RuntimeError, match='owned by an input window'):
        reader.issue_group(g, slot_id=0)
    release()
    with pytest.raises(GpuMemoryPressureError):
        reader.issue_group(group((8,), (1,)), slot_id=0)
    assert len(io.futures) == 1 and reader._budget.committed() == 4
    reader.close()


def test_direct_group_rejects_bulk_off(io):
    reader = KvikioGpuReader(bulk_read=False)
    with pytest.raises(ValueError, match='adjacent-range'):
        reader.issue_group(group(), slot_id=0)
    assert not io.calls
    reader.close()
