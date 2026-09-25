"""File lifetime across overlapping reads, chunk changes, and failure drains."""
import pytest

from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_read_plan import ReadRange, ResolvedFile
from test_gpu_bulk_read import io, desc, dm, view  # noqa: F401 (shared fixture)


def submit(reader, descriptors, paths, slot):
    if not reader.bulk_read:
        return reader.issue_batch(view(descriptors), dm(paths), slot_id=slot)
    # Inject multiple physical requests directly into the shared submission
    # engine to cover partial failure after acquiring several file references.
    table = reader._build_desc_table(descriptors)
    ranges = []
    for d, path, row in zip(descriptors, paths, table):
        identity = ResolvedFile(path, 0)
        reader._latest_files[d.stream_id] = identity
        if d.size:
            ranges.append(ReadRange(identity, d.offset, d.size, int(row[5])))
    return reader._submit_read(table, ranges, sum(d.size for d in descriptors), slot, None)


@pytest.mark.parametrize('bulk', [True, False])
def test_shared_old_file_closes_only_after_last_out_of_order_batch(io, bulk):
    io.files = {'/old': bytes(range(64)), '/new': bytes(range(64))}
    reader = KvikioGpuReader(n_slots=3, bulk_read=bulk)
    first = submit(reader, [desc(0, 0, 0), desc(1, 0, 8)], ['/old'] * 2, 0)
    second = submit(reader, [desc(2, 0, 16)], ['/old'], 1)
    new = submit(reader, [desc(3, 0, 0)], ['/new'], 2)
    old_handle, new_handle = io.handles
    reader.wait_batch(new)
    reader.wait_batch(first)
    reader.wait_batch(first)  # repeated completion must not release twice
    assert not old_handle.closed
    reader.wait_batch(second)
    assert old_handle.closed and not new_handle.closed
    assert all(f.gets == 1 for f in io.futures)
    assert not reader._pending_file_refs
    reader.close()
    reader.close()
    assert new_handle.closed


@pytest.mark.parametrize('failure', ['submit', 'get', 'short', 'open'])
def test_failure_drain_keeps_other_batches_old_file_alive(io, monkeypatch, failure):
    io.files = {'/old': bytes(range(64)), '/new': bytes(range(64))}
    reader = KvikioGpuReader(n_slots=2)
    held = submit(reader, [desc(0, 0, 0)], ['/old'], 0)
    old_handle = io.handles[0]
    # The failing batch spans a chunk change and holds two ranges on /old.
    # Its failure must not retire the separate batch's old-file reference.
    descriptors = [desc(1, 0, 8), desc(2, 0, 16), desc(3, 0, 0)]
    paths = ['/old', '/old', '/new']
    if failure == 'open':
        original = reader.kvikio.CuFile

        def fail_new(path, mode):
            if path == '/new':
                raise OSError('injected open failure')
            return original(path, mode)
        monkeypatch.setattr(reader.kvikio, 'CuFile', fail_new)
    else:
        setattr(io, {'submit': 'fail_submit', 'get': 'fail_get', 'short': 'short'}[failure],
                3 if failure == 'submit' else 1)
    with pytest.raises(RuntimeError):
        failed = submit(reader, descriptors, paths, 1)
        reader.wait_batch(failed)
    assert not old_handle.closed and held.futures[0][2].gets == 0
    assert all(f.gets == 1 for f in io.futures[1:])
    reader.wait_batch(held)
    assert old_handle.closed
    assert not reader._pending_file_refs
    reader.close()
    assert all(h.closed for h in io.handles)


def test_zero_rows_advance_latest_file_without_acquiring_a_handle(io):
    io.files = {'/old': bytes(range(32)), '/new': bytes(range(32))}
    reader = KvikioGpuReader(n_slots=2)
    old = submit(reader, [desc(0, 0, 0)], ['/old'], 0)
    empty = submit(reader, [desc(1, 0, 0, 0)], ['/new'], 1)
    reader.wait_batch(empty)
    assert len(io.handles) == 1 and not io.handles[0].closed
    reader.wait_batch(old)
    assert io.handles[0].closed and not reader._pending_file_refs
    reader.close()


def test_latest_file_shared_by_another_stream_stays_cached(io):
    io.files = {'/old': bytes(range(32)), '/new': bytes(range(32))}
    reader = KvikioGpuReader(n_slots=2)
    shared = submit(reader, [desc(0, 0, 0), desc(0, 1, 8)], ['/old'] * 2, 0)
    reader.wait_batch(shared)
    new = submit(reader, [desc(1, 0, 0)], ['/new'], 1)
    reader.wait_batch(new)
    assert not io.handles[0].closed  # stream 1 still caches /old
    empty = submit(reader, [desc(2, 1, 0, 0)], ['/new'], 0)
    reader.wait_batch(empty)
    assert io.handles[0].closed
    reader.close()


def test_close_error_does_not_release_completed_batch_twice(io, monkeypatch):
    io.files = {'/old': bytes(range(32)), '/new': bytes(range(32))}
    reader = KvikioGpuReader(n_slots=2)
    old = submit(reader, [desc(0, 0, 0)], ['/old'], 0)
    new = submit(reader, [desc(1, 0, 0)], ['/new'], 1)
    handle = io.handles[0]
    original = handle.close

    def fail_close():
        raise OSError('injected close failure')
    monkeypatch.setattr(handle, 'close', fail_close)
    for _ in range(2):
        with pytest.raises(OSError, match='close failure'):
            reader.wait_batch(old)
    assert old.futures[0][2].gets == 1
    assert not handle.closed and new.futures[0][2].gets == 0
    monkeypatch.setattr(handle, 'close', original)
    reader.close()
    assert not reader._pending_file_refs
    assert all(h.closed for h in io.handles)
