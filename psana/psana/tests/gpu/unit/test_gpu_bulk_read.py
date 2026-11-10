"""CPU fault-injection tests for bulk I/O and ordered file resolution."""

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from psana.gpu.gpu_batch import GpuReadDesc
from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError
from psana.gpu.gpu_file_epochs import FileEpoch, GpuFileEpochs
from psana.gpu.gpu_kvikio_read import KvikioGpuReader, DESC_DEVICE_OFFSET, DESC_READ_SIZE
from psana.gpu.gpu_read_plan import ResolvedDgram, ResolvedFile
from psana.gpu.gpu_stream_read_plan import build_stream_read_plan
from psana.psexp import TransitionId


def desc(event, stream, offset, size=4):
    return GpuReadDesc(event, 100 + 10 * event, stream, offset, size, 80, 1)


def dm(paths):
    return NS(xtc_files=list(paths), get_chunk_id=lambda stream: 0)


def view(descriptors):
    return NS(iter_read_descs=lambda _: iter(descriptors))


class _Stop(Exception):
    """Unwinds _process_batch once the assertion point is reached."""


def control(ts, chunk=1, filename="next.xtc2", service=TransitionId.Enable):
    dg = NS(timestamp=lambda: ts, chunkinfo={0: NS(chunkinfo=NS(chunkid=chunk, filename=filename))})
    return service, [dg]


@pytest.fixture
def io(monkeypatch):
    state = NS(files={}, calls=[], futures=[], handles=[], fail_submit=None,
               fail_get=None, short=None)

    class Future:
        def __init__(self, handle, dst, size, offset, number):
            self.handle, self.dst, self.size, self.offset = handle, dst, size, offset
            self.number, self.gets = number, 0

        def get(self):
            self.gets += 1
            assert self.gets == 1
            assert not self.handle.closed
            state.calls.append(("get", self.number))
            if self.number == state.fail_get:
                raise OSError("injected future failure")
            size = self.size - 1 if self.number == state.short else self.size
            payload = state.files[self.handle.path][self.offset:self.offset + size]
            self.dst[:len(payload)] = np.frombuffer(payload, dtype=np.uint8)
            self.dst = None  # completed fake futures retain observations, not buffers
            return len(payload)

    class File:
        def __init__(self, path, mode):
            assert mode == "r"
            self.path, self.closed = path, False
            state.handles.append(self)

        def pread(self, dst, size, file_offset, task_size=None):
            number = len(state.futures)
            state.calls.append(("submit", self.path, file_offset, size))
            if number == state.fail_submit:
                raise OSError("injected submission failure")
            f = Future(self, dst, size, file_offset, number)
            state.futures.append(f)
            return f

        def close(self):
            assert not self.closed
            self.closed = True
            state.calls.append(("close", self.path))

    monkeypatch.setitem(sys.modules, "cupy", NS(empty=np.empty, uint8=np.uint8))
    monkeypatch.setitem(sys.modules, "kvikio", NS(
        CuFile=File, DriverProperties=lambda: NS(is_gds_available=True),
    ))
    return state


def groups(descriptors, epochs):
    rows = tuple(ResolvedDgram(d.batch_event_index, d.timestamp, d.stream_id,
                 epochs[(d.batch_event_index, d.stream_id)].file, d.offset, d.size, d.smd_size)
                 for d in descriptors)
    return build_stream_read_plan(rows, n_events=max((d.batch_event_index for d in rows), default=-1) + 1,
        input_capacity_bytes=sum(d.size for d in rows),
        fence_by_event={d.batch_event_index: epochs[(d.batch_event_index, d.stream_id)].fence
                        for d in descriptors}).groups


def issue(reader, descriptors, manager, transitions=(), slot=None):
    if not reader.bulk_read:
        return reader.issue_batch(view(descriptors), manager, slot_id=slot)
    planned = groups(descriptors, GpuFileEpochs(manager).resolve(descriptors, transitions))
    assert len(planned) == 1
    return reader.issue_group(planned[0], slot_id=0 if slot is None else slot)


@pytest.mark.parametrize("bulk,requests", [(False, 6), (True, 2), (None, 2)])
def test_reader_preserves_logical_bytes_and_reports_physical_requests(io, bulk, requests):
    io.files = {"/fast": bytes(range(64)), "/slow": bytes(reversed(range(64)))}
    manager = dm(io.files)
    descriptors = [desc(0, 0, 0), desc(1, 0, 4), desc(1, 1, 0, 8),
                   desc(2, 0, 8), desc(3, 0, 12), desc(3, 1, 8, 8)]
    reader = KvikioGpuReader(**({} if bulk is None else {"bulk_read": bulk}))
    pending = ([reader.issue_group(g, slot_id=i) for i, g in enumerate(groups(
        descriptors, GpuFileEpochs(manager).resolve(descriptors, [])))] if reader.bulk_read
        else [issue(reader, descriptors, manager)])
    assert all(f.gets == 0 for f in io.futures)
    results = [reader.wait_batch(p) for p in pending]
    for result in results:
        for row in result.desc_table:
            event, stream, _, file_offset, size, offset = map(int, row)
            assert bytes(result.data_gpu[offset:offset + size]) == io.files[manager.xtc_files[stream]][file_offset:file_offset + size]
    assert sum(r.data_gpu.nbytes for r in results) == 32
    assert reader.io_stats()["total_requests"] == requests
    assert reader.io_stats()["useful_bytes"] == reader.io_stats()["requested_bytes"] == 32
    assert reader.io_stats()["total_bytes"] == 32
    assert reader.io_stats()["issue_to_complete_ns"] >= reader.io_stats()["total_ns"]
    for p in pending:
        reader.wait_batch(p)  # terminal result/statistics are idempotent
    assert reader.io_stats()["total_requests"] == requests
    reader.close()
    assert all(h.closed for h in io.handles)


@pytest.mark.parametrize("kind,number", [("submit", 1), ("get", 0), ("get", 1),
                                         ("short", 0), ("short", 1), ("short", 2)])
def test_failures_drain_every_started_future_once_and_disable_reuse(io, kind, number):
    io.files = {"/fast": bytes(range(64))}
    setattr(io, {"submit": "fail_submit", "get": "fail_get", "short": "short"}[kind], number)
    budget = _GpuBudget(12)
    reader = KvikioGpuReader(bulk_read=False, budget=budget)
    descriptors = [desc(i, 0, i * 8) for i in range(3)]
    with pytest.raises(RuntimeError, match="file=/fast") as error:
        pending = issue(reader, descriptors, dm(io.files))
        reader.wait_batch(pending)
    assert error.value.__cause__ is not None
    assert len(io.futures) == (1 if kind == "submit" else 3)
    assert all(f.gets == 1 for f in io.futures)
    assert not reader._pending
    reader.trim_free_buffers()
    # The saved exception/read still owns the failed generation after cache
    # removal. Draining and prohibiting reuse must not erase that live charge.
    assert reader.memory_bytes()['raw_input_slots'] == 0
    assert budget.committed() == 12
    assert sum(a['capacity'] for a in budget.allocation_snapshot()) == 12
    with pytest.raises(RuntimeError, match="closed or failed"):
        issue(reader, descriptors, dm(io.files))
    reader.close()
    reader.close()
    assert all(h.closed for h in io.handles)


def test_partial_submission_preserves_first_error_if_drain_also_fails(io):
    io.files = {"/fast": bytes(range(32))}
    io.fail_submit, io.fail_get = 1, 0
    reader = KvikioGpuReader(bulk_read=False)
    with pytest.raises(RuntimeError, match="injected submission failure"):
        issue(reader, [desc(0, 0, 0), desc(1, 0, 8)], dm(io.files))
    assert io.futures[0].gets == 1
    reader.close()


def test_close_drains_pending_work_before_closing_files(io):
    io.files = {"/fast": bytes(range(32))}
    reader = KvikioGpuReader(bulk_read=False)
    pending = issue(reader, [desc(0, 0, 0)], dm(io.files))
    with pytest.raises(RuntimeError, match="pending I/O"):
        reader.issue_batch(view([desc(1, 0, 4)]), dm(io.files), slot_id=0)
    reader.close()
    assert pending.completed
    assert io.calls[-2:] == [("get", 0), ("close", "/fast")]


def test_chunk_handle_survives_another_slots_new_file(io):
    io.files = {"/fast": bytes(range(32)), "/next.xtc2": bytes(range(32))}
    reader = KvikioGpuReader(bulk_read=True)
    manager = dm(["/fast"])
    resolver = GpuFileEpochs(manager)
    first = [desc(0, 0, 0)]
    old = reader.issue_group(groups(first, resolver.resolve(first, []))[0], slot_id=0)
    second = [desc(1, 0, 0)]
    new = reader.issue_group(groups(second, resolver.resolve(second, [control(105)]))[0], slot_id=1)
    reader.wait_batch(new)
    assert not io.handles[0].closed  # old read still owns its handle
    reader.wait_batch(old)
    assert io.handles[0].closed
    assert not io.handles[1].closed
    reader.close()


def test_transition_fence_prevents_merge_even_with_same_file_and_adjacent_bytes(io):
    io.files = {"/fast": bytes(range(32))}
    reader = KvikioGpuReader(bulk_read=True)
    descriptors = [desc(0, 0, 0), desc(1, 0, 4)]
    epochs = GpuFileEpochs(dm(io.files)).resolve(descriptors,
        [control(105, service=TransitionId.Disable)])
    for i, g in enumerate(groups(descriptors, epochs)):
        reader.wait_batch(reader.issue_group(g, slot_id=i))
    assert reader.io_stats()["total_requests"] == 2
    reader.close()


def test_an_absent_record_does_not_fence_coalescing(io):
    """An absent record is not a control and must NOT fence.

    `resolve()` skipped it via `not service`, which also swallowed a present
    service-0 record -- the conflation issue #156 removes. Absence is now
    None and is still skipped, deliberately: there is no control record
    there, so two reads either side of it must coalesce.

    Contrast with the test above, where a real control fences and the reads
    stay separate.
    """
    io.files = {"/fast": bytes(range(32))}
    reader = KvikioGpuReader(bulk_read=True)
    descriptors = [desc(0, 0, 0), desc(1, 0, 4)]
    epochs = GpuFileEpochs(dm(io.files)).resolve(
        descriptors, [control(105, service=None)])

    assert {e.file for e in epochs.values()} == {
        ResolvedFile(os.path.realpath("/fast"), 0)}
    for i, g in enumerate(groups(descriptors, epochs)):
        reader.wait_batch(reader.issue_group(g, slot_id=i))
    assert reader.io_stats()["total_requests"] == 1, (
        'an absent record fenced coalescing: it is not a control boundary, '
        'so the two adjacent reads should have merged')
    reader.close()


def test_an_absent_record_is_skipped_before_any_file_resolution(io):
    """`resolve` must discard an absent record before it reaches `_apply`.

    The claim is specifically about the SKIP, not about `_apply`'s filter: a
    `None` record never reaches `_apply` at all, so a chunkinfo planted on it
    is never examined and cannot demonstrate anything about which services
    switch files. `test_only_enable_applies_a_chunkinfo_file_switch` below
    covers that filter, with a service that does reach `_apply`.
    """
    io.files = {"/fast": bytes(range(32))}
    before = GpuFileEpochs(dm(io.files))._active[0]
    epochs_obj = GpuFileEpochs(dm(io.files))
    applied = []
    original = epochs_obj._apply
    epochs_obj._apply = lambda *a: applied.append(a) or original(*a)

    epochs = epochs_obj.resolve(
        [desc(0, 0, 0)],
        [control(105, chunk=7, filename="other.xtc2", service=None)])

    assert applied == [], 'an absent record must not reach _apply'
    assert epochs[(0, 0)].file == before


@pytest.mark.parametrize('service,switches', [
    (TransitionId.Enable, True),
    (TransitionId.Disable, False),
    (TransitionId.BeginStep, False),
])
def test_only_enable_applies_a_chunkinfo_file_switch(io, service, switches):
    """`_apply`'s `service != Enable` early return, with services that reach it.

    This is what the absent-record test above cannot show. Each of these is a
    present, supported control carrying chunkinfo; only Enable may remap the
    stream. Deleting `_apply`'s early return makes the non-Enable cases fail.
    """
    # Two descriptors straddling the control at ts 105: a switch shows only
    # on the one recorded AFTER it, so a single earlier descriptor cannot
    # distinguish "applied" from "not applied".
    manager = dm(["/data/old.xtc2"])
    epochs = GpuFileEpochs(manager).resolve(
        [desc(0, 0, 88), desc(1, 0, 124)],
        [control(105, chunk=7, filename="new.xtc2", service=service)])

    assert epochs[(0, 0)].file.path == "/data/old.xtc2", (
        'the descriptor before the control is never remapped')
    if switches:
        assert epochs[(1, 0)].file.path == "/data/new.xtc2", (
            'Enable with chunkinfo must remap the following descriptor')
        assert epochs[(1, 0)].file.chunk_id == 7
    else:
        assert epochs[(1, 0)].file.path == "/data/old.xtc2", (
            f'service {service} carried chunkinfo but must not switch files')


def test_capacity_is_total_input_not_last_logical_row(io):
    io.files = {"/a": bytes(range(32)), "/z": bytes(range(32))}
    reader = KvikioGpuReader(budget=_GpuBudget(8), bulk_read=False)
    descriptors = [desc(0, 1, 0), desc(1, 0, 0), desc(2, 0, 4, 0)]
    pending = issue(reader, descriptors, dm(io.files))
    assert pending.data_gpu.nbytes == 8
    assert list(pending.desc_table[:, DESC_DEVICE_OFFSET]) == [0, 4, 8]
    reader.wait_batch(pending)
    reader.close()


def test_capacity_failure_happens_before_allocation_or_io(io):
    reader = KvikioGpuReader(budget=_GpuBudget(3), bulk_read=True)
    with pytest.raises(GpuMemoryPressureError):
        issue(reader, [desc(0, 0, 0)], dm(["/fast"]))
    assert reader.memory_bytes()["raw_input_slots"] == 0
    assert not io.calls
    assert reader._budget.committed() == 0


def test_allocation_failure_rolls_back_and_submits_nothing(io):
    reader = KvikioGpuReader(budget=_GpuBudget(8), bulk_read=True)
    def fail(*args, **kwargs):
        raise MemoryError("injected allocation failure")
    reader.cp = NS(empty=fail, uint8=np.uint8)
    with pytest.raises(MemoryError):
        issue(reader, [desc(0, 0, 0)], dm(["/fast"]))
    assert not io.calls and not reader._pending
    assert reader._budget.committed() == 0


def test_resolver_maps_both_sides_of_chunk_transition_before_dm_changes():
    manager = dm(["/data/old.xtc2"])
    resolver = GpuFileEpochs(manager)
    descriptors = [desc(0, 0, 88), desc(1, 0, 24)]
    mapping = resolver.resolve(descriptors, [control(105)])
    assert mapping[(0, 0)].file.path == "/data/old.xtc2"
    assert mapping[(1, 0)].file.path == "/data/next.xtc2"
    manager.xtc_files[0] = "/data/wrong.xtc2"
    replay = resolver.resolve([desc(2, 0, 24)], [control(105)])
    assert replay[(2, 0)].file == mapping[(1, 0)].file


def test_empty_gpu_packet_advances_chunk_state_and_old_history_does_not_rewind():
    resolver = GpuFileEpochs(dm(["/data/old.xtc2"]))
    resolver.resolve([], [control(105, 2, "c002.xtc2")])
    mapping = resolver.resolve([desc(3, 0, 24)], [control(95, 1, "c001.xtc2")])
    assert mapping[(3, 0)].file.chunk_id == 2


def test_resolver_rejects_ambiguous_transition_and_preserves_state():
    resolver = GpuFileEpochs(dm(["/data/old.xtc2"]))
    bad = control(105)
    bad[1][0].chunkinfo[1] = NS(chunkinfo=NS(chunkid=2, filename="other.xtc2"))
    with pytest.raises(ValueError, match="conflicting chunkinfo"):
        resolver.resolve([desc(1, 0, 24)], [bad])
    assert resolver.resolve([desc(0, 0, 88)], [])[(0, 0)].file.chunk_id == 0


@pytest.mark.parametrize("value", ["true", 1, None])
def test_bulk_flag_rejects_nonboolean_values(value):
    from psana.psexp.ds_base import DsParms
    with pytest.raises(TypeError, match="gpu_bulk_read"):
        DsParms(5, 0, 0, False, None, "", 0, False, [], 0, [], "",
                gpu_det="jungfrau", gpu_bulk_read=value)


def test_bulk_default_requires_supported_gpu_packet_path_only_for_gpu():
    from psana.psexp.ds_base import DataSourceBase, DsParms

    class MinimalDataSource(DataSourceBase):
        def is_mpi(self):
            return False

        def runs(self):
            return iter(())

    args = (5, 0, 0, False, None, "", 0, False, [], 0, [], "")
    cpu = DsParms(*args)
    assert not cpu.gpu_enabled
    replace(cpu, intg_det="jungfrau", timestamps=np.array([100]))
    ds = MinimalDataSource(gpu_det="jungfrau")
    assert ds.gpu_bulk_read and ds.dsparms.gpu_bulk_read
    assert ds.dsparms.gpu_bulk_target_bytes == 1 << 20
    ds4 = MinimalDataSource(gpu_det="jungfrau", gpu_bulk_target_bytes=4 << 20)
    assert ds4.gpu_bulk_target_bytes == ds4.dsparms.gpu_bulk_target_bytes == 4 << 20
    p = DsParms(*args, gpu_det="jungfrau")
    assert p.gpu_bulk_read
    with pytest.raises(NotImplementedError, match="ordinary GPUBAT1"):
        replace(p, intg_det="jungfrau")
    with pytest.raises(NotImplementedError, match="ordinary GPUBAT1"):
        replace(p, timestamps=np.array([100]))
    assert not replace(p, gpu_bulk_read=False).gpu_bulk_read


def test_exclusive_smd_packet_resolves_real_chunked_fixture_before_cpu_reads(io, monkeypatch):
    from psana.gpu.gpu_batch import GpuBatchView
    from psana.gpu.gpu_events import _iter_step_events
    from psana.psexp.ds_base import DsParms
    from psana.psexp.smdreader_manager import SmdReaderManager

    # This coverage requires a packet spanning a file-chunk boundary.
    monkeypatch.setenv("PS_SMD_N_EVENTS", "1000")
    root = Path(__file__).resolve().parents[2] / "test_data/chunking"
    initial = root / "xpptut15-r0014-s000-c000.xtc2"
    io.files = {str(path.resolve()): path.read_bytes() for path in root.glob("*.xtc2")}
    manager = dm([str(initial.resolve())])
    manager.fds = [0]  # metadata index only; no CPU bigdata handle is read
    resolver = GpuFileEpochs(manager)
    params = DsParms(1000, 0, 0, False, np.empty(0, dtype=np.uint64), "", 0,
                     False, [], 0, [], "", gpu_det="xppcspad", gpu_bulk_read=True)
    params.gpu_stream_ids = [0]
    smd = root / "smalldata/xpptut15-r0014-s000-c000.smd.xtc2"
    fd = os.open(smd, os.O_RDONLY)
    reader = KvikioGpuReader(bulk_read=True)
    used, seen = set(), set()
    crossed_inside_packet = False
    try:
        source = SmdReaderManager(np.array([fd], dtype=np.int32), params)
        configs = source.get_next_dgrams()
        source.get_next_dgrams()  # BeginRun
        for batches in source:
            while True:
                try:
                    _, packets, steps = batches.next_with_gpu()
                except StopIteration:
                    break
                transitions = [item for packet, _ in steps.values()
                               for item in _iter_step_events(packet, configs)]
                packet_views = [GpuBatchView(packet) for packet, _ in packets.values()]
                descriptors = [d for v in packet_views for d in v.iter_read_descs(manager)]
                epochs = resolver.resolve(descriptors, transitions)
                identities = {e.file for e in epochs.values()}
                crossed_inside_packet |= len(identities) > 1
                used.update(identities)
                for v in packet_views:
                    for g in groups(tuple(v.iter_read_descs(manager)), epochs):
                        result = reader.wait_batch(reader.issue_group(g, slot_id=0))
                        for d, row in zip(g.dgrams, result.desc_table):
                            off, size = int(row[DESC_DEVICE_OFFSET]), int(row[DESC_READ_SIZE])
                            payload = bytes(result.data_gpu[off:off + size])
                            identity = epochs[(d.batch_event_index, d.stream_id)].file
                            assert payload == io.files[identity.path][d.file_offset:d.file_offset + d.size]
                            assert int.from_bytes(payload[:8], "little") == d.timestamp
                            assert d.timestamp not in seen
                            seen.add(d.timestamp)
    finally:
        reader.close()
        os.close(fd)
    assert len(used) == 2
    assert crossed_inside_packet
    assert seen


def test_completed_read_pin_blocks_reuse_until_all_input_owners_release(io):
    io.files = {"/fast": bytes(range(32))}
    reader = KvikioGpuReader(n_slots=1)
    result = reader.wait_batch(issue(reader, [desc(0, 0, 0)], dm(io.files)))
    first, second = result.retain_input(), result.retain_input()
    with pytest.raises(RuntimeError, match="owned by an input window"):
        issue(reader, [desc(1, 0, 4)], dm(io.files))
    first()
    first()
    with pytest.raises(RuntimeError, match="owned by an input window"):
        issue(reader, [desc(1, 0, 4)], dm(io.files))
    second()
    reader.wait_batch(issue(reader, [desc(1, 0, 4)], dm(io.files)))
    with pytest.raises(RuntimeError, match="obsolete GPU read"):
        result.retain_input()
    reader.close()


@pytest.mark.parametrize("value,error", [(True, TypeError), (1.5, TypeError),
    ("4194304", TypeError), (None, TypeError), (0, ValueError), (-1, ValueError),
    (1 << 64, ValueError)])
def test_bulk_target_validation(value, error):
    from psana.psexp.ds_base import DsParms
    with pytest.raises(error, match="gpu_bulk_target_bytes"):
        DsParms(5, 0, 0, False, None, "", 0, False, [], 0, [], "",
                gpu_bulk_target_bytes=value)


def test_absent_records_are_filtered_before_resolve(monkeypatch):
    """Pins `_process_batch`'s `transition[0] is not None` filter.

    The filter is not load-bearing for correctness -- `resolve()` skips None
    itself -- so nothing else fails if it is deleted, which makes it easy to
    lose to a "simplification". Nor is it there to save work: an instrumented
    uncapped run of mfx100852324 r77 reports an absent fraction of 0.0000
    over 80 records, so on that fixture it discards nothing. It is kept
    because it states the precondition `resolve()` relies on where the list
    is built, rather than leaving it implicit two calls away -- so this test
    is the only thing standing between it and a cleanup.

    This drives the real `_process_batch` and inspects what `resolve()`
    actually receives. Reproducing the comprehension in the test body instead
    would pass with the filter deleted, since the test would be checking its
    own copy of it.
    """
    from psana.gpu import gpu_events

    decoded = [(None, [None]),
               (TransitionId.Enable, [NS(timestamp=lambda: 105,
                                         chunkinfo={})]),
               (None, [None])]
    monkeypatch.setattr(gpu_events, '_iter_step_events',
                        lambda packet, configs: iter(decoded))

    seen = []
    m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
    m.configs = [None]
    m.dm = None
    m.dsparms = NS(gpu_bulk_read=True, max_events=0)
    m._n_events = 0
    # Stop inside resolve() itself: everything after it needs device and run
    # state this test has no business building.
    def _resolve(descs, trans):
        seen.append(list(trans))
        raise _Stop('resolve called')
    m._gpu_file_epochs = NS(resolve=_resolve)

    with pytest.raises(_Stop):
        list(m._process_batch({}, {}, {0: (b'step', [])}))

    assert seen, 'resolve() was never called'
    assert [service for service, _ in seen[0]] == [TransitionId.Enable], (
        'absent records must be filtered before resolve() sees them')
