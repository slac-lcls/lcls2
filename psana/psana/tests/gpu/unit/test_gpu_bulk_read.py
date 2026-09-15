"""CPU fault-injection tests for bulk I/O and ordered file resolution."""

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from psana.gpu.gpu_batch import GpuReadDesc
from psana.gpu.gpu_budget import _GpuBudget
from psana.gpu.gpu_file_epochs import FileEpoch, GpuFileEpochs
from psana.gpu.gpu_kvikio_read import KvikioGpuReader, DESC_DEVICE_OFFSET, DESC_READ_SIZE
from psana.gpu.gpu_read_plan import ResolvedFile
from psana.psexp import TransitionId


def desc(event, stream, offset, size=4):
    return GpuReadDesc(event, 100 + 10 * event, stream, offset, size, 80, 1)


def dm(paths):
    return NS(xtc_files=list(paths), get_chunk_id=lambda stream: 0)


def view(descriptors):
    return NS(iter_read_descs=lambda _: iter(descriptors))


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


def issue(reader, descriptors, manager, transitions=()):
    epochs = GpuFileEpochs(manager).resolve(descriptors, transitions)
    return reader.issue_batch(view(descriptors), manager, file_epochs=epochs)


@pytest.mark.parametrize("bulk,requests", [(False, 6), (True, 2), (None, 2)])
def test_reader_preserves_logical_bytes_and_reports_physical_requests(io, bulk, requests):
    io.files = {"/fast": bytes(range(64)), "/slow": bytes(reversed(range(64)))}
    manager = dm(io.files)
    descriptors = [desc(0, 0, 0), desc(1, 0, 4), desc(1, 1, 0, 8),
                   desc(2, 0, 8), desc(3, 0, 12), desc(3, 1, 8, 8)]
    reader = KvikioGpuReader(**({} if bulk is None else {"bulk_read": bulk}))
    pending = issue(reader, descriptors, manager)
    assert all(f.gets == 0 for f in io.futures)  # submission stays asynchronous
    result = reader.wait_batch(pending)
    for d, row in zip(descriptors, result.desc_table):
        offset, size = int(row[DESC_DEVICE_OFFSET]), int(row[DESC_READ_SIZE])
        assert bytes(result.data_gpu[offset:offset + size]) == io.files[manager.xtc_files[d.stream_id]][d.offset:d.offset + d.size]
    assert result.data_gpu.nbytes == 32
    assert reader.io_stats()["total_requests"] == requests
    assert reader.io_stats()["useful_bytes"] == reader.io_stats()["requested_bytes"] == 32
    assert reader.io_stats()["total_bytes"] == 32
    assert reader.io_stats()["issue_to_complete_ns"] >= reader.io_stats()["total_ns"]
    reader.wait_batch(pending)  # terminal result/statistics are idempotent
    assert reader.io_stats()["total_requests"] == requests
    reader.close()
    assert all(h.closed for h in io.handles)


@pytest.mark.parametrize("kind,number", [("submit", 1), ("get", 0), ("get", 1),
                                         ("short", 0), ("short", 1), ("short", 2)])
def test_failures_drain_every_started_future_once_and_disable_reuse(io, kind, number):
    io.files = {"/fast": bytes(range(64))}
    setattr(io, {"submit": "fail_submit", "get": "fail_get", "short": "short"}[kind], number)
    reader = KvikioGpuReader(bulk_read=True)
    descriptors = [desc(i, 0, i * 8) for i in range(3)]
    with pytest.raises(RuntimeError, match="file=/fast") as error:
        pending = issue(reader, descriptors, dm(io.files))
        reader.wait_batch(pending)
    assert error.value.__cause__ is not None
    assert len(io.futures) == (1 if kind == "submit" else 3)
    assert all(f.gets == 1 for f in io.futures)
    assert not reader._pending
    with pytest.raises(RuntimeError, match="closed or failed"):
        issue(reader, descriptors, dm(io.files))
    reader.close()
    reader.close()
    assert all(h.closed for h in io.handles)


def test_partial_submission_preserves_first_error_if_drain_also_fails(io):
    io.files = {"/fast": bytes(range(32))}
    io.fail_submit, io.fail_get = 1, 0
    reader = KvikioGpuReader(bulk_read=True)
    with pytest.raises(RuntimeError, match="injected submission failure"):
        issue(reader, [desc(0, 0, 0), desc(1, 0, 8)], dm(io.files))
    assert io.futures[0].gets == 1
    reader.close()


def test_close_drains_pending_work_before_closing_files(io):
    io.files = {"/fast": bytes(range(32))}
    reader = KvikioGpuReader(bulk_read=True)
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
    old = reader.issue_batch(view(first), manager, slot_id=0,
                            file_epochs=resolver.resolve(first, []))
    second = [desc(1, 0, 0)]
    new = reader.issue_batch(view(second), manager, slot_id=1,
                            file_epochs=resolver.resolve(second, [control(105)]))
    reader.wait_batch(new)
    assert not io.handles[0].closed  # old read still owns its handle
    reader.wait_batch(old)
    assert io.handles[0].closed
    assert not io.handles[1].closed
    reader.close()


def test_transition_fence_prevents_merge_even_with_same_file_and_adjacent_bytes(io):
    io.files = {"/fast": bytes(range(32))}
    reader = KvikioGpuReader(bulk_read=True)
    pending = issue(reader, [desc(0, 0, 0), desc(1, 0, 4)], dm(io.files),
                    [control(105, service=TransitionId.Disable)])
    reader.wait_batch(pending)
    assert reader.io_stats()["total_requests"] == 2
    reader.close()


def test_capacity_is_total_input_not_last_logical_row(io):
    io.files = {"/a": bytes(range(32)), "/z": bytes(range(32))}
    reader = KvikioGpuReader(budget=_GpuBudget(8), bulk_read=True)
    descriptors = [desc(0, 1, 0), desc(1, 0, 0), desc(2, 0, 4, 0)]
    pending = issue(reader, descriptors, dm(io.files))
    assert pending.data_gpu.nbytes == 8
    assert list(pending.desc_table[:, DESC_DEVICE_OFFSET]) == [4, 0, 0]
    reader.wait_batch(pending)
    reader.close()


def test_capacity_failure_happens_before_allocation_or_io(io):
    reader = KvikioGpuReader(budget=_GpuBudget(3), bulk_read=True)
    with pytest.raises(ValueError, match="capacity_bytes"):
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
    p = DsParms(*args, gpu_det="jungfrau")
    assert p.gpu_bulk_read
    with pytest.raises(NotImplementedError, match="ordinary GPUBAT1"):
        replace(p, intg_det="jungfrau")
    with pytest.raises(NotImplementedError, match="ordinary GPUBAT1"):
        replace(p, timestamps=np.array([100]))
    assert not replace(p, gpu_bulk_read=False).gpu_bulk_read


def test_exclusive_smd_packet_resolves_real_chunked_fixture_before_cpu_reads(io):
    from psana.gpu.gpu_batch import GpuBatchView
    from psana.gpu.gpu_events import _iter_step_events
    from psana.psexp.ds_base import DsParms
    from psana.psexp.smdreader_manager import SmdReaderManager

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
                    result = reader.wait_batch(reader.issue_batch(v, manager, file_epochs=epochs))
                    for d, row in zip(v.iter_read_descs(manager), result.desc_table):
                        off, size = int(row[DESC_DEVICE_OFFSET]), int(row[DESC_READ_SIZE])
                        payload = bytes(result.data_gpu[off:off + size])
                        identity = epochs[(d.batch_event_index, d.stream_id)].file
                        assert payload == io.files[identity.path][d.offset:d.offset + d.size]
                        assert int.from_bytes(payload[:8], "little") == d.timestamp
                        assert d.timestamp not in seen
                        seen.add(d.timestamp)
    finally:
        reader.close()
        os.close(fd)
    assert len(used) == 2
    assert crossed_inside_packet
    assert seen
