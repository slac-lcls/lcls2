"""Drive the production scheduler/reader with CPU storage and CUDA tokens."""
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from psana.gpu.gpu_events import GpuEventManager
from psana.gpu.gpu_batch import GpuBatchView, GpuReadSelection, GpuSubbatchView
from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError
from psana.gpu.gpu_file_epochs import GpuFileEpochs
from psana.gpu.gpu_input_window import InputWindow
from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from psana.gpu.gpu_stream import EventPool
from test_gpu_bulk_read import io
from test_gpu_input_window import Stream, Token


class Parser:
    """One byte of parser storage per dgram, with real InputWindow leases."""
    def __init__(self, budget, n=3):
        self.budget, self.owners, self.capacity = budget, [None] * n, [0] * n
        self.windows = []

    def estimate_batch_bytes(self, n):
        return n

    def allocation_requirements(self, n):
        return [(n, self.capacity[self.owners.index(None)])]

    def parse_window(self, read, stream, *, batch_id):
        i, n = self.owners.index(None), len(read.desc_table)
        if n > self.capacity[i]:
            self.budget.reserve(n)
            self.budget.release(self.capacity[i])
            self.capacity[i] = n
        release_raw = read.retain_input()
        def release():
            release_raw()
            self.owners[i] = None
        batch = NS(n_dgrams=n, data_gpu=read.data_gpu, walk_done=Token())
        owner = InputWindow(batch_id, len(self.windows), batch, read.desc_table, release=release)
        self.owners[i] = owner
        self.windows.append(owner)
        return owner

    def trim_free_buffers(self):
        for i, owner in enumerate(self.owners):
            if owner is None:
                self.budget.release(self.capacity[i])
                self.capacity[i] = 0

    def close(self):
        for owner in self.owners:
            if owner is not None:
                owner.close()


def manager(io, capacity=6004, *, max_events=0):
    cp = sys.modules['cupy']
    cp.cuda = NS(Stream=Stream, Event=Token)
    m = GpuEventManager.__new__(GpuEventManager)
    m.dm = NS(xtc_files=['/fast', '/slow'], get_chunk_id=lambda _: 0, fds=[0, 1])
    m.dsparms = NS(gpu_bulk_read=True, n_gpu_streams=2, max_events=max_events)
    m._gpu_budget = _GpuBudget(capacity)
    m._admission_capacity, m._admission_margin = capacity, 0
    m._subbatch_budget_bytes = capacity // 2
    m.gpu_reader = KvikioGpuReader(n_slots=3, budget=m._gpu_budget)
    m.gpu_xtc_parser = Parser(m._gpu_budget)
    m.event_pool = EventPool(n=2)
    m.gpu_detectors, m.gpu_det_names, m._d2h_pipelines = {}, [], {}
    m.configs = [None, None]
    m._gpu_file_epochs = GpuFileEpochs(m.dm)
    m._first_batch_logged, m._done, m._closed = True, False, False
    m._n_events, m._pending_gpu_read = 0, None
    m.run = NS(_handle_transition=lambda _: None)
    io.files = {'/fast': bytes(i % 256 for i in range(1000)),
                '/slow': b'x' * 10000}
    return m


def execute(m, packet):
    return m._process_batch({}, {0: (packet, [])}, {})


def test_full_fast_read_and_five_two_slow_reads(io, mixed_packet):
    m = manager(io)
    seen, fast_owners, slow_owners = [], set(), set()
    early = Token()
    for envelope in execute(m, mixed_packet()):
        state = envelope.gpu_state
        fast = state._event_dgrams[0].owner
        fast_owners.add(id(fast))
        i = state._event_dgrams.batch_event_index
        seen.append(i)
        row = fast.rows_by_event[i][0][0]
        assert int(fast.batch.data_gpu[row]) == i % 256
        assert not fast.released
        if i == 0:
            use = state._input_lease.acquire_view()
            use.register_consumer_done(early)
            use.wait_until_safe_to_reuse()
        if 1 in state._event_dgrams:
            slow_owners.add(id(state._event_dgrams[1].owner))
        assert early.waits == 0  # resident use remains open for later executions
    assert seen == list(range(1000)) and len(fast_owners) == 1
    assert len(slow_owners) == 5
    submits = [c for c in io.calls if c[0] == 'submit']
    assert [(c[1], c[3]) for c in submits] == [('/fast', 1000)] + [('/slow', 2000)] * 5
    assert early.waits == 1
    assert all(w.released for w in m.gpu_xtc_parser.windows)
    assert not m._gpu_budget._held and not m.event_pool.active_count
    m.close()


@pytest.mark.parametrize('capacity', [3500, 1500])
def test_tight_budget_preserves_all_events_and_each_input_once(io, capacity, mixed_packet):
    m = manager(io, capacity)
    timestamps = [e.dgrams[0].timestamp() for e in execute(m, mixed_packet())]
    timestamps.extend(e.dgrams[0].timestamp() for e in m.finish())
    assert timestamps == list(range(1000, 2000))
    assert m.gpu_reader.io_stats()['requested_bytes'] == 11000
    assert m._gpu_budget.committed() <= capacity and m._gpu_budget._held == 0


def test_resident_only_execution_does_not_issue_empty_reads(io, mixed_packet):
    m = manager(io, 10000)
    packet = mixed_packet(n_events=4, interval=2, slow_size=1000)
    assert len(list(execute(m, packet))) == 4
    assert len(m.gpu_xtc_parser.windows) == 1
    assert m.gpu_reader.io_stats()['total_requests'] == 2
    m.close()


@pytest.mark.parametrize('stop', ['max_events', 'generator_close', 'read_failure'])
def test_resident_cleanup_on_stop_or_failure(io, stop, mixed_packet):
    m = manager(io, max_events=3 if stop == 'max_events' else 0)
    events = execute(m, mixed_packet())
    if stop == 'read_failure':
        io.fail_get = 1  # resident succeeded; first slow read fails
        with pytest.raises(RuntimeError, match='injected future failure'):
            list(events)
    elif stop == 'generator_close':
        next(events)
        events.close()
    else:
        assert len(list(events)) == 3
    m.close()
    assert all(w.released for w in m.gpu_xtc_parser.windows)
    assert m._gpu_budget._held == 0 and not m.gpu_reader._pending
    assert not any(m.gpu_reader._input_holds.values())


def test_read_selection_keeps_original_event_and_stream_ids(mixed_packet):
    view = GpuBatchView(mixed_packet(n_events=5, interval=2, missing_fast=(2,)))
    selected = GpuReadSelection.from_view(GpuSubbatchView(view, 1, 5), NS(fds=[0, 1]), (0,), exclude=True)
    assert [(d.batch_event_index, d.stream_id) for d in selected.descriptors] == [(1, 1), (3, 1)]
    assert selected.total_read_bytes == 2000


def test_resident_read_reserves_execution_progress_before_io(io, mixed_packet):
    m = manager(io)
    view = GpuBatchView(mixed_packet())
    m._split_subbatches(view, allow_residency=True)
    # Resident input needs 2000 bytes, one planned execution needs 2002.
    # Merely fitting the fast input is insufficient to start reading it.
    m._gpu_budget._limit = 4001
    with pytest.raises(GpuMemoryPressureError):
        m._start_resident_input(view, m._last_admission_plan)
    assert not io.calls and m._gpu_budget._held == 0
    m.close()


def test_missing_streams_and_partial_tail_keep_event_identity(io, mixed_packet):
    m = manager(io)
    seen = []
    for envelope in execute(m, mixed_packet(n_events=955, missing_fast=(1, 499))):
        dgrams = envelope.gpu_state._event_dgrams
        i = dgrams.batch_event_index
        assert (0 in dgrams) == (i not in (1, 499))
        assert (1 in dgrams) == (i % 100 == 99)
        seen.append(i)
    assert seen == list(range(955))
    sizes = [c[3] for c in io.calls if c[0] == 'submit' and c[1] == '/slow']
    assert sizes == [2000, 2000, 2000, 2000, 1000]
    m.close()


def test_resident_batch_drains_before_next_batch_and_transitions(io, mixed_packet, monkeypatch):
    from psana.psexp import TransitionId
    from psana.gpu import gpu_events
    m = manager(io)
    assert len(list(execute(m, mixed_packet()))) == 1000
    first = tuple(m.gpu_xtc_parser.windows)
    transitions = []
    def dispatch(dgrams):
        assert all(w.released for w in m.gpu_xtc_parser.windows)
        assert not m.event_pool.active_count and m._resident_window is None
        transitions.append(dgrams[0].service())
    m.run._handle_transition = dispatch
    monkeypatch.setattr(gpu_events, '_iter_step_events', lambda packet, configs: iter(packet))
    for service in (TransitionId.BeginStep, TransitionId.EndRun):
        dg = NS(timestamp=lambda: 3000, service=lambda: service)
        assert list(m._process_batch({}, {}, {0: ([(service, [dg])], [])})) == []
        if service == TransitionId.BeginStep:
            assert len(list(execute(m, mixed_packet(n_events=7, interval=2)))) == 7
            assert all(w.released for w in first)
    assert transitions == [TransitionId.BeginStep, TransitionId.EndRun]
    assert m._done
    m.close()


def test_hybrid_cpu_envelopes_keep_their_payload(io, mixed_packet, monkeypatch):
    from psana.event import EventEnvelope
    from psana.gpu import gpu_events
    m = manager(io)
    cpu = [NS(timestamp=lambda i=i: 1000+i, service=lambda: 12, env=lambda: 12 << 24, detector_payload=i)
           for i in range(7)]
    class CpuEvents:
        exit_id = 0
        def __init__(self, *args):
            pass
        def __iter__(self):
            return iter(EventEnvelope([dg, None]) for dg in cpu)
    monkeypatch.setattr(gpu_events, 'EventManager', CpuEvents)
    m.max_retries, m.use_smds = 0, False
    yielded = list(m._process_batch({0: (b'cpu batch', [])},
                                    {0: (mixed_packet(n_events=7, interval=2), [])}, {}))
    assert [e.dgrams[0] for e in yielded] == cpu
    assert all(e.gpu_state is not None for e in yielded)
    m.close()
