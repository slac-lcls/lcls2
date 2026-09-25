"""Run the controller and reader with injected CUDA completion and tiny I/O."""
from types import SimpleNamespace as NS

import pytest

from test_gpu_bulk_read import io
from test_gpu_residency import manager
from test_gpu_input_window import Token
from psana.gpu.gpu_input_group import InputGroupPool
from psana.gpu.gpu_input_window import InputWindow
from psana.gpu.gpu_kvikio_read import KvikioGpuReader


class Parser:
    def __init__(self):
        self.windows = []
        self.launch_batches = []

    def estimate_batch_bytes(self, n):
        return n

    def allocation_requirements(self, n, *, groups=False):
        return []

    def parse_groups(self, reads, stream, *, batch_id):
        self.launch_batches.append(len(reads))
        windows = []
        for read in reads:
            batch = NS(data_gpu=read.data_gpu, n_dgrams=len(read.desc_table), walk_done=Token())
            window = InputWindow(batch_id, len(self.windows), batch, read.desc_table,
                                 release=read.retain_input(), defer_retirement=True)
            self.windows.append(window)
            windows.append(window)
        return windows

    def trim_free_buffers(self):
        pass

    def close(self):
        assert all(w.drain() for w in self.windows)


def build(io, monkeypatch):
    monkeypatch.setattr(Token, 'done', property(lambda self: True), raising=False)
    m = manager(io, capacity=4 * 1024**2)
    m.gpu_reader.close()
    m.gpu_reader = KvikioGpuReader(n_slots=30, budget=m._gpu_budget)
    m._group_inputs = InputGroupPool(m.gpu_reader)
    m.gpu_xtc_parser = Parser()
    io.files = {'/fast': bytes(range(120)), '/slow': b'x' * (4 * 1024**2)}
    return m


@pytest.mark.parametrize("target,requests", [(1 << 20, 5), (20, 10)])
def test_controller_interleaves_requests_and_releases_all_group_inputs(io, monkeypatch, mixed_packet, target, requests):
    m = build(io, monkeypatch)
    m.dsparms.gpu_bulk_target_bytes = target
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2, interval=3)
    observed = []
    try:
        for envelope in m._process_batch({}, {0: (packet, [])}, {}):
            observed.append(envelope.gpu_state._event_dgrams.batch_event_index)
        for envelope in m.finish():
            observed.append(envelope.gpu_state._event_dgrams.batch_event_index)
        assert observed == list(range(12))
        assert m.gpu_reader.io_stats()['total_requests'] == requests
        assert sum(m.gpu_xtc_parser.launch_batches) == requests
        if target == 1 << 20:
            assert m.gpu_xtc_parser.launch_batches[0] == 2
        assert not m._group_inputs.live_keys and m._gpu_budget._held == 0
    finally:
        m.close()


def test_early_close_cancels_future_small_group_uses(io, monkeypatch, mixed_packet):
    m = build(io, monkeypatch)
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2, interval=3)
    loop = m._process_batch({}, {0: (packet, [])}, {})
    next(loop)
    loop.close()
    m.close()
    assert not m._group_inputs.live_keys
    assert not any(m.gpu_reader._input_holds.values())
    assert m._gpu_budget._held == 0


def assert_closed(m, io):
    assert m._closed and not m.event_pool.active_count
    assert not m._group_inputs.live_keys and not m.gpu_reader._pending
    assert not any(m.gpu_reader._input_holds.values())
    assert m._gpu_budget._held == 0
    assert all(w.released and w.batch is None for w in m.gpu_xtc_parser.windows)
    assert all(f.gets == 1 for f in io.futures)
    assert all(h.closed for h in io.handles)


@pytest.mark.parametrize('failure', ['fail_submit', 'fail_get', 'short'])
def test_controller_partial_io_failure_drains_all_groups(io, monkeypatch, mixed_packet, failure):
    m = build(io, monkeypatch)
    setattr(io, failure, 1)  # first request succeeded; second request fails
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2, interval=3)
    with pytest.raises(RuntimeError, match='file='):
        list(m._process_batch({}, {0: (packet, [])}, {}))
    m.close()
    assert_closed(m, io)
    m.close()  # terminal cleanup must not retrieve futures twice
    assert_closed(m, io)


def test_controller_parser_failure_returns_holds_and_keeps_close_retryable(io, monkeypatch, mixed_packet):
    m = build(io, monkeypatch)
    original = m.gpu_xtc_parser.parse_groups

    def fail(*args, **kwargs):
        raise RuntimeError('injected parser failure before launch')

    monkeypatch.setattr(m.gpu_xtc_parser, 'parse_groups', fail)
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2, interval=3)
    with pytest.raises(RuntimeError, match='injected parser failure'):
        list(m._process_batch({}, {0: (packet, [])}, {}))
    monkeypatch.setattr(m.gpu_xtc_parser, 'parse_groups', original)
    m.close()
    assert_closed(m, io)


@pytest.mark.parametrize('max_events', [1, 3, 7])
def test_controller_max_events_retires_undelivered_planned_uses(io, monkeypatch, mixed_packet, max_events):
    m = build(io, monkeypatch)
    m.dsparms.max_events = max_events
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2, interval=3)
    delivered = list(m._process_batch({}, {0: (packet, [])}, {}))
    delivered.extend(m.finish())
    assert [e.dgrams[0].timestamp() for e in delivered] == list(range(1000, 1000 + max_events))
    assert m._done
    assert_closed(m, io)


def test_controller_missing_streams_partial_tail_and_two_batches(io, monkeypatch, mixed_packet):
    m = build(io, monkeypatch)
    m.dsparms.gpu_bulk_target_bytes = 20  # force multiple small groups per batch
    observed, identities = [], []

    def check(envelopes):
        for envelope in envelopes:
            dgrams = envelope.gpu_state._event_dgrams
            event = dgrams.batch_event_index
            assert (0 in dgrams) == (event not in (1, 5))
            assert (1 in dgrams) == (event % 3 == 2)
            observed.append(envelope.dgrams[0].timestamp())
            identities.extend(ref.owner.batch_id for ref in dgrams.values())

    for base in (1000, 2000):
        packet = mixed_packet(n_events=11, fast_size=10, slow_size=1024**2,
                              interval=3, missing_fast=(1, 5), timestamp_base=base)
        check(m._process_batch({}, {0: (packet, [])}, {}))
    check(m.finish())
    assert observed == list(range(1000, 1011)) + list(range(2000, 2011))
    assert set(identities) == {1, 2}
    assert m.gpu_reader.io_stats()['requested_bytes'] == 2 * (90 + 3 * 1024**2)
    assert_closed(m, io)


def test_controller_minimum_event_budget_rejects_before_io(io, monkeypatch, mixed_packet):
    from psana.gpu.gpu_budget import GpuMemoryPressureError
    m = build(io, monkeypatch)
    # Raw groups fit exactly, but their parser metadata and execution do not.
    m._admission_capacity = 1024**2 + 120
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2, interval=3)
    with pytest.raises(GpuMemoryPressureError):
        list(m._process_batch({}, {0: (packet, [])}, {}))
    assert not io.futures and not io.calls
    m.close()
    assert_closed(m, io)


def test_controller_retained_view_blocks_next_batch_and_close_until_released(io, monkeypatch, mixed_packet):
    from psana.gpu.gpu_budget import GpuMemoryPressureError
    m = build(io, monkeypatch)
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2, interval=3)
    loop = m._process_batch({}, {0: (packet, [])}, {})
    first = next(loop)
    owner = first.gpu_state._event_dgrams[0].owner
    child = first.gpu_state._input_lease.acquire_view()
    try:
        list(loop)
        assert not owner.released
        with pytest.raises(GpuMemoryPressureError, match='stream credit'):
            list(m._process_batch({}, {0: (packet, [])}, {}))
        with pytest.raises(RuntimeError, match='live consumers'):
            m.close()
        assert not m._closed and not owner.released
        assert any(m.gpu_reader._input_holds.values())
        assert m._gpu_budget.committed() > 0
    finally:
        child.wait_until_safe_to_reuse()
        loop.close()
        m.close()
    assert_closed(m, io)


def test_controller_transitions_drain_consumers_before_dispatch(io, monkeypatch, mixed_packet):
    from psana.psexp import TransitionId
    from psana.gpu import gpu_events
    from test_gpu_input_group import Token as Completion
    m = build(io, monkeypatch)
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2, interval=3)
    tokens, delivered, transitions = [], [], []

    def consume(envelopes):
        for envelope in envelopes:
            delivered.append(envelope.dgrams[0].timestamp())
            token = Completion(False)
            child = envelope.gpu_state._input_lease.acquire_view()
            child.register_consumer_done(token)
            child.wait_until_safe_to_reuse()
            tokens.append(token)

    def dispatch(dgrams):
        assert not m.event_pool.active_count
        assert all(t.ready for t in tokens)
        transitions.append(dgrams[0].service())

    m.run._handle_transition = dispatch
    monkeypatch.setattr(gpu_events, '_iter_step_events', lambda packet, configs: iter(packet))
    consume(m._process_batch({}, {0: (packet, [])}, {}))
    for service in (TransitionId.BeginStep, TransitionId.EndRun):
        dg = NS(timestamp=lambda: 3000, service=lambda: service)
        consume(m._process_batch({}, {}, {0: ([(service, [dg])], [])}))
    assert delivered == list(range(1000, 1012))
    assert transitions == [TransitionId.BeginStep, TransitionId.EndRun]
    assert m._done
    m.close()
    assert_closed(m, io)


def test_transition_completion_failure_preserves_owner_and_retries(io, monkeypatch, mixed_packet):
    from psana.psexp import TransitionId
    from psana.gpu import gpu_events
    from test_gpu_input_group import Token as Completion
    m = build(io, monkeypatch)
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2, interval=3)
    delivered = list(m._process_batch({}, {0: (packet, [])}, {}))
    # Register on the final delivery while the transition flushes the pool.
    class WaitFailure(Completion):
        @property
        def done(self):
            return self.ready  # readiness queries succeed; only draining fails

    token = WaitFailure(False)
    windows, dispatched = [], []
    m.run._handle_transition = lambda dgrams: dispatched.append(dgrams)
    monkeypatch.setattr(gpu_events, '_iter_step_events', lambda packet, configs: iter(packet))
    dg = NS(timestamp=lambda: 3000, service=lambda: TransitionId.BeginStep)
    steps = {0: ([(TransitionId.BeginStep, [dg])], [])}
    with pytest.raises(RuntimeError, match='injected wait failure'):
        for envelope in m._handle_steps(steps):
            delivered.append(envelope)
            if envelope.gpu_state._event_dgrams.batch_event_index == 11:
                windows.extend(envelope.gpu_state._event_dgrams.input_windows)
                child = envelope.gpu_state._input_lease.acquire_view()
                child.register_consumer_done(token)
                child.wait_until_safe_to_reuse()
                token.fail = True
    assert not dispatched and windows and all(not w.released for w in windows)
    assert m._group_inputs.live_keys and not m._closed
    token.fail = False
    assert list(m._handle_steps(steps)) == []
    assert len(dispatched) == 1 and all(w.released for w in windows)
    m.close()
    assert len(delivered) == 12
    assert_closed(m, io)
