"""Run the controller and reader with injected CUDA completion and tiny I/O."""
from types import SimpleNamespace as NS

import pytest

from test_gpu_bulk_read import io
from group_test_support import manager
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


def test_controller_yields_transitions_after_the_events_they_follow(io, monkeypatch, mixed_packet):
    """Transitions are delivered downstream, in record order.

    Since #153 PR B step 4 the manager no longer calls
    run._handle_transition() -- it yields the transition as an envelope with
    gpu_state None and the consumer dispatches it exactly once, after the
    events that preceded it. That ordering is the second half of GPU-148-01.
    """
    from psana.psexp import TransitionId
    from psana.gpu import gpu_events
    from test_gpu_input_group import Token as Completion
    m = build(io, monkeypatch)
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2, interval=3)
    tokens, order = [], []

    def consume(envelopes):
        for envelope in envelopes:
            if envelope.gpu_state is None:
                # A transition. What is asserted is its POSITION in the
                # stream, below -- not token readiness, which only becomes
                # true after this consumer releases each event, including the
                # one delivered in this same iteration.
                order.append(('transition', envelope.dgrams[0].service()))
                continue
            order.append(('event', envelope.dgrams[0].timestamp()))
            token = Completion(False)
            child = envelope.gpu_state._input_lease.acquire_view()
            child.register_consumer_done(token)
            child.wait_until_safe_to_reuse()
            tokens.append(token)

    dispatched = []
    m.run._handle_transition = lambda dgrams: dispatched.append(dgrams)
    monkeypatch.setattr(gpu_events, '_iter_step_events', lambda packet, configs: iter(packet))
    consume(m._process_batch({}, {0: (packet, [])}, {}))
    for service in (TransitionId.BeginStep, TransitionId.EndRun):
        # Bind `service` per iteration: a bare `lambda: service` closes over
        # the loop variable, so both dgrams would report the last one.
        dg = NS(timestamp=lambda: 3000, service=lambda s=service: s)
        consume(m._process_batch({}, {}, {0: ([(service, [dg])], [])}))

    # Every event, in order, then the two transitions -- never interleaved
    # the other way round. The BeginStep is released when the EndRun packet
    # flushes the pool: it stays queued until the submissions it follows have
    # retired, which is the ordering guarantee under test.
    assert [ts for kind, ts in order if kind == 'event'] == list(range(1000, 1012))
    assert [svc for kind, svc in order if kind == 'transition'] == [
        TransitionId.BeginStep, TransitionId.EndRun]
    assert order[-2:] == [('transition', TransitionId.BeginStep),
                          ('transition', TransitionId.EndRun)]
    assert dispatched == [], 'the manager must not dispatch; the consumer does'
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
    # EndRun, not BeginStep: since #153 PR B step 4 only EndRun drains the
    # pool, and this test is about a FAILED drain preserving its owners.
    dg = NS(timestamp=lambda: 3000, service=lambda: TransitionId.EndRun)
    steps = {0: ([(TransitionId.EndRun, [dg])], [])}
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
    # The retry succeeds and the transition is yielded, not dispatched.
    retried = list(m._handle_steps(steps))
    assert [e.gpu_state for e in retried] == [None]
    assert not dispatched, 'the manager must not dispatch; the consumer does'
    assert all(w.released for w in windows)
    m.close()
    assert len(delivered) == 12
    assert_closed(m, io)


def test_hybrid_cpu_envelopes_keep_their_payload(io, mixed_packet, monkeypatch):
    from psana.event import EventEnvelope
    from psana.gpu import gpu_events
    m = build(io, monkeypatch)
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
    yielded.extend(m.finish())
    assert [e.dgrams[0] for e in yielded] == cpu
    assert all(e.gpu_state is not None for e in yielded)
    m.close()


def test_packet_plan_scopes_events_before_a_trailing_beginstep(io, monkeypatch,
                                                               mixed_packet):
    """Events recorded before a trailing BeginStep keep the previous scope.

    GPU-148-01, against the real manager rather than a fixture: _handle_steps
    applies the packet's transitions and advances the generation, so a plan
    resolved afterwards would put these events in the step the BeginStep
    opens. _process_batch therefore has to plan BEFORE it dispatches.

    Timestamps matter here. mixed_packet numbers its events from 1000, and the
    trailing BeginStep is at 3000 -- after them -- which is what makes it
    "trailing" to the planner.
    """
    from psana.psexp import TransitionId
    from psana.gpu import gpu_events
    m = build(io, monkeypatch)
    m._step_generation, m._step_begin_ts = 0, 0
    packet = mixed_packet(n_events=6, fast_size=10, slow_size=1024**2,
                          interval=3)
    begin = NS(timestamp=lambda: 3000,
               service=lambda: TransitionId.BeginStep)
    captured = []
    original = gpu_events.GpuEventManager._plan_packets

    def capture(self, gpu_views, transitions):
        plans = original(self, gpu_views, transitions)
        captured.extend(plans.values())
        return plans

    monkeypatch.setattr(gpu_events.GpuEventManager, '_plan_packets', capture)
    monkeypatch.setattr(gpu_events, '_iter_step_events',
                        lambda packet, configs: iter(packet))

    list(m._process_batch({}, {0: (packet, [])},
                          {0: ([(TransitionId.BeginStep, [begin])], [])}))

    assert len(captured) == 1
    assert captured[0].scope.index == 0, (
        'events before a trailing BeginStep belong to the step already open')
    assert captured[0].next_generation == 1, 'the BeginStep still advances'
    # And the manager's own state moved on, so the NEXT packet's events are
    # planned in scope 1.
    assert m._step_generation == 1
    assert m._step_begin_ts == 3000
    list(m.finish())
    m.close()


def test_a_replayed_beginstep_does_not_advance_the_generation_twice(io,
                                                                   monkeypatch,
                                                                   mixed_packet):
    """Prepended history repeats a BeginStep this BD already applied.

    smd0 and the EB prepend missing history to every recipient, so the same
    BeginStep arrives again. _dispatch_transition used to increment per
    record seen, which made the generation diverge between BDs.
    """
    from psana.psexp import TransitionId
    from psana.gpu import gpu_events
    m = build(io, monkeypatch)
    m._step_generation, m._step_begin_ts = 1, 3000
    replayed = NS(timestamp=lambda: 3000,
                  service=lambda: TransitionId.BeginStep)
    monkeypatch.setattr(gpu_events, '_iter_step_events',
                        lambda packet, configs: iter(packet))

    list(m._process_batch({}, {},
                          {0: ([(TransitionId.BeginStep, [replayed])], [])}))

    assert m._step_generation == 1, 'a replayed BeginStep must not re-advance'
    assert m._step_begin_ts == 3000
    m.close()


def test_submission_uses_the_subbatch_scope_not_the_advanced_generation(
        io, monkeypatch, mixed_packet):
    """The whole point of GPU-148-01: submit the scope, not the manager state.

    The manager's _step_generation has already advanced to 1 by submission
    time, because _handle_steps applied the trailing BeginStep. Every
    subbatch of this packet must still be submitted with step_generation 0 --
    the step its events were actually recorded in.
    """
    from psana.psexp import TransitionId
    from psana.gpu import gpu_events
    m = build(io, monkeypatch)
    m._step_generation, m._step_begin_ts = 0, 0
    m.run = NS(runnum=77, _handle_transition=lambda _: None)

    packet = mixed_packet(n_events=6, fast_size=10, slow_size=1024**2,
                          interval=3)
    begin = NS(timestamp=lambda: 3000,
               service=lambda: TransitionId.BeginStep)
    submitted = []
    # Record the scope each submission would carry, then let the real
    # _submit_gpu run: it owns the read reservation, and skipping it leaves a
    # pre-issued read outstanding so the next subbatch fails.
    #
    # The scope is read here rather than from _task_submission() because that
    # returns {} unless a task is configured, and a stub task is not viable --
    # _event_memory walks task.inputs through metadata_bytes(). What matters
    # is which scope reaches submission; _task_submission's use of it is
    # covered directly below.
    original_submit = gpu_events.GpuEventManager._submit_gpu

    def capture(self, subbatch, gpu_read, envelopes):
        submitted.append(getattr(subbatch, 'scope', None))
        return original_submit(self, subbatch, gpu_read, envelopes)

    monkeypatch.setattr(gpu_events.GpuEventManager, '_submit_gpu', capture)
    monkeypatch.setattr(gpu_events, '_iter_step_events',
                        lambda packet, configs: iter(packet))

    list(m._process_batch({}, {0: (packet, [])},
                          {0: ([(TransitionId.BeginStep, [begin])], [])}))

    assert submitted, 'expected at least one submission'
    assert m._step_generation == 1, 'the manager did advance'
    assert all(s is not None for s in submitted)
    assert [s.index for s in submitted] == [0] * len(submitted), (
        'events before a trailing BeginStep must submit with the old scope')
    assert all(s.run == 77 and s.begin_ts == 0 for s in submitted)

    # And _task_submission turns that scope into the submitted
    # step_generation, rather than re-reading the manager's advanced one.
    m._gpu_task = NS(inputs=(), calibconst=())
    m.gpu_detector_bindings, m._task_constants = {}, None
    assert m._task_submission(submitted[0])['step_generation'] == 0
    assert m._task_submission(submitted[0])['step_scope'] is submitted[0]
    assert m._task_submission(None)['step_generation'] == 1, (
        'with no scope it falls back to the manager')
    m._gpu_task = None
    list(m.finish())
    m.close()


def test_every_subbatch_of_a_packet_carries_the_same_scope(io, monkeypatch,
                                                           mixed_packet):
    """A packet splits into several subbatches on bytes, never on steps.

    plan_packet raises if a packet's events straddle a BeginStep, so all of
    its subbatches share one scope. Pinned here because _split_subbatches
    attaches that scope to each, and a future byte-split change must not
    silently leave some without one.
    """
    from psana.psexp import TransitionId
    from psana.gpu import gpu_events
    m = build(io, monkeypatch)
    m._step_generation, m._step_begin_ts = 2, 500
    packet = mixed_packet(n_events=12, fast_size=10, slow_size=1024**2,
                          interval=3)
    monkeypatch.setattr(gpu_events, '_iter_step_events',
                        lambda packet, configs: iter(packet))

    captured = []
    original = gpu_events.GpuEventManager._split_subbatches

    def capture(self, gpu_view, scope=None):
        subbatches = original(self, gpu_view, scope)
        captured.extend(subbatches)
        return subbatches

    monkeypatch.setattr(gpu_events.GpuEventManager, '_split_subbatches',
                        capture)
    list(m._process_batch({}, {0: (packet, [])}, {}))

    assert len(captured) > 1, 'expected a byte split, or this proves nothing'
    assert all(sb.scope is not None for sb in captured)
    assert len({(sb.scope.index, sb.scope.begin_ts) for sb in captured}) == 1
    assert captured[0].scope.index == 2, 'no transitions, so the scope stands'
    list(m.finish())
    m.close()
