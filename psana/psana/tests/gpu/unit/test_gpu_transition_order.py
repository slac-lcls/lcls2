"""Delivery order of a packet's transitions relative to its events (issue #153).

Drives the real _handle_steps / _reposition_transitions / _yield_ready with a
fake pool, so the queue is tested without a GPU. Each case is a list of
packets; the expected stream is every record in timestamp order.

Two defects these cover, both found in review of the per-packet step scope:

  1. The idle-pool release at the end of _handle_steps ran before
     _process_batch had submitted the packet's events, so every transition
     went out ahead of them. The pool is idle on the final packet every time
     -- EndRun drains it first, and the EB appends EndRun to the batch
     holding the run's last events -- and both Run.events() and _steps_impl
     return on EndRun, so those events were lost entirely.
  2. _reposition_transitions counted whole subbatches, so a transition
     falling inside one was released ahead of that subbatch's earlier
     events. The EB does not break a batch on SlowUpdate, Disable or Enable,
     and a SlowUpdate arrives about once a second, so this was routine.
"""
from types import SimpleNamespace as NS

import pytest

from psana.event import EventEnvelope
from psana.gpu import gpu_events
from psana.gpu.gpu_plan import plan_packet
from psana.psexp import TransitionId as T


class _Dgram:
    def __init__(self, service, ts):
        self._service, self._ts = service, ts

    def service(self):
        return self._service

    def timestamp(self):
        return self._ts

    def env(self):
        return self._service << 24


def _manager():
    m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
    m.event_pool = NS(active_count=0)
    m.run = NS(runnum=1, _handle_transition=lambda dgrams: None)
    m.configs = [None]
    m._first_batch_logged = True
    m.log_memory = m.log_high_water = lambda *a: None
    m._flush_event_pool = lambda: iter(())    # EndRun drain: pool already idle
    m._attach_gpu = lambda envelope, *a, **k: envelope
    return m


def _submit(m, monkeypatch, records, split=()):
    """One packet through _handle_steps and _reposition_transitions; returns
    its subbatches' timestamps and anything _handle_steps yielded."""
    trans = [(s, [_Dgram(s, t)]) for s, t in records if s != T.L1Accept]
    events = [t for s, t in records if s == T.L1Accept]
    monkeypatch.setattr(gpu_events, '_iter_step_events', lambda b, c: list(trans))
    view = NS(events=[{'timestamp': t} for t in events])
    plan = plan_packet(view, trans, run=1, generation=0, begin_ts=0)
    early = list(m._handle_steps({0: (b'packet', [])}))
    bounds = [0, *split, len(events)]
    subbatches = [NS(_parent=view, _start=a, _end=b, ts=events[a:b])
                  for a, b in zip(bounds, bounds[1:]) if b > a]
    m._reposition_transitions(plan.transitions, subbatches)
    m._submit_seq = getattr(m, '_submit_seq', 0) + len(subbatches)
    m.event_pool.active_count += len(subbatches)
    return subbatches, early


def _retire(m, subbatch):
    m.event_pool.active_count -= 1
    ready = NS(event_envelopes=[EventEnvelope([_Dgram(T.L1Accept, t)])
                                for t in subbatch.ts],
               gpu_results_by_ts={}, leases_by_ts={}, pending_d2h_by_ts={},
               cached_cpu_results_by_ts={})
    return list(m._yield_ready(ready))


def _end_of_stream(m):
    """The real end-of-stream drain, as _process_batch and finish() call it."""
    return list(m._drain_transitions())


def _stream(envelopes):
    return [(e.dgrams[0].service(), e.dgrams[0].timestamp()) for e in envelopes]


L1, BS, ES, ER, DI, EN, SU = (T.L1Accept, T.BeginStep, T.EndStep, T.EndRun,
                              T.Disable, T.Enable, T.SlowUpdate)

CASES = {
    # The EndRun drain leaves the pool idle; EndRun must still follow the
    # packet's events, or Run.events()/steps() return before delivering them.
    'final_packet_on_idle_pool': [
        ([(L1, 100), (L1, 200), (DI, 250), (ES, 260), (ER, 300)], ())],
    # GPU-148-01 itself, on the first packet of a run (pool idle).
    'trailing_beginstep_on_idle_pool': [
        ([(L1, 100), (L1, 200), (ES, 260), (BS, 900)], ())],
    'trailing_beginstep_then_next_packet': [
        ([(L1, 100), (L1, 200), (ES, 260), (BS, 900)], ()),
        ([(EN, 950), (L1, 1000), (L1, 1100)], ())],
    # The EB does not break a batch on SlowUpdate/Disable/Enable.
    'slowupdate_inside_first_subbatch': [
        ([(L1, 10), (L1, 20)], ()),
        ([(L1, 100), (SU, 150), (L1, 200), (L1, 300), (L1, 400)], (3,))],
    'slowupdate_inside_second_subbatch': [
        ([(L1, 100), (L1, 200), (L1, 300), (SU, 350), (L1, 400), (L1, 500)],
         (2,))],
    'transition_only_packet_between_event_packets': [
        ([(L1, 100), (L1, 200)], ()),
        ([(DI, 250), (ES, 260), (BS, 900), (EN, 950)], ()),
        ([(L1, 1000)], ())],
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_transitions_are_delivered_in_timestamp_order(name, monkeypatch):
    m = _manager()
    delivered, submitted = [], []
    for records, split in CASES[name]:
        subbatches, early = _submit(m, monkeypatch, records, split)
        delivered += early
        submitted += subbatches
    for subbatch in submitted:
        delivered += _retire(m, subbatch)
    delivered += _end_of_stream(m)

    expected = sorted(((s, t) for records, _ in CASES[name] for s, t in records),
                      key=lambda record: record[1])
    assert _stream(delivered) == expected


def test_a_decode_mismatch_raises_rather_than_misordering(monkeypatch):
    """_reposition_transitions used to return on a count mismatch, leaving the
    "due now" positions -- which is GPU-148-01 again, silently. It must fail."""
    m = _manager()
    records = [(L1, 100), (L1, 200), (BS, 900)]
    subbatches, _ = _submit(m, monkeypatch, records)
    # A second call with a plan that disagrees with what _handle_steps queued.
    m._new_transitions = slice(0, len(getattr(m, '_queued_transitions', [])))
    with pytest.raises(RuntimeError, match='for delivery'):
        m._reposition_transitions([], subbatches)


# --- memory-pressure flushes, CPU-only packets, stale windows --------------
#
# _retire_issue_and_yield() calls _flush_event_pool() when _issue_gpu_read
# raises GpuMemoryPressureError -- mid-packet, with later subbatches of the
# same packet not yet submitted. The flush used to force-release the whole
# queue by raising _retire_seq to the queue maximum, which put a trailing
# BeginStep or EndRun ahead of those events; since Run.events() and
# _steps_impl both return on EndRun, the events were lost outright.
#
# These drive the REAL _flush_event_pool against a pool whose flush() yields
# the submissions still in flight, so the force-release cannot be simulated
# away by a stub.


class _FlushablePool:
    def __init__(self, harness):
        self._h = harness

    @property
    def active_count(self):
        return len(self._h.in_flight)

    def flush(self):
        while self._h.in_flight:
            yield self._h.ready(self._h.in_flight.pop(0))


class _Harness:
    """A manager whose pool can actually be flushed, for the pressure path."""

    def __init__(self):
        m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
        m.event_pool = _FlushablePool(self)
        m.run = NS(runnum=1, _handle_transition=lambda dgrams: None)
        m.configs = [None]
        m._first_batch_logged = True
        m.log_memory = m.log_high_water = lambda *a: None
        m._attach_gpu = lambda envelope, *a, **k: envelope
        m._group_inputs = None
        self.m, self.in_flight = m, []

    def ready(self, stamps):
        return NS(event_envelopes=[EventEnvelope([_Dgram(T.L1Accept, t)])
                                   for t in stamps],
                  gpu_results_by_ts={}, leases_by_ts={}, pending_d2h_by_ts={},
                  cached_cpu_results_by_ts={})

    def plan(self, monkeypatch, records, split=()):
        trans = [(s, [_Dgram(s, t)]) for s, t in records if s != T.L1Accept]
        events = [t for s, t in records if s == T.L1Accept]
        monkeypatch.setattr(gpu_events, '_iter_step_events',
                            lambda b, c: list(trans))
        view = NS(events=[{'timestamp': t} for t in events])
        plan = plan_packet(view, trans, run=1,
                           generation=getattr(self.m, '_step_generation', 0),
                           begin_ts=getattr(self.m, '_step_begin_ts', 0))
        early = list(self.m._handle_steps({0: (b'packet', [])} if trans else {}))
        bounds = [0, *split, len(events)]
        subbatches = [NS(_parent=view, _start=a, _end=b, ts=events[a:b])
                      for a, b in zip(bounds, bounds[1:]) if b > a]
        if trans:
            self.m._reposition_transitions(plan.transitions, subbatches)
        return subbatches, early

    def submit(self, subbatch):
        self.m._submit_seq = getattr(self.m, '_submit_seq', 0) + 1
        self.in_flight.append(subbatch.ts)

    def retire_oldest(self):
        return list(self.m._yield_ready(self.ready(self.in_flight.pop(0))))


@pytest.mark.parametrize('trailing', [T.BeginStep, T.EndRun])
def test_a_pressure_flush_keeps_a_trailing_transition_behind_its_events(
        trailing, monkeypatch):
    """The data-loss case. Pressure hits between two subbatches of one packet,
    so the second is not yet submitted when the flush runs."""
    h = _Harness()
    records = [(T.L1Accept, 100), (T.L1Accept, 200), (T.L1Accept, 300),
               (T.L1Accept, 400), (trailing, 900)]
    subbatches, delivered = h.plan(monkeypatch, records, split=(2,))

    h.submit(subbatches[0])
    delivered += list(h.m._flush_event_pool())      # GpuMemoryPressureError
    h.submit(subbatches[1])
    delivered += h.retire_oldest()
    delivered += list(h.m._drain_transitions())

    assert _stream(delivered) == sorted(records, key=lambda r: r[1])
    # The consumer's contract: Run.events() and _steps_impl return on EndRun.
    if trailing == T.EndRun:
        before_endrun = []
        for service, ts in _stream(delivered):
            if service == T.EndRun:
                break
            if service == T.L1Accept:
                before_endrun.append(ts)
        assert before_endrun == [100, 200, 300, 400], (
            'EndRun was delivered early, so the consumer returns before '
            'these events and they are lost')


def test_a_pressure_flush_keeps_a_mid_subbatch_transition_in_place(monkeypatch):
    """The seq == _retire_seq boundary.

    A SlowUpdate inside the SECOND subbatch carries seq equal to the retired
    count after the first subbatch retires. It is not behind retired work --
    it sits inside the next submission -- so a flush releasing under the
    normal `seq <= _retire_seq` gate still emits it too early. The flush must
    release nothing of its own.
    """
    h = _Harness()
    records = [(T.L1Accept, 100), (T.L1Accept, 200), (T.L1Accept, 300),
               (T.SlowUpdate, 350), (T.L1Accept, 400)]
    subbatches, delivered = h.plan(monkeypatch, records, split=(2,))

    h.submit(subbatches[0])
    delivered += h.retire_oldest()
    delivered += list(h.m._flush_event_pool())      # pressure, sb1 unsubmitted
    h.submit(subbatches[1])
    delivered += h.retire_oldest()
    delivered += list(h.m._drain_transitions())

    assert _stream(delivered) == sorted(records, key=lambda r: r[1])


def test_a_pressure_flush_never_inflates_retire_seq(monkeypatch):
    """_retire_seq means "submissions fully delivered", so it may only be
    advanced by a retirement. Raising it from queue contents left it above
    _submit_seq permanently, misordering every later packet in the run."""
    h = _Harness()
    for packet in range(3):
        base = 1000 * packet
        records = [(T.L1Accept, base + 100), (T.L1Accept, base + 200),
                   (T.BeginStep, base + 900)]
        subbatches, _ = h.plan(monkeypatch, records, split=(1,))
        h.submit(subbatches[0])
        list(h.m._flush_event_pool())
        h.submit(subbatches[1])
        h.retire_oldest()
        assert h.m._retire_seq <= h.m._submit_seq, (
            f'packet {packet}: _retire_seq {h.m._retire_seq} exceeds '
            f'_submit_seq {h.m._submit_seq}')


def test_a_cpu_only_packet_delivers_a_leading_transition_first(monkeypatch):
    """A CPU-only packet's events belong to no submission, so nothing in that
    branch advances _retire_seq. A leading BeginStep carrying an earlier
    in-flight submission's seq would come due only AFTER these events, which
    it should precede -- so the branch drains what is in flight first."""
    h = _Harness()
    # An earlier GPU packet, left in flight.
    earlier, _ = h.plan(monkeypatch, [(T.L1Accept, 100), (T.L1Accept, 200)])
    h.submit(earlier[0])

    # Now a CPU-only packet: a leading BeginStep, then its own events.
    records = [(T.BeginStep, 250), (T.L1Accept, 300), (T.L1Accept, 400)]
    trans = [(s, [_Dgram(s, t)]) for s, t in records if s != T.L1Accept]
    monkeypatch.setattr(gpu_events, '_iter_step_events',
                        lambda b, c: list(trans))
    delivered = list(h.m._handle_steps({0: (b'packet', [])}))
    # _process_batch's CPU-only branch: drain in-flight work, then interleave.
    if h.m.event_pool.active_count:
        delivered += list(h.m._flush_event_pool())
    for ts in (300, 400):
        delivered += list(h.m._release_transitions(before_ts=ts))
        delivered.append(EventEnvelope([_Dgram(T.L1Accept, ts)]))
    delivered += list(h.m._drain_transitions())

    assert _stream(delivered) == [
        (T.L1Accept, 100), (T.L1Accept, 200),   # the earlier packet, drained
        (T.BeginStep, 250),                     # before the events it opens
        (T.L1Accept, 300), (T.L1Accept, 400)]


def test_an_empty_step_dict_after_a_cpu_only_packet_does_not_raise(monkeypatch):
    """The stale-window crash. A packet with transitions but no gpu_views never
    repositions, so its window survived; if the NEXT packet had an empty
    step_dict, _handle_steps returned early without clearing it and the window
    was matched against the new plan -- RuntimeError on valid data, and
    comm.Abort(1) under MPI."""
    h = _Harness()
    trans = [(T.Enable, [_Dgram(T.Enable, 100)])]
    monkeypatch.setattr(gpu_events, '_iter_step_events',
                        lambda b, c: list(trans))
    # Packet A queues a transition and, having no gpu_views, never
    # repositions -- so its window is still set when the packet ends.
    list(h.m._handle_steps({0: (b'packet', [])}))
    assert h.m._new_transitions is not None

    # Packet B has an empty step_dict, so _handle_steps returns early. The
    # clear must happen BEFORE that return, or packet A's window survives.
    list(h.m._handle_steps({}))
    assert h.m._new_transitions is None, (
        "packet A's window survived into packet B")

    # Which is what made this raise: packet B's plan has no transitions, and
    # the stale window claimed one.
    h.m._reposition_transitions([], [])
