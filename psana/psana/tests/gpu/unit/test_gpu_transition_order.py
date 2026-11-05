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
    queued = getattr(m, '_queued_transitions', None)
    if queued:                                  # as _flush_event_pool does
        m._retire_seq = max(m._retire_seq, max(e[0] for e in queued))
    return list(m._release_transitions())


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
