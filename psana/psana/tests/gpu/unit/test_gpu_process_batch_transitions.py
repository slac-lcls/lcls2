"""Transition delivery through the REAL `_process_batch` (issue #153).

Why this file exists, separately from `test_gpu_transition_order.py`: those
tests drive `_handle_steps` / `_yield_ready` / `_flush_event_pool` directly
and emulate `_process_batch`'s ordering in the test body. That is fine for
the queue mechanics, but it cannot cover a fix that LIVES in
`_process_batch` -- the test would be exercising its own copy of the fix.

Demonstrated: deleting the CPU-only branch's drain from `_process_batch`
leaves all 313 cases in the other two files passing. These three fail.

Each test names the production line it pins, so a reviewer can delete that
line and watch exactly one test fail:

  * the CPU-only branch's `_flush_event_pool()` when work is in flight --
    and that it is skipped for a transition-only packet;
  * `_drain_transitions()` in `_process_batch`'s terminal branch;
  * `_drain_transitions()` in `finish()`.
"""
from types import SimpleNamespace as NS

import pytest

from psana.event import EventEnvelope
from psana.gpu import gpu_events
from psana.psexp import TransitionId as T


class _Dgram:
    """A dgram carrying only what the delivery path reads from it."""

    def __init__(self, service, ts):
        self._service, self._ts = service, ts

    def service(self):
        return self._service

    def timestamp(self):
        return self._ts

    def env(self):
        return self._service << 24


class _Pool:
    """A pool whose flush() yields whatever submissions are in flight."""

    def __init__(self, harness):
        self._h = harness

    @property
    def active_count(self):
        return len(self._h.in_flight)

    def flush(self):
        while self._h.in_flight:
            yield self._h.ready(self._h.in_flight.pop(0))


class _Harness:
    def __init__(self):
        m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
        m.event_pool = _Pool(self)
        m.run = NS(runnum=1, _handle_transition=lambda dgrams: None)
        m.configs = [None]
        m.dm = None
        m.max_retries = 0
        m.use_smds = []
        m._first_batch_logged = True
        m._n_events = 0
        m._done = False
        m._closed = False
        m._group_inputs = None
        m._pending_gpu_read = None
        m.gpu_reader = None
        m.dsparms = NS(max_events=0, gpu_bulk_read=False)
        m.log_memory = m.log_high_water = lambda *a: None
        m._attach_gpu = lambda envelope, *a, **k: envelope
        m._trim_gpu_caches = lambda: None
        self.m, self.in_flight = m, []

    def ready(self, stamps):
        return NS(event_envelopes=[EventEnvelope([_Dgram(T.L1Accept, ts)])
                                   for ts in stamps],
                  gpu_results_by_ts={}, leases_by_ts={}, pending_d2h_by_ts={},
                  cached_cpu_results_by_ts={})


def _fake_event_manager(monkeypatch, timestamps):
    """Make EventManager yield one L1 envelope per timestamp.

    _process_batch builds its CPU envelopes from EventManager, so a fake is
    the smallest thing that lets the real method run without xtc files.
    """
    class _EM:
        exit_id = 0

        def __init__(self, *a, **k):
            pass

        def __iter__(self):
            for ts in timestamps:
                yield EventEnvelope([_Dgram(T.L1Accept, ts)])

    monkeypatch.setattr(gpu_events, 'EventManager', _EM)


def _stream(envelopes):
    return [(e.dgrams[0].service(), e.dgrams[0].timestamp())
            for e in envelopes]


def test_a_cpu_only_packet_drains_in_flight_work_before_its_events(monkeypatch):
    """Pins the `_flush_event_pool()` in _process_batch's CPU-only branch.

    A packet with no GPU work takes the `else` branch, whose events belong to
    no submission and so advance no sequence number. With an earlier GPU
    submission still in flight, the packet's own leading BeginStep carries
    that submission's seq -- so without the drain it comes due only AFTER
    these events, which it should precede. That is GPU-148-01 in the
    direction this work set out to fix.

    Delete the `if ... active_count: yield from self._flush_event_pool()`
    lines from the CPU-only branch and this test fails; the queue-mechanics
    tests in test_gpu_transition_order.py do not notice, because they
    perform that flush themselves.
    """
    h = _Harness()
    m = h.m
    # An earlier GPU packet's submission, left in flight and unretired.
    m._submit_seq = 1
    h.in_flight.append([100, 200])

    # This packet: a leading BeginStep, then two CPU-only events.
    transitions = [(T.BeginStep, [_Dgram(T.BeginStep, 250)])]
    monkeypatch.setattr(gpu_events, '_iter_step_events',
                        lambda b, c: list(transitions))
    _fake_event_manager(monkeypatch, [300, 400])

    delivered = list(m._process_batch(
        {0: (b'smd', None)}, {}, {0: (b'step', [])}))

    assert _stream(delivered) == [
        (T.L1Accept, 100), (T.L1Accept, 200),   # the earlier submission
        (T.BeginStep, 250),                     # before the events it opens
        (T.L1Accept, 300), (T.L1Accept, 400),
    ], 'the CPU-only branch did not drain in-flight work first'


def test_a_transition_only_packet_does_not_drain_in_flight_work(monkeypatch):
    """Pins the `event_envelopes and` half of that same condition.

    A transition-only packet -- a BeginStep the EB broke on right after a
    full batch -- also takes the CPU-only branch, with NOTHING to order. The
    drain above exists to put CPU-only events behind in-flight GPU events;
    with no such events it only reinstates the step-boundary drain that
    StepScope removes (issue #153). So the pool must be left alone and the
    transition left queued: it goes out after the in-flight submission
    retires and before anything later, which the retire + drain below shows.

    Delete `event_envelopes and` from the condition and this fails: the
    packet drains the in-flight submission itself.
    """
    h = _Harness()
    m = h.m
    m._submit_seq = 1
    h.in_flight.append([100, 200])

    transitions = [(T.BeginStep, [_Dgram(T.BeginStep, 250)])]
    monkeypatch.setattr(gpu_events, '_iter_step_events',
                        lambda b, c: list(transitions))
    _fake_event_manager(monkeypatch, [])          # no CPU-only events at all

    delivered = list(m._process_batch(
        {0: (b'smd', None)}, {}, {0: (b'step', [])}))

    assert delivered == [], 'a transition-only packet must deliver nothing itself'
    assert h.in_flight == [[100, 200]], 'it must not drain the pool'
    assert [e[2] for e in m._queued_transitions] == [T.BeginStep], \
        'the BeginStep must stay queued'

    # The in-flight submission retires on its own schedule; then the stream
    # ends. The BeginStep comes out after those events, never before.
    delivered += list(m._yield_ready(h.ready(h.in_flight.pop(0))))
    delivered += list(m._drain_transitions())
    assert _stream(delivered) == [
        (T.L1Accept, 100), (T.L1Accept, 200), (T.BeginStep, 250)]


def test_end_of_stream_delivers_a_trailing_endrun_after_its_events(monkeypatch):
    """Pins `_drain_transitions()` in _process_batch's terminal branch.

    _flush_event_pool deliberately releases nothing of its own -- it also runs
    mid-packet on memory pressure -- so the terminal branch must drain
    explicitly. Without it the EndRun is never delivered at all.
    """
    h = _Harness()
    m = h.m
    transitions = [(T.EndRun, [_Dgram(T.EndRun, 900)])]
    monkeypatch.setattr(gpu_events, '_iter_step_events',
                        lambda b, c: list(transitions))
    _fake_event_manager(monkeypatch, [100, 200])

    delivered = list(m._process_batch(
        {0: (b'smd', None)}, {}, {0: (b'step', [])}))

    assert _stream(delivered) == [
        (T.L1Accept, 100), (T.L1Accept, 200), (T.EndRun, 900)]
    assert m._done, 'EndRun must terminate the stream'


def test_finish_delivers_a_transition_left_queued(monkeypatch):
    """Pins `_drain_transitions()` in finish().

    finish() is the other end-of-stream path -- reached when the consumer
    stops before EndRun -- and has the same obligation: no further submission
    can come, so whatever is queued behind one that never happened is due.
    """
    h = _Harness()
    m = h.m
    # The transition must be queued BEHIND the in-flight submission, i.e.
    # seq == the number already submitted. _handle_steps stamps an entry with
    # the current _submit_seq, so submit first and queue second -- the order
    # a trailing BeginStep actually arrives in.
    #
    # Getting this backwards makes the test vacuous: an entry with seq=0
    # precedes the submission, so _yield_ready's own tail releases it during
    # the flush and finish() has nothing left to drain. Verified by deleting
    # the drain -- with seq=0 the test still passed.
    m._submit_seq = 1
    h.in_flight.append([100, 200])
    transitions = [(T.BeginStep, [_Dgram(T.BeginStep, 250)])]
    monkeypatch.setattr(gpu_events, '_iter_step_events',
                        lambda b, c: list(transitions))
    list(m._handle_steps({0: (b'step', [])}))
    assert [e[0] for e in m._queued_transitions] == [1], (
        'precondition: the BeginStep sits behind the in-flight submission')

    m.close = lambda: None          # finish() closes in its finally
    delivered = list(m.finish())

    assert _stream(delivered) == [
        (T.L1Accept, 100), (T.L1Accept, 200), (T.BeginStep, 250)], (
        'finish() did not drain the queued transition')
    assert not m._queued_transitions, 'nothing may be left queued'
