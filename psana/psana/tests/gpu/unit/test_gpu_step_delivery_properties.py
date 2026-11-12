"""Randomized property test for transition delivery order (issue #153).

Generates whole random runs -- 1-4 steps, empty steps, SlowUpdates, EndRun --
splits them into EB-shaped packets, and drives the real `_handle_steps`,
`_reposition_transitions`, `_yield_ready`, `_flush_event_pool`,
`_release_transitions` and `_drain_transitions` against a fake pool, so no GPU
is needed. Each run must satisfy four properties:

  1. every record is delivered exactly once, in timestamp order;
  2. each L1's step scope equals the step it was recorded in;
  3. `_retire_seq <= _submit_seq` after every packet;
  4. nothing raises.

Three conditions the targeted cases in `test_gpu_transition_order.py` cover
one at a time, and which this mixes at random because their interaction is
where the defects lived:

  * **memory-pressure flushes mid-packet.** `_retire_issue_and_yield` calls
    `_flush_event_pool()` when `_issue_gpu_read` raises
    `GpuMemoryPressureError`, while later subbatches of the same packet are
    unsubmitted. The flush used to force-release the whole queue, putting a
    trailing BeginStep or EndRun ahead of those events -- and since
    `Run.events()` and `_steps_impl` both return on EndRun, losing them.
  * **CPU-only packets.** Their events belong to no submission, so nothing
    advances `_retire_seq` and a transition behind an earlier in-flight
    submission came due only after events it should precede.
  * **packets with no transitions**, which left a stale `_new_transitions`
    window and raised a spurious count-mismatch `RuntimeError` -- under MPI,
    `comm.Abort(1)` on valid data.

Scoreboard when this was written, 2000 seeds: 773 correct before the fix
(misordered x482, `retire_seq > submit_seq` x550, stale-window raise x195),
2000 after.

NOTE: this *emulates* `_process_batch`'s ordering around those methods rather
than calling `_process_batch` itself, so it is a complement to the tests that
drive the real thing, not a replacement. If the order of operations in
`_process_batch` changes, `_packet` below has to change with it.
"""
import collections
import random
from types import SimpleNamespace as NS

import pytest

from psana.event import EventEnvelope
from psana.gpu import gpu_events
from psana.gpu.gpu_plan import plan_packet
from psana.psexp import TransitionId as T

L1, BS, ES, ER, EN, DI, SU = (T.L1Accept, T.BeginStep, T.EndStep, T.EndRun,
                              T.Enable, T.Disable, T.SlowUpdate)

N_SEEDS = 300          # full 2000-seed sweep takes ~40s; see test_many_seeds
P_PRESSURE = 0.25
P_CPU_ONLY = 0.20


class _Dgram:
    def __init__(self, service, ts):
        self._service, self._ts = service, ts

    def service(self):
        return self._service

    def timestamp(self):
        return self._ts

    def env(self):
        return self._service << 24


def _random_run(rng):
    """A plausible record stream: steps of events, with SlowUpdates."""
    records, ts = [], 0

    def add(service):
        nonlocal ts
        ts += rng.randint(1, 5)
        records.append((service, ts))

    for _ in range(rng.randint(1, 4)):
        add(BS)
        add(EN)
        for _ in range(rng.choice([0, 1, 3, 7, 20])):   # 0 = an empty step
            add(L1)
            if rng.random() < 0.1:
                add(SU)
        add(DI)
        add(ES)
    add(ER)
    return records


def _eb_packets(records, target):
    """Split into packets the way the EB does: break after `target` events,
    and always after a BeginStep or EndRun (eventbuilder.pyx)."""
    out, current, n = [], [], 0
    for service, ts in records:
        if n == target:
            out.append(current)
            current, n = [], 0
        current.append((service, ts))
        if service == L1:
            n += 1
        elif service in (BS, ER):
            out.append(current)
            current, n = [], 0
    if current:
        out.append(current)
    return out


class _Pool:
    """A pool whose flush() yields whatever is still in flight."""

    def __init__(self, harness):
        self._h = harness

    @property
    def active_count(self):
        return len(self._h.in_flight)

    def flush(self):
        while self._h.in_flight:
            scope, stamps = self._h.in_flight.pop(0)
            self._h.scopes.update({ts: scope for ts in stamps})
            yield self._h.ready(stamps)


class _Harness:
    def __init__(self, rng):
        m = gpu_events.GpuEventManager.__new__(gpu_events.GpuEventManager)
        m.event_pool = _Pool(self)
        m.run = NS(runnum=1, _handle_transition=lambda dgrams: None)
        m.configs = [None]
        m._first_batch_logged = True
        m.log_memory = m.log_high_water = lambda *a: None
        m._attach_gpu = lambda envelope, *a, **k: envelope
        m._group_inputs = None
        self.m, self.rng = m, rng
        self.in_flight, self.out, self.scopes = [], [], {}

    def ready(self, stamps):
        return NS(event_envelopes=[EventEnvelope([_Dgram(L1, ts)])
                                   for ts in stamps],
                  gpu_results_by_ts={}, leases_by_ts={}, pending_d2h_by_ts={},
                  cached_cpu_results_by_ts={})

    def _deliver(self, envelopes):
        for envelope in envelopes:
            dg = envelope.dgrams[0]
            scope = (self.scopes.get(dg.timestamp())
                     if dg.service() == L1 else None)
            self.out.append((dg.service(), dg.timestamp(), scope))

    def _retire_oldest(self):
        scope, stamps = self.in_flight.pop(0)
        self.scopes.update({ts: scope for ts in stamps})
        self._deliver(self.m._yield_ready(self.ready(stamps)))

    def packet(self, records, cpu_only, depth):
        """Emulate _process_batch's order for one packet."""
        m, rng = self.m, self.rng
        transitions = [(s, [_Dgram(s, t)]) for s, t in records if s != L1]
        events = [t for s, t in records if s == L1]
        plan = None
        if not cpu_only:
            view = NS(events=[{'timestamp': t} for t in events])
            plan = plan_packet(view, transitions, run=1,
                               generation=getattr(m, '_step_generation', 0),
                               begin_ts=getattr(m, '_step_begin_ts', 0))
        end_run = any(s == ER for s, _ in records)
        # Hand _handle_steps the list directly, as _process_batch does.
        # Replacing gpu_events._iter_step_events at module level here left
        # the fake in place after the test, and test_core.py's decoder tests
        # then received this harness's last packet instead of their own bytes
        # whenever this file ran first.
        self._deliver(m._handle_steps(
            {0: (b'packet', [])} if transitions else {}, transitions))

        if cpu_only:
            # _process_batch's CPU-only branch: drain in flight only when
            # there are events to order, then interleave the packet's
            # transitions among them. A transition-only packet leaves its
            # transitions queued for the next retirement or the drain.
            if events and m.event_pool.active_count:
                self._deliver(m._flush_event_pool())
            for ts in events:
                self._deliver(m._release_transitions(before_ts=ts))
                self.out.append((L1, ts, None))
        else:
            n_cuts = (rng.randint(0, min(3, len(events) - 1))
                      if len(events) > 1 else 0)
            bounds = [0, *sorted(rng.sample(range(1, len(events)), n_cuts)),
                      len(events)] if events else [0, 0]
            subbatches = [NS(_parent=None, _start=a, _end=b, ts=events[a:b])
                          for a, b in zip(bounds, bounds[1:]) if b > a]
            m._reposition_transitions(plan.transitions, subbatches)
            for subbatch in subbatches:
                while self.in_flight and (len(self.in_flight) >= depth
                                          or rng.random() < 0.3):
                    self._retire_oldest()
                if rng.random() < P_PRESSURE:
                    # GpuMemoryPressureError on issue: flush mid-packet, with
                    # this subbatch and any after it still unsubmitted.
                    self._deliver(m._flush_event_pool())
                m._submit_seq = getattr(m, '_submit_seq', 0) + 1
                self.in_flight.append((plan.scope.index, subbatch.ts))

        if end_run:
            self._deliver(m._flush_event_pool())
            self._deliver(m._drain_transitions())
        elif not m.event_pool.active_count:
            self._deliver(m._release_transitions())

        assert getattr(m, '_retire_seq', 0) <= getattr(m, '_submit_seq', 0), (
            f"_retire_seq {getattr(m, '_retire_seq', 0)} exceeds _submit_seq "
            f"{getattr(m, '_submit_seq', 0)}: it must only be advanced by a "
            f"retirement, never assigned from queue contents")


def _run_one(seed):
    """One random run. Returns None on success, else a failure description."""
    rng = random.Random(seed)
    records = _random_run(rng)
    h = _Harness(rng)
    for packet in _eb_packets(records, rng.choice([1, 2, 5, 20])):
        has_l1 = any(s == L1 for s, _ in packet)
        h.packet(packet,
                 cpu_only=has_l1 and rng.random() < P_CPU_ONLY,
                 depth=rng.choice([1, 2, 4]))

    got = [(s, t) for s, t, _ in h.out]
    if got != records:
        n_got = len([x for x in got if x[0] == L1])
        n_want = len([x for x in records if x[0] == L1])
        kind = 'lost events' if n_got < n_want else 'misordered'
        return f'{kind}: got {got[:8]}... want {records[:8]}...'

    # Property 2: each L1's scope is the step it was recorded in.
    begun = -1
    for service, ts, scope in h.out:
        if service == BS:
            begun += 1
        elif service == L1 and scope is not None and scope != begun + 1:
            return f'scope mismatch at ts {ts}: {scope} != {begun + 1}'
    return None


@pytest.mark.parametrize('seed', range(N_SEEDS))
def test_a_random_run_delivers_every_record_in_order(seed):
    failure = _run_one(seed)
    assert failure is None, f'seed {seed}: {failure}'


@pytest.mark.slow
def test_many_seeds():
    """The full sweep, as a single case so it can be deselected.

    Run with `-m slow`. 2000 seeds was 773 correct before the delivery fix
    and 2000 after, which is the evidence that the contract -- rather than
    any one call site -- was what needed repair.
    """
    failures = collections.Counter()
    for seed in range(2000):
        failure = _run_one(seed)
        if failure is not None:
            failures[failure.split(':')[0]] += 1
    assert not failures, dict(failures)
