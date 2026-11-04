"""Step scope for one GPU packet's events (issue #153, GPU-148-01).

`_process_batch` applies a packet's transitions before submitting its events,
so a trailing BeginStep's `step_generation += 1` lands on events that belong
to the step before it. The scope is positional: a BeginStep that *precedes*
the events opens their step (prepended history), one that *follows* them does
not (a trailing BeginStep, which the EB breaks the batch on).

Position is decided by timestamp, not by record order. The decoder skips
all-empty records (#156), so there is no record cursor to walk, and GPU-only
event positions come from GPUBAT1 regardless. This mirrors
GpuFileEpochs.resolve, which already merges controls and descriptors in
timestamp order and validates that ordering.

**One scope per packet, which is narrower than #153's Design 1.** Design 1
splits subbatches at step boundaries; this assigns a single scope to all of a
packet's events and raises if a BeginStep ever falls between two of them.
That is sufficient for the packet shapes the event builder actually produces
-- its fast GPU path breaks right after appending a BeginStep
(eventbuilder.pyx), and catch-up history is prepended at the front (node.py)
-- so a mid-packet BeginStep cannot occur. Mid-packet subbatch splitting does
not exist here; do not assume it does.
"""
from bisect import bisect_left
from dataclasses import dataclass

from psana.psexp import TransitionId


@dataclass(frozen=True)
class StepScope:
    """Which step a batch of events was recorded in.

    `index` counts distinct BeginSteps applied in this run, 0 before the
    first. Counting by timestamp rather than per record seen is what makes it
    identical on every BD: smd0 and the EB prepend missing history to each
    recipient, so the same BeginStep arrives repeatedly and must not advance
    the index twice.
    """
    run: int
    index: int
    begin_ts: int


@dataclass(frozen=True)
class PacketPlan:
    """What one packet's transitions mean for its events.

    scope            StepScope for every event in the packet, or None when the
                     packet carries no events (transition-only).
    transitions      (service, dgrams, timestamp, n_events_before) in
                     timestamp order. n_events_before is how many of this
                     packet's GPUBAT1 events precede the transition, so
                     delivery yields it after exactly that many events:
                     0 for leading history, len(events) for a trailing
                     BeginStep. A count rather than a before/after flag, so a
                     mid-packet Enable or EndStep is placed too.
    next_generation  Generation after this packet's transitions, for the next.
    begin_ts         Timestamp of the latest BeginStep applied, for the next.
    """
    scope: 'StepScope | None'
    transitions: tuple
    next_generation: int
    begin_ts: int


def _transition_timestamp(dgrams):
    stamps = {int(dg.timestamp()) for dg in dgrams if dg is not None}
    if len(stamps) != 1:
        raise ValueError('GPU transition must have one aligned timestamp')
    return stamps.pop()


def plan_packet(gpu_view, transitions, *, run, generation, begin_ts):
    """Resolve the step scope for a packet's events.

    `transitions` is (service, dgrams) in record order, already filtered to
    non-L1 records as `gpu_events` does for the bulk path. `gpu_view` supplies
    the GPUBAT1 event rows.
    """
    ordered = []
    for service, dgrams in transitions:
        ordered.append((service, dgrams, _transition_timestamp(dgrams)))
    if any(a[2] > b[2] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('GPU transition history is not timestamp ordered')

    event_stamps = [int(row['timestamp']) for row in gpu_view.events]
    # The EB emits GPUBAT1 rows in timestamp order, and everything below
    # relies on it: the single-scope check compares only the endpoints, and
    # n_events_before bisects. Unsorted rows would slip an event past both --
    # events (2000, 100, 3000) around a BeginStep(900) have matching
    # endpoints, so the middle event would silently take the wrong step.
    if any(a >= b for a, b in zip(event_stamps, event_stamps[1:])):
        raise ValueError(
            'GPU events are not in strictly increasing timestamp order')
    collisions = set(event_stamps) & {ts for _, _, ts in ordered}
    if collisions:
        raise ValueError(
            f'GPU event and transition have the same timestamp: '
            f'{sorted(collisions)[0]}')

    # A BeginStep advances the generation for events recorded after it. Applied
    # by timestamp so prepended history that this BD already consumed does not
    # advance it a second time.
    next_generation, next_begin_ts = generation, begin_ts
    for service, _, ts in ordered:
        if service == TransitionId.BeginStep and ts > next_begin_ts:
            next_generation += 1
            next_begin_ts = ts

    # Where each transition falls among the events, so delivery can yield it
    # after exactly that many. Labelling the events correctly is only half of
    # GPU-148-01; the other half is not applying a trailing BeginStep to the
    # CPU before the events it follows have been delivered.
    placed = tuple((service, dgrams, ts, bisect_left(event_stamps, ts))
                   for service, dgrams, ts in ordered)

    if not event_stamps:
        return PacketPlan(None, placed, next_generation, next_begin_ts)

    def scope_at(event_ts):
        index, opened = generation, begin_ts
        for service, _, ts in ordered:
            if service == TransitionId.BeginStep and ts > opened and ts < event_ts:
                index += 1
                opened = ts
        return StepScope(run=run, index=index, begin_ts=opened)

    scope = scope_at(event_stamps[0])
    # One scope for the whole packet is a precondition of using a single value
    # per submission, not a convenience. The EB breaks its GPU batch on
    # BeginStep and prepends history at the front, so a mid-packet BeginStep
    # cannot occur today; if it ever does, the events after it would silently
    # carry the wrong step's constants. Raise instead.
    last = scope_at(event_stamps[-1])
    if last != scope:
        raise ValueError(
            f'packet spans step scopes {scope.index} and {last.index}: '
            f'a BeginStep falls between its events')

    return PacketPlan(scope, placed, next_generation, next_begin_ts)
