"""Step scope for a GPU packet's events (issue #153, GPU-148-01).

The defect: `GpuEventManager._process_batch` applies every transition in a
packet (`_handle_steps`) before submitting any of its events, so for
`[A, B, BeginStep(900)]` the BeginStep's `step_generation += 1` lands on A and
B -- which belong to the *previous* step.

The scope is positional, not per-packet. "The generation at packet start" is
right for a trailing BeginStep and wrong for prepended history, where the
BeginStep opens the step its events belong to:

    [A, B, BeginStep(900)]      events are scope 0  (capture before applying)
    [BeginStep(900), C, D]      events are scope 1  (capture after applying)

One rule covers both: the generation at packet start, plus each distinct
BeginStep in this packet whose timestamp precedes the first GPU event's.
Timestamps rather than record positions, because the decoder skips all-empty
records (#156) and GPU-only positions come from GPUBAT1 anyway.
"""
from types import SimpleNamespace as NS

import pytest

from psana.gpu.gpu_plan import StepScope, plan_packet
from psana.psexp import TransitionId


def dgram(timestamp, service):
    """A stand-in carrying what the planner reads: timestamp() and service."""
    return NS(timestamp=lambda ts=timestamp: ts,
              service=lambda svc=service: svc,
              env=lambda svc=service: svc << 24)


def transition(timestamp, service=TransitionId.BeginStep, n_streams=2):
    """One (service, dgrams) pair as `_iter_step_events` yields it."""
    return service, [dgram(timestamp, service)] * n_streams


def events(*timestamps):
    """A GPUBAT1 view exposing just the event timestamps the planner needs."""
    rows = [{'timestamp': ts} for ts in timestamps]
    return NS(events=rows, header=NS(n_events=len(rows)))


# ---------------------------------------------------------------------------
# The four shapes. These are the whole point of the module.
# ---------------------------------------------------------------------------

def test_events_before_a_trailing_beginstep_keep_the_previous_scope():
    """[A(100), B(200), BeginStep(900)] -> scope 0, not 1.

    This is GPU-148-01 itself. The BeginStep is the last record, so it opens
    the step for the NEXT packet's events; A and B were recorded in the step
    that was already open.
    """
    plan = plan_packet(events(100, 200), [transition(900)],
                       run=1, generation=0, begin_ts=0)

    assert plan.scope == StepScope(run=1, index=0, begin_ts=0)
    assert plan.next_generation == 1, 'the BeginStep still advances the run'


def test_events_after_leading_history_take_the_new_scope():
    """[BeginStep(900), C(1000), D(1100)] -> scope 1.

    Replayed history is prepended at the front of the packet (node.py
    build_smd_batch), so this BeginStep precedes its events and does open
    their step. The opposite answer to the case above, from the same rule.
    """
    plan = plan_packet(events(1000, 1100), [transition(900)],
                       run=1, generation=0, begin_ts=0)

    assert plan.scope == StepScope(run=1, index=1, begin_ts=900)


def test_a_packet_with_history_and_a_trailing_beginstep_uses_the_leading_one():
    """[BeginStep(900), C(1000), BeginStep(2000)] -> scope 1.

    Only the BeginStep that precedes the events counts toward their scope;
    the trailing one belongs to whatever comes next.
    """
    plan = plan_packet(events(1000), [transition(900), transition(2000)],
                       run=1, generation=0, begin_ts=0)

    assert plan.scope == StepScope(run=1, index=1, begin_ts=900)
    assert plan.next_generation == 2


def test_a_replayed_beginstep_is_not_counted_twice():
    """A BeginStep this BD already applied must not advance the index again.

    smd0 and the EB prepend missing history to every recipient, so the same
    BeginStep arrives repeatedly. Counting it by timestamp -- rather than
    incrementing per record seen -- is what makes the index identical on
    every BD and replay-safe.
    """
    plan = plan_packet(events(1000), [transition(900)],
                       run=1, generation=1, begin_ts=900)

    assert plan.scope == StepScope(run=1, index=1, begin_ts=900), 'not 2'
    assert plan.next_generation == 1


# ---------------------------------------------------------------------------
# Degenerate and mixed packets
# ---------------------------------------------------------------------------

def test_a_transition_only_packet_has_no_event_scope():
    """Nothing to submit, but the generation must still advance so the next
    packet's events are placed correctly."""
    plan = plan_packet(events(), [transition(900)],
                       run=1, generation=0, begin_ts=0)

    assert plan.scope is None
    assert plan.next_generation == 1
    assert plan.begin_ts == 900


def test_a_packet_with_no_transitions_keeps_the_current_scope():
    plan = plan_packet(events(100, 200), [],
                       run=1, generation=2, begin_ts=900)

    assert plan.scope == StepScope(run=1, index=2, begin_ts=900)
    assert plan.next_generation == 2


@pytest.mark.parametrize('service', [
    TransitionId.Enable,
    TransitionId.Disable,
    TransitionId.EndStep,
    TransitionId.SlowUpdate,
])
def test_only_beginstep_opens_a_scope(service):
    """All four do appear mid-packet -- the EB deliberately does not break on
    them (eventbuilder.pyx), and SlowUpdate arrives about once a second. None
    of them opens a step.
    """
    plan = plan_packet(events(1000), [transition(900, service)],
                       run=1, generation=0, begin_ts=0)

    assert plan.scope == StepScope(run=1, index=0, begin_ts=0)
    assert plan.next_generation == 0


def test_transitions_are_returned_in_timestamp_order_with_their_dgrams():
    """The delivery path needs them ordered, with dgrams intact, to yield each
    one after the events it follows."""
    plan = plan_packet(
        events(1000),
        [transition(900, TransitionId.Enable), transition(2000)],
        run=1, generation=0, begin_ts=0)

    assert [(svc, ts) for svc, _, ts, _ in plan.transitions] == [
        (TransitionId.Enable, 900), (TransitionId.BeginStep, 2000)]
    assert all(dgrams for _, dgrams, _, _ in plan.transitions)


# ---------------------------------------------------------------------------
# Piece 2: the invariant the single-scope rule depends on
# ---------------------------------------------------------------------------

def test_a_mid_packet_beginstep_is_rejected():
    """One scope per packet is a precondition, not a convenience.

    The EB breaks its GPU batch on BeginStep and prepends history at the
    front, so a BeginStep between two L1s cannot occur today. If it ever
    does, a single scope for the whole packet would be silently wrong for the
    events after it -- those events calibrated against the wrong step's
    constants, the exact failure #153 exists to remove. So: raise.
    """
    with pytest.raises(ValueError, match='spans step scopes'):
        plan_packet(events(100, 2000), [transition(900)],
                    run=1, generation=0, begin_ts=0)


def test_a_mid_packet_enable_is_allowed():
    """The same shape with a non-scope transition is fine and must not raise:
    only a scope change splits a packet."""
    plan = plan_packet(events(100, 2000),
                       [transition(900, TransitionId.Enable)],
                       run=1, generation=0, begin_ts=0)

    assert plan.scope == StepScope(run=1, index=0, begin_ts=0)


# ---------------------------------------------------------------------------
# Input validation, following GpuFileEpochs.resolve's existing contract
# ---------------------------------------------------------------------------

def test_an_event_and_transition_sharing_a_timestamp_is_rejected():
    """GpuFileEpochs.resolve already rejects this; the planner agrees rather
    than silently choosing a side."""
    with pytest.raises(ValueError, match='same timestamp'):
        plan_packet(events(900), [transition(900)],
                    run=1, generation=0, begin_ts=0)


def test_unordered_transition_history_is_rejected():
    with pytest.raises(ValueError, match='not timestamp ordered'):
        plan_packet(events(3000), [transition(2000), transition(900)],
                    run=1, generation=0, begin_ts=0)


def test_a_transition_with_disagreeing_timestamps_is_rejected():
    """Every present stream of one transition carries the same timestamp."""
    bad = (TransitionId.BeginStep, [dgram(900, TransitionId.BeginStep),
                                    dgram(901, TransitionId.BeginStep)])
    with pytest.raises(ValueError, match='one aligned timestamp'):
        plan_packet(events(1000), [bad], run=1, generation=0, begin_ts=0)


def test_absent_streams_do_not_hide_a_transition_timestamp():
    """A transition is yielded with None in the slots whose streams were
    absent; the timestamp comes from the present ones."""
    partial = (TransitionId.BeginStep,
               [None, dgram(900, TransitionId.BeginStep)])
    plan = plan_packet(events(1000), [partial],
                       run=1, generation=0, begin_ts=0)

    assert plan.scope == StepScope(run=1, index=1, begin_ts=900)


# ---------------------------------------------------------------------------
# Catch-up history: the one place scope_at increments more than once
# ---------------------------------------------------------------------------

def test_two_beginsteps_in_leading_history_advance_twice():
    """A BD that missed an empty step receives both BeginSteps as history.

    Both precede the events and both are newer than begin_ts, so the index
    advances by 2 and begin_ts becomes the later one. Every other test moves
    the index by at most one, so this is the only cover for the loop running
    twice.
    """
    plan = plan_packet(events(1000), [transition(900), transition(950)],
                       run=1, generation=0, begin_ts=0)

    assert plan.scope == StepScope(run=1, index=2, begin_ts=950)
    assert plan.next_generation == 2
    assert plan.begin_ts == 950


# ---------------------------------------------------------------------------
# n_events_before: where delivery must place each transition
# ---------------------------------------------------------------------------

def _placements(plan):
    return [(svc, n) for svc, _, _, n in plan.transitions]


def test_a_trailing_beginstep_is_placed_after_every_event():
    """[A(100), B(200), BeginStep(900)] -> 2, so delivery yields it after both.

    This is the half of GPU-148-01 that labelling the events does not fix:
    the CPU must not apply the BeginStep until A and B have been delivered.
    """
    plan = plan_packet(events(100, 200), [transition(900)],
                       run=1, generation=0, begin_ts=0)

    assert _placements(plan) == [(TransitionId.BeginStep, 2)]


def test_leading_history_is_placed_before_every_event():
    plan = plan_packet(events(1000), [transition(900)],
                       run=1, generation=0, begin_ts=0)

    assert _placements(plan) == [(TransitionId.BeginStep, 0)]


def test_a_mid_packet_enable_is_placed_between_its_neighbours():
    """A count, not a before/after flag: Enable does not open a scope, but it
    still has a position, and coalescing is fenced there."""
    plan = plan_packet(events(100, 2000),
                       [transition(900, TransitionId.Enable)],
                       run=1, generation=0, begin_ts=0)

    assert _placements(plan) == [(TransitionId.Enable, 1)]


def test_transition_only_packets_place_everything_at_zero():
    plan = plan_packet(events(), [transition(900)],
                       run=1, generation=0, begin_ts=0)

    assert _placements(plan) == [(TransitionId.BeginStep, 0)]


# ---------------------------------------------------------------------------
# GPUBAT1 ordering, which the endpoint check and bisect both rely on
# ---------------------------------------------------------------------------

def test_unordered_gpu_events_are_rejected():
    """Note this pair would be caught anyway, by the single-scope check --
    the endpoints straddle the BeginStep. It is the middle-event case below
    that the ordering check is actually load-bearing for."""
    with pytest.raises(ValueError, match='strictly increasing'):
        plan_packet(events(2000, 100), [transition(900)],
                    run=1, generation=0, begin_ts=0)


def test_duplicate_gpu_event_timestamps_are_rejected():
    """Strictly increasing, so a repeated timestamp is malformed too -- it
    would also make bisect_left's placement ambiguous."""
    with pytest.raises(ValueError, match='strictly increasing'):
        plan_packet(events(100, 100), [], run=1, generation=0, begin_ts=0)


def test_an_unsorted_middle_event_cannot_slip_past_the_scope_check():
    """The endpoint check alone is not enough.

    Events (2000, 100, 3000) around BeginStep(900): both endpoints are after
    the BeginStep, so they agree on index 1 and the single-scope check passes
    -- but the middle event at 100 precedes it and truly belongs to index 0.
    Before the ordering check this was accepted silently and that event took
    the wrong step's constants. Verified against the pre-fix code.
    """
    with pytest.raises(ValueError, match='strictly increasing'):
        plan_packet(events(2000, 100, 3000), [transition(900)],
                    run=1, generation=0, begin_ts=0)
