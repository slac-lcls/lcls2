"""Slot policy equivalence and fresh snapshots across ownership transitions."""
import random
from types import SimpleNamespace as NS

import pytest

from psana.gpu.gpu_input_group import InputGroupPool


def pool_for(capacities, busy=(), small=()):
    reader = NS(bulk_read=True, _n_slots=len(capacities), _pending=[],
                _input_holds={}, _slot_bufs=[None if c is None else NS(nbytes=c)
                                           for c in capacities])
    pool = InputGroupPool(reader)
    pool._groups = {(0, i): NS(slot=i, window=None) for i in busy}
    pool._small = {s: (0, s) for s in small}
    return pool


def reference(capacities, busy, small, groups):
    """Original selection policy, independent of the indexed implementation."""
    available = set(range(len(capacities))) - set(busy)
    credits = set(small)
    result = []
    for group in groups:
        if not available or (group.small and group.stream_id in credits):
            return None
        fitting = [i for i in available if capacities[i] is not None
                   and capacities[i] >= group.size]
        slot = min(fitting, key=lambda i: (capacities[i], i)) if fitting else min(available)
        available.remove(slot)
        result.append(slot)
        if group.small:
            credits.add(group.stream_id)
    return tuple(result)


@pytest.mark.parametrize('capacities,busy,sizes,expected', [
    ([8, 4, 4, None], (), [3, 3, 7, 9], (1, 2, 0, 3)),
    ([2, None, 8], (), [9, 1, 0], (0, 2, 1)),  # remove undersized fallback
    ([None, 0, 0], (), [0, 0, 0], (1, 2, 0)),  # allocated zero fits first
    ([4, 4, 8], (0,), [4, 5], (1, 2)),
    ([4, None], (0,), [5, 1], None),
    ([4], (0,), [], ()),
])
def test_exact_slot_policy(capacities, busy, sizes, expected):
    pool = pool_for(capacities, busy)
    groups = [NS(size=n, small=False, stream_id=i) for i, n in enumerate(sizes)]
    assert pool.plan_slots(iter(groups)) == expected


def test_generated_slot_decisions_match_original_policy_without_consuming_credits():
    rng = random.Random(250925)
    for _ in range(2000):
        capacities = [rng.choice([None, 0, 1, 4, 4, 8, 16, 64])
                      for _ in range(rng.randrange(1, 40))]
        busy = [i for i in range(len(capacities)) if rng.randrange(5) == 0]
        small = [s for s in range(4) if rng.randrange(6) == 0]
        groups = [NS(size=rng.choice([0, 1, 3, 4, 9, 64, 65]),
                     small=rng.randrange(4) == 0, stream_id=rng.randrange(4))
                  for _ in range(rng.randrange(45))]
        pool = pool_for(capacities, busy, small)
        old_groups, old_credits = dict(pool._groups), dict(pool._small)
        assert pool.plan_slots(groups) == reference(capacities, busy, small, groups)
        assert pool._groups == old_groups and pool._small == old_credits
        assert [None if b is None else b.nbytes for b in pool.reader._slot_bufs] == capacities


def test_new_plan_observes_growth_trim_and_completion(monkeypatch):
    pool = pool_for([4, 8, None], busy=(0, 1), small=(7,))
    group = NS(size=3, small=True, stream_id=7)
    assert pool.plan_slots([group]) is None

    def complete_later_group():
        # Earlier slot 0 is still busy; later slot 1 returns its stream credit.
        pool._groups.pop((0, 1), None)
        pool._small.pop(7, None)
    monkeypatch.setattr(pool, 'poll', complete_later_group)
    assert pool.plan_slots([group]) == (1,)
    pool.reader._slot_bufs[2] = NS(nbytes=4)
    assert pool.plan_slots([group]) == (2,)
    pool.reader._slot_bufs[2].nbytes = 16
    assert pool.plan_slots([group]) == (1,)
    pool.reader._slot_bufs[1] = None
    assert pool.plan_slots([group]) == (2,)
    pool.reader._slot_bufs[2] = None
    assert pool.plan_slots([group]) == (1,)
    assert (0, 0) in pool._groups
