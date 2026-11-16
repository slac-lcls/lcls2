"""Fixed six counterbalanced blocks at each length, alternating length order."""
from collections import Counter
from itertools import permutations
import random

CASES = ('A', 'B-off', 'B-on')
LENGTHS = (1000, 10000)
SEED = 20261027


def schedule(seed=SEED):
    rng = random.Random(seed)
    orders = {}
    for length in LENGTHS:
        orders[length] = list(permutations(CASES))
        rng.shuffle(orders[length])
    return [dict(events=n, repetition=rep, order=orders[n][rep-1])
            for rep in range(1, 7)
            for n in (LENGTHS if rep % 2 else LENGTHS[::-1])]


def validate(plan):
    assert len(plan) == 12
    for rep in range(1, 7):
        blocks = [b for b in plan if b['repetition'] == rep]
        assert [b['events'] for b in blocks] == list(LENGTHS if rep % 2 else LENGTHS[::-1])
    for n in LENGTHS:
        blocks = [b for b in plan if b['events'] == n]
        assert Counter(tuple(b['order']) for b in blocks) == Counter(permutations(CASES))
        for position in range(3):
            assert Counter(b['order'][position] for b in blocks) == Counter({c:2 for c in CASES})
    return plan
