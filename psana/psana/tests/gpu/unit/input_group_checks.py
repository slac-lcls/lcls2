"""Test-only consistency checks for input-group pool bookkeeping."""


def assert_pool_invariants(pool):
    assert pool._busy_slots == {group.slot for group in pool._groups.values()}, 'busy slots'
    assert set(pool._candidates) <= set(pool._groups), 'candidate keys'
    assert all(pool._groups[key] is state for key, state in pool._candidates.items()), 'candidate identity'
