"""Test-only consistency checks for input-group pool bookkeeping."""


def assert_pool_invariants(pool):
    expected_slots = {group.slot for group in pool._groups.values()}
    assert pool._busy_slots == expected_slots, (
        f'busy slots {pool._busy_slots} != live group slots {expected_slots}')
    orphan_keys = set(pool._candidates) - set(pool._groups)
    assert not orphan_keys, f'candidate keys outside live groups: {orphan_keys}'
    for key, state in pool._candidates.items():
        assert pool._groups[key] is state, f'candidate identity mismatch for {key}'
