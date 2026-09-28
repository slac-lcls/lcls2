"""Matched ordering and focused cold-cache coverage remain explicit."""
import importlib.util
from pathlib import Path


def load_runner(monkeypatch):
    scripts = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(scripts/'feespec_bulk_benchmark'))
    spec = importlib.util.spec_from_file_location('stage_regression_runner', scripts/'stage1_regression/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_focused_pairs_and_cold_spot(monkeypatch):
    runner = load_runner(monkeypatch)
    cases = list(runner.timed_cases(((1, 1), (1, 2), (1, 4)), ('warm', 'cold'),
                                   ('on',), ('stage1b', 'stage2'), 3, (4,)))
    assert len(cases) == 24
    assert sum(c[3] == 'warm' for c in cases) == 18
    assert {c[1] for c in cases if c[3] == 'cold'} == {4}
    for left, right in zip(cases[::2], cases[1::2]):
        assert left[:-1] == right[:-1]
        assert (left[-1], right[-1]) == (('stage1b', 'stage2') if left[4] % 2 else ('stage2', 'stage1b'))
    assert [c[1] for c in cases if c[4] == 2][0] == 4


def test_original_full_matrix_is_unchanged(monkeypatch):
    runner = load_runner(monkeypatch)
    cases = list(runner.timed_cases(tuple((1, b) for b in (1, 2, 3, 4)), ('cold', 'warm'),
                                   ('off', 'on'), ('stage1', 'stage1b'), 2))
    assert len(cases) == len(set(cases)) == 64
    assert cases[:32] == [(1, b, mode, cache, 1, version)
                         for b in (1, 2, 3, 4) for cache in ('cold', 'warm')
                         for mode in ('off', 'on') for version in ('stage1', 'stage1b')]
    assert cases[32:] == [(1, b, mode, cache, 2, version)
                         for b in (4, 3, 2, 1) for cache in ('warm', 'cold')
                         for mode in ('on', 'off') for version in ('stage1b', 'stage1')]


def test_controls_and_ab_are_interleaved_and_order_balanced(monkeypatch):
    runner = load_runner(monkeypatch)
    variants = ('control_a', 'control_b', 'stage1b', 'stage2')
    cases = list(runner.timed_cases(((1, 1),), ('warm',), ('on',), variants, 6))
    assert len(cases) == len(set(cases)) == 24
    for rep in range(1, 7):
        order = [c[-1] for c in cases if c[4] == rep]
        assert order == list(variants if rep % 2 else reversed(variants))
    assert sum(c[-1] == 'control_a' for c in cases) == 6
    # First sample of the A/B pair is A in odd rounds and B in even rounds.
    ab = [c[-1] for c in cases if c[-1] in ('stage1b', 'stage2')]
    assert ab[::2].count('stage1b') == ab[::2].count('stage2') == 3
    aa = [c[-1] for c in cases if c[-1].startswith('control_')]
    assert aa[::2].count('control_a') == aa[::2].count('control_b') == 3


def test_stage3_matrix_balances_pairs_and_limits_controls(monkeypatch):
    runner = load_runner(monkeypatch)
    variants = ('control_a', 'control_b', 'stage2', 'stage3')
    cases = list(runner.timed_cases(((1,1),(1,2),(1,4)), ('warm','cold'),
                                   ('on',), variants, 4, (4,), (1,)))
    assert len(cases) == len(set(cases)) == 40
    assert len([c for c in cases if c[3] == 'cold']) == 8
    assert {(c[1],c[3]) for c in cases if c[-1].startswith('control_')} == {(1,'warm')}
    for bds, cache in ((1,'warm'),(2,'warm'),(4,'warm'),(4,'cold')):
        pairs = [c[-1] for c in cases if (c[1],c[3]) == (bds,cache)
                 and c[-1] in ('stage2','stage3')]
        assert pairs[::2].count('stage2') == pairs[::2].count('stage3') == 2
    assert sum(len(runner.case_variants(variants,b,'warm',(1,))) for b in (1,2,4)) == 8
