"""The per-BD aggregate reducer (PR #270 phase 1, step C). Host-only."""
import numpy as np
import pytest

from psana.gpu.gpu_aggregate import AggregateReducer, normalise_aggregate_outputs

SPEC = normalise_aggregate_outputs({'cube': {'sum': 'sum', 'count': 'sum'}})
RI = ('xpp', 77, 123, 0)


def _reducer(n=2, cap=1 << 30):
    out = []
    return AggregateReducer(SPEC, n, out.append, cap), out


def test_merges_by_key_and_delivers_once_every_stream_finished_the_scope():
    r, out = _reducer()
    r.add('cube', RI, 900, 'a', {'sum': np.array([1.0, 2.0]), 'count': np.array([3])}, stream=0)
    r.scope_done(0, RI, 900)
    assert out == [], 'stream 1 has not finished scope 900'
    r.add('cube', RI, 900, 'a', {'sum': np.array([0.5, 0.5]), 'count': np.array([1])}, stream=1)
    r.add('cube', RI, 900, 'b', {'sum': np.array([9.0, 9.0]), 'count': np.array([9])}, stream=1)
    r.scope_done(1, RI, 900)
    assert [(x.key, x.complete, x.seq, x.streams) for x in out] == [
        ('a', True, 0, (0, 1)), ('b', True, 0, (1,))]
    assert out[0].fields['sum'].tolist() == [1.5, 2.5] and out[0].fields['count'].tolist() == [4]
    assert (out[0].name, out[0].run_instance, out[0].begin_ts) == ('cube', RI, 900)
    assert r.pending_bytes == 0

    # A scope nobody emitted for completes silently: completion is a fact,
    # records are data. Run-level records wait for every stream's end_run.
    r.scope_done(0, RI, 950); r.scope_done(1, RI, 950)
    r.add('cube', RI, None, 'total', {'sum': np.array([1.0]), 'count': np.array([1])}, stream=0)
    r.run_done(0)
    assert len(out) == 2
    r.run_done(1)
    assert out[-1].begin_ts is None and out[-1].complete and out[-1].streams == (0,)


def test_host_cap_flushes_the_oldest_scope_as_partials_and_validates_records():
    r, out = _reducer(cap=100)
    big = {'sum': np.ones(10, np.float64), 'count': np.array([1])}        # 88 bytes
    r.add('cube', RI, 900, 'a', big, stream=0)
    r.add('cube', RI, 950, 'a', big, stream=0)                             # 176 > 100: flush 900
    r.add('cube', RI, 900, 'a', big, stream=1)                             # 176 again: flush 950
    assert [(x.begin_ts, x.complete, x.final, x.seq) for x in out] == [
        (900, False, False, 0), (950, False, False, 0)]
    r.scope_done(0, RI, 900); r.scope_done(1, RI, 900)
    rec = out[-1]
    assert (rec.begin_ts, rec.complete, rec.final, rec.seq, rec.streams) == (900, False, True, 1, (1,)), \
        'the last record for a cap-flushed key is final but not complete: it is not the whole step'
    # A key the cap delivered entirely still gets a final marker, zero-valued
    # in the declared layout, so the sink can close the key; bookkeeping for a
    # finished scope is dropped.
    r.scope_done(0, RI, 950); r.scope_done(1, RI, 950)
    rec = out[-1]
    assert (rec.begin_ts, rec.complete, rec.final, rec.seq, rec.streams) == (950, False, True, 1, ())
    assert rec.fields['sum'].tolist() == [0.0] * 10 and rec.fields['count'].tolist() == [0]
    assert not r._seq and not r._done

    # Validation, on a reducer the cap has not touched.
    r, _ = _reducer()
    r.add('cube', RI, 950, 'a', big, stream=0)
    with pytest.raises(KeyError, match='not a declared'):
        r.add('other', RI, 900, 'a', big, stream=0)
    with pytest.raises(ValueError, match='expects fields'):
        r.add('cube', RI, 900, 'a', {'sum': np.zeros(1)}, stream=0)
    with pytest.raises(ValueError, match='shape'):
        r.add('cube', RI, 950, 'a', {'sum': np.zeros(3), 'count': np.array([1])}, stream=1)
    with pytest.raises(ValueError, match='unsupported op'):
        normalise_aggregate_outputs({'cube': {'sum': 'mean'}})


def test_a_key_delivered_early_keeps_its_layout():
    """Copilot #5: once the host cap has delivered part of a key, a later
    record for that key with another shape or dtype is refused, instead of
    leaving the sink partials it cannot add."""
    import pytest
    from psana.gpu.gpu_aggregate import AggregateReducer
    from psana.gpu.gpu_task_worker import RunInstance
    ri = RunInstance('xpp', 77, 1, 0)
    out = []
    red = AggregateReducer([('o', {'n': 'sum'})], n_streams=2, sink=out.append, host_bytes=1)
    red.add('o', ri, 900, 'a', {'n': np.ones(4)}, 0)
    red.add('o', ri, 950, 'b', {'n': np.ones(4)}, 0)      # over the cap: 900 delivered early
    assert [(r.begin_ts, r.final) for r in out] == [(900, False)]
    with pytest.raises(ValueError, match='already delivered for the same key'):
        red.add('o', ri, 900, 'a', {'n': np.ones(5)}, 1)
    red.add('o', ri, 900, 'a', {'n': np.ones(4)}, 1)      # same layout: accepted


# --------------------------------------------------------------------------
# PR #344 review, R6: validate every field before merging.
# --------------------------------------------------------------------------

def test_a_narrowing_dtype_is_rejected_rather_than_overflowing_silently():
    """The pre-spill path checked shape but not dtype, and np.add's default
    same-kind casting allows narrowing within one kind: an int64 contribution
    merged into an int16 accumulator wrapped instead of being refused."""
    r, out = _reducer()
    r.add('cube', RI, 900, 'a',
          {'sum': np.array([30000], dtype=np.int16), 'count': np.array([1])}, stream=0)
    with pytest.raises(ValueError, match='dtype'):
        r.add('cube', RI, 900, 'a',
              {'sum': np.array([10000], dtype=np.int64), 'count': np.array([1])}, stream=1)
    # The accumulator is untouched, so nothing overflowed.
    r.scope_done(0, RI, 900)
    r.scope_done(1, RI, 900)
    assert [int(rec.fields['sum'][0]) for rec in out] == [30000]


def test_a_mismatch_in_a_later_field_leaves_every_field_unchanged():
    """Validation used to be interleaved with the merge, so a first field
    that matched was already modified when a second field turned out not to,
    and a later delivery could expose that half-updated record."""
    r, out = _reducer()
    r.add('cube', RI, 900, 'a',
          {'sum': np.zeros(10), 'count': np.array([1])}, stream=0)
    with pytest.raises(ValueError, match='shape'):
        # count matches and would have been added first; sum does not.
        r.add('cube', RI, 900, 'a',
              {'sum': np.zeros(3), 'count': np.array([5])}, stream=1)
    r.scope_done(0, RI, 900)
    r.scope_done(1, RI, 900)
    assert len(out) == 1
    assert int(out[0].fields['count'][0]) == 1, 'count was not left half-merged'
    assert out[0].fields['sum'].shape == (10,)


def test_a_missing_field_is_rejected_against_the_recorded_layout():
    """The declared-set check already covers a wrong field set against the
    spec; this covers a set that matches the spec but not what the key has."""
    r, out = _reducer(n=1)
    r.add('cube', RI, 900, 'a', {'sum': np.zeros(4), 'count': np.array([1])}, stream=0)
    with pytest.raises(ValueError, match='expects fields'):
        r.add('cube', RI, 900, 'a', {'sum': np.zeros(4)}, stream=0)
