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
    big = {'sum': np.zeros(10, np.float64), 'count': np.array([1])}       # 88 bytes
    r.add('cube', RI, 900, 'a', big, stream=0)
    r.add('cube', RI, 950, 'a', big, stream=0)                             # 176 > 100: flush 900
    assert [(x.begin_ts, x.complete, x.seq) for x in out] == [(900, False, 0)]
    r.add('cube', RI, 900, 'a', big, stream=1)                             # late stream for 900
    r.scope_done(0, RI, 900); r.scope_done(1, RI, 900)
    assert [(x.begin_ts, x.complete, x.seq, x.streams) for x in out][-1] == (900, True, 1, (1,)), \
        'the final record for a key continues the seq the partial started'

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
