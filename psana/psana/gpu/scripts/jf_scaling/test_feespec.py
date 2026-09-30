"""Mixed-stream acceptance must survive redistribution and reject corruption."""
from copy import deepcopy

import pytest

from contract import FEESPEC_POINTS, matrix
from feespec import coalesced_requests, validate_feespec


def test_mixed_matrix():
    rows = matrix(FEESPEC_POINTS)
    assert len(rows) == len(set(rows)) == 24
    assert {r[:2] for r in rows} == {(1, 1), (1, 2), (1, 4)}
    assert rows[12] == (1, 4, 'on', 'warm', 2)


def test_request_reference_respects_gaps_fences_target_and_batches():
    rows = [dict(offset=i*8, size=8, fence=0) for i in range(5)]
    assert coalesced_requests(rows, batch_size=3, target=32) == 2
    assert coalesced_requests(rows, batch_size=5, target=16) == 3
    rows[2]['fence'] = 1
    assert coalesced_requests(rows, batch_size=5, target=100) == 3
    rows[4]['offset'] += 8
    assert coalesced_requests(rows, batch_size=5, target=100) == 4


@pytest.mark.parametrize('bad', [None, 'sum', 'timestamp', 'array', 'missing'])
def test_timestamp_matched_feespec_checks(bad):
    arrays = [dict(timestamp=t, digest=str(t)) for t in (1, 2, 3)]
    ref = dict(timestamps=[1, 2, 3], sums=[10, 20, 30], arrays=arrays)
    rows = [dict(is_bd=True, timestamps=[2], feespec_sums=[20], feespec_arrays=[deepcopy(arrays[1])]),
            dict(is_bd=True, timestamps=[1, 3], feespec_sums=[10, 30],
                 feespec_arrays=[deepcopy(arrays[0]), deepcopy(arrays[2])])]
    if bad == 'sum': rows[0]['feespec_sums'][0] = 21
    if bad == 'timestamp': rows[0]['timestamps'][0] = 1
    if bad == 'array': rows[0]['feespec_arrays'][0]['digest'] = 'bad'
    if bad == 'missing': rows[0]['feespec_sums'] = []
    if bad:
        with pytest.raises(AssertionError):
            validate_feespec(rows, ref, True)
    else:
        assert validate_feespec(rows, ref, True)['feespec_arrays'] == 3
        for row in rows:
            row['feespec_arrays'] = []
        assert validate_feespec(rows, ref, False)['feespec_arrays'] == 0
