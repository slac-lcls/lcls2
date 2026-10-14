"""Acceptance must reject incorrect aggregates and silent GPU mis-pinning."""
from copy import deepcopy
import hashlib
import struct

import pytest

from contract import KEY_POINTS, matrix, validate_result


def fixture():
    records = [dict(rank=i, is_bd=i >= 2, timestamps=[], checks=[], affinity=[0, 1], loop_s=2.)
               for i in range(6)]
    for i, r in enumerate(records[2:]):
        r.update(timestamps=[i+1], physical_gpu=i % 2, peers=[2],
            device=dict(bus=f'bus{i%2}', workers=8, task_bytes=1 << 20,
                        gds_available=False, total_bytes=100),
            counts=dict(bytes=10, requests=5, cpu_bd_reads=0,
                        peak_owned_and_held=40, budget_limit=50))
    reference = dict(payload_bytes=40,
        timestamp_sha256=hashlib.sha256(struct.pack('<4Q', 1, 2, 3, 4)).hexdigest())
    return records, reference


def test_balanced_matrix_covers_both_rounds_modes_and_cache_states():
    rows = matrix()
    assert len(rows) == len(set(rows)) == 32
    assert rows[0] == (1, 1, 'off', 'cold', 1)
    assert rows[16] == (4, 8, 'on', 'warm', 2)
    for g, b in KEY_POINTS:
        assert sum((x[0],x[1]) == (g,b) for x in rows) == 8


def test_unordered_bd_completion_preserves_global_timestamp_reference():
    records, ref = fixture()
    records[2]['timestamps'], records[5]['timestamps'] = [4], [1]
    result = validate_result(records, ref, {}, events=4, ngpus=2, check_pixels=False)
    assert result['events'] == 4 and result['events_per_s'] == 2


@pytest.mark.parametrize('bad', ['duplicate', 'wrong_timestamp', 'bytes', 'requests',
    'bus_alias', 'pin', 'peers', 'budget', 'charge', 'cpu_read', 'affinity', 'pixels'])
def test_invalid_sample_is_rejected(bad):
    records, ref = fixture()
    row = records[2]
    if bad == 'duplicate': row['timestamps'] = [2]
    elif bad == 'wrong_timestamp': row['timestamps'] = [9]
    elif bad == 'bytes': row['counts']['bytes'] = 9
    elif bad == 'requests': row['counts']['requests'] = 4
    elif bad == 'bus_alias':
        for r in records[2:]: r['device']['bus'] = 'same'
    elif bad == 'pin': row['physical_gpu'] = 1
    elif bad == 'peers': row['peers'] = [1]
    elif bad == 'budget': row['counts']['budget_limit'] = 100
    elif bad == 'charge': row['counts']['peak_owned_and_held'] = 51
    elif bad == 'cpu_read': row['counts']['cpu_bd_reads'] = 1
    elif bad == 'affinity': row['affinity'] = [1]
    else: row['checks'] = [dict(timestamp=1)]
    with pytest.raises(AssertionError):
        validate_result(records, ref, {}, events=4, ngpus=2, check_pixels=False)


def test_pixels_require_complete_exact_reference():
    records, ref = fixture()
    expected = {1: dict(timestamp=1, raw='raw-hash', calib='calib-hash')}
    with pytest.raises(AssertionError):
        validate_result(records, ref, expected, events=4, ngpus=2, check_pixels=True)
    records[2]['checks'] = [deepcopy(expected[1])]
    validate_result(records, ref, expected, events=4, ngpus=2, check_pixels=True)
    records[2]['checks'][0]['calib'] = 'wrong'
    with pytest.raises(AssertionError):
        validate_result(records, ref, expected, events=4, ngpus=2, check_pixels=True)
