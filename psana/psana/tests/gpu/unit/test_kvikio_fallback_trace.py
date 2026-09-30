"""CPU checks for the diagnostic trace's attribution and overlap accounting."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('fallback_summary', ROOT / 'gpu/scripts/summarize_kvikio_fallback.py')
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def test_overlapping_workers_partition_wall_instead_of_summing():
    # Read 0..6, transfer 4..8: 4 read-only, 2 overlap, 2 transfer, 2 idle.
    rows = np.zeros(2, dtype=summary.DTYPE)
    rows['start'] = [0, 4]
    rows['end'] = [6, 8]
    rows['kind'] = [1, 3]
    result = summary.occupancy(rows, [dict(issued_ns=0, wait_end=10)])
    assert result == dict(POSIX_only=4e-9, POSIX_and_H2D_or_wait=2e-9,
                          H2D_or_wait_only=2e-9, neither=2e-9)


def test_time_between_batches_is_excluded():
    rows = np.zeros(2, dtype=summary.DTYPE)
    rows['start'] = [2, 102]
    rows['end'] = [4, 104]
    rows['kind'] = [1, 1]
    result = summary.occupancy(rows, [dict(issued_ns=0, wait_end=10),
                                      dict(issued_ns=100, wait_end=110)])
    assert result['POSIX_only'] == 4e-9
    assert result['neither'] == 16e-9


def test_overlapping_request_windows_count_wall_time_once():
    rows = np.zeros(1, dtype=summary.DTYPE)
    rows['start'], rows['end'], rows['kind'] = 2, 12, 1
    result = summary.occupancy(rows, [dict(issued_ns=0, wait_end=10),
                                      dict(issued_ns=5, wait_end=15)])
    assert result['POSIX_only'] == 10e-9
    assert result['neither'] == 5e-9


def test_file_concurrency_counts_files_not_workers_and_excludes_idle():
    rows = np.zeros(5, dtype=summary.DTYPE)
    rows['kind'] = 1
    # Two workers on file 1 overlap; file 2 overlaps 4..6 then continues to 8.
    # A touching read 8..10 adds no overlap; idle 10..20 is excluded.
    rows['start'] = [0, 2, 4, 8, 20]
    rows['end'] = [6, 5, 8, 10, 22]
    rows['fd'] = [1, 1, 2, 1, 2]
    result = summary.file_concurrency(rows)
    assert result['single_file_s'] == 10e-9
    assert result['posix_active_s'] == 12e-9
    assert result['single_file_percent'] == pytest.approx(100*10/12)
    assert result['wall_by_distinct_files_s'] == {'1':10e-9, '2':2e-9}


@pytest.mark.parametrize('corrupt', [None, 'missing_copy', 'wrong_bytes', 'outside_window'])
def test_triplet_audit_rejects_incomplete_or_misattributed_trace(tmp_path, corrupt):
    rows = np.zeros(3, dtype=summary.DTYPE)
    rows['start'] = [1, 3, 5]
    rows['end'] = [2, 4, 6]
    rows['kind'] = [1, 2, 3]
    rows['size'] = 16
    rows['result'] = [16, 0, 0]
    rows['batch'] = rows['tid'] = 1
    if corrupt == 'missing_copy':
        rows = rows[[0, 2]]
    elif corrupt == 'wrong_bytes':
        rows['size'][1] = 15
    elif corrupt == 'outside_window':
        rows['end'][2] = 8
    binary = tmp_path/'trace.bin'
    rows.tofile(binary)
    metadata = tmp_path/'trace.json'
    metadata.write_text(json.dumps(dict(binary=str(binary), records=len(rows), record_size=64,
        batches=[dict(batch=1, issued_ns=0, issue_end=1, wait_begin=2, wait_end=7,
                      requested_bytes=16, requests=1, ranges=[dict(size=16)])])))
    if corrupt:
        with pytest.raises(AssertionError):
            summary.audit_trace(metadata)
    else:
        result = summary.audit_trace(metadata)
        assert result['operations']['POSIX']['calls'] == 1
        assert result['read_interval_s'] == 7e-9


@pytest.mark.parametrize('corrupt', [None, 'duplicate_read', 'wrong_owner_time'])
def test_overlapping_groups_use_file_ranges_not_global_submission_tag(tmp_path, corrupt):
    rows = np.zeros(6, dtype=summary.DTYPE)
    rows['kind'] = [1, 2, 3, 1, 2, 3]
    rows['start'] = [2, 6, 8, 3, 4, 5]
    rows['end'] = [6, 8, 9, 4, 5, 6]
    rows['size'] = 16
    rows['result'] = [16, 0, 0, 16, 0, 0]
    rows['batch'] = 2  # Both workers started after the second submission.
    rows['fd'] = [11, 11, 11, 12, 12, 12]
    rows['read_id'] = [0, 0, 0, 3, 3, 3]
    rows['tid'] = [1, 1, 1, 2, 2, 2]
    if corrupt == 'duplicate_read':
        rows['fd'][3:] = 11
    elif corrupt == 'wrong_owner_time':
        rows['end'][5] = 8  # File B's own wait ended at 7.
    binary = tmp_path/'groups.bin'
    rows.tofile(binary)
    metadata = tmp_path/'groups.json'
    batches = [dict(batch=i, issued_ns=i-1, issue_end=i, wait_begin=6, wait_end=end,
                    requested_bytes=16, requests=1, ranges=[dict(file=name, offset=0, size=16)])
               for i, name, end in ((1, 'A', 10), (2, 'B', 7))]
    metadata.write_text(json.dumps(dict(binary=str(binary), records=6, record_size=64,
        attribution='file_ranges', fd_paths={'11': 'A', '12': 'B'}, batches=batches)))
    if corrupt:
        with pytest.raises(AssertionError):
            summary.audit_trace(metadata)
    else:
        result = summary.audit_trace(metadata)
        assert result['read_interval_s'] == pytest.approx(10e-9)
        assert result['read_interval_sum_s'] == pytest.approx(16e-9)
        assert result['bytes'] == 32 and result['api_requests'] == 2


def test_four_mib_tasks_count_two_chunks_for_five_mib_request(tmp_path):
    rows = np.zeros(6, dtype=summary.DTYPE)
    rows['kind'] = [1, 2, 3, 1, 2, 3]
    rows['start'] = range(1, 7)
    rows['end'] = range(2, 8)
    rows['size'] = [4 << 20]*3 + [1 << 20]*3
    rows['offset'] = [0]*3 + [4 << 20]*3
    rows['result'] = [4 << 20, 0, 0, 1 << 20, 0, 0]
    rows['read_id'] = [0]*3 + [3]*3
    rows['batch'] = rows['tid'] = 1
    binary = tmp_path/'trace.bin'
    rows.tofile(binary)
    metadata = tmp_path/'trace.json'
    data = dict(binary=str(binary), records=6, record_size=64, task_size=4 << 20,
        batches=[dict(batch=1, issued_ns=0, issue_end=1, wait_begin=1, wait_end=8,
                      requested_bytes=5 << 20, requests=1, ranges=[dict(size=5 << 20)])])
    metadata.write_text(json.dumps(data))
    assert summary.audit_trace(metadata)['operations']['POSIX']['calls'] == 2
    data['task_size'] = 1 << 20
    metadata.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        summary.audit_trace(metadata)
