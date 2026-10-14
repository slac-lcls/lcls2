"""Guard native trace acceptance and steady-window counting."""
import sqlite3

import pytest

from analyze_profiles import MARKER, read_profile


def profile(path):
    with sqlite3.connect(str(path)) as db:
        db.executescript('''
            create table StringIds (id integer, value text);
            create table NVTX_EVENTS (start integer, end integer, text text, textId integer);
            create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, demangledName integer);
            create table CUPTI_ACTIVITY_KIND_RUNTIME (start integer, end integer, nameId integer, globalTid integer);
            create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer, end integer, copyKind integer, bytes integer);
            create table ENUM_CUDA_MEMCPY_OPER (id integer, label text);
            insert into StringIds values (1, 'gather'), (2, 'cuStreamSynchronize');
            insert into ENUM_CUDA_MEMCPY_OPER values (1, 'Host-to-Device');
            insert into CUPTI_ACTIVITY_KIND_KERNEL values (5, 12, 1), (10, 14, 1), (19, 22, 1), (20, 24, 1);
            insert into CUPTI_ACTIVITY_KIND_RUNTIME values (11, 13, 2, 999);
            insert into CUPTI_ACTIVITY_KIND_MEMCPY values (11, 15, 1, 32);
        ''')
        db.execute('insert into StringIds values (3, ?)', (MARKER,))
        # Registered strings are accepted as well as inline NVTX text.
        db.execute('insert into NVTX_EVENTS values (10, 20, NULL, 3)')


def test_counts_use_activity_start_and_include_worker_threads(tmp_path):
    path = tmp_path/'trace.sqlite'
    profile(path)
    result = read_profile(path)
    assert result['steady_window_ns'] == [10, 20]
    full, steady = (result['scopes'][s] for s in ('full', 'steady'))
    assert full['kernels']['gather']['count'] == 4
    assert steady['kernels']['gather'] == dict(count=2, duration_ns=7)
    assert steady['apis']['runtime/cuStreamSynchronize'] == dict(count=1, duration_ns=2)
    assert steady['copies']['Host-to-Device'] == dict(count=1, bytes=32, duration_ns=4)


@pytest.mark.parametrize('change, message', [
    ('drop table CUPTI_ACTIVITY_KIND_KERNEL', 'no CUDA kernel records'),
    ('delete from CUPTI_ACTIVITY_KIND_KERNEL', 'no full kernels'),
    ('update NVTX_EVENTS set end=NULL', 'expected one completed'),
    ('insert into NVTX_EVENTS select * from NVTX_EVENTS', 'expected one completed'),
])
def test_rejects_incomplete_or_ambiguous_traces(tmp_path, change, message):
    path = tmp_path/'trace.sqlite'
    profile(path)
    with sqlite3.connect(str(path)) as db:
        db.execute(change)
    with pytest.raises(ValueError, match=message):
        read_profile(path)
