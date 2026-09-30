"""Read an Nsight SQLite export; keep GPU execution distinct from host scopes."""
import argparse
import json
from pathlib import Path
import sqlite3


def interval_union(rows):
    """Count nanoseconds covered by start-sorted intervals, without double counting."""
    first = end = None
    covered = count = 0
    for lo, hi in rows:
        assert hi >= lo
        count += 1
        if first is None:
            first, end = lo, hi
            covered = hi - lo
        elif hi > end:
            covered += hi - max(lo, end)
            end = hi
    return dict(operations=count, covered_ns=covered,
                span_ns=0 if first is None else end-first,
                gaps_ns=0 if first is None else end-first-covered)


def summarize(path):
    db = sqlite3.connect('file:' + str(path.resolve()) + '?mode=ro', uri=True)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    kernels = 'CUPTI_ACTIVITY_KIND_KERNEL'
    assert kernels in tables, 'trace contains no CUDA kernel activity'
    summary = dict(path=str(path), gpu_kernels=[dict(name=name, calls=count, gpu_ns=ns)
        for name, count, ns in db.execute(f'''
            SELECT s.value, count(*), sum(k.end-k.start)
            FROM {kernels} k LEFT JOIN StringIds s ON s.id=k.demangledName
            GROUP BY s.value ORDER BY sum(k.end-k.start) DESC''')])
    operations = [t for t in (kernels, 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET') if t in tables]
    query = ' UNION ALL '.join('SELECT start,end FROM ' + t for t in operations) + ' ORDER BY start,end'
    summary['gpu_any_activity'] = interval_union(db.execute(query))
    summary['gpu_kernel_activity'] = interval_union(db.execute(f'SELECT start,end FROM {kernels} ORDER BY start,end'))
    summary['cuda_api'] = {}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table in tables:
            summary['cuda_api'][table] = [dict(name=name, calls=count, host_ns=ns)
                for name, count, ns in db.execute(f'''
                    SELECT s.value,count(*),sum(a.end-a.start)
                    FROM {table} a LEFT JOIN StringIds s ON s.id=a.nameId
                    GROUP BY s.value ORDER BY sum(a.end-a.start) DESC''')]
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        summary['copies'] = [dict(copy_kind=kind, calls=count, bytes=nbytes, gpu_ns=ns)
            for kind, count, nbytes, ns in db.execute('''
                SELECT e.label,count(*),sum(m.bytes),sum(m.end-m.start)
                FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind
                GROUP BY e.label''')]
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
        summary['memsets_by_size'] = [dict(bytes_per_call=size, calls=count, gpu_ns=ns, bytes=nbytes)
            for size, count, ns, nbytes in db.execute('''
                SELECT bytes,count(*),sum(end-start),sum(bytes)
                FROM CUPTI_ACTIVITY_KIND_MEMSET GROUP BY bytes''')]
    if 'DIAGNOSTIC_EVENT' in tables:
        summary['collector_diagnostics'] = [dict(severity=severity, text=text)
            for severity, text in db.execute('SELECT severity,text FROM DIAGNOSTIC_EVENT')]
    db.close()
    summary['interpretation'] = ('GPU activity is interval-union time, not SM utilization. Span is first to last '
                                 'GPU operation, not application loop time. Do not sum overlapping GPU operations '
                                 'or nested host APIs. Profiler overhead is present; this is not clean throughput.')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('database', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = json.dumps(summarize(a.database), indent=2)
    if a.output:
        a.output.write_text(result + '\n')
    else:
        print(result)
