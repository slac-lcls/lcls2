"""Audit actual fallback triplets and partition the read interval by occupancy."""
import json
from pathlib import Path
import statistics
import sys
import numpy as np

DTYPE = np.dtype([(name, '<u8') for name in ('start', 'end', 'size')] +
                 [(name, '<i8') for name in ('offset', 'result')] +
                 [(name, '<u4') for name in ('tid', 'kind', 'fd', 'batch', 'read_id', 'reserved')])


def file_concurrency(records):
    """Union wall time by active file count; excludes intervals without reads.

    File descriptors identify the six stable open input handles in this harness.
    Multiple workers reading one file still count as one active file.
    """
    reads = records[records['kind'] == 1]
    assert len(reads) and np.all(reads['end'] >= reads['start'])
    points = np.concatenate((reads['start'], reads['end']))
    fds = np.concatenate((reads['fd'], reads['fd']))
    delta = np.concatenate((np.ones(len(reads), dtype=np.int8),
                            -np.ones(len(reads), dtype=np.int8)))
    order = np.argsort(points, kind='stable')
    points, fds, delta = points[order], fds[order], delta[order]
    duration = np.diff(points)
    live = np.vstack([np.cumsum(np.where(fds == fd, delta, 0))[:-1]
                      for fd in np.unique(fds)])
    assert np.all(live >= 0)
    distinct, workers = np.count_nonzero(live, axis=0), live.sum(axis=0)
    by_file = {str(n): int(duration[distinct == n].sum()) / 1e9
               for n in np.unique(distinct) if n > 0}
    by_worker = {str(n): int(duration[workers == n].sum()) / 1e9
                 for n in np.unique(workers) if n > 0}
    active_ns = int(duration[distinct > 0].sum())
    single_ns = int(duration[distinct == 1].sum())
    assert active_ns > 0
    assert abs(sum(int(k)*v for k,v in by_worker.items()) -
               int((reads['end']-reads['start']).sum())/1e9) < 1e-5
    return dict(single_file_s=single_ns/1e9, posix_active_s=active_ns/1e9,
                single_file_percent=100*single_ns/active_ns,
                wall_by_distinct_files_s=by_file, wall_by_active_calls_s=by_worker,
                files=len(np.unique(reads['fd'])))


def occupancy(records, batches):
    """Disjoint wall seconds, including gaps inside issue-to-ready windows.

    H2D means host copy API plus existing stream wait, not CUDA device timing.
    POSIX present and transfer present can overlap across the worker threads.
    """
    starts = np.array([b['issued_ns'] for b in batches], dtype=np.int64)
    ends = np.array([b['wait_end'] for b in batches], dtype=np.int64)
    assert np.all(starts[1:] >= ends[:-1])
    n = len(records)
    points = np.concatenate((records['start'], records['end'], starts, ends)).astype(np.int64)
    category = np.concatenate((records['kind'] != 1, records['kind'] != 1,
                               np.full(len(starts), 2), np.full(len(ends), 2))).astype(np.int8)
    delta = np.concatenate((np.ones(n, dtype=np.int8), -np.ones(n, dtype=np.int8),
                            np.ones(len(starts), dtype=np.int8), -np.ones(len(ends), dtype=np.int8)))
    order = np.argsort(points, kind='stable')
    points, category, delta = points[order], category[order], delta[order]
    live = [np.cumsum(np.where(category == k, delta, 0))[:-1] for k in range(3)]
    duration = np.diff(points)
    inside = live[2] == 1
    labels = {0: 'neither', 1: 'POSIX_only', 2: 'H2D_or_wait_only', 3: 'POSIX_and_H2D_or_wait'}
    state = (live[0] > 0).astype(np.int8) + 2 * (live[1] > 0)
    result = {label: int(duration[inside & (state == key)].sum()) / 1e9 for key, label in labels.items()}
    assert abs(sum(result.values()) - int((ends-starts).sum())/1e9) < 1e-6
    return result


def audit_trace(metadata):
    path = Path(metadata)
    meta = json.loads(path.read_text())
    data = np.fromfile(meta['binary'], dtype=DTYPE)
    assert DTYPE.itemsize == meta['record_size'] == 64
    assert len(data) == meta['records']
    reads, copies, syncs = [data[data['kind'] == k] for k in (1, 2, 3)]
    read_ids = np.flatnonzero(data['kind'] == 1)
    assert len(reads) == len(copies) == len(syncs) > 0
    assert len(data) == 3 * len(reads)
    assert np.all(reads['result'] == reads['size'])
    assert np.all(copies['result'] == 0) and np.all(syncs['result'] == 0)
    copies = copies[np.argsort(copies['read_id'])]
    syncs = syncs[np.argsort(syncs['read_id'])]
    assert np.array_equal(copies['read_id'], read_ids)
    assert np.array_equal(syncs['read_id'], read_ids)
    for child in (copies, syncs):
        for key in ('tid', 'batch', 'fd', 'offset', 'size'):
            assert np.array_equal(reads[key], child[key]), key
    assert np.all(reads['end'] <= copies['start'])
    assert np.all(copies['end'] <= syncs['start'])
    assert np.all(data['start'] <= data['end'])
    batches = meta['batches']
    assert [b['batch'] for b in batches] == list(range(1, len(batches)+1))
    start = np.array([b['issued_ns'] for b in batches], dtype=np.uint64)
    end = np.array([b['wait_end'] for b in batches], dtype=np.uint64)
    assert np.all((data['batch'] >= 1) & (data['batch'] <= len(batches)))
    assert np.all(data['start'] >= start[data['batch']-1])
    assert np.all(data['end'] <= end[data['batch']-1])
    expected_bytes = sum(b['requested_bytes'] for b in batches)
    expected_ops = sum((r['size']+1048575)//1048576 for b in batches for r in b['ranges'])
    assert int(reads['size'].sum()) == int(copies['size'].sum()) == expected_bytes
    assert len(reads) == expected_ops, (len(reads), expected_ops)
    for b in batches:
        selected = reads[reads['batch'] == b['batch']]
        assert int(selected['size'].sum()) == b['requested_bytes']
    operations = {}
    for label, rows in [('POSIX', reads), ('H2D_API', copies), ('stream_wait', syncs)]:
        durations = (rows['end']-rows['start']) / 1e6
        operations[label] = dict(calls=len(rows), worker_sum_s=float(durations.sum()/1000),
            mean_ms=float(durations.mean()), p50_ms=float(np.median(durations)),
            p95_ms=float(np.percentile(durations, 95)), max_ms=float(durations.max()))
    sizes, counts = np.unique(reads['size'], return_counts=True)
    result = dict(metadata=str(path), batches=len(batches), api_requests=sum(b['requests'] for b in batches),
        bytes=expected_bytes, read_interval_s=int((end-start).sum())/1e9,
        issue_s=sum(b['issue_end']-b['issued_ns'] for b in batches)/1e9,
        before_wait_s=sum(b['wait_begin']-b['issue_end'] for b in batches)/1e9,
        completion_wait_s=sum(b['wait_end']-b['wait_begin'] for b in batches)/1e9,
        operations=operations, wall_occupancy_s=occupancy(data, batches),
        workers=len(np.unique(reads['tid'])), size_counts={str(int(s)): int(n) for s,n in zip(sizes,counts)},
        file_concurrency=file_concurrency(data))
    if 'caller_tid' in meta:
        tids = np.unique(reads['tid'])
        result['pool_workers'] = int(np.count_nonzero(tids != meta['caller_tid']))
        result['caller_reads'] = int(np.count_nonzero(reads['tid'] == meta['caller_tid']))
    path.with_suffix('.audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def summarize(root):
    from run_baseline import HERE, records, sha, verify_builds
    root = Path(root)
    results = json.loads((root/'results.json').read_text())
    preflights = json.loads((root/'preflights.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text())
    verify_builds(provenance['builds'])
    for name,digest in provenance['scripts'].items():
        assert sha(HERE/name) == digest
    variants = ('Integrated-off', 'Integrated-on')
    assert len(preflights) == 2 and {p['variant'] for p in preflights} == set(variants)
    assert len(results) == 8
    assert {(r['variant'],r['cache'],r['repetition'],r['timing_mode']) for r in results} == {
        (v,'cold',n,m) for v in variants for n in (1,2) for m in ('cpu','off')}
    details = []
    for r in results:
        assert records(Path(r['log']).read_text(), 'CACHE_BEFORE ')[0]['fraction'] <= .01
        if r['timing_mode'] == 'off':
            assert r['ranks'][0]['fallback_trace'] is None
            continue
        d = audit_trace(r['ranks'][0]['fallback_trace']['metadata'])
        assert d['batches'] == 500 and d['workers'] == 8
        assert d['bytes'] == r['payload_bytes']
        assert d['api_requests'] == (50000 if r['variant'] == 'Integrated-off' else 2885)
        d.update(variant=r['variant'], repetition=r['repetition'], loop_s=r['loop_s'])
        phases=r['ranks'][0]['phase_timing']['phases']
        d['exclusive_host_s']={}
        for label,p in phases.items():
            name=label.split('/',1)[1]
            d['exclusive_host_s'][name]=d['exclusive_host_s'].get(name,0)+p['self_ns']/1e9
        details.append(d)
    lines=['# Cold JF KvikIO fallback trace', '',
        f"Job {provenance['job_id']} on {provenance['node']}; 10,000 events, batch 20, depth 1, 8 GiB.",
        'Two rounds, one allocation, one A100/BD, 8 KvikIO workers, 1 MiB tasks, CPU fallback.',
        'All operation/byte/triplet, cold-cache, timestamp, placement and runtime audits passed.', '',
        '## Loop seconds', '', '| Variant | Control R1 / R2 | Trace R1 / R2 |', '|---|---:|---:|']
    for v in variants:
        cells=[]
        for m in ('off','cpu'):
            chosen=sorted([r for r in results if r['variant']==v and r['timing_mode']==m],key=lambda r:r['repetition'])
            cells.append(' / '.join(f"{r['loop_s']:.3f}" for r in chosen))
        lines.append('| '+v+' | '+' | '.join(cells)+' |')
    lines += ['', '## Trace medians', '', '| Metric | Bulk off | Bulk on |','|---|---:|---:|']
    metrics=[('KvikIO API requests',lambda d:d['api_requests']),('Actual POSIX reads',lambda d:d['operations']['POSIX']['calls']),
        ('Actual H2D copies',lambda d:d['operations']['H2D_API']['calls']),('Existing stream waits',lambda d:d['operations']['stream_wait']['calls']),
        ('Read issue-to-ready wall s',lambda d:d['read_interval_s']),('Submission wall s',lambda d:d['issue_s']),
        ('Host gap before wait s',lambda d:d['before_wait_s']),('Completion wait wall s',lambda d:d['completion_wait_s'])]
    metrics += [(key+' wall s',lambda d,k=key:d['wall_occupancy_s'][k]) for key in details[0]['wall_occupancy_s']]
    metrics += [(key+' worker sum s',lambda d,k=key:d['operations'][k]['worker_sum_s']) for key in details[0]['operations']]
    for label,fn in metrics:
        values=[statistics.median(fn(d) for d in details if d['variant']==v) for v in variants]
        lines.append('| '+label+' | '+' | '.join(f'{v:,.3f}' for v in values)+' |')
    lines += ['', 'Wall occupancy rows are disjoint and partition issue-to-ready time. POSIX and H2D/wait can overlap on different workers.',
        'Worker sums overlap across 8 threads and must not be added to wall time. H2D API and stream wait are host timings, not GPU copy durations.',
        'The trace adds no synchronization. Controls capture instrumentation sensitivity plus run variability.',
        'POSIX pread time includes kernel page-cache/storage and scheduling effects; this does not count physical NVMe commands.', '']
    summary=dict(samples=len(results),details=details,provenance=str(root/'provenance.json'))
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'summary.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    summarize(sys.argv[1])
