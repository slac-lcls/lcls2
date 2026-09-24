"""Audit and summarize one completed JF host-timing campaign."""
import json
from pathlib import Path
import statistics
import sys


def group(name):
    if name == 'read.wait':
        return 'Read completion wait'
    if name == 'read.pread_loop':
        return 'KvikIO pread submission loop'
    if name.startswith('sync.'):
        return 'Existing GPU synchronization'
    if name == 'allocation.owned_empty':
        return 'Owned device allocation calls'
    if name == 'cache.trim':
        return 'Cache trim'
    if name.startswith('parser.'):
        return 'Parser metadata and kernel submission'
    if name.startswith('detector.'):
        return 'Gather map, gather and calibration submission'
    if name.startswith('admission.') or name == 'resident.start':
        return 'Admission and resident setup'
    if name.startswith('read.') or name == 'manager.issue_read':
        return 'Read planning and remaining issue work'
    if name == 'upstream.next_batch':
        return 'Upstream next batch'
    if name.startswith('setup.') or name.startswith('transition.'):
        return 'Setup and transitions'
    return 'Event views, delivery, ownership and orchestration'


def summarize(root):
    from run_baseline import HERE, WORKLOADS, records, sha, verify_builds
    root = Path(root)
    results = json.loads((root / 'results.json').read_text())
    preflights = json.loads((root / 'preflights.json').read_text())
    provenance = json.loads((root / 'provenance.json').read_text())
    verify_builds(provenance['builds'])
    for name, digest in provenance['scripts'].items():
        assert sha(HERE / name) == digest, name
    for workload, references in provenance['references'].items():
        for name, digest in references.items():
            assert sha(HERE / 'references' / workload / name) == digest
    variants = ('Integrated-off', 'Integrated-on')
    assert len(preflights) == 2 and {r['variant'] for r in preflights} == set(variants)
    assert all(r['ranks'][0]['phase_timing']['phases'] for r in preflights)
    expected = {(v, c, n, m) for v in variants for c in ('warm', 'cold')
                for n in (1, 2) for m in ('cpu', 'off')}
    assert len(results) == 16
    assert {(r['variant'], r['cache'], r['repetition'], r['timing_mode']) for r in results} == expected
    details = []
    for r in results:
        manifest = json.loads((root / r['workload'] / 'stage.json').read_text())['manifest']
        assert r['events'] == manifest['events'] == 10000
        assert r['timestamp_sha256'] == manifest['timestamp_sha256']
        assert r['payload_bytes'] == manifest['payload_bytes']
        assert (r['batch_size'], r['depth'], r['budget_gib'], r['n_bds']) == (20, 1, 8, 1)
        assert (r['job_id'], r['node'], r['gpu_uuid']) == (
            provenance['job_id'], provenance['node'], provenance['gpu_uuid'])
        log = Path(r['log']).read_text()
        before, after = records(log, 'CACHE_BEFORE '), records(log, 'CACHE_AFTER ')
        assert len(before) == len(after) == 1
        if r['cache'] == 'warm':
            assert min(before[0]['fraction'], after[0]['fraction']) >= .99
        else:
            assert before[0]['fraction'] <= .01
        for point in ('PLACEMENT_BEFORE ', 'PLACEMENT_AFTER '):
            placements = records(log, point)
            assert sorted(p['rank'] for p in placements) == [0, 1, 2]
            assert all(p['affinity'] == provenance['cpu_affinity'] for p in placements)
        if r['timing_mode'] == 'off':
            assert r['ranks'][0]['phase_timing'] is None
            continue
        bd = r['ranks'][0]
        phases = bd['phase_timing']['phases']
        exclusive, inclusive, calls, grouped = {}, {}, {}, {}
        for label, p in phases.items():
            name = label.split('/', 1)[1]
            assert 0 <= p['self_ns'] <= p['total_ns']
            exclusive[name] = exclusive.get(name, 0) + p['self_ns'] / 1e9
            inclusive[name] = inclusive.get(name, 0) + p['total_ns'] / 1e9
            calls[name] = calls.get(name, 0) + p['calls']
            category = group(name)
            grouped[category] = grouped.get(category, 0) + p['self_ns'] / 1e9
        remainder = bd['elapsed'] - sum(exclusive.values())
        assert remainder >= 0
        grouped['Outside instrumented scopes'] = remainder
        assert calls['parser.parse'] == calls['detector.gather_submit'] == 500
        counts = bd['phase_timing']['counters']
        assert counts['reader']['useful_bytes'] == manifest['payload_bytes']
        assert counts['reader']['total_requests'] == (50000 if r['variant'] == 'Integrated-off' else 2885)
        details.append(dict(variant=r['variant'], cache=r['cache'], repetition=r['repetition'],
                            loop_s=r['loop_s'], bd_s=bd['elapsed'], grouped_s=grouped,
                            exclusive_s=exclusive, inclusive_s=inclusive, calls=calls, counters=counts))
    median = statistics.median
    columns = [(v, c) for v in variants for c in ('warm', 'cold')]
    lines = ['# JF warm/cold phase timing', '',
             f"Job {provenance['job_id']}, {provenance['node']}, {provenance['gpu_uuid']}.", '',
             '16 samples: two repeats of bulk off/on × warm/cold × timers enabled/disabled.',
             'Two instrumented CPU-reference preflights passed. All timestamps, cache, placement, and build audits passed.', '',
             '10,000 JF events; run 387, batch 20, depth 1, 8 GiB; one A100/BD; KvikIO CPU fallback.',
             'Each process warmed 100 events. No user/automatic D2H in measured loops.', '',
             '## Event-loop seconds', '',
             '| Variant | Cache | Timers off R1 / R2 | Timers on R1 / R2 | Median on − off |',
             '|---|---|---:|---:|---:|']
    for v, c in columns:
        selected = sorted([r for r in results if (r['variant'], r['cache']) == (v, c)], key=lambda r: r['repetition'])
        off = [r['loop_s'] for r in selected if r['timing_mode'] == 'off']
        on = [r['loop_s'] for r in selected if r['timing_mode'] == 'cpu']
        lines.append(f'| {v} | {c} | {off[0]:.3f} / {off[1]:.3f} | {on[0]:.3f} / {on[1]:.3f} | {median(on)-median(off):+.3f} |')
    lines += ['', 'Controls estimate instrumentation sensitivity plus run variability, not a pure timer-overhead measurement.', '',
              '## Exclusive BD host seconds', '',
              'Nested child time is subtracted. These rows partition BD wall time; kernel submission is not GPU execution duration.', '',
              '| Phase | Off warm | Off cold | On warm | On cold |', '|---|---:|---:|---:|---:|']
    categories = sorted({k for d in details for k in d['grouped_s']})
    for category in categories:
        values = [median(d['grouped_s'].get(category, 0) for d in details if (d['variant'], d['cache']) == pair) for pair in columns]
        lines.append('| ' + category + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines.append('| Total BD wall | ' + ' | '.join(f"{median(d['bd_s'] for d in details if (d['variant'], d['cache']) == pair):.3f}" for pair in columns) + ' |')
    lines += ['', '## Detailed phase seconds (exclusive / inclusive)', '',
              'Inclusive columns overlap and must not be added.', '',
              '| Phase | Off warm | Off cold | On warm | On cold |', '|---|---:|---:|---:|---:|']
    for name in sorted({k for d in details for k in d['exclusive_s']}):
        cells = []
        for pair in columns:
            selected = [d for d in details if (d['variant'], d['cache']) == pair]
            cells.append(f"{median(d['exclusive_s'].get(name, 0) for d in selected):.3f} / {median(d['inclusive_s'].get(name, 0) for d in selected):.3f}")
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Per-run counters', '', '| Variant | Cache | Round | Owned allocation calls | Reads |', '|---|---|---:|---|---:|']
    for d in details:
        lines.append(f"| {d['variant']} | {d['cache']} | {d['repetition']} | {d['counters']['owned_allocations']} | {d['counters']['reader']['total_requests']} |")
    lines += ['', 'Owned allocation calls count owned_empty, including memory-pool reuse; they are not cudaMalloc counts.',
              'Cold is verified Linux page-cache-cold local NVMe. These measurements do not establish true-GDS performance.',
              'No additional CUDA synchronization was introduced. CPU timers cannot determine disk/H2D/kernel overlap internally.', '']
    summary = dict(samples=len(results), instrumented=len(details), details=details, provenance=str(root / 'provenance.json'))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (root / 'summary.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    summarize(sys.argv[1])
