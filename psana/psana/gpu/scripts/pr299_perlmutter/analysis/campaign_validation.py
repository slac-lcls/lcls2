"""Extract final run contracts and measured sustained windows from existing evidence."""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path('/pscratch/sd/m/monarin/agent-work/lcls2/pr299-perlmutter-screen-20261115TKJ0du6/results/campaign-job-61375111')
DEST = Path(__file__).resolve().parent
samples = [json.loads(p.read_text()) for p in sorted(ROOT.glob('*/sample.json'))]
timed = [s for s in samples if not s['preflight']]
preflights = [s for s in samples if s['preflight']]
assert len(timed) == 48 and len(preflights) == 12
assert all(s['complete'] for s in samples)
assert {s['timestamp_sha256'] for s in timed} == {'2034208363befd6a797f889cbbff997a4c28d7297cf136de15e6b775a0597f17'}
assert all(s['events'] == (120 if s['preflight'] else 10000) for s in samples)
assert all(s['n_gpu_streams'] == 1 for s in samples)
uuids = set()
gpu_io = []
sustained = defaultdict(list)
peaks = defaultdict(list)
for s in samples:
    for row in s['rows']:
        expected_gpu = s['mode'] == 'gpu' and row['role'] == 'bd'
        assert row['runtime_provenance']['gpu_initialized'] == expected_gpu
        if row['role'] == 'bd':
            assert row['event_count'] > 0
        if expected_gpu:
            uuids.add(row['gpu_uuid'])
            io = row['cumulative_io']
            assert io['compat_mode'] and io['bulk_read'] and io['io_path'] == 'CPU-fallback'
            assert io['total_bytes'] > 0 and io['total_requests'] > 0
    if s['mode'] != 'gpu' or s['preflight']:
        continue
    gpu_io.append(sum(r['cumulative_io']['total_bytes'] for r in s['rows'] if r.get('cumulative_io')))
    peaks[s['case']].append(s['gpu_memory']['peak_used_mib'])
    for f in sorted((ROOT/s['label']).glob('progress-rank-*.jsonl')):
        rows = [json.loads(line) for line in f.read_text().splitlines()]
        # Actual logged endpoints at or just below fractional progress; no interpolation.
        last = rows[-1]['events']
        points = [max((r for r in rows if r['events'] <= last*q), key=lambda r:r['events']) for q in (.25,.5,.75,1)]
        def window(a,b):
            return dict(first_event=a['events'],last_event=b['events'],seconds=b['seconds']-a['seconds'],events_per_second=(b['events']-a['events'])/(b['seconds']-a['seconds']))
        quarters = [window(a,b) for a,b in zip(points,points[1:])]
        rates = [q['events_per_second'] for q in quarters]
        sustained[s['case']].append(dict(label=s['label'],runtime=s['runtime'],rank=rows[0]['rank'],last_reported_event=last,late_half=window(points[1],points[-1]),quarters_after_first=quarters,quarter_min_max_ratio=min(rates)/max(rates)))
assert len(uuids) == 1
assert set(gpu_io) == {335571760000}
result = dict(timed_samples=48,preflights=12,exact_timestamp_sha256=timed[0]['timestamp_sha256'],gpu_uuids=sorted(uuids),one_stream_per_bd=True,no_gds=True,cpu_roles_gpu_isolated=True,every_bd_active=True,gpu_completed_bytes_each_sample=335571760000,gpu_timed_total_completed_bytes=sum(gpu_io),sampled_gpu_memory_peak_mib={k:[min(v),max(v)] for k,v in peaks.items()},sustained=sustained,sustained_method='Per BD, use actual progress endpoints at or below 25/50/75/100% of last logged event count. BD2 may have an unlogged tail below 100 events. Do not add rates across different per-rank intervals.')
(DEST/'campaign-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sustained'},indent=2))
for case, rows in sustained.items():
    print(case,'late-half per-rank rates',min(r['late_half']['events_per_second'] for r in rows),max(r['late_half']['events_per_second'] for r in rows),'quarter stability min/max',min(r['quarter_min_max_ratio'] for r in rows),max(r['quarter_min_max_ratio'] for r in rows))
