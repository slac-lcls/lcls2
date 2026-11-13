"""Fixed-block statistics; keep all clean samples and all rank-local timings."""
import json
from pathlib import Path
import statistics
from schedule import validate, CASES, LENGTHS

METRICS = ('setup_s', 'first_event_s', 'first_batch_s', 'remaining_s', 'last_event_s',
           'tail_drain_s', 'manager_close_s', 'loop_s', 'sample_wall_s',
           'events_per_s', 'payload_GB_s', 'remaining_events_per_s')


def stats(values):
    values = list(values)
    assert len(values) == 6
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = 2.570581835636305 * sd / (6 ** .5)
    interval = [mean-half, mean+half]
    return dict(n=6, mean=mean, median=statistics.median(values), sd=sd, ddof=1,
                minimum=min(values), maximum=max(values), samples=values,
                t95_interval=interval, within_5pct=interval[0]>=-5 and interval[1]<=5)


def summarize(output):
    output = Path(output)
    provenance = json.loads((output/'provenance.json').read_text())
    plan = validate(provenance['schedule'])
    rows = json.loads((output/'results.json').read_text())
    clean = [r for r in rows if not r['diagnostic'] and not r['trace']]
    expected = [(b['events'], b['repetition'], c) for b in plan for c in b['order']]
    assert [(r['events'], r['repetition'], r['case']) for r in clean] == expected
    arms, comparisons = [], []
    for n in LENGTHS:
        groups = {c: sorted([r for r in clean if r['events']==n and r['case']==c],
                            key=lambda r:r['repetition']) for c in CASES}
        for case, group in groups.items():
            timers = sorted({key for r in group for bd in r['rows'] for key in bd['timers']})
            arms.append(dict(events=n, case=case,
                metrics={metric:stats(r[metric] for r in group) for metric in METRICS},
                # Per-sample maxima of each inclusive host timer. They overlap.
                timers={key:stats(max(bd['timers'].get(key,{}).get('total_s',0)
                                     for bd in r['rows'] if bd['role']=='bd')
                                 for r in group) for key in timers},
                ranks=[dict(tag=r['tag'], rows=[{k:bd[k] for k in
                    ('rank','events','setup_s','first_batch_s','remaining_s','remaining_events',
                     'loop_s','last_event_s','tail_drain_s','timers','snapshots','callback_sizes')}
                    for bd in r['rows'] if bd['role']=='bd']) for r in group]))
        for left, right in (('A','B-off'),('B-off','B-on')):
            changes = {}
            for metric in METRICS:
                pairs = list(zip(groups[left], groups[right]))
                changes[metric] = dict(
                    absolute=stats(b[metric]-a[metric] for a,b in pairs),
                    percent=stats(100*(b[metric]/a[metric]-1) for a,b in pairs))
            comparisons.append(dict(events=n, left=left, right=right, metrics=changes,
                pairs=[dict(repetition=a['repetition'], left_tag=a['tag'], right_tag=b['tag'])
                       for a,b in zip(groups[left],groups[right])]))
    summary = dict(clean_samples=len(clean), blocks=12, arms=arms, comparisons=comparisons,
        caveats=['Six-block t95 intervals assume independent approximately normal differences.',
                 'All valid blocks retained; multiple comparisons limit interpretation.',
                 'Rank-local remainder boundaries do not define one synchronized global window.',
                 'Inclusive host timers overlap and are not GPU execution times.',
                 'Cold Lustre may conceal compute overhead; server caches are uncontrolled.'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def report(output):
    output = Path(output)
    s = json.loads((output/'summary.json').read_text())
    p = json.loads((output/'provenance.json').read_text())
    text = f'''# PR187 startup versus remaining-loop time

Job {p['job']}, node {p['node']}. All 36 clean samples completed.
Before/after integrity accepted: {p.get('clean_complete',False)}.

This measures A, B-off and B-on on one exclusive A100 80GB node, four BDs
on GPU0, one EB and one Smd0. Each BD has a fixed 6 GiB budget, 20-event
batches and one stream; KvikIO 26.10 uses CPU fallback, eight threads and
1 MiB tasks. mfx101232725 run442, 32 Jungfrau panels, 768 MiB constants.
Runtime, source hashes, exact commands, CPU affinity, GPU UUID/PCI, inventory,
cache gates, NIC traffic and per-rank measurements are retained beside this report.

Prior evidence: [Perlmutter A/B/C report](/pscratch/sd/m/monarin/psana2-gpu/pr187-pr208-perlmutter-20261026/attempt01/REPORT.md).
Job60376655 found four-BD cold B/A +1.77% [-0.32%,+3.87%], without resolving
startup. One-BD PR208 C/B remains separately inconclusive after60393385.
Two-node functional60393368 passed; it was not repeated here. PR187 is already
merged according to the supplied handoff; these are frozen pre/post versions.

## Mean times and rates

| Events | Case | Setup s | First batch s | Remainder s | Close s | Loop s | Wall s | events/s | Payload GB/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
'''
    for arm in s['arms']:
        m=arm['metrics']
        text+=f"| {arm['events']} | {arm['case']} | "+' | '.join(f"{m[k]['mean']:.4f}" for k in
            ('setup_s','first_batch_s','remaining_s','manager_close_s','loop_s','sample_wall_s','events_per_s','payload_GB_s'))+' |\n'
    text+='''
## Paired changes

Positive duration changes mean slower. Each row uses six adjacent matched blocks.
First-batch and remainder are separately aggregated rank maxima; subtracting
these independent maxima is not an additive decomposition of total loop time.

| Events | Comparison | Metric | Mean delta s | Mean % | 95% interval % | Within ±5% |
| ---: | --- | --- | ---: | ---: | --- | --- |
'''
    for c in s['comparisons']:
        for metric in ('setup_s','first_batch_s','remaining_s','manager_close_s','loop_s','sample_wall_s'):
            m=c['metrics'][metric];pct=m['percent'];lo,hi=pct['t95_interval']
            text+=f"| {c['events']} | {c['right']} / {c['left']} | {metric} | {m['absolute']['mean']:+.4f} | {pct['mean']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {'Yes' if pct['within_5pct'] else 'Inconclusive'} |\n"
    text+='''
## Constants operations

Per-sample maximum BD inclusive host times, averaged over six samples.
Nested measurements overlap: do not sum them or equate host submission to GPU execution.

| Events | Case | Pipeline setup s | Constants refresh s | Digest s | Publish s | Subscribe s | Settle s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''
    for arm in s['arms']:
        keys=['GpuEventManager._setup_gpu_pipeline',
              'RequestedConstants.refresh' if arm['case']=='A' else 'SharedRequestedConstants.refresh',
              'digest_of','SharedRequestedConstants._publish','SharedRequestedConstants._subscribe',
              'SharedRequestedConstants._settle']
        text+=f"| {arm['events']} | {arm['case']} | "+' | '.join(f"{arm['timers'].get(k,{}).get('mean',0):.4f}" for k in keys)+' |\n'
    text+='\n## Limits and traces\n\n'+'\n'.join('- '+c for c in s['caveats'])+'\n'
    text+='\nOne first-boundary CUDA event synchronization per BD is added identically in all cases.\n'
    trace=output/'traces-status.json'
    text+='\nTrace status: '+(trace.read_text() if trace.exists() else 'not yet attempted')+'\n'
    (output/'REPORT.md').write_text(text)
