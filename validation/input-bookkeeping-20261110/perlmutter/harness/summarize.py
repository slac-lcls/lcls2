"""Summarize completed receipts; clean medians and separate timed attribution."""
import json
from pathlib import Path
from statistics import median
import sys

root = Path(sys.argv[1])
complete = json.loads((root / 'complete.json').read_text())
assert complete['status'] == 'pass' and len(complete['cases']) == 12
summary = {'clean': [], 'timing': []}
for variant in ('old', 'new'):
    for streams in (1, 2):
        rows = [r for r in complete['cases'] if r['mode'] == 'clean'
                and r['variant'] == variant and r['streams'] == streams]
        assert len(rows) == 2
        summary['clean'].append(dict(variant=variant, streams=streams,
            seconds=[r['seconds'] for r in rows],
            gib_s=[r['gib_s'] for r in rows],
            median_seconds=median(r['seconds'] for r in rows),
            median_gib_s=median(r['gib_s'] for r in rows)))
for row in complete['cases']:
    if row['mode'] != 'timing':
        continue
    phases = json.loads((root / (row['name'] + '.json.phases.json')).read_text())
    totals = {}
    for name, stats in phases['phases'].items():
        phase = name.split('/', 1)[1]
        totals[phase] = totals.get(phase, 0) + stats['self_ns'] / 1e9
    summary['timing'].append(dict(name=row['name'], variant=row['variant'],
        streams=row['streams'], seconds=row['seconds'],
        exclusive_seconds=totals, counters=phases['counters']))
print(json.dumps(summary, indent=2))
