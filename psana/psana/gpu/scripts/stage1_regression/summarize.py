"""Summarize paired samples; unfinished campaigns are explicitly provisional."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(job):
    provenance = json.loads((job/'provenance.json').read_text())
    rows = json.loads((job/'results.json').read_text()) if (job/'results.json').exists() else []
    variants = provenance['settings']['variants']
    settings = provenance['settings']
    repetitions = settings['repetitions']
    expected = (len(provenance['points']) * len(settings['modes']) *
                len(settings.get('caches', ['cold', 'warm'])) * len(variants) * repetitions)
    timed = [r for r in rows if not r['diagnostic']]
    print(f"{job.name}: {len(rows)-len(timed)} preflights, {len(timed)}/{expected} timed, "
          f"complete={provenance.get('complete', False)}, host={provenance['host']}")
    cells = defaultdict(dict)
    for row in timed:
        key = (row['nbds'], row['cache'], row['bulk'])
        cells[key].setdefault(row['variant'], []).append(row)
    print('| BDs | Cache | Bulk | Before Hz | After Hz | Change | Repetitions |')
    print('|---:|---|---|---:|---:|---:|---|')
    for (bds, cache, bulk), versions in sorted(cells.items()):
        if not all(v in versions for v in variants):
            continue
        rates = [10000/statistics.median(r['loop_s'] for r in versions[v]) for v in variants]
        change = (rates[1]/rates[0]-1)*100
        counts = '/'.join(str(len(versions[v])) for v in variants)
        print(f'| {bds} | {cache} | {bulk} | {rates[0]:.2f} | {rates[1]:.2f} | {change:+.1f}% | {counts} |')
        if all(len(versions[v]) == repetitions for v in variants) and change < -5:
            print('FOLLOWUP', bds, cache, bulk,
                  {v: [round(r['events_per_s'], 2) for r in versions[v]] for v in variants})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobs', type=Path, nargs='+')
    for path in parser.parse_args().jobs:
        summarize(path)
