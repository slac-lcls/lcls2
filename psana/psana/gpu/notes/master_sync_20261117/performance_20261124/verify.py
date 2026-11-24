"""Replay published PR331 statistics and verify compact-artifact hashes; no jobs."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent

def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, receipt in manifest['files'].items():
        content = (ROOT / name).read_bytes()
        assert len(content) == receipt['bytes'], name
        assert hashlib.sha256(content).hexdigest() == receipt['sha256'], name
    report = ROOT.parent / 'performance_20261124.md'
    assert hashlib.sha256(report.read_bytes()).hexdigest() == manifest['report_sha256']
    summary = json.loads((ROOT / 'results.json').read_text())
    with (ROOT / 'samples.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT / 'paired-loop-changes.csv').open() as stream:
        pairs = list(csv.DictReader(stream))
    assert len(rows) == 70 and len({r['sample_id'] for r in rows}) == 70
    assert len(pairs) == 30
    metrics = ['loop_seconds', 'setup_seconds', 'first_batch_seconds', 'process_seconds']
    for case in summary['cases']:
        selected = [r for r in rows if r['case'] == case['case']]
        assert len([r for r in selected if r['phase'] == 'correctness']) == 2
        ab = [r for r in selected if r['phase'] == 'ab']
        assert len(ab) == 12
        for metric in metrics:
            changes = []
            for pair in range(6):
                both = [r for r in ab if int(r['pair']) == pair]
                assert len(both) == 2
                a = next(r for r in both if r['runtime'] == 'baseline')
                b = next(r for r in both if r['runtime'] == 'candidate')
                assert (int(a['order_in_phase']) < int(b['order_in_phase'])) == (pair % 2 == 0)
                changes.append(100 * (float(b[metric]) / float(a[metric]) - 1))
            mean = statistics.mean(changes)
            half = 2.571 * statistics.stdev(changes) / math.sqrt(6)
            low, high = mean - half, mean + half
            observed = case['loop'] if metric == 'loop_seconds' else case['loop']['secondary_timings'][metric]
            assert abs(mean - observed['mean_percent_change']) < 1e-10
            assert all(abs(a - b) < 1e-10 for a, b in zip([low, high], observed['interval95']))
            assert all(abs(a - b) < 1e-10 for a, b in zip(changes, observed['paired_percent_changes']))
            verdict = ('equivalent' if low >= -5 and high <= 5 else
                       'regression' if low > 5 else 'improvement' if high < -5 else 'inconclusive')
            assert verdict == observed['acceptance']
            if metric == 'loop_seconds':
                values = sorted((p for p in pairs if p['case'] == case['case']),
                                key=lambda p: int(p['pair_index_zero_based']))
                assert len(values) == 6
                assert all(abs(x - float(p['candidate_vs_baseline_percent_change'])) < 1e-10
                           for x, p in zip(changes, values))
        print(case['case'], case['loop']['acceptance'])
    print('PASS: artifact hashes, 70 samples, 30 pairs, 20 case/metric comparisons')

if __name__ == '__main__':
    main()
