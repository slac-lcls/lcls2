"""Summarize matched controls separately from instrumented/NSight runs."""
import argparse
import json
from pathlib import Path
from statistics import median


def summarize(records):
    summary = {}
    for variant in 'ABO':
        by_mode = {}
        for mode in ('off', 'cpu-nvtx'):
            samples = [r for r in records if r['variant'] == variant and
                       r['timing_mode'] == mode and not r['nsight_capture']]
            if not samples:
                continue
            phases = {}
            keys = {key for r in samples for bd in r['ranks']
                    for key in (bd['phase_timing'] or {}).get('phases', {})}
            for key in sorted(keys):
                entries = [bd['phase_timing']['phases'].get(key, {})
                           for r in samples for bd in r['ranks'] if bd['phase_timing']]
                phases[key] = {field: median(entry.get(field, 0) for entry in entries)
                               for field in ('calls', 'total_ns', 'self_ns')}
            rates = [r['hz'] for r in samples]
            by_mode[mode] = dict(n=len(samples), median_hz=median(rates), min_hz=min(rates), max_hz=max(rates),
                                 median_loop_s=median(r['loop_s'] for r in samples), phases=phases)
        if all(mode in by_mode for mode in ('off', 'cpu-nvtx')):
            by_mode['instrumented_time_overhead_percent'] = 100 * (
                by_mode['cpu-nvtx']['median_loop_s'] / by_mode['off']['median_loop_s'] - 1)
        summary[variant] = by_mode
    if all('off' in summary[v] for v in 'BO'):
        summary['control_optimized_vs_B_throughput_percent'] = 100 * (
            summary['O']['off']['median_hz'] / summary['B']['off']['median_hz'] - 1)
    for variant in 'BO':
        if all('off' in summary[v] for v in ('A', variant)):
            summary['control_' + variant + '_vs_A_throughput_percent'] = 100 * (
                summary[variant]['off']['median_hz'] / summary['A']['off']['median_hz'] - 1)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = json.dumps(summarize(json.loads(a.results.read_text())), indent=2)
    if a.output:
        a.output.write_text(result + '\n')
    else:
        print(result)
