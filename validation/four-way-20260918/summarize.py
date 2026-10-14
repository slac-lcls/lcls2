"""Summarize matched clean timings; no hardware speed assertions."""
import argparse
import json
from pathlib import Path
from statistics import median


def summarize(root):
    results = json.loads((root / 'results.json').read_text())
    provenance = json.loads((root / 'provenance.json').read_text())
    variants = {}
    for variant, label in [('A', 'A'), ('B', 'B'), ('O', 'B+'), ('G', 'B+gather')]:
        samples = [r for r in results if r['variant'] == variant and
                   r['timing_mode'] == 'off' and not r['nsight_capture']]
        assert len(samples) == provenance['arguments']['repeats'], variant
        seconds = [s['loop_s'] for s in samples]
        variants[variant] = dict(label=label, seconds=seconds, median_s=median(seconds),
                                 min_s=min(seconds), max_s=max(seconds),
                                 hz_from_median=samples[0]['events']/median(seconds))
    comparisons = {}
    for before, after in [('A', 'B'), ('B', 'O'), ('O', 'G'), ('A', 'G'), ('B', 'G')]:
        old, new = variants[before]['median_s'], variants[after]['median_s']
        comparisons[f'{before}_to_{after}'] = dict(
            seconds_saved=old-new, elapsed_reduction_percent=100*(old-new)/old,
            throughput_increase_percent=100*(old/new-1))
    return dict(node=provenance['node'], variants=variants, comparisons=comparisons,
                orders=provenance['orders'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
