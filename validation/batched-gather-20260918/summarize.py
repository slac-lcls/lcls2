"""Summarize clean throughput separately from host scopes and device traces."""
import argparse
import json
from pathlib import Path
from statistics import median

from trace_summary import summarize as trace_summary


def summarize(root):
    records = json.loads((root / 'results.json').read_text())
    result = {}
    for variant in 'OG':
        modes = {}
        for mode in ('off', 'cpu-nvtx'):
            samples = [r for r in records if r['variant'] == variant and
                       r['timing_mode'] == mode and not r['nsight_capture']]
            if not samples:
                continue
            times = [s['loop_s'] for s in samples]
            phases = [bd['phase_timing']['phases'] for s in samples for bd in s['ranks']
                      if bd['phase_timing']]
            keys = set().union(*(set(p) for p in phases))
            modes[mode] = dict(n=len(times), seconds=times, median_s=median(times),
                               hz_from_median=samples[0]['events'] / median(times),
                               phases={k: {field: median(p[k][field] for p in phases if k in p)
                                           for field in ('calls', 'total_ns', 'self_ns')}
                                       for k in sorted(keys)})
        result[variant] = modes
    if all('off' in result[v] for v in 'OG'):
        before, after = (result[v]['off']['median_s'] for v in 'OG')
        result['clean_effect'] = dict(seconds_saved=before-after,
                                     elapsed_reduction_percent=100*(before-after)/before,
                                     throughput_increase_percent=100*(before/after-1))
    traces = {}
    for variant in 'OG':
        path = root / f'{variant}-trace-cpu-nvtx-r0.sqlite'
        if not path.exists():
            continue
        trace = trace_summary(path)
        (root / f'{variant}-trace-summary.json').write_text(json.dumps(trace, indent=2) + '\n')
        counts = {k['name']: k['calls'] for k in trace['gpu_kernels']}
        wanted = dict(walk_xtc=500, init_locators=500, locate_fields=500,
                      jungfrau_calib_kernel=10000, zero_missing_rows_kernel=10000)
        wanted['gather_locator_u16_kernel' if variant == 'O' else 'gather_canonical_u16'] = (
            320000 if variant == 'O' else 500)
        assert counts == wanted, (variant, counts)
        apis = {row['name']: row['calls'] for rows in trace['cuda_api'].values() for row in rows}
        assert apis['cuMemcpyHtoDAsync_v2'] == 370000
        assert apis['cuLaunchKernel'] == (341500 if variant == 'O' else 22000)
        assert not any(c['copy_kind'] == 'Device-to-Host' for c in trace['copies'])
        traces[variant] = dict(kernel_counts=counts, cuda_api_counts=apis,
                               copies=trace['copies'], warnings=trace.get('collector_diagnostics', []))
    result['separate_traces'] = traces
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
