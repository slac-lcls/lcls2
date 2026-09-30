"""Audit completed warm measurements and expected parser launch counts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def audit(root):
    results = json.loads((root / 'results.json').read_text())
    regular = [r for r in results if not r['nsight_capture']]
    expected = {f'{v}-warm-{mode}-r{i}' for v in 'BO'
                for mode in ('off', 'cpu-nvtx') for i in (1, 2, 3)}
    assert {r['case'] for r in regular} == expected and len(regular) == 12
    assert len(results) == 14
    stage = json.loads((root / 'stage.json').read_text())['manifest']
    caches, peaks = {}, {}
    for r in results:
        assert r['events'] == 10000 and r['timestamp_sha256'] == stage['timestamp_sha256']
        assert (r['batch_size'], r['depth'], r['budget_gib'], r['n_bds']) == (20, 1, 8, 1)
        assert r['cache'] == 'warm' and r['node'] == 'sdfampere004'
        fractions = []
        for line in (root / (r['case'] + '.log')).read_text().splitlines():
            if line.startswith(('CACHE_BEFORE ', 'CACHE_AFTER ')):
                fractions.append(json.loads(line.split(' ', 1)[1])['fraction'])
        assert fractions == [1.0, 1.0]
        caches[r['case']] = fractions
        if r['timing_mode'] == 'off':
            with (root / (r['case'] + '-gpu.csv')).open() as source:
                peaks[r['case']] = max(float(row[2]) for row in csv.reader(source))
    traces = {}
    for variant in 'BO':
        trace = json.loads((root / (variant + '-trace-summary.json')).read_text())
        kernels = {row['name']: row['calls'] for row in trace['gpu_kernels']}
        wanted = dict(walk_xtc=500, gather_locator_u16_kernel=320000,
                      jungfrau_calib_kernel=10000, zero_missing_rows_kernel=10000)
        wanted.update(dict(locate_field=96000, cupy_fill=96000) if variant == 'B'
                      else dict(init_locators=500, locate_fields=500))
        assert kernels == wanted, (variant, kernels)
        api = {row['name']: row['calls'] for rows in trace['cuda_api'].values() for row in rows}
        assert api['cuMemcpyHtoDAsync_v2'] == 370000
        assert api['cuLaunchKernel'] == (532500 if variant == 'B' else 341500)
        assert api['cudaEventRecord_v3020'] == (97008 if variant == 'B' else 1510)
        assert api['cudaMemsetAsync_v3020'] == (116000 if variant == 'B' else 20000)
        assert [r['copy_kind'] for r in trace['copies']] == ['Host-to-Device']
        traces[variant] = dict(kernels=kernels, h2d=trace['copies'][0],
                               warnings=[d for d in trace.get('collector_diagnostics', [])
                                         if d['severity'] > 1])
    assert traces['O']['h2d']['bytes'] - traces['B']['h2d']['bytes'] == 4656
    assert traces['O']['h2d']['calls'] - traces['B']['h2d']['calls'] == 2
    provenance = json.loads((root.parent / 'build-provenance.json').read_text())
    for variant in 'BO':
        for module in provenance[variant]['modules'].values():
            assert hashlib.sha256(Path(module['installed']).read_bytes()).hexdigest() == module['sha256']
    return dict(regular_samples=12, separate_traces=2, cache_fractions=caches,
                clean_peak_device_memory_mib=peaks, traces=traces,
                timestamp_sha256=stage['timestamp_sha256'], source_hashes_unchanged=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    (args.directory / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('AUDIT_PASS: 12 regular samples, 2 traces, expected launch counts, warm cache, unchanged source hashes')
