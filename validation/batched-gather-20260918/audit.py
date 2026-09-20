"""Verify the completed one-allocation B+/batched-gather comparison without speed thresholds."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def records(path, prefix):
    return [json.loads(line[len(prefix):]) for line in path.read_text().splitlines()
            if line.startswith(prefix)]


def audit(root):
    results = json.loads((root / 'results.json').read_text())
    provenance = json.loads((root / 'provenance.json').read_text())
    stage = json.loads((root / 'stage.json').read_text())['manifest']
    expected = []
    args = provenance['arguments']
    for repetition in range(1, args['repeats'] + 1):
        modes = ('off', 'cpu-nvtx') if repetition <= args['instrumented_repeats'] else ('off',)
        for mode in (modes if repetition % 2 else tuple(reversed(modes))):
            for variant in provenance['orders'][(repetition - 1) % len(provenance['orders'])]:
                expected.append(f'{variant}-warm-{mode}-r{repetition}')
    regular = [r for r in results if not r['nsight_capture']]
    assert [r['case'] for r in regular] == expected
    traces = [r for r in results if r['nsight_capture']]
    assert len(traces) == (0 if args['skip_trace'] else 2)
    constants = Path(provenance['constants'])
    assert hashlib.sha256(constants.read_bytes()).hexdigest() == provenance['constants_sha256']
    reference = records(constants.parent / 'cpu-check.log', 'CPU_CHECK ')[0]
    for variant in 'OG':
        for mode in ('off', 'cpu-nvtx'):
            checked = records(root / f'{variant}-check-{mode}-r0.log', 'CHECK_RESULT ')
            assert len(checked) == 1
            observed = sorted((c for r in checked[0]['ranks'] for c in r['checks']),
                              key=lambda c: c['timestamp'])
            assert observed == reference
    caches, peaks, placement = {}, {}, {}
    runtime_by_variant = {}
    gpu_uuids = set()
    for result in results:
        assert result['events'] == stage['events'] == 10000
        assert result['timestamp_sha256'] == stage['timestamp_sha256']
        assert result['payload_bytes'] == stage['payload_bytes']
        assert (result['batch_size'], result['depth'], result['budget_gib'], result['n_bds']) == (20, 1, 8, 1)
        assert result['cache'] == 'warm' and result['node'] == provenance['node']
        assert not result['diagnostics']
        case, variant = result['case'], result['variant']
        log = root / (case + '.log')
        fractions = [records(log, prefix)[0]['fraction']
                     for prefix in ('CACHE_BEFORE ', 'CACHE_AFTER ')]
        assert all(f >= .99 for f in fractions)
        caches[case] = fractions
        runtime = records(log, 'RUNTIME ')
        assert len(runtime) == 1
        runtime = runtime[0]
        assert runtime['psana'].startswith(provenance['prefixes'][variant] + '/')
        assert (runtime['nthreads'], runtime['task_size'], runtime['compat'], runtime['gds_available']) == (8, 1048576, True, False)
        assert runtime == runtime_by_variant.setdefault(variant, runtime)
        for module in runtime['modules'].values():
            assert hashlib.sha256(Path(module['path']).read_bytes()).hexdigest() == module['sha256']
        placement[case] = {}
        for prefix in ('PLACEMENT_BEFORE ', 'PLACEMENT_AFTER '):
            entries = records(log, prefix)
            assert sorted(e['rank'] for e in entries) == [0, 1, 2]
            assert all(e['affinity'] == provenance['cpu_affinity'] for e in entries)
            assert all(e['cupy_cache_dir'] == provenance['environment']['CUPY_CACHE_DIR'] for e in entries)
            placement[case][prefix.strip()] = entries
        with (root / (case + '-gpu.csv')).open() as source:
            samples = list(csv.reader(source))
        assert samples and len({row[1].strip() for row in samples}) == 1
        gpu_uuids.update(row[1].strip() for row in samples)
        peaks[case] = max(float(row[2]) for row in samples)
    assert len(gpu_uuids) == 1
    versions = [{k: r[k] for k in ('cupy', 'cuda', 'kvikio', 'visibility')}
                for r in runtime_by_variant.values()]
    assert all(v == versions[0] for v in versions)
    for hashes in provenance['installed_hashes'].values():
        for path, digest in hashes.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    return dict(gpu_uuid=next(iter(gpu_uuids)), samples=len(results), correctness_preflights=4, node=provenance['node'],
                cache_fractions=caches, peak_device_memory_mib=peaks,
                placement=placement, runtime_by_variant=runtime_by_variant,
                timestamp_sha256=stage['timestamp_sha256'], installed_hashes_unchanged=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    (args.directory / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"AUDIT_PASS: {result['samples']} samples, four preflights, matched workload/placement, unchanged installs")
