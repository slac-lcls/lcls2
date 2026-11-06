"""Validate protected real-data cases on current master and the fresh PR build."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path('/tmp/smd-callback-pr')
NOTES = Path(__file__).resolve().parent
steps = json.loads((ROOT / 'oracle.json').read_text())
assert list(map(len, steps)) == [1000, 1000, 1000]
assert len({ts for step in steps for ts in step}) == 3000
expected = steps[2][:50]
commands = json.loads((ROOT / 'jf/commands.json').read_text())
assert [c['case'] for c in commands] == ['master-default', 'master-10000', 'fixed-default']
reports = {}
gates = {}
results = []
for command in commands:
    case = command['case']
    path = ROOT / 'jf' / case
    r = json.loads((path / 'report.json').read_text())
    reports[case] = r
    assert not command['timed_out'] and not r['blocked_mutations']
    assert r['initial_smd_n_events'] == command['setting']
    assert r['effective_smd_n_events'] == ('10000' if command['setting'] else '1000')
    assert r['datasource_class'] == 'SerialDataSource' and len(r['smd_files']) == 10
    gates[case] = {p.name: p for p in (path / 'work1/jungfrau/block_results').glob('*.npy')}
    if case == 'master-default':
        assert command['exit_code'] == 1 and r['completed'] is False
        assert r['callback_timestamps'] == r['raw_timestamps'] == []
        assert 'UnboundLocalError' in r['exception'] and 'ievt' in r['exception']
        assert not gates[case]
    else:
        assert command['exit_code'] == 0 and r['completed'] is True
        assert r['callback_timestamps'] == r['raw_timestamps'] == expected
        assert r['raw_shape'] == [32, 512, 1024] and len(gates[case]) == 2
        assert all('-g2-gate_' in name for name in gates[case])
    assert r['name_lookups'] and all(n == dict(add_shortname=False, result='jungfrau_000005')
                                    for n in r['name_lookups'])
    for name, source in r['sources'].items():
        if source['path'].endswith('.py'):
            relative = Path('psana', *name.split('.')).with_suffix('.py')
            checkout = ROOT / ('worktree' if case == 'fixed-default' else 'master')
            original = checkout / 'psana' / relative
            assert hashlib.sha256(original.read_bytes()).hexdigest() == source['sha256']
    results.append(dict(case=case, effective_smd_n_events=r['effective_smd_n_events'],
                        exit_code=command['exit_code'], raw_events=len(r['raw_timestamps']),
                        timestamps_exact=r['raw_timestamps'] == expected,
                        callback_calls=r['callback_calls'], chunks=len(r['chunks']),
                        gate_files=len(gates[case]), exception=r.get('exception')))
    shutil.copy2(path / 'report.json', NOTES / (case + '.json'))

assert all(r['smd_files'] == reports['master-default']['smd_files'] for r in reports.values())
assert reports['master-default']['sources'] == reports['master-10000']['sources']
for name in ('smdreader', 'eventbuilder', 'dgram'):
    assert all(r['sources'][name] == reports['fixed-default']['sources'][name] for r in reports.values())
comparisons = []
assert gates['master-10000'].keys() == gates['fixed-default'].keys()
for name, a_path in sorted(gates['master-10000'].items()):
    a = np.load(a_path, mmap_mode='r')
    b = np.load(gates['fixed-default'][name], mmap_mode='r')
    assert a.shape == b.shape == (32, 512, 1024) and a.dtype == b.dtype
    assert np.isfinite(a).all() and np.isfinite(b).all() and np.array_equal(a, b)
    comparisons.append(dict(file=name, shape=list(a.shape), dtype=str(a.dtype), exact_equal=True))
output = dict(cases=results, gates=comparisons, acceptance_passed=True)
(NOTES / 'jf-results.json').write_text(json.dumps(output, indent=2) + '\n')
shutil.copy2(ROOT / 'jf/commands.json', NOTES / 'jf-commands.json')
manifest = []
for path in [ROOT / 'oracle.json', *sorted((ROOT / 'jf').glob('*/report.json')),
             *sorted((ROOT / 'jf').glob('*/stdout.log')),
             *sorted((ROOT / 'jf').glob('*/work1/**/*.npy'))]:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    manifest.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest()))
(NOTES / 'jf-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(output, indent=2))
