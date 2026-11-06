"""Check this experiment's expected negative control and two successful cases."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path('/tmp/jf-exact-command')
NOTES = Path(__file__).resolve().parent
ORACLE = Path('/tmp/jf-callback-ab/oracle.json')
steps = json.loads(ORACLE.read_text())
assert [len(step) for step in steps] == [1000, 1000, 1000]
assert len({ts for step in steps for ts in step}) == 3000
expected = steps[2][:50]
commands = json.loads((ROOT / 'commands.json').read_text())
assert [c['case'] for c in commands] == ['unfixed-default', 'unfixed-10000', 'fixed-default']
results = []
reports = {}
gates = {}
for command in commands:
    case = command['case']
    path = ROOT / case
    report = json.loads((path / 'report.json').read_text())
    reports[case] = report
    assert not command['timed_out']
    assert report['initial_smd_n_events'] == command['initial_smd_n_events']
    assert report['effective_smd_n_events'] == ('10000' if case == 'unfixed-10000' else '1000')
    assert report['datasource_class'] == 'SerialDataSource'
    assert len(report['smd_files']) == 10
    assert report['blocked_mutations'] == []
    gates[case] = {p.name: p for p in (path / 'work1/jungfrau/block_results').glob('*.npy')}
    failed = case == 'unfixed-default'
    if failed:
        assert command['exit_code'] == 1 and report['completed'] is False
        assert report['callback_timestamps'] == report['raw_timestamps'] == []
        assert 'UnboundLocalError' in report['exception'] and 'ievt' in report['exception']
        assert not gates[case]
    else:
        assert command['exit_code'] == 0 and report['completed'] is True
        assert report['callback_timestamps'] == report['raw_timestamps'] == expected
        assert report['raw_shape'] == [32, 512, 1024]
        assert len(gates[case]) == 2
        assert all('-g2-gate_' in name for name in gates[case])
        assert report['name_lookups']
        assert all(n == dict(add_shortname=False, result='jungfrau_000005')
                   for n in report['name_lookups'])
    results.append(dict(case=case, exit_code=command['exit_code'],
                        initial_smd_n_events=report['initial_smd_n_events'],
                        effective_smd_n_events=report['effective_smd_n_events'],
                        chunks=len(report['chunks']), callback_calls=report['callback_calls'],
                        selected_events=len(report['callback_timestamps']),
                        raw_events=len(report['raw_timestamps']),
                        timestamps_exact=report['raw_timestamps'] == expected,
                        raw_shape=report.get('raw_shape'), gate_files=len(gates[case]),
                        exception=report.get('exception'), seconds=command['seconds']))

assert reports['unfixed-default']['sources'] == reports['unfixed-10000']['sources']
assert all(r['smd_files'] == reports['unfixed-default']['smd_files'] for r in reports.values())
assert [(r['chunks'], r['callback_calls']) for r in results] == [(4, 4), (1, 1), (3, 1)]
assert {'psexp.callback_iterator', 'psexp.callback_stream'} <= set(reports['fixed-default']['sources'])
for module in ('smdreader', 'eventbuilder', 'dgram'):
    assert reports['fixed-default']['sources'][module] == reports['unfixed-default']['sources'][module]
for case, report in reports.items():
    checkout = ('/tmp/jf-exact-command/fixed/psana/psana/' if case == 'fixed-default'
                else '/tmp/jf-callback-ab/unfixed/psana/psana/')
    for name, source in report['sources'].items():
        if source['path'].endswith('.py'):
            assert source['path'].startswith(checkout), (case, name, source)

left = gates['unfixed-10000']
right = gates['fixed-default']
assert set(left) == set(right)
comparisons = []
for name in sorted(left):
    a = np.load(left[name], mmap_mode='r')
    b = np.load(right[name], mmap_mode='r')
    assert a.shape == b.shape == (32, 512, 1024)
    assert a.dtype == b.dtype and np.isfinite(a).all() and np.isfinite(b).all()
    equal = bool(np.array_equal(a, b))
    assert equal
    comparisons.append(dict(file=name, shape=list(a.shape), dtype=str(a.dtype),
                            exact_equal=equal, unequal_pixels=int(np.count_nonzero(a != b))))
output = dict(cases=results, gate_comparisons=comparisons, acceptance_passed=True)
(NOTES / 'results.json').write_text(json.dumps(output, indent=2) + '\n')
shutil.copy2(ROOT / 'commands.json', NOTES / 'commands.json')
for case in reports:
    shutil.copy2(ROOT / case / 'report.json', NOTES / (case + '.json'))

manifest = []
for path in [ORACLE, ROOT / 'commands.json', *sorted(ROOT.glob('*/report.json')),
             *sorted(ROOT.glob('*/stdout.log')), *sorted(ROOT.glob('*/work1/**/*.npy'))]:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    manifest.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest()))
(NOTES / 'artifact-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(output, indent=2))
