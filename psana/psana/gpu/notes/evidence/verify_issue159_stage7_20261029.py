"""Read-only reconciliation of preserved issue159 evidence; writes one JSON report.

Run from this checkout: python3 PATH/verify_issue159_stage7_20261029.py OUTPUT.json
Requires the SDF scratch artifacts named in the committed campaign evidence.
This verifies selected result artifacts, not the complete original build manifests.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics as stats
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
NOTES = Path(__file__).resolve().parent
BASE = '9706dfeeb'
B = '405f868ed380c78b6009da3001f4535cb9ae647b'
PM = Path('/sdf/scratch/users/m/monarin/gpu-validation/issue159-stage7-20261029/artifacts/perlmutter-report.md')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args]).decode().strip()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), (a, b)


def summary(values):
    return dict(n=len(values), mean=stats.mean(values), sample_sd=stats.stdev(values))


verified = {}


def verify(path, expected):
    actual = digest(path)
    assert actual == expected, str(path)
    verified[str(path)] = actual


cold = read(NOTES / 'issue159_pr246_cold_20261028.json')
field = read(NOTES / 'issue159_pr246_fields_20261028.json')
changes = git('diff', '--name-only', B, BASE).splitlines()
allowed_prefixes = ('psana/psana/gpu/notes/', 'psana/psana/gpu/docs/', 'psana/psana/tests/')
allowed_script = 'psana/psana/gpu/scripts/bench_field_retirement.py'
assert all(p.startswith(allowed_prefixes) or p == allowed_script for p in changes)
# The tree diff above shows no native-source, build-file or dependency-pin change.
# Independently check production GPU Python blobs against the measured installation.
install = Path(cold['builds']['builds']['B']['install']) / 'lib/python3.11/site-packages/psana/gpu'
gpu_blobs = {}
for path in git('ls-tree', '-r', '--name-only', BASE, 'psana/psana/gpu').splitlines():
    relative = Path(path).relative_to('psana/psana/gpu')
    if path.endswith('.py') and relative.parts[0] not in ('notes', 'docs', 'scripts'):
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE + ':' + path])
        expected = hashlib.sha256(blob).hexdigest()
        verify(install / relative, expected)
        gpu_blobs[path] = expected
assert gpu_blobs
for path, expected in cold['result_artifact_sha256'].items():
    verify(path, expected)
for relative, expected in field['results']['artifacts_sha256'].items():
    verify(Path(field['artifact_root']) / relative, expected)
verify(Path(cold['artifact_root']) / 'screen/hashes.json', cold['manifest_sha256'])
verify(Path(field['artifact_root']) / 'hashes.json', field['manifest_sha256'])
verify(PM, '16e8c46fcc814941019e03f8f4959123800daf5e95398460d137fc8de9a53f22')

raw = read(Path(cold['artifact_root']) / 'screen/job-42255812/results.json')
timed = [r for r in raw if not r['diagnostic']]
assert len(timed) == 16 and len(raw) == 20
for row in timed:
    assert row['cache'] == 'cold' and row['events'] == row['unique_timestamps'] == 10000
    assert row['nbds'] == 4 and row['ngpus'] == 1
    assert row['commit'] == cold['builds']['builds'][row['variant'][0]]['commit']
    assert all(f['resident_fraction'] <= .01 for f in row['before']['files'])
    close(row['events_per_s'], 10000 / row['loop_s'])
    close(row['loop_s'], max(r['loop_s'] for r in row['ranks'] if r['is_bd']))
    saved = next(s for s in cold['results']['samples'] if (s['variant'], s['bulk'], s['repetition']) == (row['variant'], row['bulk'], row['repetition']))
    assert saved['log_sha256'] == row['log_sha256']
    close(saved['loop_s'], row['loop_s'])

cold_stats = []
for bulk in ('off', 'on'):
    saved = next(s for s in cold['results']['summary'] if s['bulk'] == bulk)
    result = dict(bulk=bulk, variants={})
    for arm in ('A0', 'B0'):
        rows = sorted((r for r in timed if r['bulk'] == bulk and r['variant'] == arm), key=lambda r: r['repetition'])
        assert len(rows) == 4
        result['variants'][arm] = {key: summary([r[key] for r in rows]) for key in ('loop_s', 'events_per_s')}
        for key, value in result['variants'][arm].items():
            close(value['mean'], saved['variant_statistics'][arm][key]['mean'])
            close(value['sample_sd'], saved['variant_statistics'][arm][key]['std'])
    pairs = []
    for rep in range(1, 5):
        pair = {r['variant']: r['loop_s'] for r in timed if r['bulk'] == bulk and r['repetition'] == rep}
        change = 100 * (pair['B0'] / pair['A0'] - 1)
        close(change, next(p['loop_change_pct'] for p in cold['results']['pairs'] if p['bulk'] == bulk and p['repetition'] == rep))
        pairs.append(change)
    result['paired_loop_changes_pct'] = pairs
    result['median_paired_loop_change_pct'] = stats.median(pairs)
    result['mean_loop_change_pct'] = 100 * (result['variants']['B0']['loop_s']['mean'] / result['variants']['A0']['loop_s']['mean'] - 1)
    close(result['mean_loop_change_pct'], saved['mean_loop_change_pct'])
    close(result['median_paired_loop_change_pct'], saved['median_paired_loop_change_pct'])
    cold_stats.append(result)

job = Path(field['artifact_root']) / 'job-42263681'
access = {}
retire = {}
for arm in ('A', 'B'):
    for rep in range(1, 5):
        for case in read(job / ('access-%s-r%d.json' % (arm, rep)))['results']:
            shape = 'scalar' if case['shape'] == [] else '4 MiB'
            if 'isolated_byte_copy_us' in case:
                key = (arm, shape, False, 'copy')
                value = case['isolated_byte_copy_us']
            else:
                key = (arm, shape, case['aligned'], 'cached' if case['cached'] else 'first')
                value = case['host_us']['median']
            access.setdefault(key, []).append(value)
        for case in read(job / ('retire-%s-r%d.json' % (arm, rep)))['results']:
            key = (arm, 'scalar' if case['shape'] == [] else '4 MiB', case['aligned'], case['defer_retirement'], case['delay_ticks'])
            retire.setdefault(key, []).append(case)
for case in field['results']['access']:
    values = access[(case['arm'], case['shape'], case['aligned'], case['access'])]
    expected = case['host_us_or_isolated_copy_us']
    assert values == expected['process_medians']
    close(stats.mean(values), expected['mean'])
    close(stats.stdev(values), expected['sample_sd'])
retirement_samples = 0
pending_delayed = 0
for case in field['results']['retirement']:
    processes = retire[(case['arm'], case['shape'], case['aligned'], case['deferred'], case['delay_ticks'])]
    assert len(processes) == 4
    samples = [s for p in processes for s in p['samples']]
    assert len(samples) == case['n'] == 160
    retirement_samples += len(samples)
    if case['delay_ticks']:
        pending_delayed += sum(s['pending_at_finish_entry'] for s in samples)
    assert sum(s['pending_at_finish_entry'] for s in samples) == case['pending']
    assert sum(s['done_at_finish_return'] for s in samples) == case['complete']
    assert sum(s['released_at_finish_return'] for s in samples) == case['released']
    for key in ('finish_us', 'drain_us', 'consumer_span_us'):
        medians = [stats.median(s[key] for s in p['samples']) for p in processes]
        for actual, expected in zip(medians, case[key]['process_medians']):
            close(actual, expected)
        close(stats.mean(medians), case[key]['mean'])
        close(stats.stdev(medians), case[key]['sample_sd'])
assert retirement_samples == field['results']['retirement_samples'] == 3840
assert pending_delayed == field['results']['delayed_pending'] == 1920
for pair in field['results']['aligned_pairs']:
    a = access[('A', pair['shape'], True, pair['access'])]
    b = access[('B', pair['shape'], True, pair['access'])]
    close(stats.mean(100 * (y / x - 1) for x, y in zip(a, b)), pair['mean_paired_pct'])

micro = read(NOTES / 'issue159_stage2_optimized_20261019.json')
large = {}
for name, run in micro['microbenchmarks'].items():
    for measure in run['measurements']:
        close(stats.median(measure['gather_ms']), measure['median_gather_ms'])
        if measure['batch'] == 20:
            assert measure['payload_bytes'] == 671088640
            large[name] = measure['median_gather_ms']

output = dict(
    scope='Stage 7 evidence reconciliation; no new GPU run or runtime change',
    candidate_base=git('rev-parse', BASE), measured_candidate=B,
    runtime_equivalent=True, changed_paths_since_measured_candidate=changes,
    production_gpu_install_sha256=gpu_blobs, verified_artifact_sha256=verified,
    cold=dict(job=42255812, statistics=cold_stats, timed_samples=16, events_per_sample=10000,
              max_pre_residency_fraction=max(f['resident_fraction'] for r in timed for f in r['before']['files'])),
    sdf_fields=dict(job=42263681, access_cases_verified=len(access), retirement_cases_verified=len(retire),
                    retirement_samples=retirement_samples, delayed_pending=pending_delayed,
                    aligned_pairs=field['results']['aligned_pairs']),
    perlmutter=dict(job=60467642, report=str(PM), verification='Original report hash only; remote raw samples not independently reprocessed in Stage 7'),
    historical_large_gather=dict(job=41319573, candidate=micro['candidate_commit'], batch20_process_medians_ms=large,
                                 scope='Historical kernel-only evidence; not a current cumulative cold comparison'),
    decision='Reuse accepted evidence; independent Stage 7 review pending',
    limits=['Full original manifests were not rehashed file-by-file in Stage 7',
            'Field access host medians recomputed across processes, not from unavailable individual host timings',
            'Historical shifted pipeline remains warm, two pairs; no current cold shifted-throughput claim'])
Path(sys.argv[1]).write_text(json.dumps(output, indent=2) + '\n')
print('PASS: %d artifact hashes; %d production GPU modules; 16 cold samples; %d accessor cases; %d retirement samples' % (len(verified), len(gpu_blobs), len(access), retirement_samples))
