"""Verify the preserved bundle without Git, network, psana, CUDA or scratch."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


PR_RUNTIME = '67c327a148a08b2aaf972e67e1e02e367c2b298f'
EVENT_BYTES = 3358200


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def verify_inventory(root):
    manifest = read_json(root / 'manifest.json')
    require(manifest['schema_version'] == 1, 'Unsupported manifest schema')
    require(bool(manifest['files']), 'Empty manifest')
    paths = set()
    total_bytes = 0
    for entry in manifest['files']:
        relative = Path(entry['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Unsafe manifest path')
        require(relative.as_posix() not in paths, 'Duplicate manifest path')
        paths.add(relative.as_posix())
        path = root / relative
        path.resolve().relative_to(root.resolve())
        require(not path.is_symlink(), 'Symlinked artifact')
        data = path.read_bytes()
        require(len(data) == entry['bytes'], 'Size mismatch: ' + str(relative))
        require(hashlib.sha256(data).hexdigest() == entry['sha256'],
                'SHA256 mismatch: ' + str(relative))
        header = b'blob ' + str(len(data)).encode() + b'\0'
        require(hashlib.sha1(header + data).hexdigest() == entry['git_blob_sha1'],
                'Git blob mismatch: ' + str(relative))
        total_bytes += len(data)
    actual = {path.relative_to(root).as_posix()
              for campaign in ('perlmutter', 'ada-10k', 'ada-100k')
              for path in (root / campaign).rglob('*')
              if path.is_file() and '__pycache__' not in path.parts}
    require(paths == actual, 'Archived file inventory mismatch')
    return dict(files=len(paths), bytes=total_bytes)


def verify_gpu_receipt(receipt):
    require(receipt['exitcode'] == 0 and receipt['passed'] == 31
            and receipt['failed'] == receipt['skipped'] == 0, 'GPU test receipt')


def verify_perlmutter(root):
    receipts = root / 'perlmutter/receipts/native/attempt03'
    complete = read_json(receipts / 'complete.json')
    expected = read_json(receipts / 'summary.json')
    tests = read_json(receipts / 'gpu-tests.json')
    verify_gpu_receipt(tests)
    require(complete['tests'] == tests and complete['status'] == 'pass'
            and complete['input_unchanged'], 'Perlmutter completion receipt')
    require(len(complete['cases']) == 12, 'Perlmutter case count')
    observed = dict(clean=[], timing=[])
    for variant in ('old', 'new'):
        for streams in (1, 2):
            cases = [case for case in complete['cases']
                     if case['mode'] == 'clean' and case['variant'] == variant
                     and case['streams'] == streams]
            require(len(cases) == 2, 'Perlmutter clean repetitions')
            seconds = [case['seconds'] for case in cases]
            rates = [case['gib_s'] for case in cases]
            for duration, rate in zip(seconds, rates):
                require(duration > 0 and math.isclose(
                    rate, EVENT_BYTES * 10000 / 2**30 / duration), 'Perlmutter rate')
            observed['clean'].append(dict(
                variant=variant, streams=streams, seconds=seconds, gib_s=rates,
                median_seconds=median(seconds), median_gib_s=median(rates)))
    for case in complete['cases']:
        if case['mode'] != 'timing':
            continue
        phases = read_json(receipts / (case['name'] + '.json.phases.json'))
        totals = {}
        for name, values in phases['phases'].items():
            phase = name.split('/', 1)[1]
            totals[phase] = totals.get(phase, 0) + values['self_ns'] / 1e9
        observed['timing'].append(dict(
            name=case['name'], variant=case['variant'], streams=case['streams'],
            seconds=case['seconds'], exclusive_seconds=totals,
            counters=phases['counters']))
    require(observed == expected, 'Perlmutter summary mismatch')
    return dict(clean_medians=observed['clean'], timing_cases=len(observed['timing']))


def verify_ada(root, campaign, events, job):
    receipts = root / campaign / 'receipts'
    result = read_json(receipts / 'results.json')
    audit = read_json(receipts / 'audit.json')
    complete = read_json(receipts / 'complete.json')
    runtime = read_json(receipts / 'runtime-evidence.json')
    verify_gpu_receipt(read_json(receipts / 'gpu-tests.json'))
    require(result['job'] == job and result['status'] ==
            'accepted-audit-and-independent-review-pass', 'Ada result identity')
    require(audit['status'] == complete['status'] == 'pass'
            and read_json(receipts / 'cleanup.json')['removed'], 'Ada completion')
    require(runtime['pr_sha'] == PR_RUNTIME and runtime['source_is_canonical']
            and runtime['native_sources_checked'] == 225, 'Ada runtime identity')
    modules = {name: item['observed'] for name, item in runtime['modules'].items()}
    require(all(item['observed']['sha256'] == item['expected_sha256']
                for item in runtime['modules'].values()), 'Ada module hashes')
    matrix = [('preflight-stream1', 1, 1000), ('preflight-stream2', 2, 1000)]
    matrix += [('%02d-stream%d' % (index, streams), streams, events)
               for index, streams in enumerate((1, 2, 2, 1))]
    for record in (result, audit, complete):
        require([(case['name'], case['streams'], case['events'])
                 for case in record['cases']] == matrix, 'Ada case matrix')
    for case, checked in zip(result['cases'], audit['cases']):
        count = case['events']
        io = case['io']
        require(case['reader_calls'] == io['requests'] == io['completed_requests']
                == 5 * count and io['requested_bytes'] == io['completed_bytes']
                == EVENT_BYTES * count, 'Ada read accounting')
        require(case['callbacks'] == count // 100, 'Ada callback count')
        require(case['runtime_files'] == modules, 'Ada case runtime')
        require(case['seconds'] > 0 and math.isclose(
            case['gib_s'], EVENT_BYTES * count / 2**30 / case['seconds']), 'Ada rate')
        for field in ('name', 'events', 'streams', 'seconds', 'gib_s', 'hz',
                      'reader_calls', 'callbacks'):
            require(case[field] == checked[field], 'Ada audit mismatch: ' + field)
    require([summary['streams'] for summary in result['summary']] == [1, 2],
            'Ada summary matrix')
    for summary in result['summary']:
        cases = [case for case in result['cases']
                 if case['events'] == events and case['streams'] == summary['streams']]
        require(len(cases) == 2, 'Ada full repetitions')
        for field, target in (('gib_s', 'median_gib_s'), ('hz', 'median_events_s'),
                              ('seconds', 'median_seconds')):
            require(math.isclose(median(case[field] for case in cases), summary[target]),
                    'Ada median mismatch: ' + target)
        require(summary['gib_s_range'] == [min(case['gib_s'] for case in cases),
                                          max(case['gib_s'] for case in cases)],
                'Ada rate range')
    return dict(job=job, medians=result['summary'])


def verify_bundle(root):
    root = Path(root)
    inventory = verify_inventory(root)
    return dict(status='pass', inventory=inventory,
                perlmutter=verify_perlmutter(root),
                ada_10k=verify_ada(root, 'ada-10k', 10000, '44185752'),
                ada_100k=verify_ada(root, 'ada-100k', 100000, '44189224'),
                scope='Archived integrity and summary consistency only; no raw-data audit or GPU run')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    try:
        print(json.dumps(verify_bundle(arguments.root), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, 'Evidence verification failed: %s\n' % error)
