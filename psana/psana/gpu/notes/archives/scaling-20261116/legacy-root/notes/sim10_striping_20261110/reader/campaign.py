"""Run fail-closed local-page-cache-cold pilot or randomized screening."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import time
import bundle as snapshot

ROOT = Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-striping-20261110')
DATA = ROOT / 'attempt01/datasets'
COUNTS = [1, 2, 4, 8, 16, 32]
THREADS = [5, 20, 40]
REQUEST = 1048576
SEED = 20261111
STAGING_SHA256 = 'e8e45990b6ef40ac7479edd5ecf232bbe8b1ddba4fe05d51de07efc83a5171c4'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def identity(path):
    s = path.stat()
    return {k: getattr(s, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')}


def physical_cpus():
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = tuple(int((topology / name).read_text()) for name in ('physical_package_id', 'core_id'))
        cores.setdefault(key, cpu)
    result = []
    for key, cpu in sorted(cores.items()):
        nodes = sorted(int(p.name[4:]) for p in Path(f'/sys/devices/system/cpu/cpu{cpu}').glob('node[0-9]*'))
        require(len(nodes) == 1, 'cannot determine CPU NUMA node')
        result.append({'package':key[0], 'core':key[1], 'cpu':cpu, 'numa_node':nodes[0]})
    require(len(result) >= 40, 'need at least40 distinct physical cores')
    return result


def schedule(mode):
    if mode == 'pilot':
        return [{'id': 'pilot-c01-t05', 'count': 1, 'threads': 5, 'block': -1},
                {'id': 'pilot-c32-t40', 'count': 32, 'threads': 40, 'block': -1}]
    rng = random.Random(SEED)
    result = []
    for block in range(3):
        cases = list(itertools.product(COUNTS, THREADS))
        rng.shuffle(cases)
        for count, threads in cases:
            result.append({'id': f'{len(result):02d}-b{block}-c{count:02d}-t{threads:02d}',
                           'count': count, 'threads': threads, 'block': block})
    return result


def audit_reader(r, case, expected, cpus, warm):
    require(r['state'] == 'complete', 'reader not complete')
    require(r['threads'] == case['threads'] and r['request_bytes'] == REQUEST, 'wrong fixed controls')
    require(r['warm_probe'] is warm, 'wrong warm probe mode')
    require(r['confirmed_zero_scans'] == 2, 'missing repeat cold scan')
    require(1 <= r['eviction_attempts'] <= 3 and r['eviction_seconds'] >= 0, 'invalid eviction')
    require(len(r['files']) == 5 and len(r['workers']) == case['threads'], 'wrong file/worker counts')
    require(r['page_size'] > 0 and (r['page_size'] & (r['page_size'] - 1)) == 0, 'invalid page size')
    for observed, exp in zip(r['files'], expected):
        require(observed['path'] == exp['path'] and observed['identity'] == exp['identity'], 'input identity mismatch')
        require(observed['cold_resident_pages'] == 0, 'cache not cold')
        require(observed['pages'] == (exp['bytes'] + r['page_size'] - 1) // r['page_size'], 'page accounting')
        if warm:
            require(observed['warm_resident_pages'] > 0, 'mincore positive control failed')
    per_file = case['threads'] // 5
    total = 0
    for i, w in enumerate(r['workers']):
        f, part = divmod(i, per_file)
        size = expected[f]['bytes']
        chunks = (size + REQUEST - 1) // REQUEST
        begin = min(size, (chunks * part // per_file) * REQUEST)
        end = min(size, (chunks * (part + 1) // per_file) * REQUEST)
        require(w['index'] == i and w['file'] == f and w['begin'] == begin and w['end'] == end, 'range partition mismatch')
        require(w['bytes'] == end - begin, 'worker byte count')
        require(w['cpu'] == w['actual_start_cpu'] == w['actual_end_cpu'] == cpus[i], 'CPU affinity mismatch')
        require(0 <= w['start_seconds'] <= w['end_seconds'] <= r['seconds'], 'invalid worker timing')
        require(w['cpu_seconds'] >= 0 and w['calls'] >= math.ceil(w['bytes'] / REQUEST), 'invalid worker accounting')
        total += w['bytes']
    require(total == r['bytes'] == sum(e['bytes'] for e in expected), 'total byte mismatch')
    require(math.isfinite(r['seconds']) and r['seconds'] > 0, 'invalid duration')
    require(r['seconds'] == max(w['end_seconds'] for w in r['workers']), 'timing excludes final worker')
    require(math.isclose(r['gib_per_second'], total / r['seconds'] / 2**30, rel_tol=1e-12), 'throughput arithmetic')


def validate_inputs(accepted):
    require(accepted['state'] == 'complete' and [s['count'] for s in accepted['sets']] == COUNTS, 'staging incomplete')
    inputs = {}
    for group in accepted['sets']:
        count = group['count']
        rows = sorted(group['verified'], key=lambda r: r['path'])
        require(len(rows) == 5, 'five inputs required')
        copies = {r['path']: r for r in group['copies']}
        for i, r in enumerate(rows):
            path = Path(r['path'])
            require(path == DATA / f'count-{count:02d}' / f'mfx101555026-r0010-s00{i+5}-c000.xtc2', 'wrong input path')
            require(not path.is_symlink() and identity(path) == r['identity'], 'staged input changed')
            observed = subprocess.check_output(['/usr/bin/lfs', 'getstripe', str(path)], text=True)
            require(observed == copies[str(path)]['layout']['raw'], 'staged layout changed')
        inputs[count] = rows
    return inputs



def audit_global_cache(record, expected):
    require(record['state'] == 'complete' and record['confirmed_zero_scans'] == 2, 'global cache gate incomplete')
    require(len(record['files']) == len(expected), 'global cache file count')
    for observed, row in zip(record['files'], expected):
        require(observed['path'] == row['path'] and observed['identity'] == row['identity'], 'global cache input mismatch')
        require(observed['resident_pages'] == 0, 'global cache not empty')


def memory_state():
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ['MemFree','MemAvailable','Cached','Buffers','Dirty','Writeback','Slab']:
            amount, unit = value.split()
            require(unit == 'kB', 'unexpected meminfo units')
            fields[name] = int(amount)
    try:
        r = subprocess.run(['/usr/sbin/lctl','get_param','llite.*.max_cached_mb'], capture_output=True, text=True, timeout=10)
        limit = {'available': r.returncode == 0, 'stdout': r.stdout, 'stderr': r.stderr, 'returncode': r.returncode}
    except (OSError, subprocess.TimeoutExpired) as exc:
        limit = {'available': False, 'error': repr(exc)}
    return {'meminfo_kib': fields, 'lustre_max_cached_mb': limit}


def audit_memory(record):
    require(set(record['meminfo_kib']) == {'MemFree','MemAvailable','Cached','Buffers','Dirty','Writeback','Slab'}, 'missing memory fields')
    require(all(type(v) is int and v >= 0 for v in record['meminfo_kib'].values()), 'invalid memory observation')
    require(type(record['lustre_max_cached_mb']['available']) is bool, 'missing Lustre cache-limit availability')


def run(args):
    require(os.environ.get('SLURM_JOB_ID'), 'Slurm compute allocation required')
    attempt = args.attempt.resolve(strict=True)
    require(ROOT in attempt.parents and attempt.name.startswith('reader-'), 'wrong attempt root')
    output = attempt / 'results'
    output.mkdir()
    bundle = attempt / 'bundle'
    provenance = snapshot.verify(attempt)
    accepted = json.loads((bundle / 'staging-complete.json').read_text())
    require(sha(bundle / 'staging-complete.json') == STAGING_SHA256, 'accepted staging receipt changed')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(DATA)], text=True))
    require(mount['filesystems'][0]['fstype'] == 'lustre', 'inputs must be on Lustre')
    save(output / 'mount.json', mount)
    inputs = validate_inputs(accepted)
    topology = physical_cpus()
    cpus = [r['cpu'] for r in topology[:40]]
    cases = schedule(args.mode)
    save(output / 'schedule.json', {'mode': args.mode, 'seed': SEED, 'cases': cases})
    save(output / 'cpu-map.json', topology)
    status = {'state': 'running', 'mode': args.mode, 'job': os.environ['SLURM_JOB_ID'],
              'host': socket.gethostname(), 'started_unix': time.time(), 'completed': [],
              'case_count': len(cases), 'bundle_provenance': provenance,
              'cache_scope': 'local page cache cold; backend caches uncontrolled; remote bytes unverified'}
    save(output / 'status.json', status)
    try:
        for case in cases:
            start = time.time()
            all_inputs = [r for count in COUNTS for r in inputs[count]]
            eviction_path = output / (case['id'] + '.eviction.json')
            eviction_command = [str(bundle / 'reader'), '--evict-all', *[r['path'] for r in all_inputs]]
            with eviction_path.open('x') as out:
                eviction = subprocess.run(eviction_command, stdout=out, stderr=subprocess.PIPE, text=True, timeout=300)
            require(eviction.returncode == 0, 'global eviction failed: '+eviction.stderr)
            global_cache = json.loads(eviction_path.read_text())
            audit_global_cache(global_cache, all_inputs)
            memory_before = memory_state()
            paths = [r['path'] for r in inputs[case['count']]]
            selected = cpus[:case['threads']]
            warm = args.mode == 'pilot'
            command = [str(bundle / 'reader'), str(case['threads']), str(REQUEST),
                       ','.join(map(str, selected)), '1' if warm else '0', *paths]
            stdout = output / (case['id'] + '.stdout.json')
            stderr = output / (case['id'] + '.stderr.txt')
            status.update(current_case=case, updated_unix=start)
            save(output / 'status.json', status)
            with stdout.open('x') as out, stderr.open('x') as err:
                completed = subprocess.run(command, stdout=out, stderr=err, timeout=900)
            require(completed.returncode == 0, f'reader failed in {case["id"]}; retain raw logs')
            measured = json.loads(stdout.read_text())
            audit_reader(measured, case, inputs[case['count']], selected, warm)
            require(all(w['test_checksum'] == 0 for w in measured['workers']), 'test binary used for measurement')
            for r in inputs[case['count']]:
                require(identity(Path(r['path'])) == r['identity'], 'input changed after trial')
            receipt = {'case': case, 'reader': measured, 'command': command,
                       'started_unix': start, 'finished_unix': time.time(), 'raw_sha256': sha(stdout),
                       'reader_sha256': provenance['files']['reader']['sha256'], 'audit': 'pass',
                       'global_eviction': global_cache, 'eviction_sha256': sha(eviction_path),
                       'memory_before': memory_before, 'memory_after': memory_state()}
            receipt['finished_unix'] = time.time()
            save(output / (case['id'] + '.json'), receipt)
            status['completed'].append({'id': case['id'], 'count': case['count'], 'threads': case['threads'],
                                        'block': case['block'], 'gib_per_second': measured['gib_per_second'],
                                        'seconds': measured['seconds'], 'wall_seconds': receipt['finished_unix'] - start})
            status.update(updated_unix=time.time())
            save(output / 'status.json', status)
            print(json.dumps(status['completed'][-1]), flush=True)
        validate_inputs(accepted)
        status.update(state='complete', finished_unix=time.time(), input_stat_and_layout_after='pass')
        save(output / 'status.json', status)
        save(output / 'complete.json', status)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), updated_unix=time.time())
        save(output / 'status.json', status)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['pilot', 'sweep'], required=True)
    p.add_argument('--attempt', type=Path, required=True)
    run(p.parse_args())
