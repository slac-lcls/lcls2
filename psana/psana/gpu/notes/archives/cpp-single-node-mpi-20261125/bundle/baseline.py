"""C++ strong-scaling pilot. One MPI rank/node, 20/40 pinned physical-core readers."""
import argparse
import cxi
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PYTHON = '/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python'
REQUEST = 1048576


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    s = Path(path).stat()
    return {k: getattr(s, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')}


def verify_bundle():
    pins = json.loads((HERE / 'pins.json').read_text())
    for name, digest in pins.items():
        require(sha(HERE / name) == digest, 'bundle changed: ' + name)


def validate_inputs(plan):
    receipt = Path(plan['preparation_receipt'])
    require(sha(receipt) == plan['preparation_sha256'], 'preparation receipt changed')
    for rows in plan['inputs'].values():
        require(len(rows) == 5, 'expected five JF files')
        for row in rows:
            p = Path(row['path'])
            require(not p.is_symlink(), 'physical input required')
            require(identity(p) == row['identity'], 'input identity changed: ' + str(p))
            native = subprocess.check_output(['/usr/bin/lfs', 'getstripe', str(p)], text=True)
            require(native == row['layout_raw'], 'input layout changed: ' + str(p))


def physical_cpus():
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
        key = tuple(int((root / f).read_text()) for f in ('physical_package_id', 'core_id'))
        cores.setdefault(key, cpu)
    return list(cores.values())


def telemetry():
    result = {'unix': time.time(), 'meminfo': Path('/proc/meminfo').read_text(), 'network': {}}
    for p in Path('/sys/class/net').glob('*/statistics/*_bytes'):
        try:
            result['network'][str(p)] = int(p.read_text())
        except OSError:
            pass
    try:
        p = subprocess.run(['/usr/sbin/lctl', 'get_param', 'llite.*.stats'], capture_output=True, text=True, timeout=8)
        result['lustre'] = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        result['lustre'] = {'unavailable': repr(exc)}
    return result


def node(mode, case_dir):
    require(socket.gethostname().startswith('nid'), 'compute node required')
    verify_bundle()
    case_dir = Path(case_dir)
    case = json.loads((case_dir / 'case.json').read_text())
    rank = int(os.environ['SLURM_PROCID'])
    host = socket.gethostname()
    paths = [r['path'] for r in case['inputs']]
    cpus = physical_cpus()[:case['threads']]
    require(len(cpus) == case['threads'], 'insufficient physical CPUs')
    if mode == 'evict':
        command = [str(HERE / 'legacy_reader'), '--evict-all', *paths]
    else:
        command = [str(HERE / 'reader'), str(case['threads']), str(REQUEST), ','.join(map(str, cpus)), '1' if case.get('probe') else '0', *paths]
    before = telemetry()
    cxi_record = None
    if mode == 'read':
        result, cxi_record = cxi.run_reader(command,
            case_dir / ('read-%d.stdout' % rank), case_dir / ('read-%d.stderr' % rank))
    else:
        p = subprocess.run(command, capture_output=True, text=True, close_fds=False)
        (case_dir / ('%s-%d.stdout' % (mode, rank))).write_text(p.stdout)
        (case_dir / ('%s-%d.stderr' % (mode, rank))).write_text(p.stderr)
        require(p.returncode == 0, 'reader failed: ' + p.stderr + p.stdout)
        result = json.loads(p.stdout)
    after = telemetry()
    save(case_dir / ('%s-%d.json' % (mode, rank)), {'host': host, 'slurm_rank': rank, 'cpus': cpus, 'command': command, 'reader': result, 'before': before, 'after': after, 'cxi': cxi_record,
        'telemetry_scope': 'entire subprocess including cache conditioning; interface counters may omit Lustre RDMA'})


def audit(case, records):
    rows = case['inputs']
    n = case['nodes']
    require(len(records) == n and len({r['host'] for r in records}) == n, 'one distinct host per rank required')
    total = 0
    ends = [[] for _ in rows]
    durations = []
    cxi_comparisons = []
    for rank, record in enumerate(records):
        r = record['reader']
        cxi_comparisons.append(cxi.audit(record['cxi'], r))
        require(record['slurm_rank'] == r['rank'] == rank and r['ranks'] == n, 'rank placement')
        require(r['state'] == 'complete' and r['threads'] == case['threads'] and r['request_bytes'] == REQUEST, 'reader controls')
        require(r['confirmed_zero_scans'] == 2 and 1 <= r['eviction_attempts'] <= 3, 'cache gate')
        require(len(r['files']) == 5 and len(r['workers']) == case['threads'], 'file/worker count')
        for got, exp in zip(r['files'], rows):
            require(got['path'] == exp['path'] and got['identity'] == exp['identity'], 'input identity')
            require(got['cold_resident_pages'] == 0, 'client pages resident')
            if case.get('probe'):
                require(got['warm_resident_pages'] > 0, 'mincore positive control')
        subtotal = 0
        per_file = case['threads'] // 5
        for i, w in enumerate(r['workers']):
            file, part = divmod(i, per_file)
            size = rows[file]['bytes']
            chunks = (size + REQUEST - 1) // REQUEST
            index, parts = rank * per_file + part, n * per_file
            begin = min(size, chunks * index // parts * REQUEST)
            end = min(size, chunks * (index + 1) // parts * REQUEST)
            require((w['file'], w['begin'], w['end']) == (file, begin, end), 'range partition')
            require(w['bytes'] == end - begin and w['test_checksum'] == 0, 'byte count or wrong binary')
            require(w['cpu'] == w['actual_start_cpu'] == w['actual_end_cpu'] == record['cpus'][i], 'affinity')
            require(0 <= w['start_seconds'] <= w['end_seconds'] <= r['seconds'], 'worker time')
            ends[file].append((begin, end))
            subtotal += w['bytes']
        require(subtotal == r['bytes'], 'rank byte sum')
        require(math.isfinite(r['seconds']) and r['seconds'] > 0, 'duration')
        require(r['seconds'] == max(w['end_seconds'] for w in r['workers']), 'last-worker timing')
        total += subtotal
        durations.append(r['seconds'])
    for row, spans in zip(rows, ends):
        previous = 0
        for begin, end in sorted(spans):
            require(begin == previous and end >= begin, 'overlap or hole')
            previous = end
        require(previous == row['bytes'], 'incomplete coverage')
    require(total == sum(r['bytes'] for r in rows), 'global byte sum')
    elapsed = max(durations)
    for r in records:
        require(r['reader']['global_bytes'] == total and r['reader']['global_seconds'] == elapsed, 'MPI reductions')
    rx = sum(r['totals'][cxi.COUNTERS[0]] for r in cxi_comparisons)
    tx = sum(r['totals'][cxi.COUNTERS[1]] for r in cxi_comparisons)
    counter_seconds = max(r['seconds'] for r in cxi_comparisons)
    return {'cxi': {'rx_bytes': rx, 'tx_bytes': tx, 'rx_over_read': rx/total,
                    'max_rank_counter_seconds': counter_seconds,
                    'aggregate_rx_gib_s': rx/counter_seconds/2**30,
                    'rate_definition': 'summed NIC RX bytes / max rank counter window; includes barrier/sampling margin',
                    'per_rank': cxi_comparisons}, 'bytes': total, 'seconds': elapsed, 'aggregate_gib_s': total / elapsed / 2**30,
            'aggregate_gb_s': total / elapsed / 1e9, 'per_node_gib_s': total / elapsed / 2**30 / n,
            'rank_seconds': durations, 'audit': 'pass'}


def schedule(block=None):
    # Adjacent matched node subsets, cyclic node order balanced across repeats.
    rng = random.Random(20261125)
    result = []
    for repeat in range(3):
        settings = list(itertools.product([1, 2, 4, 8, 16, 32], [20, 40]))
        rng.shuffle(settings)
        for count, threads in settings:
            for nodes in ([1, 2, 4], [2, 4, 1], [4, 1, 2])[repeat]:
                result.append({'id': '%03d-r%d-c%02d-t%d-n%d' % (len(result), repeat, count, threads, nodes), 'repeat': repeat, 'count': count, 'threads': threads, 'nodes': nodes})
    require(block is None or block in [0, 1, 2], "invalid repeat block")
    return result if block is None else [c for c in result if c["repeat"] == block]


def step(mode, path, nodes, hosts, timeout):
    command = ['srun', '--mpi=cray_shasta' if mode == 'read' else '--mpi=none', '--exact', '--exclusive',
               '--nodes='+str(nodes), '--ntasks='+str(nodes), '--ntasks-per-node=1', '--cpus-per-task=128',
               '--cpu-bind=none', '--distribution=block:block', '--gpus-per-node=4', '--gpu-bind=none',
               '--nodelist='+','.join(hosts[:nodes]), '--kill-on-bad-exit=1',
               PYTHON, str(HERE / 'campaign.py'), mode, str(path)]
    save(path / (mode + '-command.json'), command)
    with (path / (mode + '-srun.log')).open('x') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def execute(case, output, hosts):
    path = output / case['id']
    path.mkdir()
    save(path / 'case.json', case)
    # Evict on all allocation clients, including clients inactive in this measurement.
    step('evict', path, 4, hosts, 300)
    for rank in range(4):
        rec = json.loads((path / ('evict-%d.json' % rank)).read_text())
        require(rec['host'] == hosts[rank], 'eviction placement')
        r = rec['reader']
        require(r['state'] == 'complete' and r['confirmed_zero_scans'] == 2, 'eviction failed')
        for got, exp in zip(r['files'], case['inputs']):
            require(got['path'] == exp['path'] and got['identity'] == exp['identity'] and got['resident_pages'] == 0, 'eviction identity/cache')
    step('read', path, case['nodes'], hosts, 900)
    records = [json.loads((path / ('read-%d.json' % rank)).read_text()) for rank in range(case['nodes'])]
    require([r['host'] for r in records] == hosts[:case['nodes']], 'unexpected measurement hosts')
    summary = dict(case)
    summary.pop('inputs')
    summary.update(audit(case, records))
    save(path / 'accepted.json', summary)
    return summary


def run(block, attempt):
    verify_bundle()
    plan = json.loads((HERE / 'plan.json').read_text())
    require(os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_JOB_NUM_NODES']) == 4, 'four-node Slurm allocation required')
    validate_inputs(plan)
    require(block in plan['blocks'] and re.fullmatch(r'attempt[0-9]{2,}', attempt), 'invalid block/attempt')
    output = Path(plan['output_root']) / ('repeat-%d' % block) / attempt
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'plan.json', plan)
    save(output / 'pins.json', json.loads((HERE / 'pins.json').read_text()))
    hosts = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], text=True).split()
    require(len(hosts) == 4 and len(set(hosts)) == 4, 'four unique hosts')
    state = {'state': 'running', 'job': os.environ['SLURM_JOB_ID'], 'hosts': hosts, 'started_unix': time.time(), 'completed': [], 'case_count': 36, 'repeat_block': block, 'cxi_required': True,
             'scope': 'fixed 400k dataset strong scaling; local client page cache cold, backend caches uncontrolled', 'timing': 'MPI barrier, CXI initial snapshots, second MPI barrier, thread release; maximum rank read duration; excludes cache conditioning'}
    save(output / 'status.json', state)
    def stop(signum, frame):
        raise RuntimeError('received signal %d' % signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        # Real four-node MPI smoke on tiny fully written files, including partial tails and empty worker spans.
        fixtures = output / 'fixtures'
        fixtures.mkdir()
        tiny = []
        for i in range(5):
            p = fixtures / ('file%d' % i)
            with p.open('xb') as f:
                f.write(bytes([i + 1]) * (REQUEST * (i + 1) + 17))
                f.flush()
                os.fsync(f.fileno())
            tiny.append({'path': str(p), 'identity': identity(p), 'bytes': p.stat().st_size})
        for n in [1, 2, 4]:
            execute({'id': 'smoke-n%d' % n, 'nodes': n, 'threads': 20, 'inputs': tiny, 'probe': True}, output, hosts)
        state['mpi_smoke'] = 'passed at 1,2,4 nodes'
        # Positive mincore controls and real-input baseline before measured randomized blocks.
        for threads in [20, 40]:
            execute({'id': 'pilot-c02-t%d' % threads, 'nodes': 1, 'threads': threads, 'inputs': plan['inputs']['2'], 'probe': True}, output, hosts)
        save(output / 'schedule.json', schedule(block))
        for case in schedule(block):
            if time.time() - state['started_unix'] > 3.7 * 3600:
                raise RuntimeError('campaign walltime budget reached; preserve partial results')
            case['inputs'] = plan['inputs'][str(case['count'])]
            state.update(current_case=case['id'], updated_unix=time.time())
            save(output / 'status.json', state)
            result = execute(case, output, hosts)
            state['completed'].append(result)
            state['updated_unix'] = time.time()
            save(output / 'status.json', state)
            print(json.dumps(result), flush=True)
        validate_inputs(plan)
        state.update(state='complete', finished_unix=time.time())
        save(output / 'status.json', state)
        save(output / 'complete.json', state)
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), updated_unix=time.time())
        save(output / 'status.json', state)
        raise


if __name__ == '__main__':
    if sys.argv[1] in ['evict', 'read']:
        node(sys.argv[1], sys.argv[2])
    elif sys.argv[1] == 'preflight':
        verify_bundle()
        validate_inputs(json.loads((HERE / 'plan.json').read_text()))
        print('Pinned bundle and all 30 physical input identities/layouts verified')
    elif sys.argv[1] == 'run':
        parser = argparse.ArgumentParser()
        parser.add_argument('--block', type=int, choices=[0, 1, 2], required=True)
        parser.add_argument('--attempt', default='attempt01')
        args = parser.parse_args(sys.argv[2:])
        run(args.block, args.attempt)
    else:
        raise ValueError('unknown mode')
