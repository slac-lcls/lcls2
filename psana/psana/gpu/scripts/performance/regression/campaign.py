"""Source-only PR regression controller. Native psana execution uses a frozen adapter."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import statistics
import subprocess
import time

SCHEMA = 'psana-pr-regression-v1'
PHASES = ('correctness', 'aa', 'ab', 'profile')
HEX = re.compile(r'^[0-9a-f]{64}$')
SHA = re.compile(r'^[0-9a-f]{40}$')
QUALIFICATION_RECEIPTS = dict(retention='retention_receipt', cross_site_identity='cross_site_identity_receipt',
    compute_access='compute_access_receipt', event_reference='event_reference',
    native_adapter='adapter_review_receipt', calibration_identity='calibration_manifest',
    nondegenerate_mask='nondegenerate_mask_receipt', numerical_reference='numerical_reference')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def validate(config, ready=False):
    require(config['schema'] == SCHEMA, 'unknown schema')
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]*', config['campaign_id']), 'unsafe campaign id')
    require(config['site'] in ('perlmutter', 'sdf'), 'unsupported site')
    require(type(config['pairs']) is int and 2 <= config['pairs'] <= 30 and config['pairs'] % 2 == 0,
            'use 2..30 even balanced pairs')
    require(type(config['timeout_seconds']) is int and config['timeout_seconds'] > 0, 'invalid timeout')
    require(config['cache_policy'] in ('client-cold', 'warm', 'uncontrolled'), 'invalid cache policy')
    require(0 < config['equivalence_percent'] < 100, 'invalid equivalence margin')
    require(config['dataset']['events'] > 0, 'invalid event count')
    require(config['warmup']['policy'] == 'separate-process-before-cache-conditioning', 'invalid warmup policy')
    require(config['warmup']['events'] > 0, 'warmup events required')
    require(config['cases'], 'no cases')
    names = [c['id'] for c in config['cases']]
    require(len(names) == len(set(names)), 'duplicate case')
    for case in config['cases']:
        require(re.fullmatch(r'[a-z0-9][a-z0-9-]*', case['id']), 'unsafe case id')
        require(case['workload'] in ('cpu-raw', 'gpu-raw', 'cpu-calib', 'gpu-calib'), 'unknown workload')
        require(case['iterator'] in ('events', 'steps'), 'unknown iterator')
        require(case['bds'] > 0 and case['ebs'] > 0 and case['gpus'] >= 0, 'invalid topology')
    blockers = []
    for key in ('retention', 'cross_site_identity', 'compute_access', 'event_reference', 'native_adapter'):
        if config['qualification'].get(key) is not True:
            blockers.append(key)
    if any('calib' in c['workload'] for c in config['cases']):
        for key in ('calibration_identity', 'nondegenerate_mask', 'numerical_reference'):
            if config['qualification'].get(key) is not True:
                blockers.append(key)
    if not HEX.fullmatch(config['dataset'].get('event_sha256') or ''):
        blockers.append('event_sha256')
    if not isinstance(config['dataset'].get('requested_bytes'), int) or config['dataset']['requested_bytes'] <= 0:
        blockers.append('requested_bytes')
    for role in ('baseline', 'candidate'):
        runtime = config['runtimes'][role]
        if not SHA.fullmatch(runtime.get('commit') or '') or not runtime.get('install'):
            blockers.append(role + '_runtime')
    required = {'dataset_manifest', 'event_reference', 'baseline_runtime', 'candidate_runtime', 'adapter'}
    if any('calib' in c['workload'] for c in config['cases']):
        required.add('calibration_manifest')
    required.update(receipt for flag, receipt in QUALIFICATION_RECEIPTS.items()
                    if config['qualification'].get(flag) is True)
    for key in sorted(required):
        if key not in config['artifacts']:
            blockers.append('artifact:' + key)
    for key, item in config['artifacts'].items():
        require(re.fullmatch(r'[a-z0-9_]+', key), 'unsafe artifact key')
        require(HEX.fullmatch(item['sha256']) is not None, 'invalid artifact digest')
        require(Path(item['path']).is_file() and file_hash(item['path']) == item['sha256'], 'artifact changed: ' + key)
    if not config.get('adapter_command'):
        blockers.append('adapter_command')
    else:
        require(isinstance(config['adapter_command'], list) and
                all(isinstance(s, str) and s for s in config['adapter_command']), 'command must be argv array')
        require(Path(config['adapter_command'][0]).is_absolute(), 'adapter executable must be absolute')
        require('{request}' in config['adapter_command'] and '{result}' in config['adapter_command'],
                'adapter command needs standalone {request} and {result} arguments')
        require(config['artifacts'].get('adapter', {}).get('path') in config['adapter_command'],
                'command must invoke the hashed adapter path')
    if ready:
        require(not blockers, 'qualification incomplete: ' + ', '.join(blockers))
    return blockers


def schedule(config, phase):
    require(phase in PHASES, 'unknown phase')
    result = []
    repetitions = 1 if phase in ('correctness', 'profile') else config['pairs']
    for case in config['cases']:
        for pair in range(repetitions):
            for side in (('left', 'right') if pair % 2 == 0 else ('right', 'left')):
                runtime = 'baseline' if side == 'left' or phase == 'aa' else 'candidate'
                result.append(dict(id=f'{case["id"]}-{phase}-{pair:02d}-{side}', case=case,
                                   pair=pair, side=side, runtime=runtime, phase=phase))
    return result


def prepare(source, output):
    config = read(source)
    # Relative artifact paths resolve against the submitted configuration.
    for item in config['artifacts'].values():
        item['path'] = str((Path(source).resolve().parent / item['path']).resolve())
    blockers = validate(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / 'artifacts'
    frozen.mkdir()
    for key, item in config['artifacts'].items():
        original = item['path']
        target = frozen / (key + Path(original).suffix)
        shutil.copy2(original, target)
        require(file_hash(target) == item['sha256'], 'artifact changed during freeze')
        item['path'] = str(target.resolve())
        config['adapter_command'] = [item['path'] if arg == original else arg
                                     for arg in config['adapter_command']]
    write(output / 'profile.json', config)
    write(output / 'plan.json', dict(schema=SCHEMA, profile_sha256=digest(config), blockers=blockers,
                                    phases={p: schedule(config, p) for p in PHASES},
                                    controller_sha256=file_hash(__file__)))
    return blockers


def verify_prepared(root, ready=True):
    config, plan = read(root / 'profile.json'), read(root / 'plan.json')
    require(plan['profile_sha256'] == digest(config), 'prepared profile changed')
    require(plan['controller_sha256'] == file_hash(__file__), 'controller changed since preparation')
    validate(config, ready=ready)
    for phase in PHASES:
        require(plan['phases'][phase] == schedule(config, phase), 'schedule changed')
    return config, plan


def make_request(config, plan, sample):
    contract = dict(dataset=config['dataset'], settings=config['settings'], case=sample['case'],
                    cache_policy=config['cache_policy'], warmup=config['warmup'], environment=config['environment'])
    return dict(schema=SCHEMA, profile_sha256=plan['profile_sha256'], sample_id=sample['id'],
                phase=sample['phase'], timeout_seconds=config['timeout_seconds'], runtime_commit=config['runtimes'][sample['runtime']]['commit'],
                runtime=config['runtimes'][sample['runtime']], case=sample['case'],
                contract=contract, contract_sha256=digest(contract), artifacts=config['artifacts'], site=config['site'])


def validate_result(config, request, row, folder):
    for key in ('schema', 'profile_sha256', 'sample_id', 'phase', 'runtime_commit'):
        require(row.get(key) == request[key], 'result identity mismatch: ' + key)
    require(row.get('status') == 'pass', 'sample did not pass')
    observed = row.get('observed_contract')
    require(isinstance(observed, dict), 'observed contract missing')
    changed = sorted(k for k in set(observed) | set(request['contract'])
                     if observed.get(k) != request['contract'].get(k))
    require(not changed and observed == request['contract'], 'observed settings differ: ' + ', '.join(changed))
    require(row.get('observed_contract_sha256') == digest(observed) == request['contract_sha256'],
            'observed contract digest differs')
    require(row.get('events') == config['dataset']['events'], 'event count differs')
    require(row.get('event_sha256') == config['dataset']['event_sha256'], 'event identity differs')
    require(row.get('requested_bytes') == config['dataset']['requested_bytes'], 'byte denominator differs')
    require(row.get('correctness_pass') is True, 'correctness failed/missing')
    for key in ('setup_seconds', 'first_batch_seconds', 'loop_seconds', 'process_seconds'):
        require(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] > 0,
                'invalid/missing timing: ' + key)
    require(row['cache']['policy'] == config['cache_policy'], 'cache policy differs')
    if config['cache_policy'] != 'uncontrolled':
        require(row['cache'].get('verified') is True, 'cache state unverified')
        resident, size = row['cache']['resident_bytes_before'], row['cache']['measured_bytes']
        require(type(resident) is int and type(size) is int and 0 <= resident <= size and size > 0,
                'invalid cache observation')
        require(resident == (0 if config['cache_policy'] == 'client-cold' else size), 'wrong cache state')
    require(row.get('warmup_completed') is True, 'separate warmup missing')
    for key in ('rx_bytes', 'tx_bytes'):
        value = row['nic'][key]
        require(value is None or (type(value) is int and value >= 0), 'invalid NIC counter delta')
    require(row.get('placement') and row['placement'].get('node') and row['placement'].get('cpu_affinity'),
            'placement evidence missing')
    if request['case']['workload'].startswith('gpu'):
        require(row['placement'].get('gpu_uuids'), 'GPU placement missing')
    evidence_paths = set()
    for key in ('runtime_provenance', 'correctness', 'cache', 'resources'):
        evidence = row['evidence'][key]
        path = Path(evidence['path'])
        require(path.is_absolute() and folder.resolve() in path.resolve().parents,
                'native evidence must be inside sample folder: ' + key)
        require(path.resolve() not in evidence_paths, 'native evidence reused: ' + key)
        evidence_paths.add(path.resolve())
        require(path.is_file() and file_hash(path) == evidence['sha256'],
                'missing/changed native evidence: ' + key)

    resources = read(row['evidence']['resources']['path'])
    for relative, sha in resources.get('native_index', {}).items():
        path = folder / relative
        require(not Path(relative).is_absolute() and folder.resolve() in path.resolve().parents,
                'native index escapes sample folder')
        require(HEX.fullmatch(sha) is not None and path.is_file() and file_hash(path) == sha,
                'native indexed file changed: ' + relative)


def run_command(argv, output, timeout):
    with Path(output).open('xb') as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except BaseException:
            # Kill the local launcher group; adapters must also cancel outstanding scheduler steps.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                pass
            raise


def execute(root, phase):
    config, plan = verify_prepared(root)
    require(os.environ.get('SLURM_JOB_ID'), 'execute inside an authorized compute allocation')
    require(phase != 'ab' or summarize(root, 'aa')['acceptance'] == 'equivalent', 'A/A gate not passed')
    if phase in ('aa', 'ab', 'profile'):
        require(summarize(root, 'correctness')['acceptance'] == 'correctness-pass', 'correctness gate not passed')
    for prerequisite in (() if phase == 'correctness' else ('correctness', 'aa') if phase == 'ab' else ('correctness',)):
        require(read(root / prerequisite / 'state.json')['job_id'] == os.environ['SLURM_JOB_ID'],
                'allocation changed; prepare a new cohort')
    destination = root / phase
    destination.mkdir(exist_ok=False)
    write(destination / 'state.json', dict(status='running', phase=phase, job_id=os.environ['SLURM_JOB_ID']))
    try:
        for sample in schedule(config, phase):
            # Recheck immutable receipts before every new process.
            verify_prepared(root)
            folder = destination / sample['id']
            folder.mkdir()
            request = make_request(config, plan, sample)
            write(folder / 'request.json', request)
            argv = [str(folder / 'request.json') if arg == '{request}' else
                    str(folder / 'result.json') if arg == '{result}' else arg
                    for arg in config['adapter_command']]
            write(folder / 'command.json', argv)
            started = time.monotonic()
            code = run_command(argv, folder / 'process.log', config['timeout_seconds'])
            write(folder / 'exit.json', dict(returncode=code, wall_seconds=time.monotonic()-started))
            require(code == 0, 'adapter failed: ' + sample['id'])
            row = read(folder / 'result.json')
            validate_result(config, request, row, folder)
            write(folder / 'accepted.json', dict(result_sha256=file_hash(folder / 'result.json'),
                                                 evidence=row['evidence']))
        write(destination / 'state.json', dict(status='complete', phase=phase, job_id=os.environ['SLURM_JOB_ID']))
        return summarize(root, phase)
    except BaseException as error:
        write(destination / 'state.json', dict(status='failed', phase=phase, error=str(error),
                                             job_id=os.environ['SLURM_JOB_ID']))
        raise


# Two-sided 95% Student-t critical values, df=1..29; at most 30 pairs.
T95 = [12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,
       2.201,2.179,2.160,2.145,2.131,2.120,2.110,2.101,2.093,2.086,
       2.080,2.074,2.069,2.064,2.060,2.056,2.052,2.048,2.045]


def paired(left, right, margin):
    require(len(left) == len(right) and len(left) >= 2, 'incomplete pairs')
    changes = [100 * (b/a - 1) for a, b in zip(left, right)]
    mean = statistics.mean(changes)
    half = T95[len(changes)-2] * statistics.stdev(changes) / math.sqrt(len(changes))
    low, high = mean-half, mean+half
    verdict = 'equivalent' if low >= -margin and high <= margin else (
        'regression' if low > margin else 'improvement' if high < -margin else 'inconclusive')
    return dict(pairs=len(changes), mean_percent_change=mean, interval95=[low,high],
                paired_percent_changes=changes, acceptance=verdict)


def summarize(root, phase):
    config, plan = verify_prepared(root)
    destination = root / phase
    require(read(destination / 'state.json')['status'] == 'complete', 'phase not complete')
    expected = schedule(config, phase)
    require({p.name for p in destination.iterdir() if p.is_dir()} == {s['id'] for s in expected},
            'missing/extra samples')
    rows = {}
    placements = {}
    node = None
    for sample in expected:
        folder = destination / sample['id']
        request, row = read(folder / 'request.json'), read(folder / 'result.json')
        require(request == make_request(config, plan, sample), 'request changed')
        require(read(folder / 'exit.json')['returncode'] == 0, 'failed process')
        accepted = read(folder / 'accepted.json')
        require(accepted['result_sha256'] == file_hash(folder / 'result.json'), 'accepted result changed')
        require(accepted['evidence'] == row['evidence'], 'accepted evidence changed')
        validate_result(config, request, row, folder)
        observed_node = row['placement']['node']
        require(node is None or node == observed_node, 'node changed within phase')
        node = observed_node
        case_id = sample['case']['id']
        require(case_id not in placements or placements[case_id] == row['placement'],
                'CPU/GPU placement changed within case')
        placements[case_id] = row['placement']
        rows[sample['id']] = row
    if phase != 'correctness':
        reference = summarize(root, 'correctness')
        require(reference['placements'] == placements, 'placement changed since correctness')
        require(read(root / 'correctness' / 'state.json')['job_id'] == read(destination / 'state.json')['job_id'],
                'allocation changed since correctness')
    if phase == 'ab':
        require(summarize(root, 'aa')['acceptance'] == 'equivalent', 'A/A gate not passed')
    if phase in ('correctness', 'profile'):
        return dict(phase=phase, acceptance='correctness-pass' if phase == 'correctness' else 'diagnostic-only',
                    samples=len(rows), placements=placements)
    cells = {}
    for case in config['cases']:
        left = [rows[f'{case["id"]}-{phase}-{i:02d}-left']['loop_seconds'] for i in range(config['pairs'])]
        right = [rows[f'{case["id"]}-{phase}-{i:02d}-right']['loop_seconds'] for i in range(config['pairs'])]
        cells[case['id']] = paired(left, right, config['equivalence_percent'])
        cells[case['id']]['network_samples'] = [
            dict(sample_id=sample_id, runtime=('baseline' if phase == 'aa' or side == 'left' else 'candidate'),
                 rx_bytes=row['nic']['rx_bytes'], tx_bytes=row['nic']['tx_bytes'],
                 rx_per_selected_byte=(row['nic']['rx_bytes']/row['requested_bytes']
                                       if row['nic']['rx_bytes'] is not None else None))
            for i in range(config['pairs']) for side in ('left','right')
            for sample_id in [f'{case["id"]}-{phase}-{i:02d}-{side}'] for row in [rows[sample_id]]]
        cells[case['id']]['secondary_timings'] = {}
        for metric in ('setup_seconds', 'first_batch_seconds', 'process_seconds'):
            a = [rows[f'{case["id"]}-{phase}-{i:02d}-left'][metric] for i in range(config['pairs'])]
            b = [rows[f'{case["id"]}-{phase}-{i:02d}-right'][metric] for i in range(config['pairs'])]
            cells[case['id']]['secondary_timings'][metric] = paired(a, b, config['equivalence_percent'])
    return dict(phase=phase, site=config['site'], profile_sha256=plan['profile_sha256'], cells=cells,
                acceptance='equivalent' if all(c['acceptance']=='equivalent' for c in cells.values()) else 'not-equivalent',
                policy='paired percent loop-time change; two-sided 95% t interval; no pooling across cells/sites')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('profile'); p.add_argument('output',type=Path)
    for name in ('execute', 'summarize'):
        p = sub.add_parser(name); p.add_argument('root',type=Path); p.add_argument('phase',choices=PHASES)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = dict(blockers=prepare(args.profile,args.output.resolve()))
    else:
        result = globals()[args.command](args.root.resolve(),args.phase)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
