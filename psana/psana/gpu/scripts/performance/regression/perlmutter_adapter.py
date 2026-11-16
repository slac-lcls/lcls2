"""Synchronous adapter_command for qualified run51 raw campaigns on Perlmutter.

CLI: python -B perlmutter_adapter.py REQUEST RESULT
No sbatch, dataset copy, calibration write, or qualification flag changes.
"""
import argparse
import copy
import os
from pathlib import Path
import pstats
import re
import signal
import subprocess
import time

from native_contract import (artifact, digest, file_hash, load_contract, native_environment,
                             read, require, write)


def command_output(argv, timeout=15):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=True)
    return result.stdout


def matching_steps(job, name, timeout=15):
    require(re.fullmatch('[0-9]+', job) is not None, 'invalid allocation id')
    output = command_output(['squeue', '--steps', '--jobs=' + job, '--noheader', '--format=%i|%j'], timeout=timeout)
    result = []
    for line in output.splitlines():
        identifier, observed_name = line.strip().split('|', 1)
        if observed_name == name:
            require(re.fullmatch(re.escape(job) + r'\.[0-9]+', identifier) is not None, 'unexpected step identity')
            result.append(identifier)
    return result


def cancel_step(job, name):
    # Total scheduler cleanup budget fits within the controller's 10 s grace.
    deadline = time.monotonic() + 6

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('native cleanup deadline exceeded; audit allocation')
        return seconds

    steps = matching_steps(job, name, timeout=remaining())
    for step in steps:
        subprocess.run(['scancel', '--signal=KILL', step], timeout=remaining(), check=True)
    while True:
        if not matching_steps(job, name, timeout=remaining()):
            return dict(cancelled=steps, no_active_steps=True)
        time.sleep(min(0.1, remaining()))


def launch(request, request_path, folder, stage, site, runtime):
    folder.mkdir(exist_ok=False)
    job = site['job_id']
    name = 'prreg-' + digest([request['profile_sha256'], request['sample_id'], stage])[:20]
    require(not matching_steps(job, name), 'native step already exists')
    cpus = site['cpu_affinity'][str(request['case']['bds'])]
    masks = ['0x%x' % sum(1 << cpu for cpu in group) for group in cpus]
    argv = ['srun', '--jobid=' + job, '--nodes=1', '--nodelist=' + site['node'], '--ntasks=' + str(len(cpus)),
            '--cpus-per-task=' + str(len(cpus[0])), '--exclusive', '--exact', '--mpi=cray_shasta',
            '--cpu-bind=mask_cpu:' + ','.join(masks), '--gpu-bind=none',
            '--gpus-per-node=' + str(request['case']['gpus']), '--kill-on-bad-exit=1', '--job-name=' + name,
            runtime['python'], '-B', '-s', str(artifact(request, 'native_worker')), str(request_path), str(folder), stage]
    environment = native_environment(request, runtime)
    started = time.monotonic()
    process, cleanup = None, None

    def interrupted(signum, frame):
        raise InterruptedError('adapter received signal %d' % signum)

    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        write(folder / 'launch.json', dict(argv=argv, job=job, name=name, runtime_environment=runtime['environment']))
        with (folder / 'native.log').open('x') as log:
            process = subprocess.Popen(argv, env=environment, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait(timeout=site['step_timeout_seconds'])
            process_seconds = time.monotonic() - started
        require(code == 0, 'native MPI step failed: ' + str(code))
        require(not matching_steps(job, name), 'srun returned while step remains active')
        step = read(folder / 'step.json')
        require(step['job'] == job and step['node'] == site['node'] and re.fullmatch('[0-9]+', step['step']), 'rank scheduler identity differs')
        step_id = job + '.' + step['step']
        final = ''
        for _ in range(10):
            final = command_output(['sacct', '-j', step_id, '--noheader', '--parsable2', '--format=JobIDRaw,State,ExitCode'])
            rows = [line.split('|') for line in final.splitlines() if line.split('|')[0] == step_id]
            if rows and all(r[1:3] == ['COMPLETED', '0:0'] for r in rows):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError('scheduler did not confirm COMPLETED/0:0 for ' + step_id)
        complete = read(folder / 'complete.json')
        require(complete['status'] == 'pass' and complete['stage'] == stage, 'native completion receipt missing')
        receipt = dict(argv=argv, step_id=step_id, returncode=code, seconds=process_seconds, scheduler_confirmation_seconds=time.monotonic()-started-process_seconds,
                       scheduler=final, no_active_steps=True, status='pass')
        write(folder / 'scheduler.json', receipt)
        return complete, receipt
    except BaseException as error:
        # Never cancel the allocation or other steps. Cancel this unique sample name only.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if process is not None and process.poll() is None:
                process.terminate()
            cleanup = cancel_step(job, name)
            if process is not None:
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.5)
        except BaseException as cleanup_error:
            cleanup = dict(error=str(cleanup_error), no_active_steps=False)
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                    process.wait(timeout=0.5)
                except BaseException as local_error:
                    cleanup['local_launcher_error'] = str(local_error)
        write(folder / 'failure.json', dict(error=repr(error), cleanup=cleanup, seconds=time.monotonic()-started))
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def component_profile(folder):
    results = []
    for path in sorted(folder.glob('rank-*.pstats')):
        stats = pstats.Stats(str(path))
        functions = []
        for (filename, line, name), (primitive, calls, self_seconds, cumulative_seconds, callers) in stats.stats.items():
            if '/psana/' in filename:
                functions.append(dict(file=filename, line=line, function=name, primitive_calls=primitive,
                                      calls=calls, self_seconds=self_seconds, cumulative_seconds=cumulative_seconds))
        results.append(dict(rank=int(path.stem.split('-')[1]), functions=functions))
    return dict(ranks=results, semantics='cProfile call counts and host wall-time profiles; nested cumulative times overlap',
                unavailable=['candidate/live-window visits', 'occupancy scan work units', 'GPU kernel counts'],
                acceptance='requires separately frozen component bounds and review; no automatic component gate')


def assemble(request, measured, warmup, scheduler, elapsed, folder, site):
    require(warmup['status'] == 'pass' and warmup['stage'] == 'warmup', 'warmup missing')
    rows = measured['rows']
    require(measured['diagnostic'] == (request['phase'] == 'profile'), 'diagnostic/clean separation failed')
    require(all(r['node'] == site['node'] for r in rows), 'multi-node or changed placement')
    require([r['affinity'] for r in rows] == [sorted(x) for x in site['cpu_affinity'][str(request['case']['bds'])]], 'placement differs')
    require([(r['affinity'], r['gpu']) for r in warmup['rows']] == [(r['affinity'], r['gpu']) for r in rows], 'warmup placement differs')
    uuids = sorted(set(r['gpu']['uuid'] for r in rows if r['gpu']))
    require(uuids == ([site['gpu_uuid']] if request['case']['gpus'] else []), 'GPU assignment differs')
    # Preserve metadata fields as frozen declarations; measured fields come from native rows.
    observed = copy.deepcopy(request['contract'])
    for row in rows:
        for key, value in row['observed_settings'].items():
            require(value == observed['settings'][key], 'resolved native setting differs')
        require({k: row['environment'][k] for k in observed['environment']} == observed['environment'], 'native environment differs')
        require(row['environment']['PS_SMD_N_EVENTS'] == str(observed['settings']['ps_smd_n_events']) and
                row['environment']['PS_PARALLEL'] == 'mpi', 'native SMD/MPI environment differs')
    observed['dataset']['events'] = measured['work']['events']
    observed['dataset']['event_sha256'] = measured['work']['event_sha256']
    observed['dataset']['requested_bytes'] = measured['work']['requested_bytes']
    observed['settings']['batch_size'] = rows[0]['observed_settings']['batch_size']
    if request['case']['gpus']:
        observed['settings'].update(rows[2]['observed_settings'])
    require(observed == request['contract'], 'observed contract differs')
    cache = measured['cache']['observations'][-1]
    require(cache['resident_bytes_before'] == 0 and cache['measured_bytes'] == observed['dataset']['requested_bytes'], 'cache evidence differs')
    evidence = {}
    values = dict(runtime_provenance=dict(rows=[dict(rank=r['rank'], before=r['provenance_before'], after=r['provenance_after']) for r in rows]),
                  correctness=dict(work=measured['work'], rows=[dict(rank=r['rank'], role=r['role'], events=len(r['timestamps']),
                                   cpu_raw_bytes=r['cpu_raw_bytes'], callback_events=r['callback_events']) for r in rows],
                                   scope='event/step identities, datagram/read bytes, callback metadata; no pixel or calibration qualification'),
                  cache=measured['cache'], resources=dict(scheduler=scheduler, rows=rows, adapter_elapsed_seconds=elapsed,
                      nic_before=measured['nic_before'], nic_after=measured['nic_after'],
                      nic_scope='selected interface byte counters over synchronized loop; node traffic, not rank/storage attribution',
                      timing='setup=max rank DataSource construction; first_batch=max BD first nonempty yield-window end (delivery or finish drain); GPU deferred retirement can move it to a later EB packet; loop=max synchronized rank loop incl drain; close separately; process=measured srun launch through srun wait return; scheduler confirmation excluded; adapter elapsed includes warmup/evidence preparation',
                      declared_metadata=['available_events', 'other-site paths', 'dataset status', 'metric wishlist', 'unused CPU-case GPU settings'],
                      qualification='implementation only; overhead, missing metrics and component bounds require hardware qualification'))
    for key, value in values.items():
        path = folder / (key + '.json')
        write(path, value)
        evidence[key] = dict(path=str(path), sha256=file_hash(path))
    row = {key: request[key] for key in ('schema', 'profile_sha256', 'sample_id', 'phase', 'runtime_commit')}
    row.update(status='pass', observed_contract=observed, observed_contract_sha256=digest(observed),
               correctness_pass=True, warmup_completed=True, **measured['work'],
               setup_seconds=max(r['setup_seconds'] for r in rows),
               first_batch_seconds=max(r['first_batch_seconds'] for r in rows[2:]),
               loop_seconds=max(r['loop_seconds'] for r in rows), process_seconds=scheduler['measured']['seconds'],
               cache={k: cache[k] for k in ('policy', 'verified', 'measured_bytes', 'resident_bytes_before')},
               nic=measured['nic'], placement=dict(node=site['node'], cpu_affinity={str(r['rank']): r['affinity'] for r in rows}, gpu_uuids=uuids),
               evidence=evidence)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    parser.add_argument('result')
    args = parser.parse_args()
    started = time.monotonic()
    request_path, result_path = Path(args.request).resolve(), Path(args.result).resolve()
    require(result_path.parent == request_path.parent and not result_path.exists(), 'result must be new and beside request')
    request = read(request_path)
    require(Path(__file__).resolve() == artifact(request, 'adapter'), 'unfrozen adapter')
    import native_contract
    require(Path(native_contract.__file__).resolve() == artifact(request, 'native_contract'), 'unfrozen helper')
    site, reference, inputs, runtime = load_contract(request)
    # The source-only controller cannot establish the truth of these qualification receipts.
    artifact(request, 'component_contract')
    folder = result_path.parent
    warmup, warm_scheduler = launch(request, request_path, folder / 'warmup', 'warmup', site, runtime)
    measured, scheduler = launch(request, request_path, folder / 'measured', 'measured', site, runtime)
    if request['phase'] == 'profile':
        write(folder / 'components.json', component_profile(folder / 'measured'))
    row = assemble(request, measured, warmup, dict(warmup=warm_scheduler, measured=scheduler), time.monotonic()-started, folder, site)
    seal_native_index(folder, row)
    write(result_path, row)


def seal_native_index(folder, row):
    # Index every native file, including diagnostics and warmup, in the accepted resources receipt.
    index = {str(p.relative_to(folder)): file_hash(p) for sub in ('warmup', 'measured') for p in sorted((folder/sub).rglob('*')) if p.is_file()}
    if (folder / 'components.json').exists():
        index['components.json'] = file_hash(folder / 'components.json')
    write(folder / 'native-index.json', index)
    # Nest the index content in an accepted receipt: the controller hashes resources.json.
    resource_path = folder / 'resources.json'
    resources = read(resource_path)
    resources['native_index'] = index
    resource_path.write_text(__import__('json').dumps(resources, sort_keys=True, indent=2, allow_nan=False) + '\n')
    row['evidence']['resources']['sha256'] = file_hash(resource_path)


if __name__ == '__main__':
    main()
