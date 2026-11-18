"""Fault-inject shared Slurm launcher/controller; direct single-node sbatch only.

No native qualification flags, data access, GPU work or job submission.
"""
import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from native_contract import digest, file_hash, read, require, write
from perlmutter_adapter import cancel_step, command_output, confirm_step, launch, matching_steps
from campaign import run_command

SOURCES = ('qualify_cleanup.py', 'cleanup_worker.py', 'perlmutter_adapter.py',
           'native_contract.py', 'campaign.py')
CASES = ('step-timeout', 'SIGTERM', 'SIGINT', 'SIGHUP', 'controller-timeout')


def wait_for(predicate, seconds, message, process=None):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        require(process is None or process.poll() is None, message + ': launcher exited')
        time.sleep(0.1)
    raise TimeoutError(message)


def identity(folder, job, ranks):
    rows = [read(folder / ('ready-%d.json' % r)) for r in range(ranks)]
    require([r['rank'] for r in rows] == list(range(ranks)), 'rank identity differs')
    require(all(r['job'] == job and re.fullmatch('[0-9]+', r['step']) for r in rows), 'invalid step identity')
    require(len({r['step'] for r in rows}) == 1, 'multiple steps in rank evidence')
    return job + '.' + rows[0]['step']


def validate_failure(failure, label, target_id):
    require(failure['cleanup']['no_active_steps'] is True, 'cleanup uncertain')
    require(set(failure['cleanup'].get('cancelled', [])) <= {target_id}, 'cancelled unrelated step')
    if label == 'step-timeout':
        require(failure['error'].startswith('TimeoutExpired('), 'wrong timeout cause')
        require(60 <= failure['seconds'] < 70, 'step timeout cleanup exceeded grace')
    else:
        number = signal.SIGTERM if label == 'controller-timeout' else getattr(signal, label)
        require(failure['error'] == repr(InterruptedError('adapter received signal %d' % number)), 'wrong signal cause')


def child(mode, request_path, folder):
    request, folder = read(request_path), Path(folder)
    if mode == '_adapter':
        write(folder.parent / 'adapter-pid.json', dict(pid=os.getpid()))
        launch(request, Path(request_path), folder, 'cleanup', request['site'], request['runtime'])
        raise RuntimeError('disposable target unexpectedly succeeded')
    started = time.monotonic()
    try:
        run_command([sys.executable, '-B', '-s', __file__, '_adapter', request_path, str(folder)],
                    folder.parent / 'controller-process.log', 60)
    except subprocess.TimeoutExpired:
        write(folder.parent / 'controller-timeout.json', dict(seconds=time.monotonic()-started))
        return
    raise RuntimeError('controller did not time out')


def run_case(root, label, runtime, site, sources):
    folder = root / label
    folder.mkdir()
    target, sentinel = folder / 'target', folder / 'sentinel'
    sentinel.mkdir()
    worker = Path(__file__).resolve().with_name('cleanup_worker.py')
    request = dict(profile_sha256=digest(sources), sample_id=digest(str(folder)),
        case=dict(bds=1, gpus=0), contract=dict(environment={'MPICH_GPU_SUPPORT_ENABLED':'0'}),
        artifacts=dict(native_worker=dict(path=str(worker), sha256=sources[worker.name])),
        runtime=runtime, site=dict(site, step_timeout_seconds=60 if label == 'step-timeout' else 180))
    request_path = folder / 'request.json'
    write(request_path, request)
    name = 'prreg-' + digest([request['profile_sha256'], request['sample_id'], 'cleanup'])[:20]
    sentinel_name = name + '-sentinel'
    require(not matching_steps(site['job_id'], name) and not matching_steps(site['job_id'], sentinel_name), 'step name collision')
    mode = '_controller' if label == 'controller-timeout' else '_adapter'
    process = watch = None
    started = time.monotonic()
    try:
        with (folder / 'adapter.log').open('x') as log, (folder / 'sentinel.log').open('x') as sentinel_log:
            process = subprocess.Popen([sys.executable, '-B', '-s', __file__, mode, str(request_path), str(target)],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_for(lambda: all((target / ('ready-%d.json' % r)).exists() for r in range(3)),
                     25, 'target readiness', process)
            target_id = identity(target, site['job_id'], 3)
            # Deliberate overlap ONLY for this non-performance isolation test.
            argv = ['srun', '--jobid='+site['job_id'], '--nodes=1', '--nodelist='+site['node'],
                    '--ntasks=1', '--cpus-per-task=1', '--overlap', '--exact', '--cpu-bind=none',
                    '--gpus-per-node=0', '--mpi=cray_shasta', '--job-name='+sentinel_name,
                    sys.executable, '-B', '-s', str(worker), str(request_path), str(sentinel), 'sentinel']
            write(folder / 'sentinel-launch.json', dict(argv=argv))
            watch = subprocess.Popen(argv, stdout=sentinel_log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_for(lambda: (sentinel / 'heartbeat').exists(), 20, 'sentinel readiness', watch)
            sentinel_id = identity(sentinel, site['job_id'], 1)
            require(sentinel_id != target_id, 'sentinel shares target step')
            (folder / 'live-steps-before.txt').write_text(command_output(['scontrol', '--oneliner', 'show', 'step', site['job_id']]))
            require(matching_steps(site['job_id'], name) == [target_id] and
                    matching_steps(site['job_id'], sentinel_name) == [sentinel_id], 'both steps must be live before injection')
            heartbeat_before = int((sentinel / 'heartbeat').read_text())
            injected = time.monotonic()
            write(folder / 'before.json', dict(target_id=target_id, sentinel_id=sentinel_id,
                  target_name=name, sentinel_name=sentinel_name, heartbeat=heartbeat_before,
                  seconds_since_start=injected-started))
            if label.startswith('SIG'):
                pid = read(folder / 'adapter-pid.json')['pid']
                require(pid == process.pid, 'adapter PID differs')
                os.kill(pid, getattr(signal, label))
            code = process.wait(timeout=80)
            finished = time.monotonic()
            require(code == (0 if label == 'controller-timeout' else 1), 'unexpected adapter exit code')
            failure = read(target / 'failure.json')
            validate_failure(failure, label, target_id)
            if label.startswith('SIG'):
                require(finished-injected < 10, 'signal cleanup exceeded controller grace')
            if label == 'controller-timeout':
                require(read(folder / 'controller-timeout.json')['seconds'] < 70, 'controller exceeded grace')
            require(not (target / 'scheduler.json').exists() and not (target / 'complete.json').exists(), 'false success receipt')
            (folder / 'live-steps-after.txt').write_text(command_output(['scontrol', '--oneliner', 'show', 'step', site['job_id']]))
            require(not matching_steps(site['job_id'], name), 'target still active')
            require(matching_steps(site['job_id'], sentinel_name) == [sentinel_id], 'sentinel cancelled')
            wait_for(lambda: int((sentinel / 'heartbeat').read_text()) > heartbeat_before,
                     3, 'sentinel heartbeat stopped', watch)
            heartbeat_after = int((sentinel / 'heartbeat').read_text())
            # Also demand a new heartbeat AFTER target cleanup, not merely before it.
            wait_for(lambda: int((sentinel / 'heartbeat').read_text()) > heartbeat_after,
                     3, 'sentinel did not progress after cleanup', watch)
            (sentinel / 'release').touch(exist_ok=False)
            require(watch.wait(timeout=20) == 0, 'sentinel failed graceful release')
            require(not matching_steps(site['job_id'], sentinel_name), 'sentinel remains active')
            accounting = confirm_step(sentinel_id, sentinel)
            write(folder / 'pass.json', dict(case=label, target_id=target_id, sentinel_id=sentinel_id,
                  failure=failure, returncode=code, injection_to_exit_seconds=finished-injected,
                  heartbeat_before=heartbeat_before, heartbeat_after=heartbeat_after,
                  sentinel_accounting=accounting, status='pass'))
    finally:
        # On harness failure, scope cleanup to these two known unique names only.
        original_error = sys.exc_info()[1]
        errors = []
        for step_name in (name, sentinel_name):
            try:
                cancel_step(site['job_id'], step_name)
            except BaseException as error:
                errors.append(dict(name=step_name, error=repr(error)))
        for proc in (process, watch):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait(timeout=2)
        write(folder / 'harness-cleanup.json', dict(errors=errors, original_error=repr(original_error) if original_error else None))
        require(not errors, 'harness cleanup uncertain; audit allocation')


def main():
    if len(sys.argv) == 4 and sys.argv[1] in ('_adapter', '_controller'):
        return child(*sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    job, node = os.environ.get('SLURM_JOB_ID', ''), os.uname().nodename
    require(re.fullmatch('[0-9]+', job) and re.fullmatch('nid[0-9]+', node), 'compute allocation required')
    require(os.environ.get('SLURM_STEP_ID', 'batch') == 'batch' and os.environ.get('SLURM_JOB_NUM_NODES') == '1', 'direct single-node batch only')
    require(file_hash(args.runtime) == args.runtime_sha256, 'runtime hash differs')
    runtime = read(args.runtime)
    require(Path(sys.executable).resolve() == Path(runtime['python']).resolve(), 'use pinned Python')
    root = Path(args.output).resolve()
    root.mkdir(mode=0o700, exist_ok=False)
    source = Path(__file__).resolve().parent
    hashes = {name:file_hash(source / name) for name in SOURCES}
    site = dict(job_id=job, node=node, cpu_affinity={'1':[[0,1,2,3],[4,5,6,7],[8,9,10,11]]}, step_cpus_per_task=16)
    write(root / 'started.json', dict(job=job, node=node, sources=hashes, site=site,
          runtime=dict(path=args.runtime, sha256=args.runtime_sha256), cases=CASES))
    (root / 'allocation.txt').write_text(command_output(['scontrol','show','job','-d',job]))
    try:
        for label in CASES:
            run_case(root, label, runtime, site, hashes)
        require(hashes == {name:file_hash(source / name) for name in SOURCES} and
                file_hash(args.runtime) == args.runtime_sha256, 'source/runtime changed')
        write(root / 'index.json', {str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()})
        write(root / 'complete.json', dict(status='pass', cases=CASES, index_sha256=file_hash(root / 'index.json'),
              scope='disposable CPU step cleanup only; no data, GPU-workload or timed qualification'))
    except BaseException as error:
        write(root / 'failed.json', dict(error=repr(error), context=repr(error.__context__) if error.__context__ else None))
        raise


if __name__ == '__main__':
    main()
