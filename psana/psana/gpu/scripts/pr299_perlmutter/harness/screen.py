import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from common import BASE, CONDA, HEAD, HEAD_INSTALL, HEAD_SOURCE, ROOT, cases, file_hash, pair_order, reference, tree_hashes, write_json


def git(source, *arguments):
    return subprocess.check_output(['git', '-C', str(source), *arguments], text=True).strip()


def verify_source(source, commit):
    if git(source, 'rev-parse', 'HEAD') != commit or git(source, 'status', '--porcelain'):
        raise RuntimeError(f'source is not clean at {commit}: {source}')


def prepare():
    verify_source(ROOT / 'base-source', BASE)
    verify_source(HEAD_SOURCE, HEAD)
    for run_number in (51, 77):
        write_json(ROOT / 'results' / f'reference-{run_number}.json', reference(run_number))
    manifest = dict(base=BASE, head=HEAD, harness=tree_hashes(ROOT / 'harness'),
                    head_install=tree_hashes(HEAD_INSTALL), cases=cases(),
                    references={str(run_number): file_hash(ROOT / 'results' / f'reference-{run_number}.json')
                                for run_number in (51, 77)}, rounds=4,
                    n_gpu_streams=1, batch_size=20, gpu_memory_budget_gb=8,
                    candidate_is_unfixed=True, cache_policy='uncontrolled Weka/cache state; no staging or cache eviction')
    write_json(ROOT / 'manifest.json', manifest)
    print(json.dumps(dict(prepared=True, cases=len(cases()), samples=64,
                          manifest_sha256=file_hash(ROOT / 'manifest.json'))))


def check_frozen(manifest):
    if tree_hashes(ROOT / 'harness') != manifest['harness']:
        raise RuntimeError('harness changed after review')
    if tree_hashes(HEAD_INSTALL) != manifest['head_install']:
        raise RuntimeError('candidate install changed after freezing')
    verify_source(ROOT / 'base-source', BASE)
    verify_source(HEAD_SOURCE, HEAD)
    for run_number in (51, 77):
        path = ROOT / 'results' / f'reference-{run_number}.json'
        if file_hash(path) != manifest['references'][str(run_number)]:
            raise RuntimeError('reference changed')
        current = reference(run_number)
        if current != json.loads(path.read_text()):
            raise RuntimeError('input files or metadata changed')


def bind_plan():
    physical = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        identity = ((topology / 'physical_package_id').read_text().strip(),
                    (topology / 'core_id').read_text().strip())
        physical.setdefault(identity, cpu)
    cores = list(physical.values())
    if len(cores) < 15:
        raise RuntimeError('at least 15 distinct allocated physical cores required')
    return [[cores[0]], [cores[1]], cores[2:8], cores[8:14]], [cores[14]]


def assigned_gpu(evidence_path):
    command = [sys.executable, '-B', str(ROOT / 'harness/gpu_identity.py')]
    evidence = dict(command=command, environment={key: os.environ.get(key) for key in
                    ('SLURM_JOB_ID', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES')})
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=45)
        evidence.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        identity = json.loads(result.stdout)
        evidence['identity'] = identity
        if result.returncode or not identity.get('complete'):
            raise RuntimeError(identity.get('error', 'GPU identity helper failed'))
        return identity['gpu_uuid'], identity['gpu_lines']
    except subprocess.TimeoutExpired as error:
        evidence.update(timeout=True, stdout=str(error.stdout or ''), stderr=str(error.stderr or ''))
        raise RuntimeError('GPU identity helper timed out') from error
    except Exception as error:
        evidence['error'] = str(error)
        raise
    finally:
        write_json(evidence_path, evidence)


def run_command(command, environment, log, timeout):
    with Path(log).open('x') as stream:
        process = subprocess.Popen(command, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f'command failed ({code}); see {log}')


def environment(runtime, affinity, gpu_uuid):
    result = dict(os.environ)
    install = HEAD_INSTALL if runtime == 'head' else ROOT / 'base-install'
    result.update(PYTHONPATH=str(install / 'lib/python3.9/site-packages'),
                  LD_LIBRARY_PATH=str(install / 'lib') + ':' + os.environ['LD_LIBRARY_PATH'],
                  CUPY_CACHE_DIR=str(ROOT / 'cache' / runtime),
                  SCREEN_AFFINITY=json.dumps(affinity), SCREEN_GPU_UUID=gpu_uuid)
    return result, install


def gpu_samples(path):
    rows = []
    for row in csv.reader(Path(path).read_text().splitlines()):
        if len(row) != 5:
            raise RuntimeError('malformed nvidia-smi sample')
        rows.append(dict(timestamp=row[0].strip(), uuid=row[1].strip(),
                         used_mib=float(row[2]), utilization_percent=float(row[3]),
                         temperature_c=float(row[4])))
    if not rows:
        raise RuntimeError('missing nvidia-smi samples')
    return dict(peak_used_mib=max(row['used_mib'] for row in rows), samples=len(rows),
                note='1Hz device-level samples across MPI process lifetime, not an exact allocation high-water mark')


def launch(case, variant, label, preflight, results, affinity, gpu_uuid, network):
    from nic import Monitor
    sample = results / label
    sample.mkdir()
    event_count = 3000 if case['run'] == 77 else (120 if preflight else 10000)
    env, install = environment(variant['runtime'], affinity, gpu_uuid)
    output = sample / 'worker.json'
    command = [str(CONDA / 'bin/mpirun'), '--nooversubscribe', '--bind-to', 'none',
               '--map-by', 'slot', '-n', str(case['bds'] + 2), str(CONDA / 'bin/python'),
               '-B', str(ROOT / 'harness/worker.py'), '--runtime', variant['runtime'],
               '--install', str(install), '--mode', variant['mode'], '--iterator', variant['iterator'],
               '--run', str(case['run']), '--events', str(event_count),
               '--reference', str(ROOT / 'results' / f"reference-{case['run']}.json"),
               '--output', str(output), '--provenance-script',
               str(HEAD_SOURCE / 'psana/psana/gpu/scripts/check_runtime_provenance.py'),
               '--nic-inventory', str(results / 'nic-inventory.json')]
    if preflight:
        command.append('--check-input')
    write_json(sample / 'command.json', dict(argv=command, runtime=variant['runtime'],
                                           install=str(install), cpu_affinity=affinity,
                                           gpu_uuid=gpu_uuid, events=event_count))
    monitor = Monitor(network, sample / 'nic.jsonl', interval=1.0)
    monitor.start()
    gpu_log = (sample / 'gpu.csv').open('x')
    gpu_process = subprocess.Popen(['nvidia-smi', '-i', gpu_uuid,
                                    '--query-gpu=timestamp,uuid,memory.used,utilization.gpu,temperature.gpu',
                                    '--format=csv,noheader,nounits', '--loop=1'],
                                   stdout=gpu_log, stderr=subprocess.STDOUT)
    started = time.monotonic()
    try:
        run_command(command, env, sample / 'worker.log', timeout=900)
    finally:
        gpu_process.terminate()
        gpu_process.wait(timeout=10)
        gpu_log.close()
        network_summary = monitor.stop()
        write_json(sample / 'nic-summary.json', network_summary)
    gpu_summary = gpu_samples(sample / 'gpu.csv')
    result = json.loads(output.read_text())
    if not result['complete'] or result['events'] != event_count or result['n_gpu_streams'] != 1:
        raise RuntimeError('sample failed completion/configuration gate')
    result.update(case=case['name'], label=label, preflight=preflight,
                  mpi_process_wall_seconds=time.monotonic() - started,
                  gpu_memory=gpu_summary, nic_process_window=network_summary)
    write_json(sample / 'sample.json', result)
    print(f"PASS {label}: {result['events_per_second']:.3f} events/s", flush=True)
    return result


def paired_result(left, right):
    if left['timestamp_sha256'] != right['timestamp_sha256'] or left['events'] != right['events']:
        raise RuntimeError('pair workloads differ')
    return dict(case=left['case'], left=left['label'], right=right['label'],
                throughput_change_percent=100 * (right['events_per_second'] / left['events_per_second'] - 1),
                loop_change_percent=100 * (right['loop_seconds'] / left['loop_seconds'] - 1))


def execute():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    approval = json.loads((ROOT / 'review-approval.json').read_text())
    if approval['manifest_sha256'] != file_hash(ROOT / 'manifest.json'):
        raise RuntimeError('review approval applies to another snapshot')
    review = json.loads(Path(approval['review_json']).read_text())
    if review['verdict'] != 'pass' or any(item['blocking'] for item in review['findings']):
        raise RuntimeError('independent review has unresolved blockers')
    check_frozen(manifest)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('run inside a Slurm allocation')
    results = ROOT / 'results' / f"job-{os.environ['SLURM_JOB_ID']}"
    results.mkdir()
    status_path = results / 'status.json'
    write_json(status_path, dict(phase='preflight', complete=False))
    try:
        affinity, monitor_affinity = bind_plan()
        gpu_uuid, gpu_lines = assigned_gpu(results / 'gpu-identity.json')
        write_json(results / 'host.json', dict(host=os.uname().nodename,
                                              job_id=os.environ['SLURM_JOB_ID'], gpu=gpu_lines,
                                              rank_affinity=affinity, monitor_affinity=monitor_affinity,
                                              manifest_sha256=file_hash(ROOT / 'manifest.json'),
                                              python=sys.version, environment={key: os.environ.get(key)
                                              for key in ('SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES',
                                                          'KVIKIO_COMPAT_MODE', 'KVIKIO_NTHREADS', 'KVIKIO_TASK_SIZE',
                                                          'OMPI_MCA_pml', 'OMPI_MCA_btl', 'OMP_NUM_THREADS')}))
        from nic import inventory, probe
        network = inventory()
        write_json(results / 'nic-inventory.json', network)
        probe_result = probe(network, results / 'weka-probe.json')
        write_json(results / 'weka-probe-result.json', probe_result)
        if (ROOT / 'base-build').exists() or (ROOT / 'base-install').exists():
            raise RuntimeError('refusing to overwrite an existing base build/install')
        if shutil.disk_usage(ROOT).free < 1024**3:
            raise RuntimeError('less than 1GiB scratch free; no home fallback')
        run_command(['bash', str(ROOT / 'base-source/build_psana.sh'), '--build-dir',
                     str(ROOT / 'base-build'), '--prefix', str(ROOT / 'base-install'),
                     '--build-type', 'release', '--jobs', '8'], dict(os.environ),
                    results / 'base-build.log', timeout=1800)
        write_json(results / 'base-install-hashes.json', tree_hashes(ROOT / 'base-install'))
        verify_source(ROOT / 'base-source', BASE)
        os.sched_setaffinity(0, monitor_affinity)
        for case in cases():
            for index, variant in enumerate(case['variants']):
                launch(case, variant, f"preflight-{case['name']}-v{index}", True,
                       results, affinity, gpu_uuid, network)
        write_json(status_path, dict(phase='timed', complete=False, preflights_passed=16))
        pairs = []
        for round_number in range(4):
            for case in (cases() if round_number % 2 == 0 else list(reversed(cases()))):
                pair = {}
                for index in pair_order(round_number):
                    label = f"r{round_number + 1}-{case['name']}-v{index}"
                    pair[index] = launch(case, case['variants'][index], label, False,
                                         results, affinity, gpu_uuid, network)
                pairs.append(paired_result(pair[0], pair[1]))
                write_json(results / 'pairs.json', dict(complete=False, pairs=pairs))
        check_frozen(manifest)
        if tree_hashes(ROOT / 'base-install') != json.loads((results / 'base-install-hashes.json').read_text()):
            raise RuntimeError('base install changed during campaign')
        summary = []
        for case in cases():
            values = [pair['throughput_change_percent'] for pair in pairs if pair['case'] == case['name']]
            if len(values) != 4:
                raise RuntimeError('incomplete pair set')
            summary.append(dict(case=case['name'], paired_throughput_median_percent=statistics.median(values),
                                paired_throughput_range_percent=[min(values), max(values)], pairs=4))
        write_json(results / 'pairs.json', dict(complete=True, pairs=pairs, summary=summary,
                                              limitation='Pre-fix screen; Weka NIC counters are node-level, not process-attributed.'))
        write_json(status_path, dict(phase='complete', complete=True, preflights_passed=16, timed_samples=64))
    except BaseException as error:
        write_json(status_path, dict(phase='failed', complete=False, error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        execute()
