"""Run51-only local performance screen. Full campaign requires separate review + pilot."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from common import BASE, HEAD, ROOT, cases, file_hash, pair_order, tree_hashes, write_json
from storage import Monitor, inventory


def verify_source(runtime):
    source = ROOT / ('source-'+runtime)
    sha = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    status = subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    if sha != {'base':BASE,'head':HEAD}[runtime] or status:
        raise RuntimeError(f'Source identity failure: {runtime} {sha} {status}')


def build_parity():
    values = {}
    required = dict(optimization='3', debug=False, b_lto=True, b_ndebug='true', cpp_std='c++20', build_daq=False)
    for runtime in ('base','head'):
        p = ROOT/f'build-{runtime}/meson-info/intro-buildoptions.json'
        options = {x['name']:x['value'] for x in json.loads(p.read_text())}
        for key,value in required.items():
            if options[key] != value:
                raise RuntimeError(f'{runtime} {key}={options[key]!r}; expected {value!r}')
        if options['prefix'] != str(ROOT/f'install-{runtime}'):
            raise RuntimeError('Wrong install prefix')
        values[runtime] = {k:v for k,v in options.items() if k not in ('prefix','python.platlibdir','python.purelibdir')}
    if values['base'] != values['head']:
        raise RuntimeError('Meson option parity failure')
    return required


def freeze():
    for runtime in ('base','head'): verify_source(runtime)
    evidence_paths = [ROOT/'logs/campaign-candidate-tests.txt', ROOT/'campaign-plan.json', ROOT/'evidence/initial-pilot-estimate.json',
                      ROOT/'results/pilot-job-61372763/status.json',
                      ROOT/'results/pilot-job-61372763/sustained-base-gpu-bd1/sample.json',
                      ROOT/'results/pilot-job-61372763/sustained-base-gpu-bd1/progress-rank-2.jsonl', ROOT/'evidence/build-environment.txt',
                      ROOT/'evidence/runtime-imports-login.json', ROOT/'evidence/provenance-login-base.json',
                      ROOT/'evidence/provenance-login-head.json', ROOT/'evidence/input-verification-51.json',
                      Path('/global/u2/m/monarin/psana-nersc/activate_psana_build_env.sh')]
    manifest = dict(evidence_sha256={str(p):file_hash(p) for p in evidence_paths},base=BASE, head=HEAD, harness=tree_hashes(ROOT/'harness'),
                    installs={v:tree_hashes(ROOT/f'install-{v}') for v in ('base','head')},
                    reference_sha256=file_hash(ROOT/'results/reference-51.json'),
                    build_options=build_parity(), cases=[c for c in cases() if c['run']==51],
                    pilot_plan='12preflights plus all6base cases at10000events, each600s cap, total30min',
                    campaign_plan=json.loads((ROOT/'campaign-plan.json').read_text()),
                    paired_rounds=4, scope='run51 only; run77 unavailable/deferred; C1 unfixed',
                    cache_policy='uncontrolled Lustre page cache; no staging or eviction',
                    monitoring='identical 2s node/client/cgroup + 1s GPU + 100-event progress for both variants')
    write_json(ROOT/'manifest.json', manifest)
    print(file_hash(ROOT/'manifest.json'))


def check_frozen():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for path,digest in manifest['evidence_sha256'].items():
        if file_hash(path) != digest: raise RuntimeError(f'Reviewed evidence changed: {path}')
    if manifest['harness'] != tree_hashes(ROOT/'harness'):
        raise RuntimeError('Harness differs from manifest')
    for v in ('base','head'):
        verify_source(v)
        if tree_hashes(ROOT/f'install-{v}') != manifest['installs'][v]:
            raise RuntimeError(f'Install changed: {v}')
    if file_hash(ROOT/'results/reference-51.json') != manifest['reference_sha256']:
        raise RuntimeError('Reference changed')
    ref = json.loads((ROOT/'results/reference-51.json').read_text())
    for item in ref['files']:
        if file_hash(item['path']) != item['sha256']:
            raise RuntimeError('SMD changed')
        st = Path(item['bigdata']).stat()
        if st.st_size != item['bytes'] or st.st_mtime_ns != item['mtime_ns']:
            raise RuntimeError('Bigdata metadata changed')
        with open(item['bigdata'],'rb') as stream: stream.read(1)
    build_parity()
    return manifest


def pilot_evidence_digest():
    import hashlib
    estimate_path = ROOT/'pilot-estimate.json'
    estimate = json.loads(estimate_path.read_text())
    sample = Path(estimate['sample_directory']).resolve()
    sample.relative_to((ROOT/'results').resolve())
    status = json.loads((sample.parent/'status.json').read_text())
    if status.get('phase') != 'pilot' or not status.get('complete'):
        raise RuntimeError('Pilot status is incomplete')
    validate_sizing_evidence(estimate, sample.parent)
    payload = dict(estimate=file_hash(estimate_path), results=tree_hashes(sample.parent))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_sizing_evidence(estimate, directory):
    expected = {case['name'] for case in cases() if case['run'] == 51}
    if set(estimate.get('case_estimates', {})) != expected:
        raise RuntimeError('Completed six-case sizing evidence required')
    for name, sizing in estimate['case_estimates'].items():
        sample_path = (directory/sizing['sample_label']/'sample.json').resolve()
        sample_path.relative_to(directory.resolve())
        result = json.loads(sample_path.read_text())
        computed = case_sizing(result)
        if result['case'] != name or sizing['measured_mpi_wall_seconds'] != computed['measured_mpi_wall_seconds']:
            raise RuntimeError('Sizing case/sample mismatch')
        if sizing['timeout_seconds'] < computed['timeout_seconds']:
            raise RuntimeError('Sizing timeout below completed-sample bound')
    return expected


def campaign_plan(manifest):
    plan = manifest['campaign_plan']
    estimate = json.loads((ROOT/'pilot-estimate.json').read_text())
    directory = Path(estimate['sample_directory']).resolve().parent
    expected = validate_sizing_evidence(estimate, directory)
    if set(plan['cases']) != expected or plan['timed_samples'] != 48 or plan['rounds'] != 4:
        raise RuntimeError('Campaign plan coverage mismatch')
    if plan['sizing_result_directory'] != str(directory):
        raise RuntimeError('Campaign plan points to different pilot')
    for name, item in plan['cases'].items():
        sizing = estimate['case_estimates'][name]
        sample = directory/sizing['sample_label']/'sample.json'
        if item['source_sample'] != str(sample) or item['source_sample_sha256'] != file_hash(sample):
            raise RuntimeError('Campaign plan sample provenance mismatch')
        if item['timeout_seconds'] < sizing['timeout_seconds']:
            raise RuntimeError('Campaign timeout below sizing bound')
    return plan


def review_gate(phase):
    approval = json.loads((ROOT/f'review-approval-{phase}.json').read_text())
    if approval['manifest_sha256'] != file_hash(ROOT/'manifest.json'):
        raise RuntimeError('Review applies to different manifest')
    review = json.loads(Path(approval['review_json']).read_text())
    if approval.get('approved_phase') != phase or review.get('approved_phase') != phase:
        raise RuntimeError('Review phase mismatch')
    if phase == 'campaign':
        digest = pilot_evidence_digest()
        if review.get('pilot_evidence_sha256') != digest or approval.get('pilot_evidence_sha256') != digest:
            raise RuntimeError('Campaign review does not bind completed pilot evidence')
    if review.get('reviewed_manifest_sha256') != file_hash(ROOT/'manifest.json'):
        raise RuntimeError('Reviewer response covers a different manifest')
    if review['reviewed_commit'] != HEAD or review['verdict'] != 'pass' or any(x['blocking'] for x in review['findings']):
        raise RuntimeError('Independent review did not pass')


def environment(runtime, gpu_uuid):
    install = ROOT/f'install-{runtime}'
    env = dict(os.environ)
    # Replace, never accumulate runtime project prefixes.
    env['PYTHONPATH'] = str(install/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
    env['LD_LIBRARY_PATH'] = str(install/'lib')+':'+os.environ['SCREEN_RUNTIME_LIBS']
    env['CUPY_CACHE_DIR'] = str(ROOT/'cache'/runtime)
    env['SCREEN_GPU_UUID'] = gpu_uuid
    env.update(PS_EB_NODES='1', PS_SRV_NODES='0', PS_PARALLEL='mpi', MPICH_GPU_SUPPORT_ENABLED='0',
               KVIKIO_COMPAT_MODE='ON', KVIKIO_NTHREADS='4', KVIKIO_TASK_SIZE='1048576',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('SIT_PSDM_OFFSITE',None)
    return env, install


def run_command(command, env, log, timeout):
    with log.open('x') as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            # Terminating srun propagates termination to its job step; allocation shell exits on failure.
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise
    if code: raise RuntimeError(f'MPI command exited {code}: {log}')


def launch(case, variant, label, count, timeout, results, gpu_uuid, network, preflight=False, diagnostics=False):
    sample = results/label; sample.mkdir()
    env, install = environment(variant['runtime'], gpu_uuid)
    output = sample/'worker.json'
    cmd = ['srun','--mpi=cray_shasta','--nodes=1',f"--ntasks={case['bds']+2}",
           f"--ntasks-per-node={case['bds']+2}",'--cpus-per-task=8','--cpu-bind=cores',
           '--gpus-per-node=1','--gpu-bind=none','--exact','--kill-on-bad-exit=1',
           sys.executable,'-B',str(ROOT/'harness/worker.py'), '--runtime',variant['runtime'],
           '--install',str(install),'--mode',variant['mode'],'--iterator',variant['iterator'],
           '--run','51','--events',str(count),'--reference',str(ROOT/'results/reference-51.json'),
           '--output',str(output),'--provenance-script',str(ROOT/'harness/check_runtime_provenance.py'),
           '--nic-inventory',str(results/'storage-inventory.json')]
    if preflight: cmd.append('--check-input')
    if diagnostics: cmd.append('--diagnostics')
    write_json(sample/'command.json',dict(argv=cmd,timeout_seconds=timeout,environment={k:env[k] for k in
               ('PYTHONPATH','LD_LIBRARY_PATH','KVIKIO_COMPAT_MODE','KVIKIO_NTHREADS','KVIKIO_TASK_SIZE','SCREEN_GPU_UUID','PS_EB_NODES','PS_SRV_NODES','PS_PARALLEL')}))
    monitor = Monitor(network,sample/'storage.jsonl'); monitor.start()
    gpu_log = (sample/'gpu.csv').open('x')
    gpu_process = subprocess.Popen(['nvidia-smi','-i',gpu_uuid,'--query-gpu=timestamp,uuid,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits','--loop=1'],stdout=gpu_log,stderr=subprocess.STDOUT)
    wall = None
    started = time.monotonic()
    try:
        run_command(cmd,env,sample/'worker.log',timeout)
        wall = time.monotonic()-started
    finally:
        gpu_process.terminate()
        try: gpu_process.wait(timeout=10)
        except subprocess.TimeoutExpired: gpu_process.kill(); gpu_process.wait()
        gpu_log.close(); monitor.stop()
        write_json(sample/'process-status.json',dict(complete=wall is not None,mpi_process_wall_seconds=wall,elapsed_seconds=time.monotonic()-started))
    result = json.loads(output.read_text())
    if not result['complete'] or result['events'] != count or result['n_gpu_streams'] != 1:
        raise RuntimeError('Worker did not complete requested workload')
    from screen import gpu_samples
    result.update(case=case['name'],label=label,preflight=preflight,mpi_process_wall_seconds=wall,
                  gpu_memory=gpu_samples(sample/'gpu.csv'),storage_note=network['scope'])
    write_json(sample/'sample.json',result)
    print(f"PASS {label}: {result['events_per_second']:.3f} events/s; MPI wall {wall:.2f}s",flush=True)
    return result


def pilot_estimate(sample):
    result = json.loads((sample/'sample.json').read_text())
    if result['events'] != 10000 or not result['complete']:
        raise RuntimeError('Pilot must finish all 10000 events')
    rows = [json.loads(s) for s in (sample/'progress-rank-2.jsonl').read_text().splitlines()]
    by_count = {r['events']:r for r in rows}
    late = (10000-5000)/(by_count[10000]['seconds']-by_count[5000]['seconds'])
    quarters = [(b-a)/(by_count[b]['seconds']-by_count[a]['seconds']) for a,b in [(2500,5000),(5000,7500),(7500,10000)]]
    if not all(math.isfinite(x) and x>0 for x in [late]+quarters):
        raise RuntimeError('Invalid sustained pilot rate')
    estimate = dict(completed_events=10000,late_events_per_second=late,quarter_rates=quarters,
                    measured_mpi_wall_seconds=result['mpi_process_wall_seconds'],
                    stability_ratio=min(quarters)/max(quarters),
                    timeout_seconds=math.ceil(max(result['mpi_process_wall_seconds'],10000/min(quarters))*3+60),
                    note='Base BD1 pilot; sizing evidence only. Cache uncontrolled; not a performance comparison.')
    return estimate


def discover_gpu(result_dir):
    command = ['srun','--mpi=none','--nodes=1','--ntasks=1','--cpus-per-task=8',
               '--cpu-bind=cores','--gpus-per-node=1','--gpu-bind=none','--exact',
               sys.executable,'-B',str(ROOT/'harness/gpu_identity.py')]
    try:
        result = subprocess.run(command,text=True,capture_output=True,timeout=45)
    except subprocess.TimeoutExpired as exc:
        def decoded(value):
            return value.decode(errors='replace') if isinstance(value,bytes) else value
        write_json(result_dir/'gpu-identity-command.json',dict(argv=command,timed_out=True,
                   timeout_seconds=45,stdout=decoded(exc.stdout),stderr=decoded(exc.stderr)))
        raise
    write_json(result_dir/'gpu-identity-command.json',dict(argv=command,returncode=result.returncode,
               stdout=result.stdout,stderr=result.stderr))
    result.check_returncode()
    gpu = json.loads(result.stdout)
    if not gpu['complete']: raise RuntimeError('GPU identity failed')
    return gpu


def case_sizing(result):
    if not result['complete'] or result['events'] != 10000 or result['runtime'] != 'base' or result['preflight']:
        raise RuntimeError('Sizing requires a completed base 10000-event sample')
    wall = result['mpi_process_wall_seconds']
    if not math.isfinite(wall) or wall <= 0: raise RuntimeError('Invalid sizing wall time')
    return dict(measured_mpi_wall_seconds=wall,loop_seconds=result['loop_seconds'],
                timeout_seconds=math.ceil(3*wall+60),sample_label=result['label'])


def allocation_identity(phase='pilot'):
    expected = {'SLURM_JOB_NUM_NODES': '1', 'SLURM_NTASKS': '4', 'SLURM_CPUS_PER_TASK': '8'}
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f'Allocation mismatch: {name}={os.environ.get(name)}')
    raw = subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--json'],text=True)
    job = json.loads(raw)['jobs'][0]
    # Slurm JSON versions represent a time_limit as either minutes or a number object.
    limit = job['time_limit']
    if isinstance(limit,dict): limit = limit['number']
    expected_minutes = 30 if phase == 'pilot' else json.loads((ROOT/'manifest.json').read_text())['campaign_plan']['allocation_minutes']
    if not 0 < int(limit) <= expected_minutes or (phase == 'campaign' and int(limit) != expected_minutes) or job['account'] != 'lcls_g' or job['qos'] not in ('premium','gpu_premium'):
        raise RuntimeError('Allocation account/QoS/time limit mismatch')
    return dict(slurm_job=job,expected_environment=expected)


def remaining_allocation_seconds():
    return int(os.environ['SLURM_JOB_END_TIME']) - time.time()


def require_remaining(seconds):
    remaining = remaining_allocation_seconds()
    if remaining < seconds+45:
        raise RuntimeError(f'Insufficient allocation time: {remaining:.1f}s left, need {seconds+45}s')


def termination_handler(signum, frame):
    raise RuntimeError(f'Allocation termination/walltime signal {signum}')


def execute(phase):
    signal.signal(signal.SIGTERM, termination_handler)
    signal.signal(signal.SIGUSR1, termination_handler)
    manifest = check_frozen(); review_gate(phase)
    plan = campaign_plan(manifest) if phase == 'campaign' else None
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Compute allocation required')
    result_dir = ROOT/'results'/f"{phase}-job-{os.environ['SLURM_JOB_ID']}"; result_dir.mkdir()
    status = result_dir/'status.json'
    write_json(status,dict(phase=phase,complete=False,manifest_sha256=file_hash(ROOT/'manifest.json')))
    try:
        allocation = allocation_identity(phase)
        write_json(result_dir/'allocation.json',allocation)
        gpu = discover_gpu(result_dir)
        write_json(result_dir/'gpu-identity.json',gpu)
        network = inventory(); write_json(result_dir/'storage-inventory.json',network)
        mount = subprocess.check_output(['findmnt','-T',str(ROOT),'-J'],text=True)
        data_mount = subprocess.check_output(['findmnt','-T','/pscratch/sd/p/psdatmgr/psdm/mfx/mfx100848724/xtc','-J'],text=True)
        if 'lustre' not in data_mount.lower(): raise RuntimeError('Expected Lustre data mount')
        write_json(result_dir/'host.json',dict(host=os.uname().nodename,python=sys.version,lustre_stats_observable=network['lustre_stats_observable'],task_mount=json.loads(mount),data_mount=json.loads(data_mount),environment={k:os.environ.get(k) for k in ['SLURM_JOB_ID','SLURM_JOB_GPUS','CUDA_VISIBLE_DEVICES','LOADEDMODULES']}))
        timed=[]
        timeout = 180 if phase == 'pilot' else plan['preflight_timeout_seconds']
        for case in manifest['cases']:
            for i,variant in enumerate(case['variants']):
                require_remaining(timeout)
                launch(case,variant,f"preflight-{case['name']}-{i}",120,timeout,result_dir,gpu['gpu_uuid'],network,preflight=True)
        if phase=='pilot':
            estimates = {}
            for index,case in enumerate(manifest['cases']):
                require_remaining(600)
                label = 'sustained-base-gpu-bd1' if index == 0 else 'sustained-base-'+case['name']
                result = launch(case,case['variants'][0],label,10000,600,result_dir,gpu['gpu_uuid'],network,diagnostics=True)
                estimates[case['name']] = case_sizing(result)
            estimate=pilot_estimate(result_dir/'sustained-base-gpu-bd1')
            estimate['sample_directory']=str(result_dir/'sustained-base-gpu-bd1')
            estimate['case_estimates']=estimates
            estimate['note']='All6base cases completed10000events. Each case bound is3x measured MPIwall+60s; no head performance claim. BD1 late-rate diagnostic retained. Cache uncontrolled.'
            estimates[manifest['cases'][0]['name']]['timeout_seconds'] = max(estimate['timeout_seconds'], estimates[manifest['cases'][0]['name']]['timeout_seconds'])
            write_json(ROOT/'pilot-estimate.json',estimate)
        else:
            estimate=json.loads((ROOT/'pilot-estimate.json').read_text())
            if estimate['stability_ratio'] < 0.7: raise RuntimeError('Pilot too unstable for full campaign')
            for case in manifest['cases']:
                timeout = plan['cases'][case['name']]['timeout_seconds']
                for round_number in range(manifest['paired_rounds']):
                    pair={}
                    for i in pair_order(round_number):
                        require_remaining(timeout)
                        pair[i]=launch(case,case['variants'][i],f"timed-{case['name']}-pair{round_number}-{i}",10000,timeout,result_dir,gpu['gpu_uuid'],network)
                    from screen import paired_result
                    timed.append(paired_result(pair[0],pair[1]))
                    write_json(result_dir/'paired-results.json',timed)
        write_json(status,dict(phase=phase,complete=True,manifest_sha256=file_hash(ROOT/'manifest.json'),paired_samples=len(timed),run77='deferred: unavailable'))
    except BaseException as exc:
        write_json(status,dict(phase=phase,complete=False,error=repr(exc),manifest_sha256=file_hash(ROOT/'manifest.json')))
        raise


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['freeze','check','pilot','campaign']); args=parser.parse_args()
    if args.action=='freeze': freeze()
    elif args.action=='check': check_frozen()
    else: execute(args.action)
if __name__=='__main__': main()
