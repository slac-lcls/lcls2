"""Three preflights, 36 clean startup samples and optional separate Nsight traces."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from common import reset_private_stage_cache, cache_inputs
from system_snapshot import snapshot
from schedule import schedule, validate, CASES, SEED
from analysis import summarize, report

ROOT = Path(os.environ['CAMPAIGN_ROOT'])
OLD = Path('/pscratch/sd/m/monarin/psana2-gpu/pr187-pr208-perlmutter-20261026/attempt01')
DEPS = Path('/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env')


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16<<20),b''):
            h.update(block)
    return h.hexdigest()


def records(text, prefix):
    return [json.JSONDecoder().raw_decode(s)[0] for s in text.split(prefix)[1:]]


def clean(variant, *args):
    return ['/usr/bin/python3', str(ROOT/'scripts/clean_exec.py'), str(ROOT), variant, *map(str,args)]


def mpi(variant, *args, minutes=15):
    return ['srun','--overlap','--distribution=block:block','--gpus-per-node=4',
            '--mpi=cray_shasta','--nodes=1','--ntasks=6','--cpus-per-task=8',
            '--cpu-bind=none','--gpu-bind=none','--kill-on-bad-exit=1',
            f'--time=00:{minutes:02}:00',*clean(variant,*args)]


def run(command, log, timeout=1000):
    with log.open('w') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def freeze(output):
    paths=[]
    for directory in ('scripts','harness','install-A','install-B'):
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for directory in ('cupy','cupyx','cupy_backends','kvikio','libkvikio','numpy','numpy.libs','mpi4py'):
        paths += [p for p in (DEPS/'lib/python3.11/site-packages'/directory).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts]
    paths += [DEPS/'bin/python', ROOT/'provenance/states.json', ROOT/'provenance/HANDOFF.md']
    paths += list(ROOT.glob('*.sbatch'))
    paths += [ROOT/n for n in ('reference.json','pixels.json','inputs-ready.json',
                              'constants-normalization.json','constants-normalized.pkl.gz')]
    states=json.loads((ROOT/'provenance/states.json').read_text())
    for variant,expected in states.items():
        source=ROOT/f'source-{variant}'
        assert subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)==''
        for name,ref in (('commit','HEAD'),('tree','HEAD^{tree}')):
            assert subprocess.check_output(['git','-C',str(source),'rev-parse',ref],text=True).strip()==expected[name]
        names=subprocess.check_output(['git','-C',str(source),'ls-files','-z']).decode().split('\0')
        paths += [source/n for n in names if n and (source/n).is_file()]
    paths=sorted({p.resolve() for p in paths})
    hashes={str(p):sha(p) for p in paths}
    # Reuse the original builds only if their accepted bytes are unchanged.
    accepted=json.loads((OLD/'triples-60376655/frozen-hashes.json').read_text())
    matched=0
    for path,digest in accepted.items():
        resolved=str(Path(path).resolve())
        if resolved in hashes:
            assert hashes[resolved]==digest, path
            matched+=1
    assert matched>1000, matched
    save(output/'inherited-runtime-check.json',dict(matched=matched,accepted=True))
    save(output/'frozen-hashes.json',hashes)
    return hashes


def external(output, phase):
    reports={}
    for variant in 'AB':
        text=subprocess.check_output(clean(variant,'env','CUDA_VISIBLE_DEVICES=0','PS_PARALLEL=none',
            'python','-B',ROOT/'scripts/external_inventory.py','--gpu'),text=True,timeout=180)
        reports[variant]=json.loads(text)
        save(output/f'external-{variant}-{phase}.json',reports[variant])
        for key in ('python','packages','mpi_library','cuda','external_libraries'):
            assert reports[variant][key]==reports['A'][key],(variant,key)
            if phase=='after':
                assert reports[variant][key]==json.loads((output/f'external-{variant}-before.json').read_text())[key]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--mode',choices=('pilot','sweep'),default='sweep')
    args=parser.parse_args()
    plan=validate(schedule())
    if args.check_only:
        print(json.dumps(dict(schedule=plan,clean_samples=36,preflights=3,traces=3),indent=2))
        return
    assert int(os.environ['SLURM_JOB_NUM_NODES'])==1
    assert os.environ['CUDA_VISIBLE_DEVICES'].split(',')==['0','1','2','3']
    output=ROOT/f'{args.mode}-{os.environ["SLURM_JOB_ID"]}'
    output.mkdir(exist_ok=False)
    shutil.copytree(ROOT/'scripts',output/'validated-source/scripts',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'harness',output/'validated-source/harness',ignore=shutil.ignore_patterns('__pycache__'))
    for launcher in ROOT.glob('*.sbatch'):
        shutil.copy2(launcher,output/'validated-source'/launcher.name)
    ready=json.loads((ROOT/'inputs-ready.json').read_text())
    assert ready['complete'] and Path(ready['stage'])==ROOT/'inputs'
    for name,key in (('pixels.json','pixels_sha256'),('reference.json','reference_sha256'),
                     ('constants-normalized.pkl.gz','constants_sha256')):
        assert sha(ROOT/name)==ready[key]
    stage=ROOT/'inputs'
    def verify_stream(entry):
        path=stage/entry['name']
        assert not path.is_symlink() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(verify_stream,ready['streams']))
    reference=json.loads((ROOT/'reference.json').read_text())
    for name,digest in reference['10000']['smd_hashes'].items():
        assert sha(stage/'smalldata'/name)==digest
    hashes=freeze(output)
    external(output,'before')
    run(['/usr/bin/python3',str(ROOT/'scripts/inventory.py'),str(output/'inventory')],output/'inventory.log')
    affinity=list(range(24))+list(range(64,88))
    assert set(affinity)<=set(os.sched_getaffinity(0))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total','--format=csv,noheader'],text=True)
    bus=next(l.split(',')[2].strip().lower().split(':',1)[-1] for l in gpu.splitlines() if l.split(',')[0].strip()=='0')
    metadata=dict(job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,mode=args.mode,
        complete=False,clean_complete=False,seed=SEED,schedule=plan,inputs=ready,gpus=gpu,
        sources=json.loads((ROOT/'provenance/states.json').read_text()),affinity=affinity,
        settings=dict(batch=20,streams=1,budget_bytes_per_bd=6*2**30,kvikio_threads=8,
                      task_bytes=1048576,compatibility='ON',bulk=True),
        cache_semantics='Client page-cache cold <=1% per measured private prefix; server caches uncontrolled.')
    save(output/'provenance.json',metadata)
    libraries=set()
    for i,variant in enumerate('ABA'):
        log=output/f'switch-{i}-{variant}.log'
        run(mpi(variant,'python','-B',ROOT/'scripts/gpu_probe.py'),log,240)
        probe=records(log.read_text(),'GPU_SWITCH_PROBE ')
        assert len(probe)==1 and len(probe[0])==6
        assert all(r['provenance']['variant']==variant for r in probe[0])
        libraries.update(path for r in probe[0] for path in r['provenance']['loaded_libraries'])
    runtime_hashes={path:sha(path) for path in sorted(libraries|{'/usr/bin/srun','/usr/bin/python3','/bin/bash'})}
    save(output/'runtime-library-hashes.json',runtime_hashes)
    hashes.update(runtime_hashes)
    results=[]

    def sample(case, events, repetition, diagnostic=False, profiler=None):
        variant='A' if case=='A' else 'B'
        trace=profiler is not None
        tag=f'{case}-n{events}-r{repetition}-'+('preflight' if diagnostic else 'trace' if trace else 'clean')
        envargs=['PSANA_GPU_SHARE_CONSTANTS='+('1' if case=='B-on' else '0'),
                 'SLURM_GPUS_ON_NODE=1','BENCH_GPUS=1','VALIDATION_CUDA_DEVICE=0',
                 'VALIDATION_CUDA_VISIBLE_DEVICES=0','BENCH_CPU_AFFINITY='+','.join(map(str,affinity))]
        target=['python','-B',ROOT/'scripts/calibration.py','--root',ROOT,'--directory',stage,
                '--variant',variant,'--case',case,'--events',events]
        if diagnostic:target+=['--diagnostic']
        if trace:
            target+=['--trace']
            destination=output/'traces'/case
            destination.mkdir(parents=True,exist_ok=False)
            target=['python','-B',ROOT/'scripts/rank_trace.py',profiler,destination,*target]
        command=mpi(variant,'env',*envargs,*target,minutes=5 if trace else 15)
        save(output/f'{tag}-command.json',command)
        before=None
        if not diagnostic:
            before=reset_private_stage_cache(stage,ranges=reference[str(events)]['prefixes'])
            save(output/f'{tag}-cache-before.json',before)
        save(output/f'{tag}-system-before.json',snapshot())
        print('BEGIN',tag,flush=True)
        started=time.perf_counter()
        with (output/f'{tag}-gpu.csv').open('w') as f:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,clocks.sm',
                                      '--format=csv,noheader,nounits','-lms','250'],stdout=f)
            try:run(command,output/f'{tag}.log',420 if trace else 1000)
            finally:monitor.terminate();monitor.wait(timeout=10)
        wall=time.perf_counter()-started
        save(output/f'{tag}-system-after.json',snapshot())
        after=None if diagnostic else cache_inputs(stage,'cold',prepare=False,ranges=reference[str(events)]['prefixes'])
        log=(output/f'{tag}.log').read_text()
        found=records(log,'PERF_RESULT ')
        assert len(found)==1
        row=found[0]
        assert (row['case'],row['events'],row['bds'])==(case,events,4)
        origins=records(log,'RUNTIME_PROVENANCE ')
        for audits in (origins,row['final_origins']):
            assert len(audits)==6 and {p['rank'] for p in audits}==set(range(6))
            assert all(p['install']==str((ROOT/f'install-{variant}').resolve()) for p in audits)
        assert all(r['affinity']==affinity and r['effective_share']==envargs[0].split('=')[1] for r in row['rows'])
        assert all(r['snapshots'][0]['pci'].lower().split(':',1)[-1]==bus for r in row['rows'] if r['role']=='bd')
        row.update(tag=tag,repetition=repetition,sample_wall_s=wall,before=before,after=after,
                   payload_GB_s=row['bytes']/row['loop_s']/1e9)
        results.append(row)
        save(output/'results.json',results)
        print('PASS',tag,row['events_per_s'],'events/s','first_batch_s',row['first_batch_s'],flush=True)
        return row

    for case in CASES:
        sample(case,200,0,diagnostic=True)
    save(output/'preflight-complete.json',dict(passed=3))
    if args.mode=='pilot':
        for case in CASES:
            sample(case,1000,0)
        external(output,'after')
        for path,digest in hashes.items():assert sha(path)==digest,path
        metadata['complete']=True
        save(output/'provenance.json',metadata)
        print('COMPLETE pilot',flush=True)
        return
    for block in plan:
        for case in block['order']:
            sample(case,block['events'],block['repetition'])
    summarize(output)
    external(output,'after')
    for path,digest in hashes.items():assert sha(path)==digest,path
    metadata['clean_complete']=True
    save(output/'provenance.json',metadata)
    report(output)
    print('CLEAN_COMPLETE 36 samples',flush=True)
    profiler=shutil.which('nsys')
    traces=dict(profiler=profiler,trace='cuda,nvtx,osrt,mpi',mpi_impl='mpich',samples=[])
    if profiler:
        run([profiler,'--version'],output/'nsys-version.log',30)
        for case in CASES:
            try:
                sample(case,1000,0,profiler=profiler)
                files=list((output/'traces'/case).glob('*.nsys-rep'))
                assert len(files)==6, [str(p) for p in files]
                traces['samples'].append(dict(case=case,complete=True,files=list(map(str,files))))
                for capture in files:
                    command=[profiler,'stats','--report','cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum,mpi_event_sum',
                             '--format','csv',str(capture)]
                    try:run(command,capture.with_suffix('.stats.csv'),180)
                    except Exception as exc:
                        traces.setdefault('stats_errors',[]).append(dict(file=str(capture),error=str(exc)))
            except Exception as exc:
                traces['samples'].append(dict(case=case,complete=False,error=str(exc)))
            save(output/'traces-status.json',traces)
    else:
        traces['blocker']='nsys unavailable in activated environment'
    save(output/'traces-status.json',traces)
    for path,digest in hashes.items():assert sha(path)==digest,path
    metadata['complete']=True
    save(output/'provenance.json',metadata)
    report(output)
    print('COMPLETE campaign',flush=True)


if __name__=='__main__':
    main()
