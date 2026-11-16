"""Exact A/B/C integrations: pixel/IPC preflights then balanced paired samples."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from common import cache_inputs, sha, reset_private_stage_cache
from memory_state import snapshot
from clean_runtime import runtime_environment
from system_snapshot import snapshot as system_snapshot


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    tmp.replace(path)


def records(text, prefix):
    # MPI stdout may concatenate a marker with another rank's newline.
    # Decode the complete JSON after each marker without depending on line starts.
    return [json.JSONDecoder().raw_decode(part)[0] for part in text.split(prefix)[1:]]


INTEGRATION = Path('/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-integration-20261026-r1')


def conditions():
    raw = [('raw', 4, cache, bulk, 'AB') for cache in ('cold','warm') for bulk in ('off','on')]
    raw += [('raw',4,'cold','off','BC')]
    raw += [('raw',4,cache,'off','AC') for cache in ('cold','warm')]
    calib = [('calibration',bds,'warm','on',pair) for bds in (1,4) for pair in ('AB','BC')]
    calib += [('calibration',4,'warm','on','AC')]
    return raw+calib


def schedule():
    for rep in (1,2,3,4):
        cases=conditions() if rep % 2 else list(reversed(conditions()))
        for workload,bds,cache,bulk,pair in cases:
            yield workload,bds,cache,bulk,pair,rep,(pair if rep % 2 else pair[::-1])


def stream_sha(path):
    import hashlib
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def verify(root, include_build_host=False):
    for manifest in (root/'hashes.json', INTEGRATION/'frozen-hashes.json'):
        hashes=json.loads(manifest.read_text())
        host_only=json.loads((root/'build-host-only-hashes.json').read_text()) if manifest.parent==INTEGRATION else {}
        assert all(path.startswith('/usr/lib64/') and hashes[path]==digest for path,digest in host_only.items())
        checked=0
        for path,digest in hashes.items():
            if path in host_only and not include_build_host:continue
            assert stream_sha(path)==digest,path
            checked+=1
        print('HASHES_VERIFIED',str(manifest),checked,'BUILD_HOST_ONLY_NOT_APPLICABLE',0 if include_build_host else len(host_only),flush=True)


def external_check(root, output, phase):
    reports = {}
    for variant in ('A','B','C'):
        env = runtime_environment(root, variant, os.environ)
        env.update(CUDA_VISIBLE_DEVICES='0', PS_PARALLEL='none')
        result = subprocess.run([sys.executable, '-B', str(root/'external_inventory.py'), '--gpu'],
                                env=env, text=True, capture_output=True, check=True, timeout=120)
        report = json.loads(result.stdout)
        reports[variant] = report
        save(output/(variant+'-external-'+phase+'.json'), report)
    for variant, report in reports.items():
        for key in ('python','packages','mpi_library','cuda','external_libraries'):
            assert report[key] == reports['A'][key], ('external mismatch',variant,key)
    if phase == 'after':
        for variant, report in reports.items():
            before = json.loads((output/(variant+'-external-before.json')).read_text())
            for key in ('python', 'packages', 'mpi_library', 'cuda', 'external_libraries'):
                assert before[key] == report[key], ('external dependency changed during job', variant, key)
    print('EXTERNAL_DEPENDENCIES_MATCH', phase, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    verify(root, include_build_host=a.check_only)
    cases = list(schedule())
    assert len(cases)==48 and sum(len(c[-1]) for c in cases)==96
    assert sum(2 for c in cases if c[0]=='raw')==56
    for condition in conditions():
        selected=[c for c in cases if c[:5]==condition]
        pair=condition[-1]
        assert [c[-1] for c in selected]==[pair,pair[::-1],pair,pair[::-1]]
    if a.check_only:
        print('SCHEDULE_CHECK_PASS: 12 conditions, four balanced pairs each, 96 timed samples')
        return
    assert os.uname().nodename.split('.')[0] == 'sdfada016'
    assert a.source is not None
    assert int(os.environ['SLURM_JOB_NUM_NODES']) == 1
    assert {str(i) for i in range(10)}.issubset(set(os.environ['SLURM_JOB_GPUS'].split(',')))
    assert os.environ['CUDA_VISIBLE_DEVICES'].split(',') == [str(i) for i in range(10)]
    job = os.environ['SLURM_JOB_ID']
    output = root/('job-'+job)
    output.mkdir(exist_ok=False)
    save(output/'system-before.json', system_snapshot())
    external_check(root, output, 'before')
    # Native provenance must pass on every rank before staging hundreds of GiB.
    for variant in ('A','B','C'):
        env = runtime_environment(root, variant, os.environ)
        env['CUDA_VISIBLE_DEVICES'] = ''
        log_path = output / (variant + '-mpi-imports.log')
        command = ['mpirun', '-n', '6', '--oversubscribe', '--bind-to', 'none',
                   sys.executable, '-B', str(root / variant / 'scripts/jf_scaling/check_runtime_provenance.py'),
                   '--install', str(root / variant / 'install'), '--mpi']
        with log_path.open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=120)
        imports = records(log_path.read_text(), 'RUNTIME_PROVENANCE ')
        assert len(imports) == 6 and {x['rank'] for x in imports} == set(range(6))
        assert all(x['install'] == str((root / variant / 'install').resolve()) for x in imports)
        save(output / (variant + '-mpi-imports.json'), imports)
        print('MPI_IMPORTS_PASS', variant, flush=True)
    manifest = json.loads((root/'reference.json').read_text())['10000']
    expected = json.loads((root/'expected-staged-sha256.json').read_text())
    capabilities={}
    for variant in ('A','B','C'):
        env=runtime_environment(root,variant,os.environ)
        env.update(CUDA_VISIBLE_DEVICES='0',PS_PARALLEL='none')
        with (output/(variant+'-adapter-check.log')).open('w') as log:
            subprocess.run([sys.executable,'-B',str(root/'check_adapter.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        probes=records((output/(variant+'-adapter-check.log')).read_text(),'GDS_PROBE ')
        assert len(probes)==1 and probes[0]['gds_available'] is False
        capabilities[variant]=probes[0]
    save(output/'capabilities.json',capabilities)
    stage = Path('/lscratch/monarin')/('jf-pr187-pr208-'+job)
    stage.mkdir(parents=True, exist_ok=False)
    affinity = sorted(os.sched_getaffinity(0))
    gpu_text = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version','--format=csv,noheader'],text=True)
    gpu0 = [l.split(',') for l in gpu_text.splitlines() if l.split(',')[0].strip() == '0'][0]
    assert 'L40S' in gpu0[3], gpu0
    metadata = dict(environment={v:runtime_environment(root,v,os.environ) for v in 'ABC'},job=job,host=os.uname().nodename,complete=False,source=str(a.source),stage=str(stage),
        plan=json.loads((root/'plan.json').read_text()),schedule=cases,affinity=affinity,gpus=gpu_text,
        gpu_topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        settings=dict(gpus_used=1,raw_bds=4,calibration_bds=[1,4],events=10000,batch=20,depth=1,workers=8,task_bytes=1<<20,
                      pairs_per_condition=4,physical_gpu=0,compatibility_mode='ON',raw_budget_gib_per_bd=6,calibration_budget='automatic'),
        filesystem=subprocess.check_output(['findmnt','-T',str(stage)],text=True))
    save(output/'provenance.json',metadata)
    rows = []
    try:
        sizes = manifest['stage_bytes']
        assert shutil.disk_usage(stage).free > sum(sizes.values())+20*1024**3
        def copy(name):
            import hashlib
            src, dest = a.source/name, stage/name
            before = src.stat(); remain = sizes[name]; h = hashlib.sha256()
            with src.open('rb',buffering=0) as f, dest.open('xb',buffering=0) as out:
                while remain:
                    chunk = f.read(min(remain,16<<20)); assert chunk, name
                    assert out.write(chunk) == len(chunk)
                    h.update(chunk);remain -= len(chunk)
                os.fsync(out.fileno())
            after = src.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            assert dest.stat().st_size == sizes[name] and h.hexdigest() == expected[name]
            print('STAGED',name,flush=True)
            return dict(name=name,bytes=sizes[name],sha256=h.hexdigest())
        with ThreadPoolExecutor(max_workers=5) as pool:
            metadata['staged'] = list(pool.map(copy,sizes))
        (stage/'smalldata').mkdir()
        for name in sizes:
            smd = name.replace('.xtc2','.smd.xtc2')
            shutil.copy2(a.source/'smalldata'/smd,stage/'smalldata'/smd)
            assert sha(stage/'smalldata'/smd) == manifest['smd_hashes'][smd]
        save(output/'provenance.json',metadata)

        def sample(variant, workload, bds, bulk, cache, rep, pair='preflight', diagnostic=False):
            build = root/variant
            tag = f'{workload}-{bds}bd-{pair}-r{rep}-{variant}-{bulk}-{cache}'
            env = runtime_environment(root, variant, os.environ)
            env.update(BENCH_GDS_AVAILABLE=str(int(capabilities[variant]['gds_available'])),
                       SLURM_GPUS_ON_NODE='1',VALIDATION_CUDA_DEVICE='0',VALIDATION_CUDA_VISIBLE_DEVICES='0',
                       BENCH_CPU_AFFINITY=','.join(map(str,affinity)))
            before = None
            if not diagnostic:
                assert cache in ('cold', 'warm')
                assert stage == Path('/lscratch/monarin') / ('jf-pr187-pr208-' + job)
                save(output/(tag+'-cache-memory-before.json'), snapshot())
                prep_start = time.perf_counter()
                reset = reset_private_stage_cache(stage, ranges=manifest['prefixes'])
                save(output/(tag+'-cache-reset.json'), reset)
                before = reset if cache == 'cold' else cache_inputs(stage,'warm',ranges=manifest['prefixes'])
                before['preparation_wall_s'] = time.perf_counter()-prep_start
                save(output/(tag+'-cache-before.json'),before)
                save(output/(tag+'-cache-memory-after-reset.json'),snapshot())
                print('CACHE_PREPARATION_PASS', tag,[r['resident_fraction'] for r in before['files']],flush=True)
            command = ['mpirun','-n',str(bds+2),'--oversubscribe','--bind-to','none',sys.executable,'-B','-u',
                       str(root/'rank_exec.py')]
            if workload=='raw':
                command += [str(build/'scripts/jf_scaling/bench.py'),'--directory',str(stage),
                    '--reference',str(root/'reference.json'),'--pixels',str(root/'pixels.json'),
                    '--constants',str(root/'constants.pkl.gz'),'--bulk',bulk,'--events','200' if diagnostic else '10000']
                if diagnostic:command += ['--check-pixels']
            else:
                command += [str(root/'calibration.py'),'--root',str(root),'--directory',str(stage),
                            '--variant',variant,'--events','200' if diagnostic else '10000']
                if diagnostic:command += ['--diagnostic']
            save(output/(tag+'-command.json'),dict(command=command,environment=env))
            save(output/(tag+'-system-before.json'), system_snapshot())
            print('BEGIN',tag,flush=True)
            started = time.perf_counter()
            with (output/(tag+'-gpu.csv')).open('w') as mon:
                monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,clocks.sm','--format=csv,noheader,nounits','-lms','250'],stdout=mon)
                try:
                    with (output/(tag+'.log')).open('w') as log:
                        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                finally:
                    monitor.terminate();monitor.wait(timeout=10)
            wall = time.perf_counter()-started
            save(output/(tag+'-system-after.json'), system_snapshot())
            # Post-sample measurement never reloads or repairs missing cache pages.
            after=None if diagnostic else cache_inputs(stage,cache,prepare=False,ranges=manifest['prefixes'])
            save(output/(tag+'-cache-after.json'),after)
            found = records((output/(tag+'.log')).read_text(),'JF_SCALE_RESULT ' if workload=='raw' else 'PERF_RESULT ')
            assert len(found)==1
            row=found[0]
            origins=records((output/(tag+'.log')).read_text(),'RUNTIME_PROVENANCE ')
            assert len(origins)==bds+2 and {p['rank'] for p in origins}==set(range(bds+2))
            assert all(p['install']==str((build/'install').resolve()) for p in origins)
            save(output/(tag+'-imports.json'),origins)
            ranks=row['ranks'] if workload=='raw' else row['rows']
            assert len(ranks)==bds+2 and all(r['affinity']==affinity for r in ranks)
            expected_bus=gpu0[2].strip().lower().split(':',1)[-1]
            if workload=='raw':
                assert (row['ngpus'],row['nbds'],row['bulk'])==(1,bds,bulk)
                assert not row['include_feespec']
                assert row['gpu_buses']['0'].lower().split(':',1)[-1]==expected_bus
                for rank in ranks:
                    if rank['is_bd']:
                        assert rank['timestamps'] and rank['physical_gpu']==0
                        assert rank['device']['kvikio']=='26.10.00' and rank['device']['cupy']=='14.0.1'
                        assert rank['counts']['budget_limit']==6*1024**3
                        assert rank['alignment']['shift']==0 and rank['alignment']['reader_allocations']>0
                audits=[r['alignment'] for r in ranks if r['is_bd']]
                if diagnostic:
                    assert sum(r['aligned_rows'] for r in audits)==200*32
                    assert sum(r['unaligned_rows'] for r in audits)==0
                    assert sum(r['audited_dgrams'] for r in audits)==200*5
                else:assert all(not r['audit'] and r['gather_calls']==0 for r in audits)
            else:
                assert row['bds']==bds
                assert all(r['snapshots'][0]['pci'].lower().split(':',1)[-1]==expected_bus for r in ranks if r['role']=='bd')
            row.update(variant=variant,workload=workload,bds=bds,bulk=bulk,cache=cache,repetition=rep,
                       pair=pair,tag=tag,diagnostic=diagnostic,sample_wall_s=wall,before=before,after=after,
                       log_sha256=sha(output/(tag+'.log')))
            rows.append(row);save(output/'results.json',rows)
            print(f'PASS {tag}: {row["events_per_s"]:.2f} events/s',flush=True)
            return row

        # Every exact runtime/configuration must pass before any timed sample.
        for variant in 'ABC':
            for bulk in ('off','on'):sample(variant,'raw',4,bulk,'uncontrolled',0,diagnostic=True)
            for bds in (1,4):sample(variant,'calibration',bds,'on','uncontrolled',0,diagnostic=True)
        save(output/'preflight-complete.json',dict(passed=12))
        print('ALL_12_GPU_PREFLIGHTS_PASS',flush=True)
        for workload,bds,cache,bulk,pair,rep,order in cases:
            for variant in order:sample(variant,workload,bds,bulk,cache,rep,pair)
        external_check(root, output, 'after')
        save(output/'system-after.json',system_snapshot())
        verify(root)
        summary=[]
        def stats(values):
            return dict(n=len(values),mean=statistics.mean(values),std=statistics.stdev(values),
                        median=statistics.median(values),samples=values,ddof=1)
        for workload,bds,cache,bulk,pair in conditions():
            selected=[r for r in rows if not r['diagnostic'] and
                      (r['workload'],r['bds'],r['cache'],r['bulk'],r['pair'])==(workload,bds,cache,bulk,pair)]
            assert len(selected)==8
            arms={v:[next(r for r in selected if r['variant']==v and r['repetition']==rep) for rep in (1,2,3,4)] for v in pair}
            changes=[100*(right['loop_s']/left['loop_s']-1) for left,right in zip(arms[pair[0]],arms[pair[1]])]
            entry=dict(workload=workload,bds=bds,cache=cache,bulk=bulk,pair=pair,
                paired_loop_percent_change=stats(changes),review_required=True,
                variants={v:{key:stats([r[key] for r in arm]) for key in
                    ('loop_s','events_per_s','setup_s','first_event_s','sample_wall_s')} for v,arm in arms.items()})
            if workload=='calibration' and pair=='BC':
                # Rank ownership and event distribution may vary: compare sorted budgets and
                # aggregate callback sizes, retaining all original per-rank evidence.
                def admission(r):
                    bd=[x for x in r['rows'] if x['role']=='bd']
                    sizes={}
                    for x in bd:
                        for k,v in x['callback_sizes'].items():sizes[k]=sizes.get(k,0)+v
                    return dict(limits=sorted(x['snapshots'][0]['budget_limit'] for x in bd),
                                subbatch=sorted(x['snapshots'][0]['subbatch_budget_bytes'] for x in bd),callback_sizes=sizes)
                entry['admission']=[dict(repetition=i+1,B=admission(left),C=admission(right),
                    exactly_equal=admission(left)==admission(right)) for i,(left,right) in enumerate(zip(arms['B'],arms['C']))]
            summary.append(entry)
        save(output/'summary.json',summary)
        metadata['complete']=True
        print('PERFORMANCE_CAMPAIGN_COMPLETE_REVIEW_REQUIRED',flush=True)
    finally:
        assert stage==Path('/lscratch/monarin')/('jf-pr187-pr208-'+job)
        shutil.rmtree(stage)
        metadata['removed_stage']=str(stage)
        save(output/'provenance.json',metadata)


if __name__=='__main__':
    main()
