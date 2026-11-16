"""Fresh-process pilot or separately requested full screen; never auto-promote."""
import argparse,json,os,random,signal,subprocess,sys,time
from pathlib import Path
from memory_watch import Watch,validate_memory
from launch_plan import check_allocation,verify_plan,allocation_deadline
from support import ROOT,HERE,COUNTS,require,write,sha,load_inputs,verify_payloads,diagnostics,select_affinity

def schedule(mode):
    fixed=dict(bds=4,streams=1,batch_size=200,workers_per_bd=4)
    if mode=='qualify':
        return [dict(id='bind-c%02d'%c,kind='binding',count=c,events=100,**fixed) for c in COUNTS]+[
            dict(id='tail-c02',kind='tail',count=2,events=1003,**fixed),
            dict(id='sizing-c02',kind='sizing',count=2,events=100000,**fixed)]
    require(mode=='screen','Unknown mode')
    import csv
    with (HERE/'sweep.csv').open() as f:rows=list(csv.DictReader(f))
    out=[]
    rng=random.Random(20261115);expected=[]
    for block in range(1,4):
        counts=list(COUNTS);rng.shuffle(counts)
        for c in counts:expected.append((block,c))
    require([(int(r['block']),int(r['stripe_count'])) for r in rows]==expected,'Frozen schedule differs')
    for i,r in enumerate(rows):
        require(all(int(r[k])==v for k,v in dict(events=100000,bds=4,active_gpus=1,cuda_streams_per_bd=1,kvikio_workers_per_bd=4,batch_size=200,stripe_size_bytes=1048576,bulk_target_bytes=524288,kvikio_task_bytes=1048576,kvikio_bounce_bytes=16777216,gpu_budget_gib_per_bd=6,proposed_bound_physical_cores=34).items()),'CSV controls differ')
        out.append(dict(id='%02d-r%s-c%02d'%(i,r['block'],int(r['stripe_count'])),kind='screen',count=int(r['stripe_count']),events=100000,block=int(r['block']),**fixed))
    return out

def launch(tasks,program,*args):
    return ['srun','--mpi=cray_shasta','--nodes=1','--ntasks='+str(tasks),'--ntasks-per-node='+str(tasks),'--cpus-per-task='+str(120//tasks),'--gpus-per-node=4','--gpu-bind=none','--cpu-bind=none','--kill-on-bad-exit=1','numactl','--interleave=all',sys.executable,str(HERE/program),*map(str,args)]

def runstep(output,name,cmd,timeout):
    write(output/(name+'.command.json'),cmd)
    start=time.time()
    try:
        with (output/(name+'.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    finally:write(output/(name+'.timing.json'),{'seconds':time.time()-start})

def verify_bundle():
    pins=json.loads((HERE/'bundle.json').read_text())
    require(all(sha(HERE/name)==h for name,h in pins['files'].items()),'Harness bundle changed')
    external=json.loads((HERE/'external-pins.json').read_text())
    require(all(sha(path)==h for path,h in external.items()),'Externalpinschanged')
    return pins

def main(attempt,mode):
    require(attempt.parent==ROOT and attempt.name.startswith(('qualify','screen')),'Unexpected attempt')
    plan=json.loads((attempt/'launch-plan.json').read_text());check_allocation(plan,(attempt/'slurm-job.txt').read_text(),mode)
    verify_plan(plan,mode)
    deadline=allocation_deadline(plan,(attempt/'slurm-job.txt').read_text())
    pins=verify_bundle();output=attempt/'results';output.mkdir(exist_ok=False)
    start=time.time();state={'state':'running','mode':mode,'job':os.environ['SLURM_JOB_ID'],'started':start,'completed':[],'bundle':pins,'launch_plan':plan,'deadline_unix':deadline}
    write(output/'status.json',state);signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:
        info=load_inputs();verify_payloads(info)
        allocation=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');require(all(x not in ('','-1') for x in allocation),'No allocation GPUs')
        os.environ['JF_GPU_UUID']=subprocess.check_output(['nvidia-smi','-i',allocation[0],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,name,memory.total','--format=csv,noheader'],text=True)
        write(output/'gpu-inventory.json',{'allocation':allocation,'selected':os.environ['JF_GPU_UUID'],'inventory':inventory})
        write(output/'diagnostics-before.json',diagnostics())
        probes=[]
        for tasks in (6,):
            runstep(output,'affinity-'+str(tasks),launch(tasks,'affinity_probe.py',output,tasks),120)
            probes.extend(json.loads((output/('affinity-%d-r%d.json'%(tasks,rank))).read_text()) for rank in range(tasks))
        cpus=select_affinity(probes);os.environ['JF_CPU_AFFINITY']=json.dumps(cpus)
        write(output/'affinity.json',{'cpus':cpus,'probes':probes})
        runstep(output,'gpu-tests',launch(1,'check_tests.py','device',output/'gpu-tests.json'),600)
        tests=json.loads((output/'gpu-tests.json').read_text());expected=json.loads((HERE/'expected-native.json').read_text())
        require(tests['exitcode']==0 and tests['collected']==expected['collected'] and tests['sources']==expected['sources'],'NativeGPUtests')
        runstep(output,'input-validation',launch(1,'validate_data.py',output/'input-validation.json'),1200)
        cases=schedule(mode);write(output/'schedule.json',{'mode':mode,'seed':20261115,'cases':cases})
        for case in cases:
            remaining=deadline-time.time()
            needed=plan['per_case_timeout']*(6 if mode=='screen' and (len(state['completed'])%6==0) else 1)
            require(remaining>=needed,'Insufficient walltime for next case/block')
            # The pilot is independently bounded; a screen requires its own submission.
            before=time.time();state['current']=case;write(output/'status.json',state)
            name=case['id'];cmd=launch(case['bds']+2,'runner.py','--count',case['count'],'--bds',case['bds'],'--streams',case['streams'],'--events',case['events'],'--kind',case['kind'],'--ranges',output/'ranges.json','--output',output/(name+'.json'))
            if case['kind']=='tail':cmd+=['--witness',str(output/'witness.json')]
            print('Starting',name,flush=True)
            watch=Watch(os.environ['JF_GPU_UUID']);watch.start()
            try:runstep(output,name,cmd,plan['per_case_timeout'])
            finally:
                sampled=watch.stop();write(output/(name+'.memory.json'),sampled)
            validate_memory(sampled)

            result=json.loads((output/(name+'.json')).read_text());require(result['state']=='pass','Casefailed')
            validate_memory(sampled,result['measurement']['started_unix'],result['measurement']['finished_unix'])
            from audit import audit_case
            audit_case(result,case,info)

            state['completed'].append({'case':case,'result_sha256':sha(output/(name+'.json')),'memory_sha256':sha(output/(name+'.memory.json')),'wall_seconds':time.time()-before,'seconds':result['seconds'],'gib_s':result['gib_s']})
            write(output/'status.json',state);print(json.dumps(state['completed'][-1]),flush=True)
        state['preflight_timing_hashes']={n:sha(output/(n+'.timing.json')) for n in ('affinity-6','gpu-tests','input-validation')}
        verify_payloads(info);verify_bundle();state.update(state='cases_passed_pending_slurm',finished=time.time());state.pop('current',None)
        write(output/'complete.json',state);write(output/'status.json',state)
    except BaseException as e:
        state.update(state='failed',error=repr(e),finished=time.time());write(output/'status.json',state);raise
    finally:write(output/'diagnostics-after.json',diagnostics())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('mode',choices=('qualify','screen'));a=p.parse_args();main(a.attempt.resolve(),a.mode)
