"""Bounded node scaling pilot or separately prepared strong-scaling screen."""
import json,os,random,signal,subprocess,sys,time
from pathlib import Path
from support import HERE,ROOT,require,write,sha,load_inputs,verify_payloads,select_affinity,campaign_affinity,diagnostics
from memory_watch import Watch,validate_memory
from nic import snapshot

SEED=20261116
LIMITS={'pilot':2700,'screen':7200}

def schedule(mode):
    def case(name,gpus,events,kind,policy='compact',block=0):
        return dict(id=name,gpus=gpus,bds=4*gpus,events=events,kind=kind,cpu_policy=policy,block=block,count=1,streams=1,batch_size=200,workers_per_bd=4)
    if mode=='pilot':
        return [case('tail-g%d'%g,g,6403,'tail') for g in range(1,5)]+[
            case('legacy-g1',1,100000,'sizing','legacy'),case('compact-g1',1,100000,'sizing')]
    require(mode=='screen','Unknown mode');out=[];rng=random.Random(SEED)
    for block in range(1,4):
        sizes=list(range(1,5));rng.shuffle(sizes)
        out.extend(case('r%d-g%d'%(block,g),g,100000,'screen',block=block) for g in sizes)
    return out

def launch(tasks,program,*args):
    # CR_CORE_MEMORY: even logical counts avoid whole-core over-allocation.
    # The actual common mask is probed and must fit the shared 42 cores plus 12 additional four-GPU cores.
    return ['srun','--mpi=cray_shasta','--nodes=1','--ntasks='+str(tasks),'--ntasks-per-node='+str(tasks),'--cpus-per-task='+str(2*(64//tasks)),'--gpus-per-node=4','--gpu-bind=none','--cpu-bind=none','--kill-on-bad-exit=1','numactl','--interleave=all',sys.executable,str(HERE/program),*map(str,args)]

def runstep(output,name,cmd,timeout):
    write(output/(name+'.command.json'),cmd);started=time.time()
    try:
        with (output/(name+'.log')).open('x') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    finally:write(output/(name+'.timing.json'),{'seconds':time.time()-started})

def verify_bundle():
    pins=json.loads((HERE/'bundle.json').read_text())
    require(all(sha(HERE/name)==h for name,h in pins['files'].items()),'Bundle changed')
    require(all(sha(path)==h for path,h in json.loads((HERE/'external-pins.json').read_text()).items()),'External pins changed')
    return pins

def native_ok(result):
    expected=json.loads((HERE/'expected-native.json').read_text())
    require(result['exitcode']==0 and result['collected']==expected['collected'] and result['sources']==expected['sources'],'Native GPU test inventory/source')
    require(not any(r['outcome'] in ('failed','skipped') for r in result['reports']) and sum(r['when']=='call' and r['outcome']=='passed' for r in result['reports'])==len(expected['collected']),'Native GPU test calls')

def deadline(plan,slurm):
    import re
    from datetime import datetime,timezone
    def field(name):
        found=re.findall(r'(?:^|\s)'+name+r'=(\S+)',slurm);require(len(found)==1,'Slurm '+name);return found[0]
    require(field('Account')=='lcls_g' and field('QOS')=='gpu_premium' and field('NumNodes')=='1','Allocation account/QOS/nodes')
    start=datetime.strptime(field('StartTime'),'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    end=datetime.strptime(field('EndTime'),'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    require(end-start==LIMITS[plan['mode']]==plan['requested_seconds'],'Allocation duration')
    return end-90

def main(attempt):
    require(attempt.parent==ROOT and attempt.name.startswith('scale-'),'Unexpected attempt')
    pins=verify_bundle();plan=json.loads((attempt/'plan.json').read_text());mode=plan['mode']
    require(plan['commit']==pins['commit'],'Prepared commit mismatch')
    from prepare import validate_plan
    validate_plan(plan)
    until=deadline(plan,(attempt/'slurm-job.txt').read_text())
    output=attempt/'results';output.mkdir(exist_ok=False)
    state=dict(state='running',mode=mode,job=os.environ['SLURM_JOB_ID'],bundle=pins,plan=plan,deadline_unix=until,started=time.time(),completed=[])
    write(output/'status.json',state);signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:
        info=load_inputs();verify_payloads(info);snapshot() # Require actual hardware counters before costly preflight.
        allocation=os.environ.get('CUDA_VISIBLE_DEVICES','');require(all(x not in ('','-1') for x in allocation.split(',')),'Missing GPUs')
        uuids=subprocess.check_output(['nvidia-smi','-i',allocation,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
        require(len(uuids)==len(set(uuids))==4,'Four allocated GPUs required')
        inventory=subprocess.check_output(['nvidia-smi','-i',allocation,'--query-gpu=uuid,pci.bus_id,name,memory.total','--format=csv,noheader'],text=True)
        # Numeric PCI order, fixed nested subsets on this node.
        uuids=[row.split(',')[0].strip() for row in sorted(inventory.strip().splitlines(),key=lambda r:r.split(',')[1].strip())]
        write(output/'gpu-inventory.json',dict(allocation=allocation,uuids=uuids,inventory=inventory))
        write(output/'diagnostics-before.json',diagnostics());probes={}
        for tasks in (6,10,14,18):
            runstep(output,'affinity-'+str(tasks),launch(tasks,'affinity_probe.py',output,tasks),120)
            probes[tasks]=[json.loads((output/('affinity-%d-r%d.json'%(tasks,r))).read_text()) for r in range(tasks)]
        # Shared ranks retain the same cores; new BDs use available step cores.
        maps=campaign_affinity(probes)
        write(output/'affinity.json',dict(maps=maps,probes=probes))
        for index,uuid in enumerate(uuids):
            os.environ['JF_GPU_UUID']=uuid
            runstep(output,'gpu-tests-'+str(index),launch(1,'check_tests.py','device',output/('gpu-tests-%d.json'%index)),600)
            test_result=json.loads((output/('gpu-tests-%d.json'%index)).read_text());native_ok(test_result)
            require(test_result['device']['uuid']==uuid,'Native GPU test assignment')
        runstep(output,'input-validation',launch(1,'validate_data.py',output/'input-validation.json'),1200)
        cases=schedule(mode);write(output/'schedule.json',dict(seed=SEED,cases=cases))
        from audit import audit_case
        for index,case in enumerate(cases):
            require(until-time.time()>=600*(4 if mode=='screen' and index%4==0 else 1),'Insufficient time for case/block')
            os.environ['JF_GPU_UUIDS']=json.dumps(uuids[:case['gpus']]);os.environ['JF_CPU_POLICY']=case['cpu_policy']
            os.environ['JF_CPU_AFFINITY']=json.dumps(maps['legacy' if case['cpu_policy']=='legacy' else str(case['gpus'])])
            state['current']=case;write(output/'status.json',state);name=case['id'];before=time.time()
            cmd=launch(case['bds']+2,'runner.py','--count',1,'--bds',case['bds'],'--gpus',case['gpus'],'--cpu-policy',case['cpu_policy'],'--streams',1,'--events',case['events'],'--kind',case['kind'],'--ranges',output/'ranges.json','--output',output/(name+'.json'))
            if case['kind']=='tail':cmd+=['--witness',str(output/'witness.json')]
            watches=[];sampled=[]
            try:
                for uuid in uuids[:case['gpus']]:
                    watch=Watch(uuid);watch.start();watches.append(watch)
                print('Starting '+name,flush=True);runstep(output,name,cmd,600)
            finally:
                errors=[]
                for watch in watches:
                    try:sampled.append(watch.stop())
                    except BaseException as e:errors.append(repr(e))
                write(output/(name+'.memory.json'),dict(devices=sampled,errors=errors))
            require(not errors and len(sampled)==case['gpus'],'Memory sampler completeness')
            result=json.loads((output/(name+'.json')).read_text());audit_case(result,case,info)
            for receipt in sampled:validate_memory(receipt,result['measurement']['started_unix'],result['measurement']['finished_unix'])
            state['completed'].append(dict(case=case,result_sha256=sha(output/(name+'.json')),memory_sha256=sha(output/(name+'.memory.json')),wall_seconds=time.time()-before,seconds=result['seconds'],gib_s=result['gib_s']))
            write(output/'status.json',state);print(json.dumps(state['completed'][-1]),flush=True)
        verify_payloads(info);verify_bundle();state.update(state='cases_passed_pending_slurm',finished=time.time());state.pop('current',None)
        write(output/'complete.json',state);write(output/'status.json',state)
    except BaseException as e:
        state.update(state='failed',error=repr(e),finished=time.time());write(output/'status.json',state);raise
    finally:write(output/'diagnostics-after.json',diagnostics())

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
