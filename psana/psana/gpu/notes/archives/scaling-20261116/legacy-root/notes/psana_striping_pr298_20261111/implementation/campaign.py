"""Fresh-process pilot or separately requested full screen; never auto-promote."""
import argparse,json,os,random,signal,subprocess,sys,time
from pathlib import Path
from support import ROOT,HERE,COUNTS,require,write,sha,load_inputs,verify_payloads,diagnostics,select_affinity

def schedule(mode):
    if mode=='pilot':
        rows=[{'id':'bind-c%02d'%c,'kind':'binding','count':c,'bds':1,'streams':1,'events':100} for c in COUNTS]
        for kind,n in [('tail',1003),('pilot',1000)]:
            rows += [{'id':'%s-b%d-s%d'%(kind,b,s),'kind':kind,'count':2,'bds':b,'streams':s,'events':n} for s in (1,2) for b in (1,4)]
        return rows
    require(mode=='screen','Unknownmode');rng=random.Random(20261111);out=[]
    base=[(c,s,b) for c in COUNTS for s in (1,2) for b in (1,4)]
    for block in range(3):
        shuffled=base.copy();rng.shuffle(shuffled)
        for c,s,b in shuffled:out.append({'id':'%02d-r%d-c%02d-b%d-s%d'%(len(out),block,c,b,s),'kind':'screen','count':c,'streams':s,'bds':b,'events':100000,'block':block})
    return out

def launch(tasks,program,*args):
    return ['srun','--mpi=cray_shasta','--nodes=1','--ntasks='+str(tasks),'--ntasks-per-node='+str(tasks),'--cpus-per-task='+str(120//tasks),'--gpus-per-node=4','--gpu-bind=none','--cpu-bind=none','--kill-on-bad-exit=1','numactl','--interleave=all',sys.executable,str(HERE/program),*map(str,args)]

def runstep(output,name,cmd,timeout):
    write(output/(name+'.command.json'),cmd)
    with (output/(name+'.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)

def verify_bundle():
    pins=json.loads((HERE/'bundle.json').read_text())
    require(all(sha(HERE/name)==h for name,h in pins['files'].items()),'Harness bundle changed')
    external=json.loads((HERE/'external-pins.json').read_text())
    require(all(sha(path)==h for path,h in external.items()),'Externalpinschanged')
    return pins

def main(attempt,mode):
    require(attempt.parent==ROOT and attempt.name.startswith(('pilot','screen')),'Unexpected attempt')
    pins=verify_bundle();output=attempt/'results';output.mkdir(exist_ok=False)
    start=time.time();state={'state':'running','mode':mode,'job':os.environ['SLURM_JOB_ID'],'started':start,'completed':[],'bundle':pins}
    write(output/'status.json',state);signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:
        info=load_inputs();verify_payloads(info)
        allocation=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');require(all(x not in ('','-1') for x in allocation),'No allocation GPUs')
        os.environ['JF_GPU_UUID']=subprocess.check_output(['nvidia-smi','-i',allocation[0],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,name,memory.total','--format=csv,noheader'],text=True)
        write(output/'gpu-inventory.json',{'allocation':allocation,'selected':os.environ['JF_GPU_UUID'],'inventory':inventory})
        write(output/'diagnostics-before.json',diagnostics())
        probes=[]
        for tasks in (3,6):
            runstep(output,'affinity-'+str(tasks),launch(tasks,'affinity_probe.py',output,tasks),120)
            probes.extend(json.loads((output/('affinity-%d-r%d.json'%(tasks,rank))).read_text()) for rank in range(tasks))
        cpus=select_affinity(probes);os.environ['JF_CPU_AFFINITY']=','.join(map(str,cpus))
        write(output/'affinity.json',{'cpus':cpus,'probes':probes})
        runstep(output,'gpu-tests',launch(1,'check_tests.py','device',output/'gpu-tests.json'),600)
        tests=json.loads((output/'gpu-tests.json').read_text());require(tests['exitcode']==0,'NativeGPUtests')
        runstep(output,'input-validation',launch(1,'validate_data.py',output/'input-validation.json'),1200)
        cases=schedule(mode);write(output/'schedule.json',{'mode':mode,'seed':20261111,'cases':cases})
        for case in cases:
            # The pilot is independently bounded; a screen requires its own submission.
            before=time.time();state['current']=case;write(output/'status.json',state)
            name=case['id'];cmd=launch(case['bds']+2,'runner.py','--count',case['count'],'--bds',case['bds'],'--streams',case['streams'],'--events',case['events'],'--kind',case['kind'],'--ranges',output/'ranges.json','--output',output/(name+'.json'))
            if case['kind']=='tail':cmd+=['--witness',str(output/'witness.json')]
            print('Starting',name,flush=True);runstep(output,name,cmd,600 if mode=='pilot' else 1200)
            result=json.loads((output/(name+'.json')).read_text());require(result['state']=='pass','Casefailed')
            state['completed'].append({'case':case,'result_sha256':sha(output/(name+'.json')),'wall_seconds':time.time()-before,'seconds':result['seconds'],'gib_s':result['gib_s']})
            write(output/'status.json',state);print(json.dumps(state['completed'][-1]),flush=True)
        verify_payloads(info);verify_bundle();state.update(state='cases_passed_pending_slurm',finished=time.time());state.pop('current',None)
        write(output/'complete.json',state);write(output/'status.json',state)
    except BaseException as e:
        state.update(state='failed',error=repr(e),finished=time.time());write(output/'status.json',state);raise
    finally:write(output/'diagnostics-after.json',diagnostics())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('mode',choices=('pilot','screen'));a=p.parse_args();main(a.attempt.resolve(),a.mode)
