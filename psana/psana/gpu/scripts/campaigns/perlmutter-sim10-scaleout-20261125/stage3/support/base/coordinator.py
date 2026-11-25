"""One immutable Stage 2 allocation; serial fresh MPI processes, fail closed."""
from pathlib import Path
import datetime,fcntl,json,math,os,re,shutil,signal,socket,subprocess,sys,time
from support import require,sha,write,verify_inputs,gpu_inventory
from qualified_support import select_affinity,diagnostics
from memory_watch import Watch,validate_memory
from audit import audit_case
from gate import preconditions
HERE=Path(__file__).resolve().parent
ROOT=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-stage2-20261124')
LOCK=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115/count2-use.lock')

def verify(attempt):
 plan=json.loads((attempt/'plan.json').read_text())
 require(attempt==Path(plan['attempt']) and attempt.parent==ROOT and HERE==attempt/'bundle','Prepared path')
 manifest=json.loads((HERE/'bundle.json').read_text())
 require(sha(HERE/'bundle.json')==plan['bundle_sha256'],'Manifest pin')
 require(all(sha(HERE/name)==value for name,value in manifest['files'].items()),'Bundle pin')
 require(all(sha(name)==value for name,value in manifest['external'].items()),'External pin')
 require(sum(plan['budget'].values())==plan['walltime_seconds']<=14400,'Budget')
 return plan

def launch(attempt,tasks,program,args,timeout):
 return ['srun','--overlap','--mpi=cray_shasta','--nodes=1','--ntasks='+str(tasks),'--ntasks-per-node='+str(tasks),'--cpus-per-task='+str(120//tasks),'--gpus-per-node=4','--gpu-bind=none','--cpu-bind=none','--kill-on-bad-exit=1','--time='+str(math.ceil(timeout/60)),'numactl','--interleave=all','/usr/bin/python3',str(HERE/'clean_exec.py'),str(attempt),'python',str(HERE/program),*map(str,args)]

def step(output,name,command,timeout,deadline):
 require(deadline-time.time()>=timeout+120,'Insufficient remaining time for complete step')
 write(output/(name+'.command.json'),command)
 started=time.time()
 try:
  with (output/(name+'.log')).open('x') as log:
   process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:require(process.wait(timeout=timeout)==0,'Step failed: '+name)
   except BaseException:
    if process.poll() is None:
     os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=15)
    raise
 finally:write(output/(name+'.timing.json'),dict(started=started,finished=time.time(),seconds=time.time()-started))

def main(attempt):
 plan=verify(attempt);text=(attempt/'slurm-job.txt').read_text()
 def field(k):
  values=re.findall(r'(?:^|\s)'+k+r'=(\S+)',text);require(len(values)==1,'Slurm '+k);return values[0]
 require(field('Account')=='lcls_g' and field('QOS') in ('interactive','gpu_interactive'),'Account/QoS')
 require(field('NumNodes')=='1' and field('TimeLimit')==plan['walltime'],'Allocation size/time')
 require(field('JobId')==os.environ['SLURM_JOB_ID'],'Job identity')
 deadline=datetime.datetime.fromisoformat(field('EndTime')).replace(tzinfo=datetime.timezone.utc).timestamp()
 require(0<deadline-time.time()<=plan['walltime_seconds']+2,'Allocation deadline')
 output=attempt/'results';output.mkdir()
 state=dict(status='running',job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),plan_sha256=sha(attempt/'plan.json'),completed=[])
 lock=LOCK.open('a')
 try:
  write(output/'status.json',state)
  write(output/'predecessors.json',preconditions(plan))
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  info=json.loads((attempt/'inputs.json').read_text());ranges=json.loads((attempt/'ranges.json').read_text())
  verify_inputs(info);require(shutil.disk_usage(attempt).free>=4<<30,'4 GiB output reserve')
  for role in plan['roles']:
   os.environ['SIM10_RUNTIME']=role;os.environ['SIM10_NUMERICAL_REFERENCE']='0'
   step(output,'runtime-'+role,launch(attempt,1,'runtime_check.py',[output/('runtime-'+role+'.json')],300),300,deadline)
  uuids=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
  require(len(uuids)==len(set(uuids))==4,'Four GPUs')
  pci=gpu_inventory(subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id','--format=csv,noheader'],text=True))
  require(set(pci)==set(uuids),'Granted inventory')
  write(output/'hardware.json',dict(uuids=uuids,pci_by_uuid=pci,diagnostics=diagnostics()))
  probes=[]
  for tasks in (3,6):
   step(output,'affinity-%d'%tasks,launch(attempt,tasks,'affinity_probe.py',[output,tasks],30),30,deadline)
   probes.extend(json.loads((output/('affinity-%d-r%d.json'%(tasks,r))).read_text()) for r in range(tasks))
  common=set.intersection(*(set(p['cpus']) for p in probes));six=[dict(p,cpus=sorted(common)) for p in probes[3:]]
  selected=select_affinity(six);mapping=[selected[str(r)] for r in range(6)]
  write(output/'affinity.json',dict(mapping=mapping,probes=probes))
  for role in plan['roles']:
   os.environ['SIM10_RUNTIME']=role
   for index,uuid in enumerate(uuids):
    os.environ['JF_GPU_UUID']=uuid
    name='device-%s-%d'%(role,index)
    step(output,name,launch(attempt,1,'device_check.py',[output/(name+'.json')],120),120,deadline)
  for case in plan['cases']:
   os.environ['SIM10_RUNTIME']=case['role'];os.environ['SIM10_NUMERICAL_REFERENCE']='1' if case['numerical'] else '0'
   gpu_plan=dict(rank_uuids=['','']+uuids[:case['bds']],pci_by_uuid=pci)
   cpu_plan=mapping[:case['bds']+2]
   os.environ['SIM10_GPU_PLAN']=json.dumps(gpu_plan);os.environ['SIM10_CPU_PLAN']=json.dumps(cpu_plan)
   name=case['name'];state['current']=case;write(output/'status.json',state)
   args=['--inputs',attempt/'inputs.json','--output',output/(name+'.json')]
   for key in ('bds','batch','stage','events','kind'):args+=['--'+key,case[key]]
   watches=[Watch(uuid) for uuid in uuids[:case['gpus']]]
   for watch in watches:watch.start()
   try:step(output,name,launch(attempt,case['bds']+2,'runner.py',args,case['timeout']),case['timeout'],deadline)
   finally:
    memory=[w.stop() for w in watches];write(output/(name+'.memory.json'),memory)
   for receipt in memory:validate_memory(receipt)
   report=json.loads((output/(name+'.json')).read_text())
   require(report['job']==state['job'] and {r['host'] for r in report['rows']}=={state['node']},'Case job/node')
   context=output/(name+'.audit-context.json')
   write(context,dict(case=case,inputs=str(attempt/'inputs.json'),ranges=str(attempt/'ranges.json'),mapping=cpu_plan,gpu_plan=gpu_plan))
   step(output,name+'-audit',[sys.executable,str(HERE/'audit.py'),str(context),str(output/(name+'.json')),str(output/(name+'.audit.json'))],20,deadline)
   verdict=json.loads((output/(name+'.audit.json')).read_text())
   numerical=None
   if case['numerical']:
    receipts=[r['sampled_numerical_reference'] for r in report['rows'][2:]]
    require(all(r and r['status']=='pass' and r['panels']==32 and r['bins']==100 for r in receipts),'All BD numerical receipts')
    require(sum(r['events'] for r in receipts)==case['events'],'Numerical event coverage')
    numerical=dict(status='pass',receipts=receipts)
   else:require(all(r['sampled_numerical_reference'] is None for r in report['rows']),'No numerical observer in timing')
   state['completed'].append(dict(case=case,audit=verdict,result_sha256=sha(output/(name+'.json')),seconds=report['seconds'],numerical=numerical))
   write(output/'status.json',state)
  drift=[]
  if plan['phase']!='qualification':
   for position in ('left','right'):
    entries=[e for e in state['completed'] if e['case']['position']==position]
    before=next(e['seconds'] for e in entries if e['case']['kind']=='full' and e['case']['stage']=='S')
    after=next(e['seconds'] for e in entries if e['case']['kind']=='anchor')
    ratio=before/after;require(abs(ratio-1)<=0.05,'Staging drift exceeds 5%')
    drift.append(dict(position=position,rate_ratio=ratio))
  write(output/'summary.json',dict(status='pass_pending_scheduler',job=state['job'],node=state['node'],phase=plan['phase'],block=plan['block'],bds=plan['bds'],drift=drift))
  verify_inputs(info);verify(attempt)
  state.pop('current',None);state.update(status='passed_pending_scheduler_review',summary_sha256=sha(output/'summary.json'))
  write(output/'complete.json',state);write(output/'status.json',state)
 except BaseException as error:
  state.update(status='failed',error=repr(error));write(output/'status.json',state);raise
 finally:lock.close()
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
