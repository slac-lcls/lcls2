"""Bounded host validation; no GPU work and no scheduler submission."""
from pathlib import Path
import ast,datetime,hashlib,json,os,signal,subprocess,sys,time
from compose import compose,HERE
from design import plans,launch
SCRATCH=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-stage3-preparation-20261124')
def run(command,log,cwd,env,limit):
 started=time.monotonic()
 with log.open('x') as stream:
  p=subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   ret=p.wait(timeout=limit)
   if ret:raise RuntimeError('Failed: '+str(log))
  except BaseException:
   if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
 return time.monotonic()-started
repo=Path(subprocess.check_output(['git','-C',str(HERE),'rev-parse','--show-toplevel'],text=True).strip())
SCRATCH.mkdir(exist_ok=True)
root=SCRATCH/('validation-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
root.mkdir();bundle=root/'bundle';bundle.mkdir();(root/'tmp').mkdir();(root/'cupy-cache').mkdir()
files=compose()
for name,data in files.items():
 p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 if p.suffix=='.py':ast.parse(data,filename=name)
 if p.suffix=='.sh':subprocess.run(['bash','-n',str(p)],check=True)
report=dict(status='running',root=str(root),bundle_hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},runtimes={},limitations=['Runtime/import integrity and test collection only; data unavailable, GPU and multi-node execution not performed.'])
(HERE/'evidence/validation-attempt.json').write_text(json.dumps(report,indent=2)+'\n')
try:
 for role in ('C',):
  env=dict(os.environ,SIM10_RUNTIME=role,SIM10_NUMERICAL_REFERENCE='0')
  command=['/usr/bin/python3',str(bundle/'clean_exec.py'),str(root),'python',str(bundle/'runtime_check.py'),str(root/('runtime-'+role+'.json')),'--runtime-only']
  runtime_seconds=run(command,root/('runtime-'+role+'.log'),bundle,env,300)
  d=json.loads((root/('runtime-'+role+'.json')).read_text());report['runtimes'][role]={k:v for k,v in d.items() if k not in ('kvikio_identity','cupy_identity')}
  command=['/usr/bin/python3',str(bundle/'clean_exec.py'),str(root),'python',str(bundle/'check_tests.py'),'collect-device',str(root/('native-collection-'+role+'.json'))]
  collection_seconds=run(command,root/('native-collection-'+role+'.log'),bundle,env,180)
  actual=json.loads((root/('native-collection-'+role+'.json')).read_text());expected=json.loads((bundle/('expected-native-'+role+'.json')).read_text())
  assert actual['collected']==expected['collected'] and actual['sources']==expected['sources'],'Native collection mismatch: '+role
  report['runtimes'][role].update(native_collected=len(actual['collected']),runtime_seconds=runtime_seconds,collection_seconds=collection_seconds)
  print(role,'runtime and collection pass',flush=True)
 report['status']='pass'
except BaseException as error:
 report['status']='failed';report['error']=repr(error)
 raise
finally:
 (HERE/'evidence/host-validation.json').write_text(json.dumps(report,indent=2)+'\n')
 # The reviewer can inspect the actual generated source within its checkout.
 snapshot=HERE/'evidence/composed-bundle'
 import shutil
 if snapshot.exists():shutil.rmtree(snapshot)
 shutil.copytree(bundle,snapshot)
 print(json.dumps(dict(status=report['status'],root=str(root))),flush=True)
