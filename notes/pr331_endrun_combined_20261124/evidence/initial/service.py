import os,sys,json,time,subprocess,signal,hashlib,traceback,re,datetime
import xml.etree.ElementTree as ET
from pathlib import Path
S=Path(__file__).resolve().parent
ROOT=Path('/pscratch/sd/m/monarin/agent-work/lcls2/pr331-endrun-combined-20261124-attempt01');SRC=ROOT/'source';TESTS=SRC/'psana/psana/tests'
BASE=Path('/pscratch/sd/m/monarin/agent-work/lcls2/gpu-master-sync-20261119-attempt02/candidate/install')
job=os.environ['SLURM_JOB_ID'];PY=sys.executable

def steps():
 p=subprocess.run(['scontrol','show','step',job],capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr)
 return sorted(set(re.findall(r'StepId=('+job+r'\.\d+)\s',p.stdout)))
def clean():
 ids=steps()
 if ids:
  subprocess.run(['scancel',*ids],check=True)
  for _ in range(40):
   if not steps():return ids
   time.sleep(1)
  raise RuntimeError('residual Slurm steps did not drain')
 return []
def execute(name,selection,prefix,timeout=300,expected_failures=0,extra=(),bootstrap=False,trace=False,expected_tests=None):
 R=S/'results'/name;R.mkdir()
 site=prefix/'lib/python3.11/site-packages'
 env={k:v for k,v in os.environ.items() if not k.startswith('PS_')}
 env.update(PYTHONPATH=str(site),PATH=f'{S}/bin:{prefix}/bin:{Path(PY).parent}:'+env['PATH'],LD_LIBRARY_PATH=f'{prefix}/lib:{Path(PY).parent.parent}/lib:/global/u2/m/monarin/.local/share/psana2-gpu/cuda12shim/lib:'+env.get('LD_LIBRARY_PATH',''),CUDA_PATH='/global/u2/m/monarin/.local/share/psana2-gpu/cuda12shim',PS_PARALLEL='none',PS_EB_NODES='1',PS_SRV_NODES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',LAUNCH_LOG=str(R/'launches.jsonl'),CUPY_CACHE_DIR=str(S/'cupy-cache'))
 if bootstrap:env['PS_PARALLEL']='mpi'
 if trace:env['PYTHONPATH']=str(S/'diagnostic-hooks')+':'+env['PYTHONPATH']
 clean()
 probe="import psana,psana.psexp.mpi_ds,json;print(json.dumps({'psana':psana.__file__,'mpi_ds':psana.psexp.mpi_ds.__file__}));assert psana.__file__.startswith("+repr(str(prefix))+')'
 p=subprocess.run([PY,'-B','-c',probe],env=env,cwd=R,capture_output=True,text=True,timeout=30)
 (R/'imports.log').write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError('import failed')
 argv=[PY,'-B','-m','pytest','-p','no:cacheprovider','-v','-rs',*map(str,selection),*extra,'--basetemp='+str(R/'tmp'),'--junitxml='+str(R/'pytest.xml')]
 if bootstrap:argv=[PY,'-B',str(S/'pytest_with_test_package.py'),*argv[4:]]
 start=time.monotonic();timedout=False
 with (R/'pytest.log').open('w') as f:
  child=subprocess.Popen(argv,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=child.wait(timeout=timeout)
  except subprocess.TimeoutExpired:timedout=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
 killed=clean()
 counts={k:0 for k in ('tests','failures','errors','skipped')}
 if (R/'pytest.xml').exists():
  for suite in ET.parse(R/'pytest.xml').getroot().iter('testsuite'):
   for k in counts:counts[k]+=int(suite.get(k,0))
 accepted=not timedout and counts['tests']>0 and counts['skipped']==0 and (expected_tests is None or counts['tests']==expected_tests) and counts['errors']==0 and counts['failures']==expected_failures and rc==(1 if expected_failures else 0)
 result=dict(id=name,job=job,argv=argv,install=str(prefix),returncode=rc,controller_timeout=timedout,expected_failures=expected_failures,accepted=accepted,counts=counts,cleanup_steps=killed,elapsed=time.monotonic()-start,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime_hashes={x:hashlib.sha256((site/'psana/psexp'/x).read_bytes()).hexdigest() for x in ('mpi_ds.py','node.py')})
 (R/'verdict.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 return result
rows=[]
try:
 rows.append(execute('01-cpu-lifetime',[TESTS/'test_steps_iterator_lifetime.py'],ROOT/'install',expected_tests=6))
 rows.append(execute('02-exact-ci-mpi',[str(TESTS/'test_callback_stream.py')+'::test_fresh_callback_per_run[mpi]'],ROOT/'install',timeout=150,expected_tests=1))
 rows.append(execute('03-callback-remainder',[TESTS/'test_callback_stream.py'],ROOT/'install',timeout=360,extra=('-k','not (test_fresh_callback_per_run and mpi)'),expected_tests=16))
 rows.append(execute('04-gpu-unit',[TESTS/'gpu/unit'],ROOT/'install',timeout=300,bootstrap=True,expected_tests=1398))
 rows.append(execute('05-gpu-device',[TESTS/'gpu/integration'/x for x in ('test_gpu_steps_device.py','test_gpu_step_scope_device.py','test_gpu_task_lifecycle_device.py','test_bulk_lifecycle_device.py')],ROOT/'install',timeout=300,expected_tests=15))
 rows.append(execute('06-pr326-multirun',[ROOT/'test_pr326_multirun.py'],ROOT/'install',timeout=240,expected_tests=2))
 clean()
 success=all(r['accepted'] for r in rows)
 (S/'results/summary.json').write_text(json.dumps(dict(service_complete=True,all_expected=success,job=job,rows=rows),indent=2)+'\n')
except BaseException as exc:
 (S/'service-error.json').write_text(json.dumps({'error':repr(exc)})+'\n');traceback.print_exc()
 try:clean()
 except Exception:traceback.print_exc()
 raise
sys.exit(0 if success else 1)
