"""Run the pinned PR suites and persist actual collected node IDs and reports."""
import json,os,sys
from pathlib import Path
from support import SOURCE,ROOT,write,sha,require,runtime
kind,output=sys.argv[1],Path(sys.argv[2])
os.environ['CUDA_VISIBLE_DEVICES']='' if kind=='unit' else os.environ.get('JF_GPU_UUID','')
import psana,pytest,types
# The installer excludes tests. Expose only the pinned test package, never a
# production source-tree overlay. Some unit tests explicitly load PR source.
module=types.ModuleType('psana.tests');module.__path__=[str(SOURCE/'psana/psana/tests')];sys.modules['psana.tests']=module
class Receipt:
    def __init__(self):self.collected=[];self.reports=[]
    def pytest_collection_finish(self,session):self.collected=[i.nodeid for i in session.items]
    def pytest_runtest_logreport(self,report):self.reports.append({'nodeid':report.nodeid,'when':report.when,'outcome':report.outcome})
r=Receipt();base=SOURCE/'psana/psana/tests/gpu'
files=[base/'unit'] if kind=='unit' else [base/'integration'/f for f in ['test_gpu_input_group_device.py','test_gpu_input_window_device.py','test_gpu_group_ownership_device.py','test_gpu_group_schedule_device.py','test_multiowner_transitions.py','test_gpu_task_batch_device.py']]
device_identity=None
if kind=='device':
    import cupy as cp
    require(cp.cuda.runtime.getDeviceCount()==1,'One visible GPU')
    import subprocess
    pci=cp.cuda.runtime.deviceGetPCIBusId(0)
    uuid=subprocess.check_output(['nvidia-smi','-i',pci,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    require(uuid==os.environ['JF_GPU_UUID'],'Native test physical GPU mismatch')
    device_identity=dict(uuid=uuid,pci=pci)
ret=pytest.main(['-q','--basetemp='+str(output.with_suffix('.tmp')),*(['--collect-only'] if kind=='collect-device' else []),*map(str,files)],plugins=[r])
pins=json.loads((ROOT/'build-provenance.json').read_text())['tests'];current={str(p.relative_to(SOURCE)):sha(p) for p in (base/'unit').rglob('*.py')} if kind=='unit' else {str(p.relative_to(SOURCE)):sha(p) for p in files}
require(all(pins[k]==v for k,v in current.items()),'Test sources changed')
result={'device':device_identity,'exitcode':int(ret),'kind':kind,'collected':r.collected,'reports':r.reports,'sources':current,'runtime':runtime(allow_test_source=True)}
write(output,result)
require(ret==0 and r.collected and not any(x['outcome'] in ('failed','skipped') for x in r.reports),'Required tests failed/skipped')
if kind!='collect-device':require(sum(x['when']=='call' and x['outcome']=='passed' for x in r.reports)==len(r.collected),'Tests not all called')
