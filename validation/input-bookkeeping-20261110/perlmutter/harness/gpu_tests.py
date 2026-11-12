"""Run native ownership tests on the same isolated three-file overlay."""
import json
import os
from pathlib import Path
import sys
import importlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'sim10_stream_length_20261109'))
os.environ['CUDA_VISIBLE_DEVICES']=os.environ['JF_GPU_UUID']
sys.path.insert(0,os.environ['SIM10_OVERLAY'])
import cupy as cp
assert cp.cuda.runtime.getDeviceCount()==1
from runtime_identity import CHANGED
for name in CHANGED:
    module=importlib.import_module('psana.gpu.'+name)
    assert Path(module.__file__).resolve()==Path(os.environ['SIM10_OVERLAY'])/'psana/gpu'/(name+'.py')
import pytest
class Receipt:
    def __init__(self):self.passed=self.failed=self.skipped=0
    def pytest_runtest_logreport(self,report):
        if report.failed:self.failed+=1
        if report.skipped:self.skipped+=1
        if report.when=='call' and report.passed:self.passed+=1
receipt=Receipt()
base=HERE.parents[1]/'psana/psana/tests/gpu/integration'
files=['test_gpu_input_group_device.py','test_gpu_input_window_device.py',
       'test_gpu_group_ownership_device.py','test_gpu_group_schedule_device.py',
       'test_multiowner_transitions.py','test_gpu_task_batch_device.py']
result=pytest.main(['-q','--basetemp='+sys.argv[2],*[str(base/f) for f in files]],plugins=[receipt])
Path(sys.argv[1]).write_text(json.dumps(dict(exitcode=int(result),passed=receipt.passed,
    failed=receipt.failed,skipped=receipt.skipped,files=files),indent=2)+'\n')
assert result==0 and receipt.passed>0 and receipt.skipped==receipt.failed==0
