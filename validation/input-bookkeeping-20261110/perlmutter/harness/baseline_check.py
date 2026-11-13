"""Explicit reproduction of a stale test on unchanged install-B, not acceptance."""
import hashlib
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']=os.environ['JF_GPU_UUID']
import cupy as cp
import psana.gpu.gpu_events as events
assert cp.cuda.runtime.getDeviceCount()==1
source=Path(events.__file__)
assert '/install-B/' in str(source)
expected=HERE.parent/'sim10_trace_20261110/frozen/gpu_events.py'
assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(expected.read_bytes()).digest()
# Frozen historical test imports helpers from the maintained integration suite.
sys.path.insert(0,str(HERE.parents[1]/'psana/psana/tests/gpu/integration'))
import pytest
reports=[]
class Record:
    def pytest_runtest_logreport(self,report):
        r=report
        reports.append(dict(nodeid=r.nodeid,when=r.when,passed=r.passed,failed=r.failed,
            skipped=r.skipped,error=str(r.longrepr) if r.failed else None))
result=pytest.main(['-q',str(HERE/'baseline_test/test_multiowner_transitions.py'),
                   '--basetemp='+sys.argv[2]],plugins=[Record()])
failures=[r for r in reports if r['failed']]
reproduced=(result==1 and len(failures)==1 and failures[0]['when']=='call'
    and failures[0]['nodeid'].endswith('before_dispatch[6]')
    and 'array(1.) == 2' in failures[0]['error']
    and sum(r['when']=='call' and r['passed'] for r in reports)==1
    and not any(r['skipped'] for r in reports))
Path(sys.argv[1]).write_text(json.dumps(dict(expected_failure_reproduced=reproduced,
    runtime=str(source),runtime_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    exitcode=int(result),reports=reports),indent=2)+'\n')
assert reproduced,'Baseline failure differs from stale BeginStep-refresh assertion'
