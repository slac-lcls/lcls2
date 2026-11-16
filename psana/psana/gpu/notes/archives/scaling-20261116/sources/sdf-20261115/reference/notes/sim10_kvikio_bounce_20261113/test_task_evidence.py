import pytest
from task_evidence import parse_trace, pilot_evidence


def test_thread_interleaving_and_setup_exclusion(tmp_path):
    p=tmp_path/'trace'
    p.write_text('10 100.0 pread64(7, ""..., 20, 0) = 20\n'
                 '11 102.1 pread64(7,  <unfinished ...>\n'
                 '12 102.2 pread64(8, ""..., 13, 0) = 13\n'
                 '11 102.3 <... pread64 resumed>""..., 17, 20) = 17\n'
                 '11 104.0 +++ exited with 0 +++\n')
    assert parse_trace(p,101,103)==dict(calls=2,bytes=30,size_histogram={'13':1,'17':1})


@pytest.mark.parametrize('text',[
    '10 102.0 pread64(7, "", 17, 0) = 16\n',
    '10 102.0 pread64(7, "", 17, 0) = -1 EIO\n',
    '10 102.0 pread64(7, <unfinished ...>\n',
    '10 102.0 <... pread64 resumed>"", 17, 0) = 17\n',
    'unexpected pread64 format\n',
])
def test_reject_missing_or_failed_trace(tmp_path,text):
    p=tmp_path/'trace';p.write_text(text)
    with pytest.raises(ValueError):parse_trace(p,101,103)


def test_missing_payload_not_accepted(tmp_path):
    p=tmp_path/'case.json'
    p.with_name(p.name+'.rank2.strace').write_text('10 102.0 pread64(7, "", 20, 0) = 20\n')
    result={'rows':[{}, {},dict(rank=2,timestamps=[1],trace_window=dict(start=101,end=103,tracer_pid=10))]}
    with pytest.raises(ValueError,match='coverage'):pilot_evidence(p,result,1,{'3358200':1})


def test_actual_strace_wrapper(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    payload=tmp_path/'payload';payload.write_bytes(b'x'*8192)
    info=tmp_path/'inputs.json';info.write_text(json.dumps({'files':[str(payload)]}))
    program=tmp_path/'reader.py'
    program.write_text('import os,time,json,sys\nfrom pathlib import Path\n'
        'fd=os.open(sys.argv[-1],os.O_RDONLY)\n'
        'start=time.time()\na=os.pread(fd,4096,0)\nb=os.pread(fd,4096,4096)\nend=time.time()\n'
        'pid=int(next(s.split(":")[1] for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("TracerPid:")))\n'
        'Path(sys.argv[sys.argv.index("--output")+1]).write_text(json.dumps(dict(start=start,end=end,pid=pid)))\n')
    output=tmp_path/'result.json'
    args=[sys.executable,str(Path(__file__).with_name('trace_runner.py')),str(program),
          '--inputs',str(info),'--output',str(output),'--events','1000',str(payload)]
    env=dict(os.environ,OMPI_COMM_WORLD_RANK='2',
             SIM10_CPU_PLAN=json.dumps({'ranks':[[],[],sorted(os.sched_getaffinity(0))]}))
    subprocess.run(args,env=env,check=True,timeout=30)
    r=json.loads(output.read_text());assert r['pid']>0
    observed=parse_trace(str(output)+'.rank2.strace',r['start'],r['end'])
    assert observed==dict(calls=2,bytes=8192,size_histogram={'4096':2})
    args[args.index('1000')]='100000'
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'forbidden' in result.stderr
