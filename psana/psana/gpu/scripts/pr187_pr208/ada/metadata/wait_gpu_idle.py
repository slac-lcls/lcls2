import csv,datetime,json,os,subprocess,sys,time
from pathlib import Path
out=Path(sys.argv[1]);history=[];quiet_since=None;start=time.monotonic()
assert os.uname().nodename.split('.')[0]=='sdfada016'
while time.monotonic()-start<600:
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).strip()
    rows=list(csv.reader(raw.splitlines()));assert len(rows)==10 and all('L40S' in r[1] for r in rows),rows
    quiet=not procs and all(int(r[2])<=1 and int(r[3])<=16 for r in rows)
    now=time.monotonic()
    quiet_since=(quiet_since if quiet_since is not None else now) if quiet else None
    history.append(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpus=rows,processes=procs,quiet=quiet))
    result=dict(passed=quiet_since is not None and now-quiet_since>=30,observations=history)
    out.write_text(json.dumps(result,indent=2)+'\n')
    if result['passed']:
        print('ADA_STARTUP_IDLE_GATE_PASS',flush=True);sys.exit(0)
    time.sleep(5)
raise RuntimeError('Ada GPUs did not remain idle for 30 seconds within 10 minutes; no benchmarks started')
