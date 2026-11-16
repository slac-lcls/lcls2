"""Wait within the active session for timing to finish, then discover counters."""
import os,subprocess,sys,time
from pathlib import Path
attempt=Path(sys.argv[1]);job=sys.argv[2];deadline=time.monotonic()+1800
while not (attempt/'results/complete.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('No complete matrix; no counter probe run')
    time.sleep(5)
command=['/usr/bin/python3',str(Path(__file__).with_name('clean_exec.py')),str(attempt),'srun','--jobid='+job,'--overlap','--mpi=cray_shasta','--nodes=1','--ntasks=1','--cpus-per-task=1','--gpus-per-node=1','--gpu-bind=none','--cpu-bind=none','bash',str(Path(__file__).with_name('counter_probe.sh'))]
with (attempt/'counter-probe.log').open('x') as log:
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
print('Post-measurement counter discovery completed')
