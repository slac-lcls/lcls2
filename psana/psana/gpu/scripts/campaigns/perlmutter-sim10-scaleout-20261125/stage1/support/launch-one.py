from pathlib import Path
import datetime,json,socket,subprocess,sys,time
root=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115')
repeat=int(sys.argv[1])
assert repeat in (1,2,3)
attempt=root/('grouped-replay-20261123-r%d'%repeat)
execution=Path(__file__).resolve().parent
assert not (attempt/'job-started.txt').exists()
assert not (attempt/'results').exists()
command=json.loads((attempt/'launch-command.json').read_text())
record=dict(repeat=repeat,attempt=str(attempt),command=command,hostname=socket.gethostname(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (execution/('launch-r%d.json'%repeat)).open('x') as stream:json.dump(record,stream,indent=2)
start=time.monotonic()
with (execution/('salloc-r%d.log'%repeat)).open('x') as log:
    process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
record.update(returncode=process.returncode,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (execution/('exit-r%d.json'%repeat)).open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record),flush=True)
