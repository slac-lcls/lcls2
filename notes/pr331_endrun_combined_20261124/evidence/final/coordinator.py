"""Single authorized Interactive allocation, durable PTY; no automatic retry."""
import datetime,errno,json,os,pty,re,select,signal,subprocess,time,traceback
from pathlib import Path
S=Path(__file__).resolve().parent

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def new(name,value):
    with (S/name).open('x') as f:json.dump(value,f,indent=2)
def status(**value):
    p=S/'status.tmp';p.write_text(json.dumps(dict(time_utc=now(),**value),indent=2));p.replace(S/'status.json')

def main():
    argv=['salloc','--account=lcls_g','--qos=interactive','--constraint=gpu','--nodes=1',
          '--ntasks=5','--cpus-per-task=1','--gpus-per-node=4','--gres-flags=disable-binding',
          '--licenses=scratch','--mem=32G','--exclusive','--time=00:20:00','--immediate=360',
          '--job-name=pr331-endrun-fix','--chdir='+str(S)]
    new('submission-intent.json',dict(time_utc=now(),argv=argv,scope='verify PR331 plus EndRun fix with PR326',automatic_retries=False))
    status(phase='requesting_allocation')
    pid,fd=pty.fork()
    if pid==0:os.execvp(argv[0],argv)
    job=None;sent=False;buffer='';started=time.monotonic();child_status=None
    try:
        with (S/'interactive.log').open('x') as log:
            while child_status is None:
                if time.monotonic()-started > (20*60+900 if sent else 600):
                    raise TimeoutError('allocation/startup exceeded bounded wait')
                if select.select([fd],[],[],1)[0]:
                    try:data=os.read(fd,65536)
                    except OSError as e:
                        if e.errno!=errno.EIO:raise
                        data=b''
                    if data:
                        text=data.decode(errors='replace');log.write(text);log.flush();buffer=(buffer+text)[-32768:]
                        m=re.search(r'Granted job allocation (\d+)',buffer)
                        if m and job is None:
                            job=m.group(1);new('submission.json',dict(job=job,time_utc=now(),qos='interactive',account='lcls_g'));status(phase='allocated',job=job)
                        if not sent and job and re.search(r'monarin@nid\d+:',buffer):
                            os.write(fd,('exec bash '+str(S/'allocation.sh')+'\n').encode());sent=True;status(phase='running',job=job)
                waited,result=os.waitpid(pid,os.WNOHANG)
                if waited:child_status=result
        rc=os.waitstatus_to_exitcode(child_status)
        new('coordinator-exit.json',dict(job=job,command_sent=sent,returncode=rc,time_utc=now()))
        status(phase='allocation_exited',job=job,returncode=rc)
        if job:
            raw=''
            for _ in range(24):
                raw=subprocess.check_output(['sacct','-j',job,'--parsable2','--format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList,AllocTRES'],text=True,timeout=30)
                queue=subprocess.check_output(['squeue','-h','-j',job],text=True,timeout=30)
                parent=[r.split('|') for r in raw.splitlines() if r.startswith(job+'|')]
                if not queue.strip() and parent and parent[0][2] not in ('RUNNING','PENDING','COMPLETING'):
                    break
                time.sleep(5)
            (S/'terminal-accounting.txt').write_text(raw)
        summary=S/'results/summary.json'
        success=bool(job and sent and rc==0 and summary.exists())
        if success:
            rows=json.loads(summary.read_text())
            success=rows['service_complete']
        status(phase='workload_complete' if success else 'stopped',job=job,returncode=rc,
               service_complete=success)
    except BaseException as exc:
        status(phase='stopped',job=job,error=repr(exc));traceback.print_exc()
        if child_status is None:
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise
    finally:os.close(fd)

if __name__=='__main__':main()
