"""Authorized one-shot sequence: audit terminal success before each successor."""
from pathlib import Path
import datetime,fcntl,json,os,re,socket,subprocess,sys,time
from launch_one import read,save,verify,require
HERE=Path(__file__).resolve().parent
EXPECTED=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-stage2-20261124/execution-01')

def granted_job(exit_record,group,console):
 require(exit_record['returncode']==0,'Allocation failed; successors stopped')
 require(exit_record['name']==group['name'] and exit_record['attempt']==group['attempt'],'Launch identity')
 jobs=re.findall(r'Granted job allocation (\d+)',console)
 require(len(jobs)==1,'Exactly one granted allocation')
 return jobs[0]

def terminal_accounting(job,text):
 rows=[r.split('|') for r in text.splitlines() if r.strip()]
 return any(len(r)>=4 and r[0]==job for r in rows) and all(len(r)>=4 and r[1] not in ('RUNNING','PENDING','COMPLETING','CONFIGURING') for r in rows)

def main():
 require(HERE==EXPECTED,'Deployed execution directory')
 lock=(HERE/'sequence.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(HERE/'sequence-start.json',dict(pid=os.getpid(),host=socket.gethostname(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 index=-1
 try:
  groups=read(HERE/'prepared-attempts.json')['groups'];require(len(groups)==6,'Six fixed groups')
  for index,g in enumerate(groups):
   a,plan=verify(g)
   if index>0:
    subprocess.run([sys.executable,str(HERE/'launch_one.py'),str(index)],check=True,timeout=plan['walltime_seconds']+600)
   launch=read(HERE/('launch-%d.json'%index))
   deadline=datetime.datetime.fromisoformat(launch['started_utc']).timestamp()+plan['walltime_seconds']+600
   exit_path=HERE/('exit-%d.json'%index)
   while not exit_path.exists():
    require(time.time()<deadline,'Allocation command exceeded its bounded wait');time.sleep(15)
   job=granted_job(read(exit_path),g,(HERE/('salloc-%d.log'%index)).read_text())
   for poll in range(9):
    accounting=subprocess.check_output(['sacct','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,NodeList'],text=True,timeout=30)
    if terminal_accounting(job,accounting):break
    if poll<8:time.sleep(15)
   require(terminal_accounting(job,accounting),'Terminal scheduler records missing')
   (HERE/('sacct-%d.txt'%index)).write_text(accounting)
   sys.path.insert(0,str(a/'bundle'))
   from gate import completed
   current,state,summary=completed(a)
   require(state['job']==job,'Granted job matches completed results')
   verify(g)
   save(HERE/('successor-gate-%d.json'%index),dict(status='pass_for_successor',name=g['name'],job=job,node=state['node'],cases=len(state['completed'])))
  save(HERE/'sequence-complete.json',dict(status='all_six_completed_pending_result_report'))
 except BaseException as error:
  save(HERE/'sequence-failed.json',dict(index=index,error=repr(error),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));raise
if __name__=='__main__':main()
