"""Durable sequential submission requested by user; fail closed, never retry jobs."""
import datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from design import plans
from gate import completed
from data_contract import require,sha
HERE=Path(__file__).resolve().parent
INPUT=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-stage3-input-20261124/attempt02')
ROOT=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-stage3-sequence-20261124/attempt03')


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def ordered_groups(groups,launch,check):
    results=[]
    for plan in groups:
        launch(plan)
        results.append(check(plan))
    return results


def wait_accounting(job,deadline):
    while True:
        rows=[line.split('|') for line in subprocess.check_output(['sacct','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],text=True,timeout=30).splitlines() if line.strip()]
        require(all(len(r)>=3 and (r[0]==job or r[0].startswith(job+'.')) for r in rows),'Accounting job identity')
        active={'PENDING','RUNNING','COMPLETING','CONFIGURING'}
        require(not any(r[1] not in active|{'COMPLETED'} or (r[1]=='COMPLETED' and r[2]!='0:0') for r in rows),'Scheduler failure')
        if any(r[0]==job for r in rows) and all(r[1:3]==['COMPLETED','0:0'] for r in rows):return rows
        require(time.time()<deadline,'Accounting completion deadline')
        time.sleep(15)


def main():
    require(not ROOT.exists(),'No sequence restart or overwrite')
    require(json.loads((HERE/'status.json').read_text())['submission_authorized'],'User submission authorization')
    review=json.loads((HERE/'evidence/approved-sequence-review.json').read_text())
    require(review['verdict']=='pass' and not any(f['blocking'] for f in review['findings']),'Reviewed submission controller')
    repo=Path(subprocess.check_output(['git','-C',str(HERE),'rev-parse','--show-toplevel'],text=True).strip())
    tracked=subprocess.check_output(['git','-C',str(repo),'ls-files',str(HERE.relative_to(repo))],text=True).splitlines()
    code=[name for name in tracked if Path(name).suffix in ('.py','.sh')]
    require(not subprocess.check_output(['git','-C',str(repo),'diff',review['reviewed_commit'],'--',*code],text=True),'Code changed after controller review')
    pins={str(repo/name):sha(repo/name) for name in code}
    def verify():require(all(sha(p)==v for p,v in pins.items()),'Source code changed during sequence')
    ROOT.mkdir(mode=0o700,parents=True)
    state=dict(status='waiting_input',pid=os.getpid(),started=time.time(),input_job=None,completed=[],source_pins=pins)
    def save():write(ROOT/'status.json',state)
    save()
    try:
        # The explicitly revised JF-only qualification reuses the verified copy.
        started=json.loads((INPUT/'launcher.json').read_text())['started']
        while not (INPUT/'launcher-exit.json').exists():
            require(time.time()<started+14400,'Input launcher completion deadline');time.sleep(15)
        require(json.loads((INPUT/'launcher-exit.json').read_text())['returncode']==0,'Input launcher failed')
        from requalify import verify_reuse
        verify_reuse(INPUT/'requalification-plan.json')
        qualified=json.loads((INPUT/'qualified/qualification.json').read_text())
        state['input_job']=qualified['job'];save()
        rows=wait_accounting(state['input_job'],time.time()+180);write(ROOT/'input-scheduler.json',rows)
        require(qualified['streams']==list(range(5,10)) and len(qualified['files'])==5,'Five qualified Jungfrau streams')
        require(qualified['status']=='pass' and qualified['job']==state['input_job'] and qualified['qualified_events']==400000,'Data qualification pass and job')
        verify();state['status']='preparing_measurements';save()
        subprocess.run([sys.executable,str(HERE/'prepare.py'),str(INPUT/'qualified')],check=True,timeout=300)
        def launch(plan):
            verify();attempt=Path(plan['attempt']);state.update(status='running_group',current=plan['name']);save()
            account=json.loads(subprocess.check_output(['iris','user',os.environ['USER'],'--json'],text=True,timeout=60))
            write(ROOT/(plan['name']+'-account.json'),account)
            command=json.loads((attempt/'launch-command.json').read_text())
            with (attempt/'allocation.log').open('x') as log:
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['launcher_pid']=proc.pid;save()
                try:require(proc.wait(timeout=plan['walltime_seconds']+600)==0,'Measurement allocation failed: '+plan['name'])
                except BaseException:
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    raise
        def check(plan):
            attempt=Path(plan['attempt']);state['status']='checking_group';save()
            done=json.loads((attempt/'results/complete.json').read_text())
            wait_accounting(done['job'],time.time()+180)
            p,s,summary=completed(attempt)
            result=dict(name=plan['name'],job=s['job'],summary=summary)
            state['completed'].append(result);save();return result
        ordered_groups(plans(),launch,check)
        verify()
        result=json.loads(subprocess.check_output([sys.executable,str(HERE/'analyze.py')],text=True,timeout=180))
        write(ROOT/'scaling-results.json',result);state.update(status='complete',finished=time.time());state.pop('current',None);save()
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished=time.time());save();raise


if __name__=='__main__':main()
