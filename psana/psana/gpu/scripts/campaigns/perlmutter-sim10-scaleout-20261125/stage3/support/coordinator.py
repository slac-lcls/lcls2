"""Stage 3 serial cases on one fixed allocation; no retries or successors."""
from pathlib import Path
import datetime,fcntl,json,math,os,re,signal,socket,subprocess,sys,time
from support import require,sha,write,verify_inputs
from topology import build,rank_hosts
from gate import preconditions
from launch_steps import step_command,step_environment
HERE=Path(__file__).resolve().parent


def verify(attempt):
    plan=json.loads((attempt/'plan.json').read_text())
    require(attempt==Path(plan['attempt']) and HERE==attempt/'bundle','Prepared location')
    manifest=json.loads((HERE/'bundle.json').read_text())
    require(sha(HERE/'bundle.json')==plan['bundle_sha256'],'Manifest pin')
    require(all(sha(HERE/n)==v for n,v in manifest['files'].items()),'Bundle bytes')
    require(all(sha(p)==v for p,v in manifest['external'].items()),'External pins')
    require(sum(plan['budget'].values())==plan['walltime_seconds']<=14400,'Walltime budget')
    return plan


def command(attempt,hosts,program,args,timeout,ranked=False):
    return step_command(HERE,attempt,hosts,program,args,timeout,ranked)


def step(output,name,argv,timeout,deadline,hostfile=None):
    require(deadline-time.time()>=timeout+120,'Insufficient remaining time for entire step')
    write(output/(name+'.command.json'),argv)
    env=step_environment(os.environ,hostfile)
    require(('--distribution=arbitrary' in argv)==bool(hostfile),'Arbitrary step requires its rank hostfile')
    started=time.time()
    try:
        with (output/(name+'.log')).open('x') as log:
            proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:require(proc.wait(timeout=timeout)==0,'Step failed: '+name)
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=15)
                raise
    finally:write(output/(name+'.timing.json'),dict(started=started,finished=time.time()))


def main(attempt):
    plan=verify(attempt);text=(attempt/'slurm-job.txt').read_text()
    def field(key):
        matches=re.findall(r'(?:^|\s)'+key+r'=(\S+)',text)
        require(len(matches)==1,'Slurm field '+key);return matches[0]
    require(field('Account')=='lcls_g' and field('QOS') in ('interactive','gpu_interactive'),'Account/QoS')
    require(int(field('NumNodes'))==plan['nodes'] and field('TimeLimit')==plan['walltime'],'Allocation resources')
    require(field('JobId')==os.environ['SLURM_JOB_ID'],'Job identity')
    deadline=datetime.datetime.fromisoformat(field('EndTime')).replace(tzinfo=datetime.timezone.utc).timestamp()
    hosts=subprocess.check_output(['scontrol','show','hostnames',field('NodeList')],text=True).split()
    require(len(hosts)==plan['nodes'] and hosts[0]==socket.gethostname(),'Coordinator on first allocated node')
    output=attempt/'results';output.mkdir()
    state=dict(status='running',job=os.environ['SLURM_JOB_ID'],hosts=hosts,
               plan_sha256=sha(attempt/'plan.json'),completed=[])
    lock=(attempt.parent/'run4-use.lock').open('a')
    try:
        write(output/'status.json',state);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(output/'predecessor.json',preconditions(plan))
        info=json.loads((attempt/'inputs.json').read_text());verify_inputs(info)
        hostfile=output/'rank-hosts.txt';hostfile.write_text('\n'.join(rank_hosts(hosts))+'\n')
        step(output,'probe',command(attempt,hosts,'probe.py',[output],120,True),120,deadline,hostfile)
        probes=[json.loads((output/('probe-r%d.json'%r)).read_text()) for r in range(plan['ranks'])]
        placement=build(probes,hosts);write(output/'placement.json',placement)
        os.environ.update(SIM10_GPU_PLAN=json.dumps(placement),SIM10_CPU_PLAN=json.dumps(placement['mapping']),
                          SIM10_PLACEMENT=str(output/'placement.json'),SIM10_RUNTIME='C',SIM10_NUMERICAL_REFERENCE='0')
        step(output,'runtime',command(attempt,hosts,'runtime_node.py',[output],300),300,deadline)
        step(output,'devices',command(attempt,hosts,'device_node.py',[output],480),480,deadline)
        for case in plan['cases']:
            os.environ['SIM10_NUMERICAL_REFERENCE']='1' if case['numerical'] else '0'
            name=case['name'];state['current']=case;write(output/'status.json',state)
            args=['--inputs',attempt/'inputs.json','--output',output/(name+'.json')]
            for key in ('bds','batch','stage','events','kind'):args.extend(['--'+key,case[key]])
            step(output,name,command(attempt,hosts,'runner.py',args,case['timeout'],True),case['timeout'],deadline,hostfile)
            report=json.loads((output/(name+'.json')).read_text())
            require(report['job']==state['job'],'Case job identity')
            context=output/(name+'.audit-context.json')
            write(context,dict(case=case,inputs=str(attempt/'inputs.json'),ranges=str(attempt/'ranges.json'),
                               mapping=placement['mapping'],gpu_plan=placement))
            step(output,name+'-audit',[sys.executable,str(HERE/'audit.py'),str(context),str(output/(name+'.json')),str(output/(name+'.audit.json'))],60,deadline)
            audit=json.loads((output/(name+'.audit.json')).read_text());require(audit['status']=='pass','Audit')
            receipts=[r['sampled_numerical_reference'] for r in report['rows'][2:]]
            if case['numerical']:
                require(all(r and r['status']=='pass' and r['events']>0 and r['panels']==32 and r['bins']==100 for r in receipts),'All BD numerical acceptance')
                require(sum(r['events'] for r in receipts)==case['events'],'Numerical coverage')
            else:require(all(r is None for r in receipts),'No numerical observer in primary case')
            from analyze import windows
            diagnostic=windows([r['delivery_ticks'] for r in report['rows'][2:]],report['seconds'])
            state['completed'].append(dict(case=case,audit=audit,result_sha256=sha(output/(name+'.json')),
                                           loop_diagnostic=diagnostic,
                                           seconds=report['seconds'],node_events=report['node_events'],
                                           numerical=receipts if case['numerical'] else None))
            write(output/'status.json',state)
            if case['kind']=='anchor':
                before=next(e['seconds'] for e in state['completed'] if e['case']['name']=='r%d-S'%case['repeat'])
                require(abs(before/report['seconds']-1)<=.05,'S-anchor drift exceeds 5%')
        from analyze import summarize
        summary=summarize(state);write(output/'summary.json',summary)
        verify_inputs(info);verify(attempt)
        state.pop('current',None);state.update(status='passed_pending_scheduler_review',summary_sha256=sha(output/'summary.json'))
        write(output/'complete.json',state);write(output/'status.json',state)
    except BaseException as error:
        state.update(status='failed',error=repr(error));write(output/'status.json',state);raise
    finally:lock.close()


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
