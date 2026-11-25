"""Completion gates between separately submitted Stage 3 allocations."""
import hashlib,json,subprocess
from pathlib import Path


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def completed(attempt):
    plan=read(attempt/'plan.json');state=read(attempt/'results/complete.json')
    require(state==read(attempt/'results/status.json'),'Final state changed')
    require(state['status']=='passed_pending_scheduler_review','Coordinator completion')
    require(state['plan_sha256']==sha(attempt/'plan.json'),'Plan pin')
    require([e['case'] for e in state['completed']]==plan['cases'],'Exact ordered cases')
    rows=subprocess.check_output(['sacct','-n','-P','-j',state['job'],'--format=JobIDRaw,State,ExitCode,NodeList'],text=True,timeout=30).splitlines()
    require(any(r.split('|')[0]==state['job'] for r in rows),'Allocation accounting')
    for line in rows:
        if not line.strip():continue
        row=line.split('|')
        require(len(row)>=4 and (row[0]==state['job'] or row[0].startswith(state['job']+'.')) and row[1:3]==['COMPLETED','0:0'],'All job steps completed')
        hosts=subprocess.check_output(['scontrol','show','hostnames',row[3]],text=True,timeout=30).split()
        require(set(hosts)==set(state['hosts']) if row[0]==state['job'] else bool(hosts) and set(hosts)<=set(state['hosts']),'Scheduler node identity')
    for e in state['completed']:
        name=e['case']['name']
        require(sha(attempt/'results'/(name+'.json'))==e['result_sha256'],'Result bytes')
        require(read(attempt/'results'/(name+'.audit.json'))==e['audit'] and e['audit']['status']=='pass','Audit receipt')
        if e['case']['numerical']:require(all(r['status']=='pass' for r in e['numerical']),'Numerical acceptance')
    summary=read(attempt/'results/summary.json')
    require(sha(attempt/'results/summary.json')==state['summary_sha256'],'Summary bytes')
    require(summary['status']=='pass' and all(abs(r['rate_ratio']-1)<=.05 for r in summary['drift']),'Drift gates')
    return plan,state,summary


def identity(plan):
    attempt=Path(plan['attempt']);manifest=read(attempt/'bundle/bundle.json')
    require(sha(attempt/'bundle/bundle.json')==plan['bundle_sha256'],'Manifest pin')
    return dict(candidate=plan['candidate'],files=manifest['files'],inputs=sha(attempt/'inputs.json'),ranges=sha(attempt/'ranges.json'))


def preconditions(plan):
    if plan['predecessor'] is None:return dict(predecessor=None)
    previous=Path(plan['attempt']).parent/plan['predecessor']
    p,s,u=completed(previous);require(identity(p)==identity(plan),'Common runtime/harness/inputs')
    return dict(predecessor=p['name'],job=s['job'],status='pass')
