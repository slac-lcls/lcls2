"""Exercise successor rejection on incomplete, changed or failed evidence."""
from pathlib import Path
import copy
import json
import tempfile
from sequence import audit, digest

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))

with tempfile.TemporaryDirectory(prefix='sequence-gates-',dir='/pscratch/sd/m/monarin/psana2-gpu/sim10-grouped-preparation-20261123') as tmp:
    attempt=Path(tmp)
    cases=[dict(name='case-%d'%i) for i in range(13)]
    write(attempt/'bundle/bundle.json',dict(files={},external={}))
    write(attempt/'plan.json',dict(bundle_sha256=digest(attempt/'bundle/bundle.json'),source_commit='source',cases=cases))
    completed=[]
    for case in cases:
        report=attempt/'results'/(case['name']+'.json')
        write(report,dict(job='123',rows=[dict(host='nid000001')]*3))
        verdict=dict(status='pass')
        write(attempt/'results'/(case['name']+'.audit.json'),verdict)
        completed.append(dict(case=case,audit=verdict,result_sha256=digest(report)))
    summary=dict(status='pass_pending_scheduler_and_independent_result_review',job='123',node='nid000001',repeat=1,staging_drift=[dict(after_over_before=1)]*2)
    write(attempt/'results/comparison.json',summary)
    state=dict(status='passed_pending_slurm_and_independent_result_review',job='123',source_commit='source',completed=completed,summary_sha256=digest(attempt/'results/comparison.json'))
    def save_state(s):
        write(attempt/'results/complete.json',s);write(attempt/'results/status.json',s)
    save_state(state)
    exit_record=dict(returncode=0,repeat=1,attempt=str(attempt))
    console='salloc: Granted job allocation 123\n'
    accounting='123|COMPLETED|0:0|nid000001\n123.0|COMPLETED|0:0|nid000001\n'
    def check(e=exit_record,c=console,a=accounting):return audit(attempt,1,e,c,a)
    assert check()['status']=='pass_for_successor'
    rejects=[]
    def reject(name,func):
        try:func()
        except (ValueError,KeyError,FileNotFoundError):rejects.append(name)
        else:raise AssertionError('Accepted invalid evidence: '+name)
    reject('allocation exit failed',lambda:check(e=dict(exit_record,returncode=1)))
    reject('wrong launch attempt',lambda:check(e=dict(exit_record,attempt='/wrong')))
    reject('duplicate allocation',lambda:check(c=console+console))
    reject('missing allocation accounting',lambda:check(a='123.0|COMPLETED|0:0|nid000001\n'))
    reject('scheduler still running',lambda:check(a=accounting.replace('COMPLETED','RUNNING')))
    reject('scheduler failed step',lambda:check(a=accounting.replace('123.0|COMPLETED|0:0','123.0|FAILED|1:0')))
    reject('wrong scheduler node',lambda:check(a=accounting.replace('nid000001','nid000002')))
    s=copy.deepcopy(state);s['completed'].pop();save_state(s)
    reject('incomplete group',check);save_state(state)
    s=copy.deepcopy(state);s['completed'][0]['audit']['status']='failed';save_state(s)
    reject('failed audit',check);save_state(state)
    write(attempt/'results/case-0.json',dict(job='123',rows=[dict(host='wrong')]))
    reject('changed result hash',check)
    write(attempt/'results/case-0.json',dict(job='123',rows=[dict(host='nid000001')]*3))
    altered=dict(summary,repeat=2);write(attempt/'results/comparison.json',altered)
    reject('changed summary hash',check)
    s=copy.deepcopy(state);s['summary_sha256']=digest(attempt/'results/comparison.json');save_state(s)
    reject('wrong summary repeat with valid hash',check)
    altered=dict(summary,staging_drift=[dict(after_over_before=1/1.06)]*2)
    write(attempt/'results/comparison.json',altered)
    s['summary_sha256']=digest(attempt/'results/comparison.json');save_state(s)
    reject('six percent staging rate drift',check)
    write(attempt/'results/comparison.json',summary);save_state(state)
    assert check()['status']=='pass_for_successor'
    print(json.dumps(dict(status='pass',positive_cases=2,rejection_cases=rejects,limitations=['Synthetic evidence fixtures; no live scheduler or submission mocked as a pass.'])))
