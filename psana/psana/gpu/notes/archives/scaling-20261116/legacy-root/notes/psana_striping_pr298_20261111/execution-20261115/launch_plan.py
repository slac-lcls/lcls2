"""Prepare bounded allocation plans; never submit jobs or infer authorization."""
import argparse,csv,json,math,re
from pathlib import Path
from support import require,sha,write

CAP=21600
BASE_JOBS={'61131843','61132571','61132790','61133510'}

def duration(text):
    day,sep,rest=text.partition('-')
    if not sep:rest=day;day='0'
    parts=list(map(int,rest.split(':')))
    require(len(parts)==3,'Expected Slurm HH:MM:SS duration')
    h,m,s=parts;return int(day)*86400+h*3600+m*60+s

def make_plan(mode,records,sizing=None):
    parents=[r for r in records if '.' not in r['JobID']]
    ids=[r['JobID'] for r in parents]
    require(len(ids)==len(set(ids)) and BASE_JOBS<=set(ids),'Incomplete phase ledger')
    require(all(r['Account']=='lcls_g' and r['NNodes']=='1' and r['State'].split()[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for r in parents),'Unfinished or wrong ledger job')
    used=sum(int(r['ElapsedRaw']) for r in parents)
    require(used>=3042 and used<CAP,'Phase ledger budget')
    require(mode in ('qualify','screen'),'Bad mode')
    if mode=='qualify':requested=2700;per_case=900;source=None
    else:
        require(sizing and sizing['audit']['state']=='pass' and sizing['audit']['mode']=='qualify','Accepted qualification required')
        require(sizing['audit']['job'] in ids,'Sizing job missing from ledger')
        case=next(c for c in sizing['audit']['cases'] if c['kind']=='sizing')
        require(case['events']==100000 and case['bds']==4 and case['streams']==1 and case['count']==2,'Wrong sizing profile')
        per_case=max(120,math.ceil(case['whole_case_seconds']*1.5))
        # Full metadata validation and native tests repeat in the screen.
        setup=sizing['preflight_seconds']
        require(setup>0 and case['whole_case_seconds']>case['seconds']>0,'Sizing durations')
        requested=math.ceil((setup*1.5+18*per_case+300)/60)*60
        source=sizing['source']
    require(requested<=CAP-used,'Requested allocation exceeds remaining phase cap')
    return {'mode':mode,'phase_cap_seconds':CAP,'prior_node_seconds':used,'remaining_seconds':CAP-used,'requested_seconds':requested,'per_case_timeout':per_case,'ledger_job_ids':sorted(ids),'sizing_source':source,'submission_authorized':False}

def check_allocation(plan,slurm_text,mode):
    require(plan['mode']==mode and plan['requested_seconds']<=plan['remaining_seconds']<=CAP,'Launch plan mode/budget')
    matches=re.findall(r'(?:^|\s)TimeLimit=(\S+)',slurm_text)
    require(len(matches)==1 and duration(matches[0])==plan['requested_seconds'],'Slurm walltime differs from bounded plan')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('qualify','screen'));p.add_argument('ledger',type=Path);p.add_argument('output',type=Path);p.add_argument('--sizing',type=Path);a=p.parse_args()
    records=list(csv.DictReader(a.ledger.open(),delimiter='|'));sizing=None
    if a.sizing:
        af=a.sizing/'results/audit.json';cf=a.sizing/'results/complete.json'
        audit=json.loads(af.read_text());complete=json.loads(cf.read_text())
        sizing={'audit':audit,'preflight_seconds':sum(json.loads((a.sizing/'results'/(n+'.timing.json')).read_text())['seconds'] for n in ('affinity-6','gpu-tests','input-validation')),
                'source':{'audit':str(af),'audit_sha256':sha(af),'complete_sha256':sha(cf),'bundle_commit':complete['bundle']['commit']}}
    plan=make_plan(a.mode,records,sizing);plan.update(ledger_path=str(a.ledger.resolve()),ledger_sha256=sha(a.ledger));write(a.output,plan);print(json.dumps(plan,indent=2))
