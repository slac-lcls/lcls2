"""Prepare bounded allocation plans; never submit jobs or infer authorization."""
import argparse,csv,json,math,re
from datetime import datetime,timezone
from pathlib import Path
from support import HERE,require,sha,write

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
        # Conservative failure bound, not a claim that one layout predicts all others.
        per_case=max(600,math.ceil(case['whole_case_seconds']*6))
        # Full metadata validation and native tests repeat in the screen.
        setup=sizing['preflight_seconds']
        require(setup>0 and case['whole_case_seconds']>case['seconds']>0,'Sizing durations')
        requested=math.ceil((setup*2+18*per_case+300)/60)*60
        source=sizing['source']
    require(requested<=CAP-used,'Requested allocation exceeds remaining phase cap')
    return {'mode':mode,'phase_cap_seconds':CAP,'prior_node_seconds':used,'remaining_seconds':CAP-used,'requested_seconds':requested,'per_case_timeout':per_case,'ledger_job_ids':sorted(ids),'sizing_source':source,'submission_authorized':False}

def check_allocation(plan,slurm_text,mode):
    require(plan['mode']==mode and plan['requested_seconds']<=plan['remaining_seconds']<=CAP,'Launch plan mode/budget')
    matches=re.findall(r'(?:^|\s)TimeLimit=(\S+)',slurm_text)
    require(len(matches)==1 and duration(matches[0])==plan['requested_seconds'],'Slurm walltime differs from bounded plan')

def load_sizing():
    seal_path=HERE/'sizing-evidence.json'
    seal=json.loads(seal_path.read_text())
    require(all(sha(path)==h for path,h in seal['files'].items()),'Sizing evidence changed')
    attempt=Path(seal['attempt']);output=attempt/'results'
    audit=json.loads((output/'audit.json').read_text())
    complete=json.loads((output/'complete.json').read_text())
    require(complete['bundle']['commit']==seal['qualification_commit'] and
            complete['state']=='cases_passed_pending_slurm' and audit['job']==complete['job'],
            'Qualification binding')
    # These exact receipts were committed before the passing results review.
    # The seal binds legacy timings without modifying the accepted raw evidence.
    records=list(csv.DictReader((attempt/'accounting-utc.txt').read_text().splitlines(),delimiter='|'))
    job=audit['job']
    require(all(r['State']=='COMPLETED' and r['ExitCode']=='0:0' for r in records),'Sizing accounting')
    timings={}
    for step,name in enumerate(('affinity-6','gpu-tests','input-validation')):
        seconds=json.loads((output/(name+'.timing.json')).read_text())['seconds']
        elapsed=int(next(r for r in records if r['JobID']==job+'.'+str(step))['ElapsedRaw'])
        require(math.isfinite(seconds) and seconds>0 and abs(seconds-elapsed)<=10,'Sizing preflight timing')
        timings[name]=seconds
    case=next(c for c in audit['cases'] if c['kind']=='sizing')
    receipt=next(c for c in complete['completed'] if c['case']['kind']=='sizing')
    require(case['whole_case_seconds']==receipt['wall_seconds'] and case['seconds']==receipt['seconds'],'Sizing case timing')
    return {'audit':audit,'preflight_seconds':sum(timings.values()),
            'source':{'seal_sha256':sha(seal_path),'qualification_commit':seal['qualification_commit'],
                      'results_review_commit':seal['results_review_commit'],'files':seal['files'],
                      'preflight_seconds':timings}}

def build_plan(mode,ledger):
    records=list(csv.DictReader(ledger.read_text().splitlines(),delimiter='|'))
    plan=make_plan(mode,records,load_sizing() if mode=='screen' else None)
    plan.update(ledger_path=str(ledger.resolve()),ledger_sha256=sha(ledger))
    return plan

def verify_plan(plan,mode):
    require(plan==build_plan(mode,Path(plan['ledger_path'])),'Plan differs from ledger/sizing evidence')

def allocation_deadline(plan,slurm_text):
    def stamp(key):
        matches=re.findall(r'(?:^|\s)'+key+r'=(\S+)',slurm_text)
        require(len(matches)==1,'Missing/duplicate Slurm '+key)
        return datetime.strptime(matches[0],'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    start=stamp('StartTime');end=stamp('EndTime')
    require(end-start==plan['requested_seconds'],'Slurm start/end duration differs')
    return end-90

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('qualify','screen'));p.add_argument('ledger',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    plan=build_plan(a.mode,a.ledger);write(a.output,plan);print(json.dumps(plan,indent=2))
