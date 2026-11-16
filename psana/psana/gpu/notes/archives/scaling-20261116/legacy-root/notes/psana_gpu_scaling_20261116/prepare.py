"""Freeze an exact committed harness and write a plan. Never calls sbatch."""
import argparse,json,re,subprocess
from pathlib import Path
from support import HERE,ROOT,require,sha,write
from campaign import LIMITS
PREFIX='notes/psana_gpu_scaling_20261116/'

def validate_plan(plan):
    require(plan['mode'] in LIMITS and plan['requested_seconds']==LIMITS[plan['mode']] and plan['per_case_timeout']==600,'Plan limits')
    require(re.fullmatch('[0-9a-f]{40}',plan['commit']) is not None,'Full reviewed commit required')
    if plan['mode']=='pilot':require(plan['pilot'] is None,'Pilot has no predecessor')
    else:
        from audit import audit
        predecessor=plan['pilot'];attempt=Path(predecessor['attempt'])
        require(attempt.parent==ROOT and attempt.name.startswith('scale-'),'Pilot path')
        require(sha(attempt/'results/audit.json')==predecessor['audit_sha256'],'Pilot audit changed')
        # Recompute from raw receipts and Slurm accounting, not a claimed pass field.
        result=audit(attempt,plan['commit'])
        require(result['state']=='pass' and result['mode']=='pilot','Accepted same-commit pilot required')
        require(result==json.loads((attempt/'results/audit.json').read_text()),'Pilot audit receipt differs')

def prepare(commit,name,mode,pilot=None):
    require(re.fullmatch('scale-[a-z0-9-]+',name) is not None,'Attempt name must start scale-')
    plan=dict(commit=commit,mode=mode,requested_seconds=LIMITS[mode],per_case_timeout=600,pilot=None)
    if mode=='screen':
        require(pilot is not None,'Screen requires --pilot accepted attempt')
        plan['pilot']=dict(attempt=str(pilot.resolve()),audit_sha256=sha(pilot/'results/audit.json'))
    else:require(pilot is None,'Unexpected pilot argument')
    validate_plan(plan)
    files={}
    for path in sorted(HERE.iterdir()):
        if path.suffix not in ('.py','.sh','.sbatch') and path.name not in ('external-pins.json','expected-native.json'):continue
        data=subprocess.check_output(['git','show',commit+':'+PREFIX+path.name],cwd=HERE)
        require(data==path.read_bytes(),'Working file differs from commit: '+path.name);files[path.name]=data
    require(files,'No bundle files');attempt=ROOT/name;attempt.mkdir(exist_ok=False)
    bundle=attempt/'bundle';bundle.mkdir();hashes={}
    for name,data in files.items():
        path=bundle/name;path.write_bytes(data);path.chmod(0o444);hashes[name]=sha(path)
    write(bundle/'bundle.json',dict(commit=commit,files=hashes));(bundle/'bundle.json').chmod(0o444);bundle.chmod(0o555)
    write(attempt/'plan.json',plan)
    hours,rem=divmod(plan['requested_seconds'],3600);minutes,seconds=divmod(rem,60)
    cmd=['sbatch','--time=%02d:%02d:%02d'%(hours,minutes,seconds),'--output='+str(attempt/'slurm-%j.out'),str(bundle/'run.sbatch'),str(attempt)]
    write(attempt/'submit-command.json',cmd)
    import shlex
    print('Prepared only. After submission is requested and Iris balances are checked:')
    print(shlex.join(cmd));return attempt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('commit');p.add_argument('name');p.add_argument('mode',choices=tuple(LIMITS));p.add_argument('--pilot',type=Path);a=p.parse_args();prepare(a.commit,a.name,a.mode,a.pilot)
