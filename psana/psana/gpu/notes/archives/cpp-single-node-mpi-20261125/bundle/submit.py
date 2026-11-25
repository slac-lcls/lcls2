"""Metadata-only dry run by default; --submit launches one single-node comparison."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import baseline as base

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', default='attempt01')
    p.add_argument('--submit', action='store_true')
    args=p.parse_args()
    base.require(re.fullmatch(r'attempt[0-9]{2,}',args.attempt), 'invalid attempt')
    base.verify_bundle()
    plan=json.loads((HERE/'plan.json').read_text());base.validate_inputs(plan)
    out=Path(plan['output_root'])/args.attempt
    launch=HERE/('launch-'+args.attempt);bundle=launch/'bundle'
    base.require(not out.exists() and not launch.exists(),'attempt already exists')
    cmd=['salloc','--no-bell','--account='+plan['account'],'--constraint=gpu','--qos='+plan['qos'],
         '--nodes=1','--ntasks=1','--cpus-per-task=128','--gpus-per-node=4','--exclusive','--mem=0',
         '--time='+plan['time'],'--job-name=cxi-node-mpi','--chdir='+str(bundle),
         '/bin/bash',str(bundle/'run.sh'),'--attempt',args.attempt]
    print(json.dumps({'mode':'submit' if args.submit else 'dry_run','command':cmd,
                     'output':str(out),'cases':24},indent=2),flush=True)
    if not args.submit:return
    try:
        account=json.loads(subprocess.check_output(['iris','user',os.environ['USER'],'--json'],text=True,timeout=30))
        row=next(r for r in account if r['Project']==plan['account'])
        for scope,a,c in [('personal','Allocated(user)','Charged(user)'),('project','Allocated','Charged')]:
            base.require(row[a]>0,'zero allocation')
            remaining=row[a]-row[c];pct=100*remaining/row[a]
            print('%s %s: %.3f allocated, %.3f charged, %.3f remaining hours (%.2f%%), %s'%
                  (plan['account'],scope,row[a],row[c],remaining,pct,datetime.datetime.now(datetime.timezone.utc).isoformat()),flush=True)
            if pct<20:print('LOW ALLOCATION BALANCE: '+scope,flush=True)
    except (OSError,ValueError,KeyError,StopIteration,RuntimeError,subprocess.SubprocessError) as exc:
        account={'unavailable':repr(exc)};print('Allocation balance unavailable: '+repr(exc),flush=True)
    launch.mkdir();bundle.mkdir()
    pins=json.loads((HERE/'pins.json').read_text())
    for name,digest in pins.items():
        shutil.copy2(HERE/name,bundle/name);base.require(base.sha(bundle/name)==digest,'snapshot hash mismatch')
    for name in ['pins.json','README.md','validation.json','compiler.txt']:
        if (HERE/name).exists():shutil.copy2(HERE/name,bundle/name)
    base.save(launch/'account-before.json',account)
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_'))}
    env['MPICH_GPU_SUPPORT_ENABLED']='0'
    with (launch/'allocation.log').open('x') as f:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                              env=env,start_new_session=True)
    base.save(launch/'submission.json',{'state':'allocation_requested','pid':proc.pid,
              'submitted_unix':time.time(),'output':str(out),'command':cmd})
    print('Allocation requested; '+str(launch/'allocation.log'),flush=True)


if __name__=='__main__':main()
