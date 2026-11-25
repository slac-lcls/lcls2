"""Explicit JF-only recovery; reuse verified data without overwriting attempt01."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from data_contract import identity,require,sha
HERE=Path(__file__).resolve().parent
PRIOR=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-stage3-input-20261124/attempt01')
ROOT=PRIOR.with_name('attempt02')


def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def verify_reuse(plan_path):
    plan=json.loads(Path(plan_path).read_text())
    for path,digest in plan['pins'].items():require(sha(path)==digest,'Pinned recovery input changed: '+path)
    staged=json.loads(Path(plan['staging_receipt']).read_text())
    require(staged['status']=='pass' and staged['job']=='61939667' and len(staged['files'])==20,'Original copy completed')
    selected=[row for row in staged['files'] if int(Path(row['destination']).name.split('-s')[1][:3]) in range(5,10)]
    require(len(selected)==10,'Five JF payload/SMD pairs')
    for row in selected:require(identity(row['destination'])==row['destination_identity'],'Staged JF file changed')
    return plan


def freeze():
    from compose import compose
    require(json.loads((HERE/'status.json').read_text())['submission_authorized'],'Submission authorized')
    review=json.loads((HERE/'evidence/approved-sequence-review.json').read_text())
    require(review['verdict']=='pass' and not any(f['blocking'] for f in review['findings']),'Recovery review pass')
    repo=Path(subprocess.check_output(['git','-C',str(HERE),'rev-parse','--show-toplevel'],text=True).strip())
    tracked=subprocess.check_output(['git','-C',str(repo),'ls-files',str(HERE.relative_to(repo))],text=True).splitlines()
    code=[name for name in tracked if Path(name).suffix in ('.py','.sh')]
    require(not subprocess.check_output(['git','-C',str(repo),'diff',review['reviewed_commit'],'--',*code],text=True),'Code changed after recovery review')
    bundle=compose();hashes={n:hashlib.sha256(b).hexdigest() for n,b in bundle.items()}
    require(hashes==json.loads((HERE/'evidence/host-validation.json').read_text())['bundle_hashes']==json.loads((HERE/'evidence/approved-review.json').read_text())['bundle_hashes'],'Measurement bundle unchanged')
    require(not ROOT.exists(),'Never overwrite qualification attempts')
    ROOT.mkdir(parents=True,mode=0o700)
    for name in ('code','bundle','tmp','cupy-cache'):(ROOT/name).mkdir()
    for name in tracked:
        src=repo/name;rel=src.relative_to(HERE)
        if rel.parts[0]=='reviews' or str(rel).startswith('evidence/composed-bundle/'):continue
        dst=ROOT/'code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    for name,raw in bundle.items():
        dst=ROOT/'bundle'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
    shutil.copyfile(PRIOR/'site.json',ROOT/'site.json')
    prior_plan=json.loads((PRIOR/'input-plan.json').read_text())
    # Preserve all original evidence pins, without re-reading non-JF payloads.
    pins=dict(prior_plan['pins']);pins[str(PRIOR/'input-plan.json')]=sha(PRIOR/'input-plan.json')
    pins[str(PRIOR/'staging.json')]=sha(PRIOR/'staging.json')
    pins[str(ROOT/'site.json')]=sha(ROOT/'site.json')
    pins.update({str(p):sha(p) for folder in ('code','bundle') for p in (ROOT/folder).rglob('*') if p.is_file()})
    plan=dict(attempt=str(ROOT),site=str(ROOT/'site.json'),staging_receipt=str(PRIOR/'staging.json'),pins=pins,reviewed_commit=review['reviewed_commit'])
    write(ROOT/'requalification-plan.json',plan);verify_reuse(ROOT/'requalification-plan.json')
    command=['salloc','-A','lcls_g','-q','interactive','-C','gpu','-N','1','--ntasks=1','--cpus-per-task=128','--gpus-per-node=4','--exclusive','--mem=0','--licenses=scratch,cfs','--time=02:00:00','--immediate=360','--job-name=sim10-s3-jf-qual','--chdir='+str(ROOT),'srun','--mpi=none','--nodes=1','--ntasks=1','--cpus-per-task=128','--gpus-per-node=4','--cpu-bind=none','--gpu-bind=none','timeout','--signal=TERM','--kill-after=30s','6600s','/usr/bin/python3',str(ROOT/'bundle/clean_exec.py'),str(ROOT),'python',str(ROOT/'code/requalify.py'),str(ROOT/'requalification-plan.json')]
    write(ROOT/'launch-command.json',command)
    for folder in ('code','bundle'):
        for p in (ROOT/folder).rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        (ROOT/folder).chmod(0o555)
    for name in ('site.json','requalification-plan.json','launch-command.json'):(ROOT/name).chmod(0o444)
    print(ROOT)


if __name__=='__main__':
    if sys.argv[1:] == ['--freeze']:freeze()
    else:
        from qualify_data import qualify
        plan=verify_reuse(Path(sys.argv[1]));root=Path(plan['attempt'])
        require(Path(__file__).resolve()==root/'code/requalify.py','Frozen recovery code only')
        qualify(json.loads(Path(plan['site']).read_text()),root/'qualified')
