"""Freeze a new one-node staging/qualification attempt after independent review."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from compose import compose,HERE
from data_contract import require,sha,checksums,published_files
ROOT=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-stage3-input-20261124/attempt01')
SOURCE=Path('/global/cfs/cdirs/lcls/projecta/lcls/psdm/prj/prjdatpsana/xtc')


def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


def verify_publication(publication):
    receipt=json.loads((publication/'receipt.json').read_text())
    expected={'SHA256SUMS','completion.json','postcheck.json','r0004-validation.json',
              'calibration/after.json','calibration/verification.json'}
    require(set(receipt['files'])==expected,'Exact fetched publication receipt')
    for name,digest in receipt['files'].items():
        require(sha(publication/name)==digest,'Fetched publication record changed: '+name)


def freeze():
    require(json.loads((HERE/'status.json').read_text())['submission_authorized'],'Submission authorized')
    review=json.loads((HERE/'evidence/approved-input-review.json').read_text())
    require(review['verdict']=='pass' and not any(f['blocking'] for f in review['findings']),'Reviewed input workflow')
    repo=Path(subprocess.check_output(['git','-C',str(HERE),'rev-parse','--show-toplevel'],text=True).strip())
    relative=HERE.relative_to(repo)
    require(not subprocess.check_output(['git','-C',str(repo),'diff',review['reviewed_commit'],'--',str(relative/'stage_input.py'),str(relative/'normalize_selection.py'),str(relative/'freeze_input.py'),str(relative/'on_input_allocation.sh'),str(relative/'qualify_data.py'),str(relative/'data_contract.py'),str(relative/'prepare.py'),str(relative/'evidence/publication'),str(relative/'status.json')],text=True),'Input code changed after review')
    files=compose();hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    require(hashes==json.loads((HERE/'evidence/host-validation.json').read_text())['bundle_hashes']==json.loads((HERE/'evidence/approved-review.json').read_text())['bundle_hashes'],'Timing bundle unchanged')
    verify_publication(HERE/'evidence/publication')
    sums=checksums(HERE/'evidence/publication/SHA256SUMS')
    published=json.loads((HERE/'evidence/publication/postcheck.json').read_text())
    require(published['status']=='pass','Published postcheck')
    expected={r['relative_path']:dict(bytes=r['bytes'],sha256=r['sha256']) for r in published['files'] if r['relative_path'] in published_files()}
    require(set(expected)==set(published_files()),'Twenty published records')
    for name,row in expected.items():
        require(row['sha256']==sums[name] and (SOURCE/name).stat().st_size==row['bytes'],'Published source metadata')
    require(not ROOT.exists(),'No overwrite or implicit retry')
    ROOT.mkdir(mode=0o700,parents=True)
    for name in ('bundle','code','tmp','cupy-cache'):(ROOT/name).mkdir()
    # Snapshot only tracked task files; exclude raw reviewer outputs.
    tracked=subprocess.check_output(['git','-C',str(repo),'ls-files',str(relative)],text=True).splitlines()
    for name in tracked:
        src=repo/name;rel=src.relative_to(HERE)
        if rel.parts[0]=='reviews' or str(rel).startswith('evidence/composed-bundle/'):continue
        dst=ROOT/'code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    for name,raw in files.items():
        dst=ROOT/'bundle'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
    # Re-normalize against the immutable copied raw records for durable paths.
    from normalize_selection import normalize
    snapshot=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-transfer-20261115/package/calibration')
    selection=ROOT/'calibration-selection.json';write(selection,normalize(ROOT/'code/evidence/publication',snapshot))
    site=dict(exp='prjdatpsana',run=4,xtc_directory=str(ROOT/'xtc'),sha256sums=str(ROOT/'code/evidence/publication/SHA256SUMS'),calibration_snapshot=str(snapshot),calibration_selection=str(selection))
    write(ROOT/'site.json',site)
    pins={str(p):sha(p) for folder in ('code','bundle') for p in (ROOT/folder).rglob('*') if p.is_file()}
    pins.update({str(p):sha(p) for p in (selection,ROOT/'site.json',snapshot/'manifest.json')})
    plan=dict(attempt=str(ROOT),source=str(SOURCE),files=expected,pins=pins,reviewed_commit=review['reviewed_commit'],walltime='03:50:00',stage_timeout=6600,qualification_timeout=6600)
    write(ROOT/'input-plan.json',plan)
    command=['salloc','-A','lcls_g','-q','interactive','-C','gpu','-N','1','--ntasks=1','--cpus-per-task=128','--gpus-per-node=4','--exclusive','--mem=0','--licenses=scratch,cfs','--time=03:50:00','--immediate=360','--job-name=sim10-s3-input','--chdir='+str(ROOT),'srun','--mpi=none','--nodes=1','--ntasks=1','--cpus-per-task=128','--gpus-per-node=4','--cpu-bind=none','--gpu-bind=none','/bin/bash',str(ROOT/'code/on_input_allocation.sh'),str(ROOT),str(ROOT/'code')]
    write(ROOT/'launch-command.json',command)
    for folder in ('code','bundle'):
        for p in (ROOT/folder).rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        (ROOT/folder).chmod(0o555)
    for name in ('calibration-selection.json','site.json','input-plan.json','launch-command.json'):(ROOT/name).chmod(0o444)
    print(str(ROOT))


if __name__=='__main__':freeze()
