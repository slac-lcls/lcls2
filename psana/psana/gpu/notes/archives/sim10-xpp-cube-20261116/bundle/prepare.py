"""Create a unique immutable task bundle and attempt. Never submit."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from support import HERE, TASK_ROOT, ROOT, PACKAGE, require, sha, write
from plan import ATTEMPT, cases, WALLTIME, BUDGET, OUTPUT_RESERVE

def main(name):
    require(name==ATTEMPT, 'Only the reviewed full comparison is prepared')
    repo=Path(subprocess.check_output(['git','-C',str(HERE),'rev-parse','--show-toplevel'],text=True).strip())
    attempt=TASK_ROOT/name
    require(not attempt.exists(), 'Never overwrite an attempt')
    TASK_ROOT.mkdir(exist_ok=True)
    attempt.mkdir()
    (attempt/'tmp').mkdir()
    (attempt/'cupy-cache').mkdir()
    bundle=attempt/'bundle'
    bundle.mkdir()
    for source in HERE.rglob('*'):
        if source.is_file() and source.suffix in ('.py','.sh','.sbatch','.json') and not {'evidence','review','__pycache__'}.intersection(source.relative_to(HERE).parts):
            require(not source.is_symlink(), 'Bundle symlink')
            committed=subprocess.check_output(['git','-C',str(repo),'show','HEAD:'+str(source.relative_to(repo))])
            require(committed==source.read_bytes(), 'Uncommitted bundle source: '+str(source))
            destination=bundle/source.relative_to(HERE)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
    subprocess.run([sys.executable,str(HERE/'check_reuse.py'),str(attempt/'reuse-check.json')],check=True)
    reuse=json.loads((attempt/'reuse-check.json').read_text())
    write(attempt/'inputs.json',reuse['inputs'])
    shutil.copy2(ROOT/'qualify01/results/ranges.json',attempt/'ranges.json')
    commit=subprocess.check_output(['git','-C',str(HERE),'rev-parse','HEAD'],text=True).strip()
    pins={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    external={str(p):sha(p) for p in [attempt/'reuse-check.json',attempt/'inputs.json',attempt/'ranges.json',
              ROOT/'inputs.json',ROOT/'build-provenance.json',PACKAGE/'PACKAGE.json',
              PACKAGE/'calibration/manifest.json']}
    write(bundle/'bundle.json',dict(commit=commit,files=pins,external=external))
    write(attempt/'plan.json',dict(walltime_seconds=WALLTIME,budget=BUDGET,cases=cases(),
          output_reserve_bytes=OUTPUT_RESERVE,submission_authorized=False,source_commit=commit,
          bundle_sha256=sha(bundle/'bundle.json')))
    for path in bundle.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    bundle.chmod(0o555)
    print(attempt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');main(p.parse_args().name)
