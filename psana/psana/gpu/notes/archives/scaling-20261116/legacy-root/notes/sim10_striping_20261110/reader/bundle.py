"""Reviewed snapshot creation and fixed production identity verification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE_NAMES = ['reader.cpp','campaign.py','audit.py','run.sbatch','build.sh','bundle.py','build-pins.json']
KEYS = set(SOURCE_NAMES + ['reader','compiler.txt','staging-complete.json'])
FLAGS = ['-O3','-std=c++17','-pthread','-Wall','-Wextra','-Werror']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(attempt):
    root=attempt/'bundle'
    p=json.loads((attempt/'bundle-provenance.json').read_text())
    if set(p['files']) != KEYS or p['production_flags'] != FLAGS:
        raise RuntimeError('bundle file set or flags mismatch')
    if len(p['commit']) != 40 or any(c not in '0123456789abcdef' for c in p['commit']):
        raise RuntimeError('invalid reviewed source commit')
    for name,r in p['files'].items():
        if digest(root/name) != r['sha256']:
            raise RuntimeError('bundle changed: '+name)
    pins=json.loads((root/'build-pins.json').read_text())
    for name in ['reader','reader.cpp','compiler.txt']:
        if p['files'][name]['sha256'] != pins['sha256'][name]:
            raise RuntimeError('reviewed build pin mismatch: '+name)
    if p['files']['reader']['sha256'] == pins['test_binary_sha256']:
        raise RuntimeError('test binary in production bundle')
    if p['production_flags'] != pins['production_flags']:
        raise RuntimeError('build flags disagree with reviewed pins')
    return p


def create(commit,build,attempt):
    repo=Path(__file__).resolve().parents[3]
    campaign=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-striping-20261110')
    attempt=attempt.resolve()
    if attempt.parent != campaign or not attempt.name.startswith('reader-'):
        raise RuntimeError('unexpected attempt location')
    # Verify every source byte against the nominated reviewed commit before writes.
    base='notes/sim10_striping_20261110/reader/'
    data={}
    for name in SOURCE_NAMES:
        raw=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+base+name])
        if raw != (Path(__file__).parent/name).read_bytes():
            raise RuntimeError('working source differs from reviewed commit: '+name)
        data[name]=raw
    pins=json.loads(data['build-pins.json'])
    for name in ['reader','compiler.txt']:
        raw=(build/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pins['sha256'][name]:
            raise RuntimeError('build does not match reviewed pins: '+name)
        data[name]=raw
    data['staging-complete.json']=subprocess.check_output(['git','-C',str(repo),'show',commit+':notes/sim10_striping_20261110/staging/native/complete.json'])
    attempt.mkdir(mode=0o700)
    root=attempt/'bundle';root.mkdir(mode=0o700)
    p={'commit':commit,'production_flags':FLAGS,'files':{}}
    for name,raw in data.items():
        path=root/name;path.write_bytes(raw);path.chmod(0o555 if name=='reader' else 0o444)
        p['files'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    (attempt/'bundle-provenance.json').write_text(json.dumps(p,indent=2)+'\n')
    root.chmod(0o555)
    verify(attempt)
    print(attempt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True);a=p.parse_args()
    create(a.commit,a.build,a.attempt)
