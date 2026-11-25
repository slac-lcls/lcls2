"""Restore byte-exact historical bundle into a NEW directory; never submit jobs."""
import argparse,hashlib,json
from pathlib import Path

def restore(repo,group,dest):
    archive=repo/'psana/psana/gpu/notes/archives/sim10-scaleout-20261125/groups'
    groups={p.name:p for p in archive.iterdir() if p.is_dir()}
    if group not in groups:raise ValueError('Unknown archived group')
    mapping=json.loads((groups[group]/'bundle-map.json').read_text())
    payload={}
    for rel,item in mapping.items():
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe bundle path')
        source=(repo/item['path']).resolve();source.relative_to(repo.resolve())
        raw=source.read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Hash mismatch: '+rel)
        payload[rel]=raw
    dest.mkdir(parents=True,exist_ok=False)
    for rel,raw in payload.items():
        p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    print('Restored',len(payload),'files; historical paths require adaptation. No job submitted.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);p.add_argument('--group',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();restore(a.repository.resolve(),a.group,a.output)
