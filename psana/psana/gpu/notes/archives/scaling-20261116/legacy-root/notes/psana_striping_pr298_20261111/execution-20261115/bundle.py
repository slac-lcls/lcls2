"""Snapshot reviewed harness source; immutable bundle and explicit external pins."""
import argparse,json,subprocess
from pathlib import Path
from support import ROOT,HERE,sha,write,require

def create(commit,name):
    require(name.startswith('bundle') and '/' not in name,'Bundle name')
    dest=ROOT/name;dest.mkdir(exist_ok=False);prefix='notes/psana_striping_pr298_20261111/execution-20261115/'
    files={}
    for p in sorted(HERE.iterdir()):
        if p.suffix not in ('.py','.sh','.sbatch') and p.name not in ('external-pins.json','expected-native.json','sweep.csv'):continue
        data=subprocess.check_output(['git','show',commit+':'+prefix+p.name]);require(data==p.read_bytes(),'Working file differs from reviewed commit: '+p.name)
        target=dest/p.name;target.write_bytes(data);target.chmod(0o444);files[p.name]=sha(target)
    write(dest/'bundle.json',{'commit':commit,'files':files});(dest/'bundle.json').chmod(0o444);dest.chmod(0o555);print(dest)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('commit');p.add_argument('name');a=p.parse_args();create(a.commit,a.name)
