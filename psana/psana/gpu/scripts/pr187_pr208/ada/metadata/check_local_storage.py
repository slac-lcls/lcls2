import json,os,shutil,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads(p.read_text())
assert os.uname().nodename.split('.')[0]=='sdfada016'
assert not d['commands']['mount']['returncode']
m=json.loads(d['commands']['mount']['stdout'])['filesystems'][0]
assert m['target']=='/lscratch' and m['fstype'] in ('xfs','ext4'),m
top=m['source'].split('/')[-1];assert top in d['blocks'],top
members=d['blocks'][top]['slaves'] or [top]
assert members and all(x.startswith('nvme') and x in d['blocks'] for x in members),members
assert d['nvme'] and all(d['blocks'][x]['pci_chain'] for x in members)
required=sum(json.loads(Path(sys.argv[2]).read_text())['10000']['stage_bytes'].values())
space=shutil.disk_usage('/lscratch');assert space.free>required+20*1024**3,(space,required)
result=dict(passed=True,host=d['host'],mount=m,top_device=top,members=members,
 total_bytes=space.total,free_bytes=space.free,required_stage_bytes=required)
Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n')
print('ADA016_LOCAL_NVME_GATE_PASS',json.dumps(result),flush=True)
