"""Read-only Perlmutter packaging audit; requires the original external attempt."""
import hashlib
import json
from pathlib import Path
import sys

RAW=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115/comparison02')
REPO=Path(sys.argv[1]).resolve()
ARC=REPO/'psana/psana/gpu/notes/archives/sim10-xpp-cube-20261116'
sys.path.insert(0,str(RAW/'bundle'))
from audit import audit_case
from comparison import summarize
from memory_watch import validate_memory
from plan import placement

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
bundle=read(ARC/'bundle/bundle.json')
for name,h in bundle['files'].items():
 assert sha(ARC/'bundle'/name)==sha(RAW/'bundle'/name)==h
for path,h in bundle['external'].items():assert sha(Path(path))==h
info=read(RAW/'inputs.json');ranges=read(RAW/'ranges.json');hardware=read(RAW/'results/hardware.json');mapping=read(RAW/'results/affinity.json')['mapping'];native=read(RAW/'results/complete.json')
records=[]
for compact,done in zip(read(ARC/'case-receipts.json'),native['completed']):
 c=compact['case'];name=c['name'];p=RAW/'results'/(name+'.json');raw=read(p)
 assert sha(p)==compact['raw_sha256']==done['result_sha256']
 gpu=placement(hardware['uuids'],c);gpu['pci_by_uuid']=hardware['pci_by_uuid']
 audit=audit_case(raw,c,info,ranges,mapping[:c['bds']+2],gpu)
 assert audit==compact['audit']==done['audit']==read(RAW/'results'/(name+'.audit.json'))
 for field in ['seconds','events','hz','payload_gib_s','end_to_end_seconds','processing_instrumentation','cube_exported','numerical_reference_checked']:
  assert compact[field]==raw[field],field
 assert compact['cache_gate']==raw['observation']['cache_gate']
 assert compact['storage_evidence']==raw['observation'].get('storage_evidence')
 for cr,rr in zip(compact['ranks'],raw['rows']):
  for key,value in cr.items():assert value==(len(rr['timestamps']) if key=='event_count' else rr[key]),key
 memory=read(RAW/'results'/(name+'.memory.json'))
 for entry in memory:validate_memory(entry)
 records.append(dict(case=c,seconds=raw['seconds'],audit='pass',job=raw['job'],node=raw['rows'][0]['host'],cache_attempts=len(raw['observation']['cache_gate']['attempts']),conditioning_seconds=raw['observation']['conditioning_seconds']))
assert len(records)==len(native['completed'])==35
summary=summarize(records);summary['source_commit']=native['source_commit']
assert summary==read(RAW/'results/comparison.json')
index=read(ARC/'raw-index.json')['files']
for item in index:
 p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
print(json.dumps(dict(status='pass',cases=len(records),bundle_files=len(bundle['files']),external_pins=len(bundle['external']),raw_files=len(index),compact_extraction='matches raw',native_case_audit='all pass',native_summary='matches',scope='Read-only current raw correspondence and original native case/resource audits; no GPU kernel re-execution or XTC payload rehash'),indent=2))
