"""One-time, local packaging of existing results; never submits or modifies raw evidence."""
import csv, hashlib, json, shutil, subprocess, os
from pathlib import Path
ROOT=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-results-recording-20261116/implementation')
SRC=Path('/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115')
RAW=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115/comparison02')
GPU=ROOT/'psana/psana/gpu'
ARC=GPU/'notes/archives/sim10-xpp-cube-20261116'
REC=GPU/'docs/performance/campaigns/perlmutter/sim10-xpp-cube/2026-11-16-se-ec-ecc-100k'
REV=GPU/'notes/sim10_results_recording_20261116'
ID='perlmutter-sim10-xpp-cube-20261116-100k'
FRAME='67c327a148a08b2aaf972e67e1e02e367c2b298f'
def read(p):return json.loads(p.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def copy(s,d):d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(s,d);assert digest(s)==digest(d)
def rel(p):return str(p.relative_to(ROOT))
def link(p):return os.path.relpath(p,REC)
for p in (ARC,REC,REV):p.mkdir(parents=True,exist_ok=True)
b=read(RAW/'bundle/bundle.json');e=read(SRC/'evidence/job61409171-results.json');plan=read(RAW/'plan.json')
assert e['source_commit']==b['commit']==plan['source_commit']
for name,h in b['files'].items():
 assert digest(RAW/'bundle'/name)==h
 copy(RAW/'bundle'/name,ARC/'bundle'/name)
copy(RAW/'bundle/bundle.json',ARC/'bundle/bundle.json')
external=[]
for path,h in b['external'].items():
 p=Path(path);assert digest(p)==h
 row=dict(path=path,bytes=p.stat().st_size,sha256=h,availability='verified at packaging; external scratch, retention not guaranteed')
 if p.stat().st_size<200000:
  dest=ARC/'external'/p.parent.name/p.name;copy(p,dest);row['archived_path']=rel(dest)
 external.append(row)
write(ARC/'external-index.json',external)
for name in ['job61409171-results.json','job61409171-sacct.txt']:
 copy(SRC/'evidence'/name,ARC/'evidence'/name)
for name in ['RESULTS-61409171.md','RESULTS-61391587.md','PLAN.md','PLAN-comparison01.md','SUBMISSION-comparison02.md','status.json']:
 copy(SRC/name,ARC/'history'/name)
for name in ['comparison02-results-round01.json','cache-retry-round02.json','cache-retry-response02.md']:
 copy(SRC/'review'/name,ARC/'reviews'/name)
for name in ['plan.json','presubmit.json','cpu.txt','slurm-job.txt','submission.json']:
 copy(RAW/name,ARC/'execution'/name)
for name in ['complete.json','status.json','comparison.json','affinity.json','hardware.json']:
 copy(RAW/'results'/name,ARC/'execution/results'/name)
inputs=read(RAW/'inputs.json');write(ARC/'input-identities.json',{k:v for k,v in inputs.items() if k!='timestamps'})
raw_index=[]
for directory in [RAW,RAW/'results']:
 for p in sorted(directory.iterdir()):
  if p.is_file():raw_index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
write(ARC/'raw-index.json',dict(scope='Top-level attempt files and direct results files, excluding caches, temporary directories and bundle (separately pinned). XTC/native/calibration payloads remain external.',files=raw_index))
records=[];runs=[]
for rr in e['records']:
 c=rr['case'];n=c['name'];p=RAW/'results'/(n+'.json');d=read(p);assert digest(p)==rr['result_sha256']
 audit=read(RAW/'results'/(n+'.audit.json'));assert audit['status']==d['status']=='pass'
 assert audit['logical_payload_bytes']==c['events']*3358200
 assert d['seconds']==rr['seconds']
 gate=d['observation']['cache_gate'];assert all(f['resident_pages']==0 for s in gate['attempts'][-1]['scans'] for f in s['files'])
 compact=dict(case=c,job=d['job'],node=d['rows'][0]['host'],seconds=d['seconds'],events=d['events'],hz=d['hz'],payload_gib_s=d['payload_gib_s'],end_to_end_seconds=d['end_to_end_seconds'],whole_step_seconds=rr['whole_step_seconds'],audit=audit,raw_path=str(p),raw_sha256=digest(p),cache_gate=gate,processing_instrumentation=d['processing_instrumentation'],cube_exported=d['cube_exported'],numerical_reference_checked=d['numerical_reference_checked'],min_sampled_free_gib=rr['min_sampled_free_gib'],storage_evidence=d['observation'].get('storage_evidence'),ranks=[])
 for row in d['rows']:
  compact['ranks'].append(dict(rank=row['rank'],host=row['host'],affinity=row['affinity'],device=row['device'],event_count=len(row['timestamps']),cube_verification=row['cube_verification'],drain=row['drain'],sampled_memory=row['sampled_memory'],task_copy=row['task_copy'],memory_ledger=row['memory_ledger']))
 records.append(compact)
 runs.append(dict(case_id=n,job_id=d['job'],block=1,repeat=c['repeat'],phase=c['cohort'],stage=c['stage'],site='perlmutter',node=compact['node'],storage='lustre',gpus_allocated=4,gpus_active=c['gpus'],bds=c['bds'],workers_per_bd=4,cpu_policy='7 physical cores per BD, one SMT thread/core',stripe_count=2,batch=c['batch'],events_global=c['events'],requested_bytes=audit['logical_payload_bytes'],loop_seconds=d['seconds'],whole_case_seconds=d['end_to_end_seconds'],whole_step_seconds=rr['whole_step_seconds'],payload_gib_s=d['payload_gib_s'],cache_state='client-page-cold',cache_attempts=len(gate['attempts']),conditioning_seconds=gate['elapsed_seconds'],nic_rx_bytes='',nic_tx_bytes='',raw_path=str(p),raw_sha256=digest(p)))
write(ARC/'case-receipts.json',records)
with (REC/'runs.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(runs[0]),lineterminator="\n");w.writeheader();w.writerows(runs)
copy(RAW/'results/comparison.json',REC/'summary.json')
(ARC/'README.md').write_text('''# Frozen Sim10/XPP-cube completion evidence

Exact comparison02 bundle and compact evidence for job 61409171. The bundle's
49 files are byte-identical to the execution bundle at fd63dfb327b154b06e3e37bf39abf698a5495f89.
The original seven external pins are verified and indexed; three small manifests
are copied, while large identity/range/install inventories remain external.

Do not launch this relocated snapshot. Its absolute site paths, historical status,
submission authorization and prepared manifests describe the original execution;
they are not current authorization or a portable installation. Reproduction needs
a new reviewed bundle and the qualified PR298 runtime, data and calibration.

case-receipts.json extracts scalar timing, memory, affinity, device, cache and
per-bin cube receipts from each indexed raw result. It omits event timestamp
arrays, pixel data and verbose diagnostics. raw-index.json pins original locations,
sizes and hashes. Hashes cannot recover deleted scratch files; no durable-storage
transfer or retention guarantee is claimed. No cube payload was exported.

history/ and execution/ retain original wording/status, including pre-review
pending labels. The dated campaign manifest and latest history/status.json record
the terminal scheduler and independent results-review disposition separately.
Old report-relative links in these byte-exact historical documents retain their
original context; use the campaign report for maintained navigation.
''')
timing='Processing includes delivery, drain, barrier and two memory queries per callback; excludes setup/JIT/constants, cache conditioning, post-loop cube validation and teardown.'
manifest=dict(schema_version=2,id=ID,title='Perlmutter Sim10 S/E/EC/ECC: repeated 100k completion',kind='application',site='perlmutter',workload='sim10-xpp-cube',status='complete',snapshot_utc='2026-11-16',jobs=['61409171'],framework_commit=FRAME,source_commit=b['commit'],record_commit='a2affb7f9',application_source=dict(bundle=rel(ARC/'bundle'),files='vendor/sim10_compute_cube_20261115/stages.py and cube_check.py; hashes in bundle.json'),analysis_source='psana/psana/gpu/scripts/performance/analysis/sim10_cube.py',events_per_full_case=100000,full_cases=24,total_cases=35,replicates=3,metric=dict(name='application_stage_processing_rate',unit='events/s',timing=timing,derived_payload_bandwidth=dict(unit='GiB/s',bytes_per_event=3358200,meaning='logical requested input bytes, not expanded/calibrated/cube output or NIC traffic')),topology='One exclusive node, four allocated A100 40GB GPUs. Primary profiles: 1BD/1GPU/batch50 and 4BD/4GPU/batch200. Four KvikIO workers/BD; one SMD0 and one EB. Three 4BD/1GPU/batch200 S bridges.',configuration=dict(workers_per_bd=4,streams_per_bd=1,task_bytes=1048576,bounce_bytes_per_worker=16777216,bulk_target_bytes=524288,input_budget_gib_per_bd=6,pinned_d2h_budget_bytes=67108864,smd_events=1000,smd_chunk_bytes=268435456,compat_mode='ON',auto_direct_io_read=False,true_gds=False,cpu_affinity='3 physical cores/helper, 7 physical cores/BD; one SMT thread/core, distinct across ranks; 13/34 cores total',numa='interleave=all',derived_and_geometry_sharing=False),storage=dict(media=['lustre'],stripe_count=2,stripe_size_bytes=1048576,cache='two zero-residency host-page scans; at most three pre-timing attempts',nic='unavailable; blank CSV values, never zero; remote traffic unverified'),dataset=dict(exp=inputs['exp'],run=inputs['run'],identities=rel(ARC/'input-identities.json'),bytes_per_event=3358200),cube_policy=dict(layout='distributed per-BD GPU FP32 (100,32,512,1024)',binning='global event ordinal modulo 99; bin99 empty',cube_exported=False,mpi_cube_reduced=False,numerical_reference_checked=False,checks=['shape/dtype/bytes','per-bin counts','all finite','nonzero populated bins','zero empty bins']),review=dict(automated='all35 audited; exact event/read/resource checks; see original coordinator receipt',implementation=dict(verdict='pass',commit=b['commit']),independent_results=dict(verdict='pass',commit='890761707f117c515125fac60ba7f86ddfd0e940',receipt=rel(ARC/'reviews/comparison02-results-round01.json'))),lifecycle=dict(execution='complete',correctness='basic GPU shape/content/count checks passed; numerical reference not checked',results_review='pass at 890761707f117c515125fac60ba7f86ddfd0e940; scope/limitations in archived verdict',archive_verification='offline cube-complete-v1 checks; packaging review recorded separately'),analysis='cube-complete-v1',script_roots=[rel(ARC/'bundle')],supersedes='perlmutter-sim10-xpp-cube-20261115',raw_locations=[str(RAW)],archive_root=rel(ARC),record_path=rel(REC),retention='External scratch can expire; raw index is not a durable copy.',failed_attempts=[dict(job='61373306',status='failed',note='historical bootstrap timeout; no throughput accepted; original pending snapshot retained'),dict(job='61391587',status='failed',note='7 cases completed before strict cache gate failure; no pooling with completion job; failed-run reporting review passed')])
write(REC/'manifest.json',manifest)
(REV/'scope.md').write_text('''# Scope: completed Sim10 results recording

Record job61409171 on results/psana2-gpu-scaling using a new dated application
record, preserving the old pending snapshot byte-for-byte. Freeze exact scripts,
compact receipts, input identities, raw index, terminal accounting and previous
review verdicts. Verify archive integrity, matrix coverage, arithmetic, links,
source-only isolation and native receipt correspondence. No new GPU runs, runtime
changes, pushes or branch retirement. Three independent Claude rounds maximum.
Scoped external review remains authorized by the user's Sim10 review consent.
A pass requires all checks and no unresolved blocking findings on latest code.
''')
write(REV/'status.json',dict(scope='sim10_results_recording',phase='implementation',round=0,round_limit=3,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),unresolved_ids=[],artifact_root=str(ROOT.parent)))
print('packaged',len(records),'cases;',len(b['files']),'bundle files;',len(raw_index),'raw index files')
