import json,hashlib,shutil
from pathlib import Path
R=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-results-recording-20261116/implementation');G=R/'psana/psana/gpu';A=G/'notes/archives/sim10-xpp-cube-20261116';D=G/'docs/performance/campaigns/perlmutter/sim10-xpp-cube/2026-11-16-se-ec-ecc-100k';V=G/'notes/sim10_results_recording_20261116'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def memory_summary(samples):
 return dict(samples=len(samples),minimum_free_bytes=min((s['free'] for s in samples),default=None),maximum_pool_used_bytes=max((s['used'] for s in samples),default=None),maximum_pool_reserved_bytes=max((s['reserved'] for s in samples),default=None))
records=read(A/'case-receipts.json')
for r in records:
 raw=read(Path(r['raw_path']))
 for cr,rr in zip(r['ranks'],raw['rows']):
  cr.pop('sampled_memory');cr['sampled_memory_summary']=memory_summary(rr['sampled_memory']);cr['host_memory']=rr['host_memory']
write(A/'case-receipts.json',records)
ranks=[r for c in records for r in c['ranks']]
host=max(s['VmHWM'] for r in ranks for s in r['host_memory'].values() if 'VmHWM' in s)
pinned=max(r['task_copy']['pinned_capacity_high'] for r in ranks if r['task_copy'])
summary=dict(max_recorded_per_process_rss_high_water_bytes=host,max_task_d2h_destination_capacity_bytes=pinned,scope='Maximum across rank receipts, not summed across processes or a simultaneous node total. VmHWM is the kernel-reported process RSS high-water mark at recorded checkpoints. D2H capacity is the harness-observed destination capacity, not total pinned memory; POSIX bounce and other allocations are separate.')
write(A/'memory-summary.json',summary)
m=read(D/'manifest.json');m['record_commit_note']='Commit in slac-lcls/lcls2 on work/perlmutter-sim10-baseline-20261115 that recorded completed-job results and the passing independent results-review disposition. It is the source record imported here, not this results-branch commit.'
m['artifacts']=[x for x in m['artifacts'] if x['path'] not in [str((A/'case-receipts.json').relative_to(R)),str((A/'README.md').relative_to(R))]]
p=A/'README.md';p.write_text(p.read_text().replace('scalar timing, memory, affinity','scalar timing, memory extrema/checkpoints, affinity').replace('It omits event timestamp\narrays, pixel data and verbose diagnostics.','It omits event timestamp\narrays, pixel data, per-callback memory sample sequences and verbose diagnostics.\nPer-rank sample counts/minimum free/maximum pool use are retained, together with\nhost RSS checkpoints and task D2H destination-capacity ledgers. Full memory\nsequences remain in the indexed raw results.'))
for p in [A/'case-receipts.json',A/'memory-summary.json',A/'README.md']:
 m['artifacts'].append(dict(role='frozen_evidence',path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
write(D/'manifest.json',m)
p=D/'README.md';s=p.read_text().replace('The results-branch HEAD is not the measured code.','The source-record commit `a2affb7f9144b92846097ffd9da4c19971c5538b` on\n`work/perlmutter-sim10-baseline-20261115` recorded the completed results and\npassing results-review disposition imported here. The results-branch HEAD is\nnot the measured code.')
s=s.replace('Host/GPU/pinned memory\nreceipts are retained without treating sampled readings as complete peak bounds.',f'The largest recorded per-process RSS high-water mark was **{host/2**30:.3f} GiB**\n({host} bytes); this is a process maximum across rank checkpoints, not a summed\nnode peak. The largest observed task D2H destination capacity was **{pinned} bytes**.\nThat capacity excludes POSIX bounce buffers and other pinned allocations, so it\nis not a total pinned-memory peak. [Memory summary](../../../../../../notes/archives/sim10-xpp-cube-20261116/memory-summary.json)\nand per-rank host checkpoints, device-sample extrema and D2H ledgers are retained.\nFull per-callback device sample sequences remain in indexed raw files.')
p.write_text(s)
# Persist a provenance script for this one-time transformation, not a general exporter.
shutil.copyfile(R.parent/'compact_memory.py',V/'compact_memory.py')
for name in ['context-01.md','exact-review01-catalog.json','exact-review01-scope.json']:
 shutil.copyfile(R.parent/name,V/name)
shutil.copyfile(R.parent/'claude-01/review.json',V/'review-01.json')
(V/'response-01.md').write_text('''# Round1 response

Reviewed bf2ecfa62fdd220051370c78398878161dc8948d: PASS with two low findings.

- SR-R1-01 accepted/fixed: manifest note and report explain a2affb7f9 as the
  implementation branch source-record commit, distinct from the measured
  framework, harness, original results review and current packaging commits.
- SR-R1-02 accepted/fixed: add observed per-process RSS high-water and task D2H
  destination-capacity maxima, explicitly not simultaneous node/total pinned
  peaks. Pin a memory summary and retain per-rank host checkpoints.
- Coordinator follow-up: the initial extraction unintentionally retained all
  callback memory samples (19MB). Replace them with exact per-rank extrema and
  sample counts, verified against the raw arrays. The raw sequences stay indexed;
  exact execution bundle, case timings, counts and original receipts unchanged.
- Authored CSV CRLF was fixed before review. The full whitespace diff retains
  one blank EOF line in the byte-exact archived Slurm receipt by design.

Re-review is required because the archive extraction and verifier changed.
''')
print(summary);print('compact receipt bytes',(A/'case-receipts.json').stat().st_size)
