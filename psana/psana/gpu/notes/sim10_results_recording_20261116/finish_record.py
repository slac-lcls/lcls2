import json,sys,hashlib,shutil
from pathlib import Path
R=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-results-recording-20261116/implementation');G=R/'psana/psana/gpu';P=G/'docs/performance';A=G/'notes/archives/sim10-xpp-cube-20261116';D=P/'campaigns/perlmutter/sim10-xpp-cube/2026-11-16-se-ec-ecc-100k';V=G/'notes/sim10_results_recording_20261116';T=G/'scripts/performance/analysis'
sys.path.insert(0,str(T));from sim10_cube import table
summary=json.loads((D/'summary.json').read_text());arc='../../../../../../notes/archives/sim10-xpp-cube-20261116'
report='''# Perlmutter Sim10 S/E/EC/ECC: repeated 100k completion

Job **61409171** completed on **nid001020**, 2026-11-16 16:07:29–17:36:20 UTC,
Slurm COMPLETED, exit 0:0, elapsed 01:28:51. All 35 cases passed: 24 primary
comparisons, two short acceptance cases, three staging bridges and six anchors.
The independent results review passed at `890761707f117c515125fac60ba7f86ddfd0e940`;
its scope and limitations are preserved in the linked verdict. This is a dated
application baseline, not a PR baseline/candidate regression or branch-HEAD measurement.

## Primary measurements

Three 100,000-global-event repetitions per cell, on the same node. Values are
medians, with observed minimum–maximum processing seconds. S stages/delivers the
Sim10 seed; E adds expansion to 32 uint16 (512,1024) panels; EC adds offset-enabled
FP32 calibration using the prepared CPU-v3-equivalent algorithm; ECC adds a
persistent distributed FP32 cube. The algorithm description is not a numerical
CPU-reference validation claim.

<!-- results-table -->
TABLE
<!-- /results-table -->

Input GiB/s = globally requested input bytes / 2**30 / processing seconds;
3,358,200 bytes/event, 335,820,000,000 bytes per full case. This denominator is
seed input, not expanded/calibrated images, cube output or measured NIC traffic.
Only the 24 primary cases enter these tables. Acceptance cases use 201/1003 events.

The timer includes delivery, GPU drain, an MPI barrier, and the existing two
CUDA free/pool memory queries per callback. It excludes setup/JIT/constants,
cache conditioning, post-loop GPU cube checks and teardown. `whole_case_seconds`
in runs.csv is the runner's end-to-end interval after initial MPI/device setup;
`whole_step_seconds` is the surrounding launch-step wall time. Neither is the
processing denominator. Runs are untraced; no isolated kernel times are claimed.

## Matched interpretation

- Median paired ECC/S processing increases: **24.45%** for 1BD and **8.47%** for
  4BD (ranges 24.19–24.45% and 7.82–8.85%). Median additional times: 36.912 s and
  5.110 s. Ratios pair stages with S in the same repeat/profile before aggregation.
- Post-block S anchor drift: +0.21% to +1.55% for 1BD; −0.92% to +0.99% for 4BD.
  Small E/S differences are comparable to this drift, so they do not establish
  isolated expansion cost. Differences measure whole-pipeline overhead.
- The 4BD/1GPU/batch200 S bridge median was **60.055 s / 5.208 GiB/s**, close to
  the primary 4BD/4GPU S result. Changing GPUs alone did not raise staging rate
  in these controls. The two primary profiles change BD count, GPUs and batch
  together and cannot establish GPU-only scaling.

Stage order rotates E/EC/ECC after S; profile order alternates between repeats.
This is deterministic blocking, not randomized ordering. Three repeats on one
node give descriptive spread, not a universal optimum or hardware ceiling.

## Configuration and provenance

Framework/native PR298: `67c327a148a08b2aaf972e67e1e02e367c2b298f`, reusing
job61359629 qualification. Harness: `fd63dfb327b154b06e3e37bf39abf698a5495f89`.
The results-branch HEAD is not the measured code. Exact application sources and
per-file hashes are in the 49-file execution bundle; seven external pins are
recorded separately. CuPy 14.0.1, KvikIO 26.10.00, CUDA runtime 12.9, Cray MPICH.

One exclusive Perlmutter node, AMD EPYC 7763, four allocated A100-SXM4-40GB GPUs;
active GPUs are one/four for the primary profiles and one for bridges. One SMD0,
one EB and one/four BDs. Four KvikIO workers per BD, one CUDA stream per BD,
1 MiB tasks, 16 MiB bounce per worker, 512 KiB bulk target, 6 GiB input budget
per BD and 64 MiB pinned D2H budget. KvikIO compatibility mode ON; automatic
direct reads disabled, true GDS inactive. SMD chunks: 1000 events/256 MiB.
Derived calibration/geometry sharing disabled. GPU UUID/PCI and actual CPU maps
are retained. Helpers have three physical cores each; each BD has seven distinct
physical cores with one SMT thread per core: 13 or 34 physical cores total.
Memory placement uses `numactl --interleave=all`.

Fixture: mfx101555026 run10, Sim10 five Jungfrau streams s005–s009, Lustre
**stripe count2 / 1 MiB**, with per-file OST placement and input hashes in the
input identities. The [earlier staging scaling report](../../sim10-staging/2026-11-16-gpu-scaling/README.md)
reached **9.842 GiB/s with 16BD/4GPU/batch200**, four BDs/GPU and 64 workers total.
This application campaign's four-BD case has 16 workers total. The earlier run
also used count1, three physical cores with both SMT threads per BD, and node
nid001008. Its 4BD/1GPU control was 5.075 GiB/s. Those results provide context;
they are not pooled or treated as a controlled speedup comparison here.

## Validation, cache and limitations

All 20 ECC banks (15 full-run, five acceptance) passed FP32 shape
**(100,32,512,1024)**, exact per-bin counts, every pixel finite, at least one
nonzero pixel per populated bin, and zero pixels in empty bins. Bins use global
event ordinal modulo99, with bin99 empty, not scientific XPP-coordinate binning.
Cubes remain distributed per BD. No cube payload was exported, no MPI cube
reduction occurred, and no CPU cube/numerical-reference comparison was performed.
Counts and scalar flags return to the host; bounded per-event error publication
remains enabled. These are basic content checks, not numerical scientific validation.

Every case passed two zero-residency host-page scans after eviction, with at most
three attempts and a 180 s conditioning deadline. 32 passed attempt1; three
passed attempt2: full-r2-bd4-E (one residual page), bridge-r3 (eight), and
anchor-r3-bd1-S (five). Their conditioning took 44.540, 47.249 and 46.031 s,
excluded from timing. The cause of residual pages is unproven. Backend Lustre
cache state and wire traffic were unverified; NIC CSV fields are blank, not zero.

The original coordinator re-audited all 35 cases for exact event identities,
read ranges/bytes, mapping and resource drain. Four GPUs each passed 31 native
tests and ten descriptor guards. Minimum sampled whole-device free memory was
12.809 GiB; sampling does not guarantee transient peaks. Host/GPU/pinned memory
receipts are retained without treating sampled readings as complete peak bounds.
The original independent reviewer checked compact receipts and arithmetic;
raw hashes, exact event/bin ownership and hardware checks relied on coordinator
evidence. Archive verification remains a separate lifecycle axis.

## History and reproduction

The [November15 preparation snapshot](../2026-11-15-baseline/README.md) is unchanged.
Bootstrap61373306 timed out under an earlier export/reduction design. Comparison
61391587 later stopped after seven cases on a strict cache gate. Neither supplies
samples to this report; the failed-run report and final retry implementation
review are preserved. Low-priority historical deferred suggestions are retained
in the status/review records and do not change the accepted measurement scope.

Evidence: [manifest](manifest.json), [35 case rows](runs.csv), [native summary](summary.json),
[compact case receipts](ARC/case-receipts.json),
[original coordinator audit](ARC/evidence/job61409171-results.json),
[terminal accounting](ARC/evidence/job61409171-sacct.txt),
[independent results verdict](ARC/reviews/comparison02-results-round01.json),
[exact execution bundle](ARC/bundle),
[input identities](ARC/input-identities.json), [raw index](ARC/raw-index.json),
[archive guidance](ARC/README.md).

Native summary and bundle status strings were captured before later scheduler
and review completion; they are preserved byte-for-byte. The terminal receipts
and this manifest record the subsequent completed/reviewed state.

Offline validation (Python3.9+, no GPU, data access or submission):

```bash
python3 -B psana/psana/gpu/scripts/performance/analysis/check_catalog.py
python3 -B psana/psana/gpu/scripts/performance/analysis/test_sim10_cube.py -v
```

The dedicated validator checks retained artifact hashes, matrix coverage,
arithmetic, cube receipt consistency and the authored results table. It does not
repeat kernels or the native raw-data audit. Full raw event arrays, builds, data
and calibration payloads remain at indexed site paths; scratch can expire and
hashes cannot restore deleted evidence. No durable-storage transfer is claimed.
Do not execute the relocated archived launcher. A new run requires a new reviewed
bundle against the pinned runtime and qualified inputs.
'''.replace('TABLE',table(summary)).replace('ARC',arc)
(D/'README.md').write_text(report)
# All archive assets are frozen; the campaign manifest pins every one.
m=json.loads((D/'manifest.json').read_text());art=[]
for p in sorted(A.rglob('*')):
 if p.is_file():art.append(dict(role='frozen_evidence',path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
for p in [D/'runs.csv',D/'summary.json']:
 art.append(dict(role='measurement',path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
m['artifacts']=art;(D/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
c=json.loads((P/'catalog.json').read_text());c['campaigns'].append(dict(id=m['id'],kind='application',record_path=str(D.relative_to(R)),manifest=str((D/'manifest.json').relative_to(R)),report=str((D/'README.md').relative_to(R)),script_roots=m['script_roots']));(P/'catalog.json').write_text(json.dumps(c,indent=2)+'\n')
p=T/'check_catalog.py';s=p.read_text().replace("def check_artifacts(record, report):","def check_cube(record, report):\n    from sim10_cube import validate\n    repository_path(record['archive_root'])\n    return validate(record, report, REPO)\n\n\ndef check_artifacts(record, report):");s=s.replace("'cube-pending-v1': check_pending,", "'cube-pending-v1': check_pending, 'cube-complete-v1': check_cube,");p.write_text(s)
url='https://github.com/slac-lcls/lcls2/blob/results/psana2-gpu-scaling/'+str((D/'README.md').relative_to(R))
p=P/'README.md';s=p.read_text().replace('five scaling/application records','six scaling/application records');s=s.replace('| Sim10/XPP-cube application baseline | Full S/E/EC/ECC results pending; prepared scripts preserved |','| Sim10/XPP-cube preparation snapshot | Historical pending state; preserved unchanged |');s=s.replace('C++ full-file reads, psana staging,',f'Completed Sim10 application campaign: [100k S/E/EC/ECC report]({url}). All35 cases passed; independent results review passed. Four-BD S/E/EC/ECC input rates: 5.181 / 5.156 / 4.966 / 4.780 GiB/s. No cube export or numerical CPU-reference comparison.\n\nC++ full-file reads, psana staging,');p.write_text(s)
p=P/'campaigns/README.md';p.write_text(p.read_text()+f"| {m['id']} | application | [Completed report](perlmutter/sim10-xpp-cube/2026-11-16-se-ec-ecc-100k/README.md) |\n")
p=P/'history.md';s=p.read_text().replace('| November 16, 2026 | Perlmutter Sim10 psana GPU scaling |',f'| November 16, 2026 | Sim10 S/E/EC/ECC completion | [Job61409171]({url}): 24 primary 100k cases plus11 controls/checks; four workers/BD, count2. Four-BD S 5.181 GiB/s, ECC 4.780 GiB/s; independent results review passed, basic cube content checks only. |\n| November 16, 2026 | Perlmutter Sim10 psana GPU scaling |');s=s.replace('The Sim10/XPP-cube preparation remains a pending\napplication snapshot and is not a completed performance result.','The Sim10/XPP-cube preparation remains a historical pending snapshot; the\nseparate November16 completion record above contains accepted application results.');p.write_text(s)
p=G/'scripts/performance/analysis/README.md';s=p.read_text().replace('full results pending','historical pending snapshot preserved');s+=f'\n## Completed Sim10 application baseline\n\n[Job61409171 report]({url}) retains the new exact execution bundle and compact receipts.\nThe catalog dispatches `cube-complete-v1` to `sim10_cube.py`: all35 cases, eight\nstage/profile summaries, paired ratios, anchors, cache attempts and cube counts\nare checked offline. Run `python3 -B psana/psana/gpu/scripts/performance/analysis/test_sim10_cube.py -v`\nfor deliberate-corruption tests. This does not re-execute the frozen harness.\n';p.write_text(s)
p=P/'FORMAT.md';s=p.read_text().replace('Current staging batch and sharing choices','The completed November16 application record retains this no-export policy.\nCurrent staging batch and sharing choices');p.write_text(s)
shutil.copyfile(R.parent/'package_record.py',V/'package_record.py');shutil.copyfile(R.parent/'finish_record.py',V/'finish_record.py')
print('report and catalog written;',len(art),'artifact hashes')
