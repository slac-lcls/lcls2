# JF + feespec, one GPU and 1/2/4 BigData ranks

**Current attempt: job 39178621 RUNNING on sdfampere031**, checked 2026-09-26
at handoff (about nine minutes elapsed). All six pixel preflights and the first
cold 1-BD bulk-off sample passed. Complete results remain pending. This retry
uses bounded missing-page warm-cache repair. Frozen campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-feespec-scale-20260926-r2`.
All six pixel preflights and 24 timed samples are rerun; no samples are reused
from the failed attempt. Production runtime, references, requested resources,
and all benchmark settings are unchanged. Estimated runtime remains
**65–90 minutes after allocation**, excluding queue time, with a **1h59m** limit.

## Warm-cache repair

After initial warming, any prefix below 99% residency is scanned with `mincore`.
The helper coalesces absent pages into byte ranges and reads only those ranges,
clipped to the measured prefix. Repair preserves NUMA interleaving even when
only the small feespec file needs repair. Up to **three** repair passes are
allowed; **all six files are rechecked after every pass**. Persistent residency
failure still aborts before timing. Post-timing checks never repair cache state.
Logs record each retry and repaired byte count; accepted cache snapshots retain
`warm_retries`. Cold preparation and its acceptance gate are unchanged.

Validation: **33 cache/harness tests passed**, including nine new repair cases.
The frozen retry passed **35 scaling/cache tests**, cache-helper preflight and
all **930 file hashes** before and after testing. A real NUMA-interleaved repair
restored an 8 MiB+17-byte prefix from **1.56% to 100% residency**, reading only
its **8,257,553 absent bytes**; the file contents were unchanged.

## Prior failed attempt

2026-09-26. Full sweep **39174520 FAILED on sdfampere011**, exit **1:0**,
after **17m40s**. All six pixel preflights passed, followed by two cold timed
samples (1 GPU/1 BD, first repetition): **197.85 events/s off**,
**189.12 events/s on**. Both passed timestamp, feespec sum, byte/request and
resource checks. No warm timed samples or complete summary were produced.

The failure occurred before the first warm sample: after reading the six
prefixes, feespec stream s000 had **91.393%** page residency (21,736 of 23,783
pages), below the required **99%**. The cache gate correctly rejected this
condition. This was a cache-preparation failure, not a pixel or GPU-processing
failure. The exact cause of the page loss has not yet been established. The
private stage was removed and partial results remain preserved on shared scratch.

Device preflight **39174462 COMPLETED**, exit **0:0**, in **2m45s** on
`sdfampere038`. Both modes passed with 1 and 4 BDs: each case validated
200 feespec arrays, 200 sums, three JF raw/calibrated samples and exact read
counts (1,200 off / 1,012 on). The `afterok` gate released the full sweep.
Full-campaign results are incomplete. Original runtime estimate **65–90 minutes after allocation**;
walltime limit **1h59m**.
The estimate is based on the prior 86-minute JF-only campaign with 32 samples;
this campaign has 24 samples and adds a small feespec GPU consumer.

## Controlled workload

Uses the same runtime as the completed [JF-only baseline](jungfrau_current_scaling.md),
commit **ad8d454d10e203d3ef02d9c75069da48a31de182** including cleanup.
Production source and native dependencies are unchanged. All frozen parent
hashes were checked before creating the new campaign.

- `mfx101210926`, run **387**, first **10,000 events**.
- JF streams **005–009**, plus shared feespec stream **000**.
- **One A100**, **1, 2, 4 BDs**, bulk off/on, cold/warm, two fresh-process
  repetitions with reversed topology/cache/mode order: **24 timed samples**.
- Same settings as JF-only: batch **20**, depth **1**, **8 KvikIO workers/BD**,
  **1 MiB** KvikIO task and bulk target, automatic per-BD GPU budget,
  `PS_SMD_N_EVENTS=1000`, one SMD0/one EB, CPU fallback, no automatic D2H.
- An exclusive node with 112 CPUs and 700 GiB host memory is requested, with
  one GPU requested and used. All BDs share the allocation's CPU affinity.
- Same private node-local `/lscratch` staging and cache controls. Cold requires
  less than 1% page residency before timing; warm requires greater than 99%
  before and after. Warm preparation uses NUMA interleaving.

Feespec uses the existing benchmark-only exclusive shared-stream routing
exception. Complete s000 datagrams are read; other CPU detectors in that
stream are not consumed. This does not change production routing policy.

JF calibration remains active. The timed consumer reads timestamps and feespec
`raw.hproj` via the public GPU field view, then computes a per-event int64 GPU
sum. Field aliases are cleared before advancing the iterator. Compact sums are
copied after timing; full feespec array and JF image copies occur only in the
separate diagnostic preflights. The public field API's internal locator access
remains part of the timed path. Therefore this is the combined detector and
feespec-consumer workload, not a pure comparison of read submission overhead.

## Validation

- Six 200-event pixel preflights, one per topology/mode, before any timed sample.
  Each checks **all 200 feespec arrays** and **three JF raw/calibrated samples**
  against CPU references. The separate submission gate checks 1 and 4 BDs,
  both read modes, using the same settings against shared input.
- Every timed sample validates all 10,000 timestamp-associated feespec sums
  across MPI ranks, exact global timestamp hash, bytes/read counts, zero CPU
  bigdata reads, GPU identity/sharing, affinity, owned allocation budgets and
  successful GPU resource cleanup.
- Independent CPU references decode feespec arrays and sums, read six-stream
  SMD extents, and count coalescing across adjacent file offsets while respecting
  batch-20 boundaries, transition fences and the 1 MiB target.
- Expected 10k payload **335,669,114,744 bytes**, **60,000 reads off** and
  **50,577 reads on**. At 200 events: **6,713,382,256 bytes**, **1,200 off** and
  **1,012 on**. Both modes retain all shared-stream bytes.
- Before submission: **26 CPU harness tests passed**, launchers passed shell
  syntax checks, and **929 frozen file hashes** were verified. The cache helper
  preflight remains enabled before staging to catch missing dependencies early.

## Artifacts

Current: `/sdf/scratch/users/m/monarin/gpu-validation/jf-feespec-scale-20260926-r2`.

- `run.sbatch`, `job-39178621.log`, `job-39178621/`.
- `cache-repair.patch`, `retry.json`: exact repair changes and retry provenance.
- Failed job `39174520` and successful device preflight `39174462` remain
  unchanged in the sibling `jf-feespec-scale-20260926-r1` directory.
- `python/`, `scripts/`, `source-commit.txt`, `source.patch`, `hashes.json`,
  `campaign.json`: frozen runtime, maintained harness and provenance.
- `reference.json`: six-stream SMD and CPU feespec reference; `parent-reference.json`
  retains the JF-only references used to cross-check timestamps and JF payload.
- `pixels.json`, `constants.pkl.gz`: unchanged JF pixel reference and shared
  immutable calibration snapshot.

Use per-job `results.json` for validated partial results. A complete
`summary.json` is written only after all samples and the final file hash check
pass. Require Slurm exit 0, `CAMPAIGN_COMPLETE` and provenance `complete: true`
before declaring the full campaign accepted.
