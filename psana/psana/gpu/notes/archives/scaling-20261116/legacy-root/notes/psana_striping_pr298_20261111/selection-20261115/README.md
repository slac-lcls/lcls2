# Revised stripe-test configuration, November 15, 2026

Select **four BD ranks sharing one GPU, four KvikIO workers per BD, batch 200,
and one CUDA execution stream per BD** for the next psana2-GPU stripe screen.
This replaces the proposed 72-case design with **18 cases**, varying stripe
count only. The existing pilot results and immutable bundles remain historical
evidence; their batch-100/eight-worker harness has not been adapted or validated
for this revised plan. No job was submitted for this revision.

Coordinator document review and consistency checks passed. Independent Claude
review **passed in round 2**, with no remaining findings, at commit
`1bdbd6b01bb5ce05a9047f663e1babf545baf2cc`. Round 1's evidence correction and two
clarifications are resolved. See [review-02.json](review-02.json) and
[responses-02.md](responses-02.md). This is plan acceptance, not native runtime
or upstream cohort acceptance. The user explicitly approved the external review
after its initial automatic approval rejection; review-block.json retains that history.

## Evidence and decision

Reviewed the supplied [report](evidence/REPORT.md) and all four pages of the
[heat map](evidence/heatmaps.pdf). The upload contains these two documents only;
the linked raw CSVs, provenance, tests and underlying independent reviews were
not included. This is a review of the supplied synthesis and a plan selection,
not a new audit of its 102 native runs. Source hashes are in evidence/provenance.json.
The figures label nodes sdfampere017/018/023/036/040, so record those exact names
despite the user's broader description of the SDF environment. Weka results
guide the starting profile; they do not predict Lustre throughput or stripe count.

| Within-allocation evidence | Finding | Decision |
|---|---|---|
| Workers, job 44770441: four BDs, batch 200 | 4 workers/BD: 1635.45 events/s (5.1150 GiB/s); 8 workers/BD: 1606.41 events/s | Choose 4 workers/BD; 1.81% difference is descriptive, not a proven speedup. |
| Same worker job: fixed 16 total workers, batch 100 | 1 BD × 16: 616.48; 2 × 8: 1087.46; 4 × 4: 1551.44 events/s | Independent BD issuers help; worker count alone does not substitute for BDs. Core allowances also differ. |
| Extended batch, job 44321906: four BDs | Batch 200: 1585.47 events/s, +3.95% over batch 100 | Choose batch 200; the separate initial batch cohort also ranks it highest. |
| Task and bounce cohorts, jobs 44852752 / 44870463 | Four-BD gains of 1.15% for 4 MiB tasks and 1.28% for 64 MiB bounce, each in its own cohort | Retain 1 MiB tasks and 16 MiB bounce; no combined gain or unique optimum established. |
| Storage/GPU cohort, job 45028778 | Weka 1/2/4 active GPUs: 4.7839/4.6073/4.8049 GiB/s | One active GPU is a reasonable starting point for staging. |

The complete selected parameter combination was measured in the worker cohort;
it is not assembled from isolated best cells. Its later Weka observation of
1529.58 events/s (4.7839 GiB/s) is a separate cohort, not a replicate to pool with
5.1150 GiB/s. The latest one-GPU Weka cell has 16.4% repeat spread (PDF page 3,
Storage × GPUs panel, Weka row, one-GPU column); small apparent
effects need caution. According to the report, the first four cohorts have result
review, while task, bounce and storage/GPU cohorts await it; the synthesis itself
had no independent verdict. Our plan review does not close those upstream reviews.

At batch 100 in the same worker cohort, the four-vs-eight-worker ordering reverses:
four workers/BD was about 1.8% slower. Sixteen workers/BD at four BDs was not
tested. The practical reason to choose four workers is lower provisioning at
similar observed throughput, not a demonstrated throughput advantage.

## Fixed configuration

| Parameter | Selected value |
|---|---|
| Source | PR298 exact commit 67c327a148a08b2aaf972e67e1e02e367c2b298f; existing isolated build, identity rechecked before use |
| Workload | Five JF files, 100,000 unique event timestamps globally; descriptor staging, no expansion/calibration/cube/payload D2H |
| MPI roles | 1 SMD0 + 1 EB + 4 BDs; zero SRV |
| Active GPU | One A100 40 GB shared by four BDs; helpers GPU-masked |
| CUDA execution streams | 1 per BD, fixed; no stream-count or overlap comparison |
| KvikIO workers | 4 per BD, 16 total |
| batch_size | 200 |
| gpu_bulk_read / gpu_bulk_target_bytes | True / 524288 (512 KiB) |
| PS_SMD_N_EVENTS / PS_SMD_CHUNKSIZE | 1000 / 268435456 (256 MiB) |
| gpu_memory_budget_gb | 6 per BD, 24 GiB aggregate input reservation |
| KvikIO task / bounce | 1048576 / 16777216 bytes (1 / 16 MiB) |
| KvikIO compatibility / auto-direct | ON / OFF; CPU fallback, not GDS |
| CPU binding target | 34 distinct physical cores: 3 per helper and 7 per BD; disjoint rank masks, one logical CPU/core, NUMA interleave |
| CPU allocation | At least the binding target; allow scheduler rounding, e.g. 36 physical cores; exclusive node may reserve more |
| Other controls | PS_PARALLEL=mpi, PS_FAKESTEP_FLAG=0, OpenMP/BLAS threads 1 |
| Lustre stripe size | 1 MiB, existing verified copies, same pool/layout type |
| Account / QoS | lcls_g / premium |

The CPU assignment is a proposed Perlmutter mapping using the report's allowance
formula, 6 + BDs × (workers + 3); it is not evidence of busy cores or savings in
an exclusive allocation. It differs from our pilot's shared 56-core masks.
Before execution, derive masks from actual Slurm rank allowances and NUMA/core
inventory; verify every rank's assigned CPUs lie within its allowed set and masks
are disjoint. If this mapping cannot be satisfied, stop and revise the documented
binding rather than silently fall back. Freeze the same map for all stripe cases.
Do not attribute a comparison against the old pilot solely to batch or workers.

The estimated worker bounce demand is 16 × 16 MiB = 256 MiB; it is not a measured
pinned-memory total. CUDA contexts, pools and other overhead are additional to
the 24 GiB input reservation. Retain actual memory-headroom and drain checks.

## Stripe matrix and selection

Use counts **1, 2, 4, 8, 16, 32**, three randomized complete blocks, 100k events
per case, with all controls above fixed. The proposed order is in [sweep.csv](sweep.csv):
one random.Random(20261115), a fresh ascending count list shuffled sequentially
for each block. All 18 cases use one allocation/node/GPU, fresh MPI processes
per case, the same event set and existing full SHA-verified count copies.

Preserve the original plan's local-cache and binding gates: compile/setup outside
timing; owner-scoped eviction and two zero mincore scans across all 30 payload
files, immediate selected-five zero check, actual KvikIO and DgramManager handle
identity checks, exact global events/read ranges/bytes, and complete GPU drain.
Backend caches remain uncontrolled and wire traffic unverified unless measured
with validated counters. The primary metric is completed useful JF GiB divided
by the common event-loop-through-drain interval, with events/s and full case
seconds also reported. Each full case must retain 500,000 reads and
335,820,000,000 useful bytes (312.756747 GiB).

Require all three complete blocks for ranking. Report all rates, per-count
median/min/max and block-paired ratios to count 1. Retain count 1 and up to two
non-1 counts with the highest geometric mean paired ratios for later placement
confirmation. A smallest count within 5% of the best median may be a provisional
operational candidate, not a confirmed optimum. Overlapping or unstable rankings
remain inconclusive. A later independent-placement confirmation should use this
same fixed profile, replacing the old four-BD/two-stream target. No confirmation
copies or jobs are part of this revision. Multinode and full-cube validation
remain necessary before transferring the stripe recommendation to those workloads.

## Before running

Adapt and review the harness and auditor for the new batch, worker count, CPU
masks and schedule. Validate expected callback sizes independently rather than
just replacing hardcoded 100 by 200; 1003-event correctness and exact timestamp/
payload witnesses must cover the tail. The old small pilot does not qualify this
configuration. Keep the existing exact-PR correctness requirements and all six
stripe binding checks. Implement the already deferred peak device-memory telemetry
and bounded sizing/screen launcher before any submission.

Then qualify the selected configuration on Lustre with a small correctness run
and one separate count-2, four-BD, one-stream, batch-200 full-100k sizing run,
excluded from ranking. This replaces the old one-BD/two-stream sizing proposal.
Success establishes local usability, not a proven joint parameter optimum on Lustre.
Use its full-case duration to budget the 18-case screen. The historical campaign
used 3042 node-seconds, leaving 18558 seconds (5h09m18s) of the six-node-hour cap
before any new work; recheck the ledger and Iris before submission. Do not infer
an ETA from Weka rates. No new allocation, code adaptation or sweep submission
is authorized or performed by this document-selection step alone.

If a later one-BD diagnostic is needed, the supplied report favors 8 workers and
batch 50. It is outside this 18-case stripe matrix and would not be a pure BD-count
comparison because its batch and worker settings differ.
