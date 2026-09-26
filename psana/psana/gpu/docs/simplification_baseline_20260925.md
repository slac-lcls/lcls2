# GPU pipeline simplification baseline

Updated **2026-09-26** with the completed JF-only campaign and mixed-detector
retry status. Prepared as context for a code-size simplification task.
Performance rates summarize two repetitions;
they are not confidence bounds. This document does not authorize changing benchmark definitions or
weakening correctness checks to obtain a simpler implementation.

## Source and evidence

- Worktree: `/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline`.
- Branch: `codex/psana2-gpu-bulk-batched-integration`.
- Measured production runtime: `ad8d454d10e203d3ef02d9c75069da48a31de182`.
- Current benchmark runtime includes cleanup `cc4451b3c` and is frozen at the
  campaign below. The retry changed only the harness, not the production runtime.
- The handoff checkpoint containing this document tracks the scaling harness,
  reports, historical-note relocation and cache-repair tests. Its additional
  changes are benchmark/test/documentation changes, not production runtime
  changes. The original worktree still contains unrelated untracked historical
  artifacts; use a separate worktree from the pushed checkpoint.
- Current campaign:
  `/sdf/scratch/users/m/monarin/gpu-validation/jf-current-scale-20260925-r4`.
  Read `source-commit.txt`, `source.patch`, `harness.patch`, `retry.json`,
  `hashes.json`, and `run.sbatch` for exact provenance and launch configuration.
  Keep this frozen baseline unchanged when preparing a simplified candidate.

## Starting the simplification task

Create a new branch/worktree from the pushed checkpoint on
`origin/codex/psana2-gpu-bulk-batched-integration`; use the exact checkpoint hash
provided with this handoff to avoid later moving branch tips. Scope the work to
reducing production GPU pipeline code size and duplicated control flow while
preserving its public behavior, ownership, memory accounting and performance.
Do not count deleting tests or benchmark evidence as implementation simplification.

Start with the GPU README's current-design documents, the completion checklist,
and the cleanup coverage mapping linked below. Inspect the source before
choosing targets: the old residency policy and bulk adapter are already gone.
Record before/after production line counts for the same explicit file set and
explain structural reductions. Keep changes reviewable and tie any removed
branch to equivalent coverage.

Run relevant CPU and A100 lifetime/pixel acceptance, then compare independently
frozen baseline/candidate runs under the same benchmark settings. Report
performance variability rather than inferring equivalence from one sample.
Keep builds and generated outputs on scratch; give the candidate a separate
installation. Frozen campaigns contain symlinks to shared native dependencies:
do not rebuild into or modify those referenced installations or calibration files.

## Mixed-detector benchmark in progress

At handoff, job **39178621** is **RUNNING on sdfampere031**. All six pixel
preflights passed, followed by the first cold 1-BD bulk-off sample. Full results
remain pending. This is a baseline run of the same production runtime, not a
simplified candidate. It uses `mfx101210926/r0387`, adds stream 000 to JF streams
005–009, and runs one GPU with 1/2/4 BDs, cold/warm and off/on, twice (24 samples).
Batch 20, depth 1, eight workers/BD, 1 MiB tasks/target and automatic budgets
match the JF-only baseline; the timed consumer adds feespec GPU sums.

Artifacts:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-feespec-scale-20260926-r2`.
The benchmark-only exclusive shared-stream override is in
`scripts/jf_scaling/feespec.py`; do not promote it into production routing.
Expected 10k payload is 335,669,114,744 bytes, 60,000 reads off / 50,577 on.
Every event's feespec sum is checked against CPU by timestamp; pixel preflights
check all 200 feespec arrays and three JF raw/calibrated samples per case.

The prior attempt failed warm-cache residency, not GPU correctness. The retry
repairs missing pages before timing with at most three passes, rechecks every
file, and retains the 99% gate and read-only postchecks. Validation: 33 shared
cache/harness tests and 35 frozen scaling/cache tests passed (overlapping suites),
plus a real missing-page repair test. Check Slurm, `results.json`, `summary.json`
and `CAMPAIGN_COMPLETE` before treating this campaign as complete. See the
[mixed-detector report](performance/jf_feespec_single_gpu_scaling.md).

## Implemented and validated behavior

The runtime uses independently owned, bounded stream-read groups and batched
multi-buffer GPU parsing/gathering. Bulk-on submits resolved groups through
`issue_group`; bulk-off retains per-datagram `issue_batch` reads.

All three measured CPU-overhead optimizations are implemented:

1. Per-plan raw-slot capacity index: over 93% less profiled slot-selection time.
2. Per-file pending-read reference counts: replace repeated ownership scans.
3. Direct group submission: avoid rebuilding legacy plans; 26–33% less profiled
   group-submission time in the recorded optimization campaign.

These are local profiling improvements, not additive end-to-end speedups.
Old whole-stream residency policy/orchestration and the obsolete reader bulk
adapter have already been removed. Do not restore them from historical tools.

Recorded final cleanup acceptance: **430 CPU checks** (406 unit + 24 harness),
**53 A100 tests**, and all 41 timing instrumentation patches installed. Device
coverage includes an 18-case retained-view/tight-budget/delayed-consumer matrix,
depth 1/2, 4/8 MiB working budgets, automatic D2H, independent CUDA consumers,
early exit, failures, and transition drain/retry behavior. These tests were
recorded by the prior cleanup task, not rerun for this handoff. See
[completion checklist](stream_read_refactor_cleanup.md),
[ownership acceptance](stream_read_ownership_acceptance.md), and
[cleanup coverage mapping](stream_read_legacy_cleanup.md).

## Current Jungfrau-only scaling campaign

Job **39104724**, node **sdfampere030**, **COMPLETED**, exit **0:0**, elapsed
**1h25m58s**. **8/8 pixel preflights and 32/32 timed samples passed**, followed
by final source-hash verification. `summary.json`, `CAMPAIGN_COMPLETE` and
`provenance.json` with `complete: true` are present. The private stage was removed.
Rates below are **10,000 divided by median loop seconds across two repetitions**.

| GPUs | BDs | Cold off (events/s) | Cold on (events/s) | Warm off (events/s) | Warm on (events/s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 200.70 | 192.06 | 306.46 | 323.64 |
| 1 | 4 | 302.16 | 289.00 | 475.87 | 498.42 |
| 2 | 4 | 298.18 | 303.30 | 618.49 | 592.33 |
| 4 | 8 | 303.00 | 301.75 | 680.58 | 750.93 |

Cold rates cluster around 300 events/s with four or more BDs, consistent with
a storage limit but not proof of disk saturation. Warm throughput scales to
750.93 events/s (23.47 GiB/s useful input) at 4 GPUs/8 BDs, bulk on.
Bulk-on warm changes versus off are +5.61%, +4.74%, -4.23% and +10.34% for the
four topologies. There is no universal bulk-on throughput win.

Retain individual repetitions for simplification comparisons: 2 GPU/4 BD warm
bulk-off varied from **712.12 to 546.61 events/s**, while bulk-on ranged from
603.39 to 581.67. At 4 GPU/8 BD warm, bulk-on exceeded off in both repetitions
(743.23 vs 695.13, and 758.80 vs 666.63). Two repetitions do not establish
statistical confidence for small changes. Full per-sample evidence remains in
`job-39104724/results.json`; aggregate rates are in `summary.json`.

### Reproduction settings

- Dataset: `mfx101210926/r0387`, JF streams 005–009 only, no feespec override.
- 10,000 events per timed sample; topologies 1 GPU/1 BD, 1/4, 2/4, 4/8;
  bulk off/on, cold/warm, two fresh-process repetitions with reversed order.
- One SMD0 and one EB; batch 20; execution depth 1; eight KvikIO workers per
  BD; 1 MiB KvikIO task size and bulk target; automatic per-BD GPU budgets.
- KvikIO compatibility mode ON (CPU fallback); no automatic D2H in timed runs.
- Timed consumer reads timestamps. The iterator includes input reads, GPU
  parsing/calibration and lazy setup; staging, cache preparation, setup before
  iteration and teardown are outside the timing. This is not a user-kernel
  benchmark or an isolated disk-bandwidth measurement.
- Eight separate 200-event preflights check three CPU-reference raw/calibrated
  pixel samples each, covering every topology/mode. Their rates are diagnostic.
- Both modes read exactly **335,571,760,000 bytes in 50,000 API requests** for
  10,000 events. Each JF datagram exceeds the bulk target, so bulk mode does
  not reduce the API request count for this dataset.
- Acceptance checks unique timestamps, bytes/requests, zero CPU bigdata reads,
  physical GPU assignments and sharing, CPU affinity, owned-memory budgets,
  cleanup, cache residency, reference pixels and source hashes.

### Storage and units

Private input prefixes are staged onto XFS `/lscratch`, `/dev/md0`, a RAID0
array of two KIOXIA CD6-R 3.84 TB NVMe drives (`KCD6XLUL3T84`). RAID stripes
the files across both drives; BDs are not individually assigned to drives.
Cold samples evict the private files from Linux page cache and require less
than 1% residency before reading. Warm samples preload the prefixes with NUMA
interleaving and require greater than 99% residency. CPU-fallback reads go
through host memory before host-to-GPU copies.

Useful input is **32.0026169 MiB/event**. Multiply events/s by
`33557176 / 2**30` for GiB/s: **305.92 events/s = 9.56 GiB/s**.
The manufacturer's 6,200 MB/s per-drive rating gives a theoretical additive
pair rate of **11.55 GiB/s**, or about **370 events/s** of this input.
The measured event-loop rate is about 83% of that rating; controller/PCIe
topology and actual disk utilization have not been established by that ratio.
[KIOXIA specifications](https://www.kioxia.com/content/dam/kioxia/shared/business/ssd/data-center-ssd/asset/productbrief/dSSD-CD6-R-product-brief.pdf).

### Harness correction

Earlier job 39100314 failed after eight successful pixel preflights and two
cold samples: the frozen warm-cache helper omitted `memory_state.py`.
The `-r4` retry includes and hashes it, requires all cache helpers in the
manifest, and exercises the real warm-cache subprocess before staging.
Pre-submission verification: **19 CPU harness tests**, 924 file hashes,
real cache preflight, 1 MiB warm-prefix subprocess, and shell syntax passed.
The current job has completed warm samples, confirming the original failure
is resolved. Failed `-r3` evidence is preserved separately.

## Separate completed mixed-detector comparison

Job **39084570** completed eight audited JF+feespec 10k samples before final
cleanup. Median rates:

| Cache | Bulk off (events/s) | Bulk on (events/s) | Bulk-on loop-time increase |
|---|---:|---:|---:|
| Cold | 134.15 | 127.26 | 5.42% |
| Warm | 304.32 | 245.88 | 23.77% |

This used one A100/one BD, batch 100, depth 1, an 8 GiB budget, 4 MiB bulk/task
sizes, eight workers and CPU fallback on Weka-backed input. Requests dropped
from 60,000 to 50,182, but throughput worsened. Do not compare these rates
directly with the current local-storage JF-only campaign or describe them as
post-cleanup measurements. See [full report](performance/stream_read_current_10k.md).

## Constraints for simplification

- Preserve input ownership until every planned use and consumer completion
  token finishes, including retained public field views, independent consumers,
  out-of-order completion, parser aliases and delayed D2H.
- Preserve byte-budget admission, rounded allocation growth, old-plus-new
  replacement peaks, fixed/cache allocations, live aliases and multi-BD budgets.
- Preserve small-stream credits across EB batches, execution splits needed for
  progress, and independent stream scheduling; avoid self-deadlock under pressure.
- Preserve batched parser/gather launches, event identity, missing-data behavior,
  bulk-off parity and canonical detector segment ordering.
- Drain before BeginStep calibration replacement and EndRun dispatch; retain
  ownership after failed drains so retries and early close remain safe.
- Simplify representation/control flow only with replacement coverage for the
  invariants above. Existing complexity often represents asynchronous lifetime
  or failure-recovery requirements, not obsolete policy.
- Validate with CPU and real-device acceptance, then compare a separately frozen
  candidate using the same workload and controls. JF-only throughput does not
  validate mixed-rate stream progress, true GDS, live data or user kernels.
- Do not edit the frozen baseline, change performance gates, or delete
  historical evidence. Use matched repeated baseline/candidate runs and inspect
  variability before treating small rate differences as regressions or gains.

Maintained harness: [jf_scaling/README.md](../scripts/jf_scaling/README.md).
Current campaign report: [jungfrau_current_scaling.md](performance/jungfrau_current_scaling.md).
