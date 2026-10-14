# User-kernel Stage 1 / 1b Jungfrau performance regression check

Status (September 27): Stage 1b, the intended input-only runtime, is accepted
for proceeding to Stage 2. Both full sweeps and bulk-off repeats completed;
native launch/copy/synchronization checks passed. The historical Stage 1
calibration-path timing finding remains documented below and is not an
acceptance claim for that removed path.

This is a matched comparison following the Stage 1 and Stage 1b correctness
acceptance. It uses the settings and cache controls in
[Jungfrau current scaling](jungfrau_current_scaling.md), with **one A100 and
1, 2, 3, 4 BDs**, JF only. Historical rates are context, not matched controls.

## Comparisons and measurement scope

| Comparison | Before | After | Workload |
|---|---|---|---|
| Stage 1 extraction | `480f7074c` | `7d0b5941e` | Existing read/parse/gather/calibration; no output D2H |
| Stage 1b removal | `7d0b5941e` | `137e2902f` | Identical read/parse/dense-raw-gather request on each version; no constants, calibration or published outputs |

Stage 1b's default is parse-only. For this comparison a benchmark-only adapter
requests dense Jungfrau raw preparation on both versions, using their existing
budget and slot retirement. The old version needs a `process_batch` shim that
returns no published results; the new version uses its input-preparer map.
This is a controlled internal preparation benchmark, not callback acceptance.

`mfx101210926/r0387`, streams 005–009; 10,000 events, batch 20, depth 1,
eight KvikIO workers/BD, 1 MiB task/bulk target, automatic per-BD quotas,
CPU-fallback I/O. Input source:
`/sdf/data/lcls/drpsrcf/ffb/users/monarin/jf-feespec-bulk-38995226/xtc`.
Private local prefixes, NUMA-interleaved page warming, cold residency below 1%
and warm residency above 99% match the current scaling campaign. One exclusive
node per comparison, 112 CPUs and 700 GiB host memory; only GPU 0 is used.

Each comparison has 64 timed samples: four BD counts × two bulk modes × two
cache states × two versions × two repetitions. Versions run adjacent and all
orders reverse in repetition two. Each fresh MPI process must validate all
10,000 unique timestamps, 335,571,760,000 useful input bytes and 50,000 reads,
no CPU BD payload reads, correct GPU assignment/peer quotas, and clean retirement.
Sixteen separate 200-event preflights per comparison check three CPU-reference
pixels and kernel counts. Profile/preflight times are excluded from throughput.

Primary throughput follows the existing report: 10,000 / maximum rank loop
seconds. Setup before `run.events()` is separate; lazy setup within it remains
included. Per-rank first-delivery times and a descriptive post-first-delivery
rate help identify startup effects. Two repetitions give variability evidence,
not confidence bounds. A slowdown exceeding 5% is a follow-up trigger, not a
hardware-independent test threshold; overlapping repetition ranges need care.

Separate native Nsight traces cover 200 events, bulk-on at 1 and 4 BDs.
The steady NVTX marker starts after the first delivered event on each BD and
ends after iterator cleanup and the benchmark's final device synchronization.
It excludes each BD's first batch: nine batches remain at 1 BD and six total
at 4 BDs. Counts include worker-thread CUDA calls. They are window counts,
not attribution to individual events; durations may overlap. Python preflight
counters separately record inclusive host submission/join times.

## Provenance and progress

Runtime/native provenance is recorded in frozen manifests. No native source
changed across these checkpoints. All three Python snapshots use the same
verified native installation inherited by Stage 1b correctness validation.
Maintained harness: [stage1_regression](../../scripts/stage1_regression/README.md).
Its reused CPU acceptance gates passed: **19 tests**.

- Initial attempt: `/sdf/scratch/users/m/monarin/gpu-validation/jf-stage1-regression-20260926-r1`.
  Stage 1 calibration preflight `39191848`, `sdfampere023`: all 16 samples
  passed, with identical launch inventories on both versions: 10 each of
  walk/init/locate/gather and 200 each of calibration/missing-row cleanup.
  Stage 1b preflight `39191849`, `sdfampere027`, failed before timing because
  the new benchmark adapter used a nonexistent binding attribute. Runtime
  code was unchanged; the adapter now uses `binding.field(...).field_handles_by_segment`.
- Corrected attempt: `/sdf/scratch/users/m/monarin/gpu-validation/jf-stage1-regression-20260926-r2`.
  Stage 1b preflight `39192030`: all 16 samples passed. Every sample had exactly
  10 walk, 10 locator initialization, 10 field-location and 10 gather launches
  for 200 events, unchanged across versions, BD counts and bulk modes.
- Full Stage 1 comparison: job `39192035`, `sdfampere023`, after successful `39191848`.
- Full Stage 1b comparison: job `39192036`, `sdfampere027`, after successful `39192030`.
- Native profiles: `39192192` produced no report with CUDA-profiler API range
  capture. Full-capture retry `39192299` produced reports, but their diagnostics
  state “Could not load the CUPTI library” and “CUDA injection initialization
  failed”; there are no CUDA trace tables. It was canceled after inspecting
  those diagnostics. Both attempts were on `sdfampere029`, Nsight 2026.1.1,
  under `r2/profiles-r1` and `r2/profiles-r2`. These are **not accepted CUDA
  profiles**. The harness now rejects reports with no CUDA kernel records.
  Further isolated attempts (`39192816`, `39192977`, `39193248`) tried
  scratch-local CUPTI 12.9.79 and 13.1.115 with `NSYS_CUPTI_LIBRARY_PATH`.
  The override requires a directory; after correcting that, initialization
  still failed. The original environment and throughput jobs were untouched.
  These attempts yielded no accepted native counts. The later r6 retry below
  resolves profiling. Timed snapshots are unaffected.

Each job directory contains `results.json`, per-sample logs/GPU monitoring and
`provenance.json`. `summary.json` and `complete: true` are written only after
all samples and final frozen-source verification pass. NIC byte deltas are
recorded outside timing; the measured payload is node-local. Generated traces,
logs, snapshots and staged data remain on scratch.


## Resumed native profiling acceptance

Job **39197391**, `sdfampere033`, completed all eight 200-event bulk-on profiles
in 3 min 38 s under `r2/profiles-r6/job-39197391`. The shared installation
`/sdf/group/lcls/ds/tools/nsight-2025.3.1/bin/nsys` includes its CUPTI libraries;
the local 2026.1.1 installation used in the failed attempts does not. No runtime,
throughput snapshot, or global environment was changed for this retry.

The extractor requires nonempty CUDA kernel tables and one completed
`psana.benchmark.steady` NVTX range per BD. All 3,560 frozen runtime/harness
manifest entries and seven profile-script entries verified unchanged after the
run. [Native counts](user_kernel_stage1_native_counts_20260926.json) preserve
per-rank and aggregate evidence. Five extractor tests passed, covering range
boundaries, worker-thread calls, missing CUDA data and incomplete/ambiguous
markers; together with 15 existing benchmark contract tests, **20 passed**.
These tests run standalone because source-tree psana imports require
its built native installation; no production source changed.

**Every matched pair has identical full and steady kernel counts, CUDA API
counts, H2D counts/bytes, and zero D2H copies.** Durations vary and are not used
as throughput evidence. Steady aggregate counts are:

| Comparison | BDs | Before / after kernels | H2D copies, both versions | H2D bytes, both versions | Driver stream waits, both versions |
|---|---:|---:|---:|---:|---:|
| Stage 1 calibration | 1 | 396 / 396 | 6,687 | 6,040,435,680 | 6,660 |
| Stage 1 calibration | 4 | 264 / 264 | 4,458 | 4,026,957,120 | 4,440 |
| Stage 1b dense inputs | 1 | 36 / 36 | 6,687 | 6,040,435,680 | 6,660 |
| Stage 1b dense inputs | 4 | 24 / 24 | 4,458 | 4,026,957,120 | 4,440 |

The four framework kernels each launch nine times for one BD and six times
across four BDs in these steady ranges. Calibration adds 180/120 launches each
for calibration and missing-row cleanup. Runtime stream waits are 18/12 and
terminal device waits 1/4; calibration cases also have 400 event waits. Counts
match within each comparison. Each BD's marker begins after its first delivery,
so four BDs exclude four initial subbatches. These are activity-start window
counts including worker threads, not event-attributed counts; API durations
can overlap and must not be added into wall time. The large H2D/driver-wait
counts belong to the observed CPU-fallback I/O path.

This clears the native launch-structure check for Stages 1 and 1b. It does not
establish a throughput result or validate the still-pending user callback API.

Home-space check at resumption: 18 GiB used of 30 GiB (58%), 13 GiB available;
no cleanup was needed. New traces remain on shared scratch.

## Native scheduling and copies

[Complete native counts](user_kernel_stage1_native_counts_20260926.json) preserve
per-rank and aggregate full/steady results. All **kernel, CUDA API and copy
counts match exactly** between versions within each workload/BD comparison,
in both full and steady capture. All copies are H2D; no D2H appears in these
profiles. This checks the requested work with automatic D2H disabled, not the
future user callback/publication API.

| Steady-window count, summed over BDs | 1 BD, calibration | 4 BDs, calibration | 1 BD, dense inputs | 4 BDs, dense inputs |
|---|---:|---:|---:|---:|
| Each of walk/init/locate/gather | 9 | 6 | 9 | 6 |
| Calibration / missing-row cleanup, each | 180 | 120 | 0 | 0 |
| H2D copies | 6,687 | 4,458 | 6,687 | 4,458 |
| H2D bytes | 6,040,435,680 | 4,026,957,120 | 6,040,435,680 | 4,026,957,120 |
| KvikIO `cuStreamSynchronize` | 6,660 | 4,440 | 6,660 | 4,440 |
| `cudaStreamSynchronize` | 18 | 12 | 18 | 12 |
| `cudaEventSynchronize` | 400 | 400 | 0 | 0 |
| `cudaStreamWaitEvent` | 1,809 | 1,206 | 1,809 | 1,206 |
| Final benchmark `cudaDeviceSynchronize` | 1 | 4 | 1 | 4 |

Each table entry applies to both versions of its matched comparison. The high
fallback H2D/synchronization counts are unchanged. Stage 1b introduces no extra
per-field launches or synchronization. This is structural evidence; timing
conclusions come from the separate unprofiled 10,000-event samples.

## Completed throughput results (September 27)

Both original sweeps completed all 64 timed samples and 16 pixel preflights.
Both same-node bulk-off follow-ups completed all 16 timed samples and four
preflights. All four provenance records have `complete: true`: **160 timed
samples and 40 preflights** in total. These completion checks include event
identity, payload/read counts, budgets, cache residency, and frozen-source
verification; successful execution does not imply a throughput improvement.

Report job `39199741` produced the paired summary and plots under
`r2/reports-r1`. Maintained evidence:
[full sweep data](user_kernel_stage1_throughput_20260926.json),
[scaling figure](user_kernel_stage1_throughput_20260926.svg), and
[repeat summaries](user_kernel_stage1_repeats_20260926.json).

The full sweeps show Stage 1 cold-cache changes from -1.5% to +4.3%, and Stage
1b from -2.8% to +1.2%. Warm/bulk-on results at four BDs improve by 10.6% for
Stage 1 (390.66 to 432.03 events/s) and 15.8% for Stage 1b (444.47 to 514.52).
These are within-workload before/after comparisons, not cross-stage speedups.

The targeted warm/bulk-off follow-ups used four alternating-order repetitions
per version, keeping the original node, snapshots, workload and cache controls:

| Comparison | Job / node | BDs | Before events/s | After events/s | Change |
|---|---|---:|---:|---:|---:|
| Stage 1 | 39200360 / sdfampere023 | 2 | 352.80 | 368.31 | +4.4% |
| Stage 1 | 39200360 / sdfampere023 | 4 | 400.37 | 386.26 | -3.5% |
| Stage 1b | 39200361 / sdfampere027 | 2 | 401.07 | 442.90 | +10.4% |
| Stage 1b | 39200361 / sdfampere027 | 4 | 408.73 | 430.64 | +5.4% |

Thus the original bulk-off slowdowns did not recur above the predefined 5%
follow-up threshold. Stage 1b has no remaining greater-than-5% slowdown that
reproduced in its follow-ups. This is not a statistical proof of zero regression;
repeat results and the original variability are both retained.

## Historical Stage 1 finding and Stage 1b acceptance

The completed full sweep exposed another configuration not included in the
bulk-off repeats: **two BDs, warm cache, bulk on**. Stage 1's primary rate fell
from **356.40 to 320.18 events/s (-10.2%)**, with before repetitions
325.67/393.53 and after repetitions 312.52/328.21. This case still needs checking.

Startup timing is a likely contributor: the earliest first delivery was
9.34/1.74 seconds in the before runs versus 9.31/9.27 seconds after. The
descriptive post-first-delivery rates were 467.88/422.33 before and
440.76/471.73 after, which do not show the same slowdown. That metric is not a
synchronized warmup and does not replace primary event-loop timing or prove
causation. Native launch counts remain unchanged.

Job **39267471** was prepared to repeat this case on `sdfampere023`, using four
alternating-order repetitions per version (eight samples plus two preflights). Frozen
root: `r2/followup-stage1-bd2-bulkon`. The original completed job's dependency ID
had expired in Slurm, so submission uses `run-nodep.sbatch` after independently
verifying its complete results. The runtime and measurement contract are unchanged.

After reviewing the three implementation checkpoints, the user chose to proceed
with Stage 1b as the intended path. Job 39267471 was canceled while still pending;
no measurement from it is claimed. Stage 1b removes both the calibration buffer
allocation and automatic calibration that remain in the historical Stage 1 test.
Its controlled dense-input comparison disables calibration on both versions;
its follow-ups did not reproduce a greater-than-5% slowdown. This supports moving
to Stage 2 without requiring the removed calibration path to pass another sweep.

Acceptance is scoped to the Stage 1b input runtime, its validated correctness,
and measured matched workload. It does not establish a speedup from removing
calibration, statistical equivalence, true-GDS performance, or callback support.
The Stage 1 -10.2% primary timing result is retained, not reclassified as a pass.

The Stage 1 code audit found that extraction moved calibrated-buffer allocation
from before the gather to after it on first use/growth. Setup and BeginStep
calibration operations were otherwise moved into helpers; MPI and reader code
were unchanged. That ordering difference is absent in Stage 1b. It is a possible
startup contributor, not a demonstrated cause of the measured timing difference.
