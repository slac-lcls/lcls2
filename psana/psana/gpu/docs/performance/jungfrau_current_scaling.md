# Current-runtime Jungfrau GPU/BD scaling

**Status: job 39104724 COMPLETED, exit 0:0; all acceptance gates passed.**
Verified 2026-09-26. Elapsed **1h25m58s**, node **sdfampere030**.
This campaign follows the user's request to check Jungfrau-only scaling before
moving to internal user-kernel support. It does not change the production
runtime. The final results below establish this frozen runtime's JF-only baseline,
followed by the completed single-GPU JF+feespec comparison.

## Completed results

All **8 pixel preflights and 32 timed samples passed**, along with final source
hash verification. The log ends with `CAMPAIGN_COMPLETE`, provenance records
`complete: true`, and the private local stage was removed successfully.
Rates are **10,000 / median loop seconds** across two repetitions per cell.

| GPUs | BDs | Cold off (events/s) | Cold on (events/s) | Warm off (events/s) | Warm on (events/s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 200.70 | 192.06 | 306.46 | 323.64 |
| 1 | 4 | 302.16 | 289.00 | 475.87 | 498.42 |
| 2 | 4 | 298.18 | 303.30 | 618.49 | 592.33 |
| 4 | 8 | 303.00 | 301.75 | 680.58 | 750.93 |

Cold throughput clusters near 300 events/s with four or more BDs, consistent
with a storage limit but not proof of saturation. The fastest warm configuration
is 4 GPUs/8 BDs, bulk on: **750.93 events/s**, **23.47 GiB/s** of useful input,
10.34% above bulk-off. Bulk-on is not faster in every configuration.

Variability matters: 2 GPU/4 BD warm bulk-off was **712.12 and 546.61 events/s**;
bulk-on was **603.39 and 581.67**. For 4 GPU/8 BD warm, off was **695.13 and
666.63**, versus on **743.23 and 758.80**. Individual repetitions remain in
`job-39104724/results.json`. Two repetitions are not statistical confidence
bounds; keep matched controls when evaluating simplification.

The PDF supplied with the request corresponds to the existing report formerly
at `notes/psana2_gpu_performance_20260830.md`, now at
[Jungfrau single-node scaling](jungfrau_single_node_sdf.md) and linked from the
GPU documentation index. Those measurements use historical commit `e18cf6bb7`.
The report was present on this branch under `notes`, but absent from `gpu/docs`.

## Completed JF+feespec single-GPU comparison

Job **39178621 COMPLETED**, exit **0:0**, on **sdfampere031** in
**1h07m57s**, verified 2026-09-26. All **6 pixel preflights and 24 timed
samples passed**, including every event's timestamp-associated feespec sum
against the CPU reference. Final source-hash verification passed, the log records
`CAMPAIGN_COMPLETE`, provenance has `complete: true`, and the private stage was
removed. This supersedes the partial results from the failed first attempt.

The production runtime remains **`ad8d454d1`**. The dataset is the same
`mfx101210926/r0387`, adding feespec stream 000 to JF streams 005–009. The sweep
uses one A100 with 1/2/4 BDs, cold/warm, bulk off/on, and two repetitions in
reversed order. Batch 20, depth 1, eight KvikIO workers/BD, 1 MiB tasks and bulk
target, automatic per-BD budgets, CPU-fallback reads, and private local-storage
cache controls match the JF-only campaign.

The mixed timed consumer accesses feespec's public GPU field and computes an
int64 sum per event in addition to the JF calibration path. Field-access metadata
D2H is inside timing; the compact sums are copied for validation after timing.
The shared-stream exclusive routing override is benchmark-only. Each separate
200-event preflight checks all feespec arrays and three JF raw/calibrated samples
against CPU references. See the [mixed benchmark setup and retry history](jf_feespec_single_gpu_scaling.md).

Rates are **10,000 / median loop seconds** across both repetitions:

| GPUs | BDs | Cold off (events/s) | Cold on (events/s) | Warm off (events/s) | Warm on (events/s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 198.80 | 187.93 | 315.89 | 291.23 |
| 1 | 2 | 281.84 | 272.73 | 342.18 | 402.51 |
| 1 | 4 | 305.87 | 313.06 | 408.40 | 416.99 |

Best warm throughput is **416.99 events/s**, **13.04 GiB/s** of useful input;
best cold is **313.06 events/s**, **9.79 GiB/s**. The exact 10k payload is
**335,669,114,744 bytes** in both modes, with **60,000 API requests off** and
**50,577 on** (15.7% fewer). Feespec adds only about **0.03%** to the useful
input bytes, but adds parsing, field access and GPU reduction work. This
campaign does not isolate the cost of those operations.

Warm bulk-on throughput changes versus off are **-7.8%, +17.6%, and +2.1%**
at 1/2/4 BDs. The first repetition's apparent bulk-on gains did not persist
uniformly: 1-BD warm off was **277.44 / 366.71 events/s**, versus on
**298.82 / 284.01**; 4-BD warm off was **380.25 / 441.05**, versus on
**417.69 / 416.30**. Two repetitions do not establish confidence for small
differences or a universal bulk benefit.

Relative to the JF-only baseline above, warm bulk-on is **10.0% slower at
1 BD** and **16.3% slower at 4 BDs**; warm bulk-off is **3.1% faster at
1 BD** and **14.2% slower at 4 BDs**. Cold throughput still reaches roughly
300 events/s with four BDs. The campaigns used different nodes and timed
consumers, and exhibit repetition variability; these are workload comparisons,
not an isolated measurement of feespec overhead or a runtime regression.

Artifacts:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-feespec-scale-20260926-r2`

- `job-39178621/{summary,results,provenance}.json`: final aggregates, individual
  samples and completion evidence; per-sample logs and GPU CSVs are alongside.
- `job-39178621.log`, `run.sbatch`, `hashes.json`, `reference.json`: execution,
  launch settings, frozen-source hashes and CPU reference evidence.
- `retry.json`, `cache-repair.patch`: bounded missing-page repair before warm
  timing, retaining the 99% residency gate and read-only postchecks. Warm cases
  completed successfully in this retry; the failed first attempt remains in
  the sibling `jf-feespec-scale-20260926-r1` directory.

## Requested JF-only comparison

- Current source **`ad8d454d1`**, including cleanup `cc4451b3c`, frozen with native
  dependencies from the prior validated Integrated installation.
- `mfx101210926/r0387`, Jungfrau streams 005–009 only; no feespec routing override.
- Key historical points: **1 GPU/1 BD, 1/4, 2/4, 4/8**. The full 11-point matrix
  remains available in the harness; it is not part of this initial submission.
- **Bulk off and on**, cold and warm, **10,000 events**, two repetitions: 32
  timed samples. Eight separate 200-event preflight processes check raw and
  calibrated pixels against the existing frozen CPU references.
- Historical settings: batch 20, depth 1, eight workers/BD, 1 MiB task size,
  no D2H in timed runs. Current bulk target is 1 MiB; current automatic per-BD
  budgets divide each GPU among its assigned BD peers.
- One exclusive four-A100 node, 112 CPUs and 700 GiB host memory requested.
  Actual GPU, filesystem and local drive identities are recorded by the job;
  hardware is not assumed identical to the historical allocation.
- Private node-local prefixes reproduce the local-storage/page-cache experiment,
  rather than the recent mixed-detector Weka experiment. Exact useful payload:
  **335,571,760,000 bytes**; **50,000 requests** in either read mode.

The [maintained harness](../../scripts/jf_scaling/README.md) retains timestamp,
byte/request, GPU assignment, BD-sharing, memory-budget, cache and pixel gates.
Its CPU gate tests pass: **15 tests**. Preflight **39099796** recorded
`SMOKE_PASS off` and `SMOKE_PASS on`: each mode completed 200 events with four
BDs on one GPU, 1,000 reads, and three raw/calibrated CPU-reference pixel checks.
These diagnostic runs are not throughput measurements. Slurm confirmed the
preflight completed with exit code 0 in 1 minute 10 seconds on `sdfampere011`.
The original full campaign was submitted with `afterok:39099796` as job
39100314. Its failure and corrected resubmission are recorded below.

Actual execution time was **1h25m58s**, exceeding the initial 40–60 minute
estimate but within the **1h59m** walltime limit. The total includes staging
approximately 336 GB, repeated cache preparation, fresh MPI processes and eight
pixel preflights, in addition to the timed event loops.

## Artifacts

`/sdf/scratch/users/m/monarin/gpu-validation/jf-current-scale-20260925-r4`

- `run.sbatch`, `job-39104724.log`: launcher and scheduler transcript.
- Prior preflight evidence remains in the sibling `-r3` campaign:
  `smoke-39099796.log`, `smoke-39099796/{off,on}.log`.
- `python/`, `scripts/`, `source-commit.txt`, `source.patch`, `hashes.json`:
  frozen runtime/harness identity and validated shared calibration dependency.
- `retry.json`, `harness.patch`: retry provenance and the cache-helper startup
  validation change; the production runtime is identical to `-r3`.
- `reference.json`, `pixels.json`: SMD-derived event/extent checks and CPU pixel
  digests. The 10k timestamp SHA-256 is
  `23f9051ac9f98208310171f734251b0ccd1f40fa048d1748553cbd38cce559d6`.
- `job-39104724/`: per-sample logs, GPU monitor CSVs, results and provenance
  written when the job starts.
  `summary.json` is written only after every sample and final hash check pass.

Initial preflight job 39099425 found a harness wrapper that did not forward
`calib_leader`; no timed comparison ran. The original queued campaign 39099160
was canceled. Second preflight 39099606 passed the bulk-off checks but stalled
in MPI teardown; queued campaign 39099625 was canceled. The `-r3` harness
explicitly closes psana's shared-memory windows after all GPU work and peer
checks complete. Both modes then passed preflight 39099796. Production runtime
sources were unchanged throughout these harness corrections.

## Failed campaign and corrected resubmission

Job **39100314** failed on `sdfampere014` after **14m03s**, exit `1:0`.
All eight topology/mode pixel preflights passed. Two of 32 timed samples passed:
1 GPU/1 BD, cold, repetition 1, **206.95 events/s bulk off** and
**198.68 events/s bulk on**. These are preliminary individual samples, not a
completed scaling comparison. No warm or multi-GPU timed samples completed.

The first warm-cache preparation failed because the frozen harness included
`warm_cache.py` but omitted its imported `memory_state.py`. The failed attempt
and partial results remain unchanged in `jf-current-scale-20260925-r3`.

The `-r4` campaign includes and hashes that dependency. The maintained runner
now requires all three cache helper files in the manifest and runs the actual
NUMA-interleaved warm-cache subprocess with an empty prefix list before staging
input or starting GPU work. Pre-submission validation passed **19 CPU tests**
(15 acceptance tests and four dependency-manifest cases), all **924 file hashes**,
the actual cache preflight, a **1 MiB prefix warm subprocess**, and shell syntax.

Job **39104724** reran all eight pixel preflights and all 32 timed samples with
the original runtime, topology matrix, cache controls and benchmark settings.
The corrected campaign completed successfully; its final results are above.
Historical rates are comparison context only.
