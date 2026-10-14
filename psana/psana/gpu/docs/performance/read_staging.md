# JF and JF+feespec read/staging performance

Latest accepted staging campaigns: September 28, 2026, runtime `10df4c6e3`.
These measure JF read, GPU XTC parsing and dense raw gathering **without a
GpuTask or calibration**. A benchmark-only input adapter requests the gather.
Mixed runs also perform the existing per-event feespec `raw.hproj` GPU int64
sum. Thus these are pipeline-staging rates, not isolated storage bandwidth.
The production no-task path does not automatically request that dense gather.

## Configuration and timing definition

Run 387, 10,000 events, full 32-panel JF across five streams; mixed runs add
feespec. One EB, A100 GPUs, batch 20, execution depth 1, eight KvikIO workers per
BD, 1 MiB task/group targets, automatic per-GPU peer budgets and **CPU fallback**.
Each topology uses bulk off/on and cold/warm cache preparation on private
node-local staged prefixes. Two fresh-process repetitions reverse ordering.

Rates below are **10,000 / median event-loop seconds**, including lazy work
inside the loop. Explicit DataSource/run setup is separate. Staging, cache
preparation, imports and MPI startup are excluded from loop time. Both sample
loop durations, setup, cache checks, GPU placement and resource counters are
retained in [the evidence](evidence/read_staging.json). Two repetitions support
these descriptive medians, not narrow confidence intervals.

## Full JF

Job **39369005**, `sdfampere035`, completed in **3h07m01s**, exit 0.
22 preflights and 88 timed samples passed; 1,289 frozen file hashes verified.

| GPUs | BDs | Cold bulk off | Cold bulk on | Warm bulk off | Warm bulk on |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 221.58 | 211.13 | 437.29 | 382.33 |
| 1 | 2 | 314.45 | 310.28 | 494.16 | 458.29 |
| 1 | 4 | 341.64 | 351.64 | 484.13 | 489.12 |
| 1 | 6 | 355.81 | 355.22 | 456.77 | 463.21 |
| 1 | 8 | 357.91 | 358.70 | 432.26 | 433.81 |
| 2 | 2 | 320.36 | 313.20 | 729.66 | 606.74 |
| 2 | 4 | 354.06 | 353.18 | 689.48 | 658.26 |
| 2 | 6 | 352.20 | 345.57 | 715.47 | 781.78 |
| 4 | 4 | 353.86 | 356.99 | 1007.14 | 847.35 |
| 4 | 8 | 357.54 | 354.89 | 918.04 | 848.15 |
| 4 | 12 | 360.10 | 359.92 | 914.20 | 882.14 |

Peak warm median is **1007.14 events/s** at four GPUs/four BDs, bulk off.
Cold rates approach **360 events/s** across the largest configurations.
Increasing BDs does not monotonically improve warm throughput; bulk grouping
also does not consistently win for these large JF dgrams.

## Partial JF+feespec: one GPU, 1–4 BDs

Job **39369006**, `sdfampere033`, completed in **57m03s**, exit 0.
6 preflights and 24 timed samples passed; 1,289 frozen file hashes verified.
“Partial” denotes the smaller topology sweep, not a reduced JF panel count.

| GPUs | BDs | Cold bulk off | Cold bulk on | Warm bulk off | Warm bulk on |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 205.93 | 188.87 | 377.41 | 336.26 |
| 1 | 2 | 282.74 | 263.74 | 447.31 | 456.40 |
| 1 | 4 | 305.36 | 305.60 | 500.29 | 489.88 |

Peak warm median is **500.29 events/s** at four BDs, bulk off; the best cold
median is **305.60 events/s** at four BDs, bulk on. The mixed workload includes
feespec reduction and its validation; it is not JF-only I/O with another label.

## Correctness and provenance

Preflights checked three JF raw arrays against the independent CPU reference;
mixed preflights checked all 200 feespec arrays. Each timed mixed run validated
10,000 timestamp-associated sums. Timestamp identities, payload/read counts,
GPU placement, quotas and drained resources gated acceptance. No calibrated-pixel
comparison is claimed for this staging workload.

Frozen roots under `/sdf/scratch/users/m/monarin/gpu-validation/`:

- `jf-user-kernel-full-scale-20260928-r1`
- `jf-feespec-user-kernel-scale-20260928-r1`

The [JSON evidence](evidence/read_staging.json) preserves sample measurements,
accepted summaries, source/log hashes and scheduler results. Historical baseline
metadata inside it is provenance only. Earlier automatic-calibration reports
measure different work; their ratios are not matched regression results.
Maintained measurement harness: [jf_scaling](../../scripts/jf_scaling/README.md).

Use [user-kernel performance](user_kernels.md) for matched scheduling and the
latest calibration/integration scaling. Neither report establishes true-GDS
performance or a universally optimal bulk setting.

The mixed benchmark uses an explicit benchmark-only routing exception for the
shared feespec stream; co-resident CPU detectors are not consumed. Production
applications needing those consumers must use the documented hybrid routing
contract, including its duplicate I/O cost.
