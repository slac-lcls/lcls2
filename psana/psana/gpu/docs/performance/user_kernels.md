# Batched user-kernel performance

Latest accepted campaigns: September 28–29, 2026, runtime/algorithms
`6ba5fa586`. Both scheduling variants run the same external Jungfrau calibration
and radial-integration kernels and produce independent float64 `(3, 64)` host
histograms (mean, sum, count). This is the matched comparison for moving user
work from the event loop into the pipeline.

## Matched scheduling comparison

Job **39380082** completed all **16 diagnostics and 38 matched pairs** (76 timed
samples). Numerical outputs and timestamp-associated hashes matched. Topology,
input preparation, original constants, quota and cache state were matched.
Pairs alternated order; the main warm case has six pairs, other cases four.

- **Event loop:** user analysis and its two kernels run per event. A
  benchmark-only adapter exposes leased dense inputs; its overhead remains
  included in loop time. The framework callback captures input/constants,
  so framework callback count is distinct from user analysis count.
- **Batched task:** the same analysis runs once per selected execution subbatch;
  psana copies the resulting contiguous publication and delivers host rows.

For a full 20-event execution this changes **20 user calls / 40 kernel launches**
to **1 user call / 2 launches**. Diagnostics checked actual launch/copy counts;
per-event host result delivery still occurs in both paths. `batch_size=1` is not
the event-loop reference: it also changes upstream batching.

Run 387, 10,000 events per timed sample, one EB, A100 GPUs, bulk reads **on**,
eight KvikIO workers/BD, 1 MiB tasks, CPU fallback and private node-local input
prefixes. Warm/cold residency gates are >99%/<1%. The fixed radial map is from
the validated run-51 example; it is a matched workload, not run-387 q-space or
beam calibration. User scratch reserves 80 MiB/event times batch and depth plus
2 GiB/BD outside the explicit framework quota.

Rates are events/s from median loop durations, excluding explicit initialization
but including lazy first-use work. Paired percentage changes are medians of
within-pair changes, not ratios of separately reported medians.

| GPUs | BDs | Batch | Depth | Cache | Event loop | Batched task | Paired loop change | Pair-change range |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 1 | 20 | 2 | warm | 101.30 | 355.28 | -71.07% | -73.54% to -67.32% |
| 1 | 1 | 5 | 2 | warm | 98.58 | 327.66 | -69.53% | -71.36% to -68.70% |
| 1 | 1 | 20 | 1 | warm | 101.74 | 303.31 | -66.04% | -67.70% to -63.85% |
| 1 | 2 | 20 | 2 | warm | 114.81 | 429.14 | -73.49% | -73.95% to -72.52% |
| 1 | 4 | 20 | 2 | warm | 124.61 | 446.35 | -72.27% | -73.20% to -70.74% |
| 2 | 2 | 20 | 2 | warm | 194.27 | 568.14 | -65.80% | -68.27% to -54.34% |
| 2 | 4 | 20 | 2 | warm | 220.56 | 694.89 | -68.31% | -68.93% to -67.50% |
| 4 | 4 | 20 | 2 | warm | 354.39 | 771.92 | -54.99% | -56.49% to -53.19% |
| 1 | 1 | 20 | 2 | cold | 81.70 | 194.13 | -57.92% | -58.78% to -54.42% |

The main warm case reduced median loop time from **98.721 s to 28.147 s**.
Explicit setup medians were **12.180 s and 12.026 s**; whole-sample wall medians
were **118.351 s and 47.346 s**, respectively. Wall time includes MPI/imports,
initialization and cleanup, but excludes prior cache preparation. Separate
medians need not add exactly. All cases improved in this measured workload;
that does not establish a general no-regression guarantee for other callbacks.

## Independent kernel timing check

Job **39381257** repeats the first real run-387 event with hot buffers. CUDA event
intervals exclude allocation, I/O and D2H. Batch-20 histograms exactly match
repeated batch-1 results; 16,377,224 pixels contribute and all 64 sums are nonzero.

| Kernel | One event/launch | 20 events/launch | Batch-20 per event |
| --- | ---: | ---: | ---: |
| Calibration | 4.142 ms | 7.114 ms | 0.356 ms |
| Integration | 2.567 ms | 5.237 ms | 0.262 ms |

These checks support timer plausibility. More integration blocks and possible
constant-cache reuse are code-based explanations, not profiler-proven attribution.

## Full JF calibration/integration scaling

Job **39391686**, `sdfampere040`, completed in **3h49m27s**, exit 0:
22 diagnostics and 88 timed samples. Batch **20**, depth **1**, 10,000 events,
one EB, CPU fallback, bulk off/on and cold/warm, two repetitions. Scratch reserve
is 80 MiB/event plus 1 GiB/BD outside the framework quota. Actual callback sizes
can shrink under memory admission. These settings differ from the matched
main case's depth 2 and bulk-on-only sweep.

| GPUs | BDs | Cold bulk off | Cold bulk on | Warm bulk off | Warm bulk on |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 192.86 | 177.71 | 323.19 | 303.36 |
| 1 | 2 | 257.17 | 253.44 | 414.92 | 381.46 |
| 1 | 4 | 274.64 | 269.72 | 485.98 | 463.53 |
| 1 | 6 | 274.43 | 268.36 | 461.37 | 459.12 |
| 1 | 8 | 267.09 | 268.60 | 440.00 | 450.89 |
| 2 | 2 | 261.28 | 242.86 | 556.97 | 472.14 |
| 2 | 4 | 263.37 | 270.51 | 707.84 | 627.77 |
| 2 | 6 | 276.75 | 266.96 | 698.40 | 698.27 |
| 4 | 4 | 269.11 | 268.42 | 876.65 | 720.85 |
| 4 | 8 | 268.59 | 272.18 | 918.86 | 895.44 |
| 4 | 12 | 266.09 | 266.70 | 886.08 | 851.62 |

Best warm median: **918.86 events/s**, four GPUs/eight BDs, bulk off.
Warm throughput scales through that point, then declines at twelve BDs.
Cold rates level near 270 events/s across many larger configurations.

## Partial JF+feespec calibration/integration scaling

Job **39391687**, `sdfampere014`, completed in **1h06m20s**, exit 0:
6 diagnostics and 24 timed samples. Same settings as the full scaling campaign,
plus per-event feespec GPU int64 sums. Partial denotes the one-GPU/1–4-BD sweep.

| GPUs | BDs | Cold bulk off | Cold bulk on | Warm bulk off | Warm bulk on |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 195.85 | 179.32 | 289.57 | 270.15 |
| 1 | 2 | 260.87 | 245.22 | 301.23 | 317.67 |
| 1 | 4 | 282.28 | 276.16 | 332.76 | 326.75 |

Best warm median: **332.76 events/s**, four BDs, bulk off. Full and mixed
campaigns have matching Jungfrau output hashes. Both bulk settings are useful
measurements; neither is uniformly faster in this matrix.

## Evidence and reproduction

[Compact evidence](evidence/user_kernels.json) retains paired variability,
setup/loop/wall medians, timing-consistency checks, scaling samples' loop times,
output hashes, job identities and source-artifact hashes. The
[harness guide](../../scripts/stage5c/README.md) defines adapters, acceptance and
memory reserves. Job **39395582** verified both completed scaling campaigns.

Frozen roots under `/sdf/scratch/users/m/monarin/gpu-validation/`:

- `user-kernel-stage5c-20260928-r3` (matched pairs)
- `user-kernel-stage5c-kernel-check-20260928-r1` (isolated kernel check)
- `user-kernel-stage5c-followup-20260928-r1` (paired acceptance)
- `user-kernel-stage5c-scale-jf-20260928-r1`
- `user-kernel-stage5c-scale-mixed-20260928-r1`
- `user-kernel-stage5c-scaling-report-20260928-r1/results` (scaling acceptance)

These are calibration-plus-integration measurements. The
[read/staging results](read_staging.md) omit this science workload and must not
be used as a matched denominator. Scratch paths identify full artifacts; compact
accepted evidence is retained here in Git. Correctness scope is recorded in
[design validation](../design.md#validation).

The mixed benchmark uses an explicit benchmark-only routing exception for the
shared feespec stream; co-resident CPU detectors are not consumed. Production
applications needing those consumers must use the documented hybrid routing
contract, including its duplicate I/O cost.
