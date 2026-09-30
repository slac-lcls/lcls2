# Stage 5c: matched calibration/integration performance

**Status:** Complete and accepted: 16 diagnostics and 38 matched pairs (76 timed samples). Job **39380082**
uses runtime **`6ba5fa586`** plus benchmark-only harness additions. No user
algorithm or production runtime changes were made for this comparison.

Campaign and frozen provenance:
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage5c-20260928-r3`.
The retry started September 28 at 17:28 Pacific on `sdfampere040`.

Initial launch **39378791** stopped before its first diagnostic completed because
the launcher omitted `PS_EB_NODES` and `PS_SRV_NODES`. No timed samples were
collected. The retry sets both variables and checks the MPI environment before
input staging. The failed attempt's logs remain in the `-r1` campaign directory.

Retry **39379015** passed 12/16 diagnostics on `sdfampere012`, then stopped at
17:24 Pacific because one BD received no events in the two-GPU/four-BD
200-event diagnostic. The failure was the all-BDs-active coverage gate, with
no timed samples collected. The `-r3` retry uses 1,000 diagnostic events with an
independent SMD-derived reference; all BDs must still participate. Performance
samples remain 10,000 events and all other matrix settings remain unchanged.

The [harness and measurement definitions](../scripts/stage5c/README.md) describe
16 correctness/diagnostic samples followed by 76 timed samples (38 balanced
pairs). It compares the same calibration/integration kernels and compact
histograms scheduled in the per-event public loop versus a batched GpuTask.
Input staging and original constant uploads are matched; the baseline's
benchmark-only dense-input exposure is documented and timed.

The matrix covers one BD/GPU, batch 5 versus 20, depth 1 versus 2, shared GPUs,
and two/four GPUs. The main single-BD case has six warm pairs and four cold
pairs; other cases have four warm pairs each. Numerical/checksum or cache-gate
failure stops the campaign before a result is accepted.

The host-only paired-result acceptance suite passed **8 tests**. The first A100
diagnostic pair (200 events, one BD/GPU, batch 20, depth 2) passed numerical
checks with identical timestamp-associated histogram SHA-256
`51a4eeddfdc614a9cb40a3f628b3f61a9bd6339679a608c9fc5ca069dd19a036`:

| Path | User analysis calls | Actual kernel launches | Output copy groups |
| --- | ---: | ---: | ---: |
| Per-event loop | 200 | 400 | 200 |
| Batched task | 10 | 20 | 10 |

Those first-pair counts are evidence from `-r2`; `-r3` repeated every diagnostic
with 1,000 events before timing. All 16 passed. At the requested ten-minute
checkpoint (17:53 Pacific), the job remained running and two timed samples had
completed: the first warm batch-20/depth-2 pair took 98.699 s per-event versus
32.260 s batched. This is one pair, not an accepted full-campaign conclusion.
Diagnostic rates
include CPU reference checks and synchronization and are not throughput results.
Estimated campaign
runtime is roughly 2–3 hours after allocation; file staging and all diagnostics
precede the main timing matrix. Results and pairs are saved incrementally.

This is run-387 JF calibration/integration, using a fixed validated run-51 radial
bin map as the matched performance workload. It is not run-387 q-space/beam
calibration and is not comparable as identical work to the staging-only scaling
campaign. Kernel/copy timing diagnostics are separate from throughput samples;
setup, lazy first-use work, loop time and complete sample wall time are recorded
with distinct definitions.

## Independent timing check and scheduled follow-up

Job **39381257** completed successfully on `sdfampere027` in 1m12s. The
isolated check uses the first real run-387 event, original constants, the same
bin map, and the exact user kernels. It repeats hot buffers, excludes I/O,
allocation and D2H from CUDA event intervals, and alternates batch 1/20 order
over 12 rounds. Batch-20 histograms exactly match the repeated batch-1 result.
There are 16,377,224 contributing pixels and all 64 sums are nonzero.

| Kernel | One event / launch | 20 events / launch | Batch-20 time per event |
| --- | ---: | ---: | ---: |
| Calibration | 4.142 ms | 7.114 ms | 0.356 ms |
| Integration | 2.567 ms | 5.237 ms | 0.262 ms |

These independent times agree with the pipeline diagnostic timers and support
their plausibility. They do not establish all causes of the batching benefit.
In particular, the integration grid grows from 64 blocks to 1,280 blocks on an
A100 with 108 SMs; the calibration kernel can also change constant-cache reuse
across events. These are mechanisms suggested by the code, not profiler-proven
attribution. All scheduled end-to-end pairs subsequently passed the acceptance gates below.

Scheduled CPU job **39381542** completed its review at **21:25 Pacific**. It
checked Stage 5c, requiring exit
0, all 16 diagnostics/38 pairs, matching outputs, and verified frozen source
and log hashes. It compares single-BD diagnostic kernel times with the isolated
check (0.65–1.65 ratio), rejects implausibly short loops and unexplained large
slowdowns, and preserves slower batched results without a speedup requirement.
Any failed gate stops the follow-up before scaling submission. These are
conservative automated consistency checks, not a replacement for a detailed
performance interpretation.

After acceptance, it submitted the **batched calibration + integration**
scaling campaigns: full JF on 1/2/4 GPUs with the previous 11 GPU/BD points,
and JF+feespec on one GPU with 1/2/4 BDs. Both sweep bulk OFF/ON and cold/warm
with two repetitions, batch 20, depth 1, and 10,000 events. Mixed runs retain
the prior per-event feespec int64 sum. User scratch reserves 80 MiB per event
plus 1 GiB per BD for tables/context/allocation slack; this reserve is outside
the explicit framework quota. High-sharing diagnostics use 4,000 events and
other diagnostics 1,000; every BD must participate. This differs in workload
from both the earlier staging-only and historical automatic-calibration reports.

The follow-up's status, review, report and submitted job IDs are saved under
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage5c-followup-20260928-r1/`
(`checks.json`, `review.json`, `plausibility.json`, `report.md`, `launches.json`).
The kernel check and the two prepared scaling campaigns have separate frozen
scratch directories. Acceptance and follow-up tests passed **40 host tests**,
including failure-to-launch prevention and duplicate-submission prevention.

### Scaling completion follow-up

Stage 5c completed at 21:24 Pacific with all 16 diagnostics and 76 timed samples
accepted. Its review completed successfully and submitted **39391686** (full JF)
and **39391687** (JF+feespec) at 21:25. Scheduled CPU job **39395582** uses
`afterany:39391686:39391687`, so it runs after both terminate, including failures.
It verifies scheduler exits, complete matrices, output hashes, diagnostic
launch/copy counts, source/log hashes and reported medians. It writes a summary
and evidence to
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage5c-scaling-report-20260928-r1/results/`.
Failed or incomplete campaigns are reported without claiming acceptance.
This follow-up makes no code changes, commits, pushes, or further test launches;
the next steps remain for discussion with the user.

## Accepted final measurements (September 29)

[Compact accepted evidence](user_kernel_stage5c_20260928.json) preserves the
paired measurements, scaling summaries, timing checks and source-artifact hashes.

All 16 diagnostics and 38 matched pairs passed; frozen sources and sample log hashes verified.
Kernel times passed the independent real-input hot-buffer consistency checks. These checks do not require batching to be faster.

| GPUs / BDs / batch / depth | Cache | Event-loop rate | Batched rate | Median paired loop change |
| --- | --- | ---: | ---: | ---: |
| 1 / 1 / 20 / 2 | warm | 101.30 | 355.28 | -71.07% |
| 1 / 1 / 5 / 2 | warm | 98.58 | 327.66 | -69.53% |
| 1 / 1 / 20 / 1 | warm | 101.74 | 303.31 | -66.04% |
| 1 / 2 / 20 / 2 | warm | 114.81 | 429.14 | -73.49% |
| 1 / 4 / 20 / 2 | warm | 124.61 | 446.35 | -72.27% |
| 2 / 2 / 20 / 2 | warm | 194.27 | 568.14 | -65.80% |
| 2 / 4 / 20 / 2 | warm | 220.56 | 694.89 | -68.31% |
| 4 / 4 / 20 / 2 | warm | 354.39 | 771.92 | -54.99% |
| 1 / 1 / 20 / 2 | cold | 81.70 | 194.13 | -57.92% |

Rates are events/s from median loop duration; startup is recorded separately in review.json.
Paired changes are medians of within-pair percentage changes, not ratios of separate medians.
The kernel sanity check repeats the first real event with hot buffers and excludes I/O, allocation and D2H.
It supports timer plausibility without identifying every cause of the batching benefit.


### Completed scaling

Final report job **39395582** passed after both scaling campaigns completed; no further performance jobs are pending.

Workload: Jungfrau calibration plus radial integration; mixed runs add the per-event feespec sum.
Rates are events/s from the median of two loop durations, excluding explicit initialization.
Batch 20, depth 1, 10,000 events, CPU-fallback KvikIO reads. Bulk refers to file-read grouping.
Historical staging-only and automatic-calibration rates measure different work; they are not matched regressions.

### JF: accepted

Job 39391686, sdfampere040, elapsed 03:49:27. 22 diagnostics and 88 timed samples passed.

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

### JF+feespec: accepted

Job 39391687, sdfampere014, elapsed 01:06:20. 6 diagnostics and 24 timed samples passed.

| GPUs | BDs | Cold bulk off | Cold bulk on | Warm bulk off | Warm bulk on |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 195.85 | 179.32 | 289.57 | 270.15 |
| 1 | 2 | 260.87 | 245.22 | 301.23 | 317.67 |
| 1 | 4 | 282.28 | 276.16 | 332.76 | 326.75 |

Cross-campaign Jungfrau output hashes: match.


Stage 6 lifecycle acceptance is recorded in [the closeout checklist](user_kernel_stage6_20260929.md). These timings use KvikIO CPU fallback and do not establish true-GDS throughput.
