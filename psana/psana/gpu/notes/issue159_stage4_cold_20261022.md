# Issue 159 Stage 4: cold A/B performance screen

Job **41604951 completed successfully** on **sdfampere026** in **43m10s**,
exit `0:0`. All 16 timed samples and four pixel preflights passed. This is a
cold-cache screen, using private files staged once on node-local `/lscratch`.
A and B showed no repeatable slowdown in this cold dense-gather workload;
this is not an equivalence claim or a measurement of field-access cost.

| Variant | Runtime revision | Meaning |
| --- | --- | --- |
| A | `f425e601e450a292cda440fe9b67820b5dc7b4eb` | Accepted Stage 3 |
| B | `ef0a453d46a02210616dccd6f73f8fffccd164cd` | Stage 4 aligned public field views |

Both frozen native installations are reused at their original paths. No runtime
code changes. Prior validation job 41494246 passed 399 CPU, 323 CUDA and 305
memcheck checks. The new screen repeats six-rank native provenance, input hashes
and four 200-event pixel preflights before timing.

## Sampling and cache procedure

One exclusive A100 node; four BDs share physical GPU 0. Jungfrau run 387, 32
panels, five input streams, dense raw gather with no calibration/user kernel.
Batch 20, depth 1, eight KvikIO workers, 1 MiB tasks/groups, CPU fallback and
6 GiB/BD budget remain identical across variants.

Four pairs per bulk-off/on condition, alternating AB/BA and condition order:
**16 fresh-process timed samples of 10,000 events**, or 160,000 timed events.
Each A/B/bulk set therefore has n=4 observations. Pixel diagnostics are excluded
from all performance statistics and labeled uncontrolled cache.

1. Stage/hash five private file prefixes once, with source changes checked.
2. Before each timed sample, run the guarded private-file reset: fsync and
   `POSIX_FADV_DONTNEED`, then verify every measured prefix is <=1% resident.
3. Time the event loop. Staging, reset, residency checks and explicit setup are
   outside that metric. Do not evict during the sample.
4. Record post-run residency without eviction. It may rise as files are read;
   there is no requirement that the files stay cold or become fully warm.
5. Repeat the reset independently for the next sample, including between A/B.

No shared-source eviction, global cache dropping, warm repair, or NUMA-bound
warmer is used. NIC/GPU/memory evidence and private-stage cleanup remain enabled.
A cold result does not establish unchanged warm performance; the dense no-field
workload also does not replace the completed field-access microbenchmarks.

## Reporting mean and standard deviation

Measured results; mean +/- sample standard deviation, n=4 per row:

| Bulk | Variant | n | Loop time, mean +/- sample std (s) | Rate, mean +/- sample std (events/s) |
| --- | --- | ---: | --- | --- |
| Off | A | 4 | 56.4147 +/- 0.0744 | 177.259 +/- 0.234 |
| Off | B | 4 | 56.3419 +/- 0.0526 | 177.488 +/- 0.166 |
| On | A | 4 | 56.3544 +/- 0.0422 | 177.449 +/- 0.133 |
| On | B | 4 | 56.3739 +/- 0.0910 | 177.387 +/- 0.286 |

Positive paired B/A loop-time change means Stage 4 is slower:

| Bulk | Median paired change | Pair range | Sample std of paired changes |
| --- | ---: | ---: | ---: |
| Off | -0.1430% | -0.2911% to +0.0609% | 0.1456 percentage points |
| On | -0.0124% | -0.0496% to +0.2131% | 0.1209 percentage points |

Signs were mixed: B was faster in three of four bulk-off pairs and two of four
bulk-on pairs. The largest absolute paired change was 0.2911%. The screen does
not show a consistent incremental Stage 4 regression; four pairs do not prove
zero overhead. Keep the separately measured field-access overhead in the
[Stage 4 report](issue159_stage4_20261021.md) visible, and retain final cumulative
comparison against the original frozen integrated A.

Sample standard deviation is `sqrt(sum((x - mean(x))**2) / (n - 1))` (ddof=1).
Calculate rate statistics from each run's measured rate, not by inverting the
mean loop time. Preserve individual observations and median paired B/A time
changes/range; also record the sample standard deviation of paired changes.
Do not pool A/B or bulk conditions. Partial reporting must use actual n and
leave standard deviation unavailable with fewer than two observations.

The frozen harness emits these values in `summary.json` under each condition's
`variant_statistics`, with explicit n/ddof and all samples. Analytical checks
verified ddof=1, A/B and bulk separation, exclusion of pixel diagnostics, and
rate calculations on known synthetic values. The 16-sample balanced schedule,
Python/shell syntax, unchanged helper contents and all 1,912 hashes passed
before submission.

Artifacts:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159/stage4-cold-20261022-r1/`.
Logs: `logs/screen-41604951.log`; results: `screen/job-41604951/`.
The job has a 90-minute allocation limit. Queue delay is additional.

[Submission evidence](evidence/issue159_stage4_cold_20261022.json).

## Completed gates and evidence

All 160,000 timed events passed identity/count checks. Both variants passed
six-rank native provenance, four 200-event pixel checks passed in total, staged
input hashes matched, and all 1,912 frozen artifact hashes verified before and
after the screen. The private stage was removed successfully.

The largest pre-sample per-file residency was **0.00028643%**, well below
1%. Post-run residency was observed without repair; the lowest per-file value
was 80.33130%, which is permitted for a cold run. Peak tracked owned/held
allocation was **1.2518 GiB per BD**, below the 6 GiB limit.

The evidence JSON preserves individual rates/loop times, all paired changes,
per-file pre/post residency, per-BD read/budget counters, setup/preparation times,
provenance and result-log hashes. Means/stds were independently recomputed from
individual timed rows and matched the frozen harness summary. No pixel
preflight or cache-preparation time enters the reported loop statistics.

## Historical absolute throughput remains unresolved

The near-zero Stage 4/Stage 3 difference does not explain the older ~350 Hz
cold baseline versus ~177 Hz here. The [historical audit](issue159_cold_rate_history_20261022.md)
traces ~390 Hz on October 16, ~300 Hz October 17 and ~261 Hz October 19, with
unchanged runtime source across the first two drops. The latest additional
loss appears in read-future wait time. A matched old/current comparison is
needed before attributing the absolute gap to a code revision.
