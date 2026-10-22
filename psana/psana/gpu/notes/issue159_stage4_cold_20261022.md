# Issue 159 Stage 4: cold A/B performance screen

Job **41604951** submitted; waiting for resources at submission. No measured
standard deviations are available yet. This is a cold-cache screen, using
private files staged once on the allocated node's `/lscratch`.

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

Report these four rows after all gates pass:

| Bulk | Variant | n | Loop time, mean +/- sample std (s) | Rate, mean +/- sample std (events/s) |
| --- | --- | ---: | --- | --- |
| Off | A | 4 planned | Pending | Pending |
| Off | B | 4 planned | Pending | Pending |
| On | A | 4 planned | Pending | Pending |
| On | B | 4 planned | Pending | Pending |

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
