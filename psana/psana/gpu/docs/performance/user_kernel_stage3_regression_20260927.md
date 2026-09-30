# User-kernel Stage 3 performance regression

Status: initial input-only regression failed cache conditioning before timing;
retry 39315680 cancelled at the user's request after 16m04s on sdfampere033.
The allocation, batch, and extern steps all report CANCELLED. Callback
characterization completed successfully for the original per-event design.
Stage 3 acceptance now requires the
[batched scheduling correction](../proposals/user_kernel_batched_scheduling_20260927.md).
Any partial retry results are incomplete evidence, not performance acceptance.

The input-only comparison is Stage 2 `f5b4cfb0e` against Stage 3 `c64fcb2ba`.
The later Stage 2 acceptance commits change benchmark/docs only. Runtime sources
remain frozen while these jobs execute. No production code changed to run them.

| Job | Purpose | Node | Estimate after start |
|---|---|---|---|
| 39311913 | Initial balanced regression | sdfampere026 | Failed, exit 1, 9m35s; eight preflights passed, zero timed samples |
| 39315680 | Same regression, excluding failed node | sdfampere033 | Cancelled at user request, 16m04s |
| 39312035 | Internal callback submission/retirement characterization | sdfampere020 | Completed, exit 0, 32 seconds |

## Input-only comparison

Frozen root:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage3-regression-20260927-r1`.
Retry root:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage3-regression-20260927-r2`.

One exclusive A100-node allocation runs four balanced rounds with warm cache at
1/2/4 BDs and cold cache at 4 BDs, bulk on. Warm 1 BD additionally interleaves
identical Stage 2 `control_a`/`control_b` aliases. Their resolved installation
paths and commits must match Stage 2 before staging. There are 40 timed samples
(32 A/B plus eight A/A), preceded by eight pixel/launch preflights. Both version
and comparison-pair ordering reverse on even rounds.

The workload retains the established run-387 contract: 10,000 events, batch 20,
depth 1, eight KvikIO workers per BD, 1 MiB tasks/bulk target, CPU-fallback I/O,
automatic budgets, and identical benchmark-only dense preparation with
`gpu_fn=None`. Each sample must deliver the independent expected timestamps and
335,571,760,000 useful bytes in 50,000 requests. Measured warm/cold cache
residency, pixel preflights, GPU assignment, allocation charges, manifest
verification and node-local staging cleanup remain enforced.

Primary rate is events divided by maximum rank loop time. Retain setup,
first-delivery and read-wait timings separately. Investigate a repeatable
slowdown above 5%; evaluate it alongside the A/A control and paired results.
The harness passed 35 standalone tests, including original matrix preservation
and the new 40-sample balanced coverage check. The 3,696-entry manifest passed
verification before submission. No native source rebuild was needed.

### Initial allocation failure and retry

Job 39311913 stopped at 20:01:55 Pacific on September 27, after all eight
pixel/launch preflights passed. Before the first timed sample, warm-cache
residency remained below the 99% requirement after three repair attempts. The
final failing stream-007 prefix had 12,656,578 of 12,801,093 pages resident
(98.8711%). Earlier repairs improved residency but caused other prefixes to
fall below the gate. This is a cache-conditioning failure, not an observed
Stage 3 throughput regression. No timed A/B or A/A result is available from r1.
The runner removed its private node-local stage on failure.

Job 39315680 retried the identical runtime snapshots, scripts, matrix and cache
thresholds, adding sdfampere026 to the excluded nodes. Previously successful
nodes sdfampere014/027 were occupied when the retry was prepared, so the scheduler
may choose another eligible node. All 3,743 inherited/new manifest entries were
verified before submission. The retry was cancelled to correct per-event
dispatch before further acceptance testing. Previous completion estimates no
longer apply. The successful callback measurement is preserved as a historical
reference; the corrected batch callback requires new measurements.

## Internal callback costs

Frozen root:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage3-callback-cost-20260927-r1`.
It pins the Stage 3 runtime from the main campaign and its own measurement script.

Use the real-device acceptance fixture's 900-pixel uint16 input, batch sizes
1/20, pool depths 1/2, and six alternating-order repetitions of 200 submissions.
Compare no task, empty callback, registered scratch plus one kernel, and scalar
publication plus one kernel. Separate preflights validate values, callback
counts, scalar publication metadata, and one gather without repeated parsing.
No diagnostic wrappers remain enabled during timing.

Compilation, parsing and initial preparation are outside timing. Fresh input
windows reference immutable parsed fixture storage, avoiding dependency growth
from indefinitely reusing one resident window. Timings distinguish submit,
retirement/drain and total loop cost; post-drain allocation values are not peaks.
This is a synthetic internal producer measurement with no disk I/O or automatic
publication D2H. It does not establish full Jungfrau callback throughput or
public result-delivery performance, which remains a Stage 4 measurement.

Callback job 39312035 completed all 16 separate correctness/launch preflights and
96 timed samples, followed by final manifest verification. The
[saved results](user_kernel_stage3_callback_cost_20260927.json) preserve the
per-sample measurements and provenance. Median host submission microseconds per
event were:

| Batch | Depth | No task | Empty callback | Scratch + kernel | Scalar publication + kernel |
|---|---|---:|---:|---:|---:|
| 1 | 1 | 169.23 | 194.21 | 226.09 | 225.91 |
| 1 | 2 | 167.10 | 191.66 | 223.46 | 223.54 |
| 20 | 1 | 23.54 | 29.84 | 48.13 | 49.51 |
| 20 | 2 | 23.34 | 29.55 | 48.84 | 56.85 |

For this synthetic workload, the empty callback adds roughly 25 microseconds per
event at batch 1 and 6.2–6.3 at batch 20 to host submission. This includes the
task context and ownership bookkeeping. Scratch/publication cases additionally
allocate user storage and launch one user kernel per event. These are feature
costs within Stage 3, not a Stage 2/3 throughput comparison. Preflights preserved
one gather per submission and zero repeated parser launches for the already
parsed fixture; the main regression checks the full read/parse/gather path.

At submission, home had 13 GiB free and shared scratch 81 GiB free. Generated
artifacts and compiler caches use scratch; the large staged input copy uses the
exclusive allocation's node-local storage. The estimates use previous matched
campaign durations and include staging/cache preparation, not queue delay.
