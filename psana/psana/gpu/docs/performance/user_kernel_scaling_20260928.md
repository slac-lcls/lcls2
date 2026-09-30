# User-kernel support: scaling and scheduling comparisons

**Status:** Scheduling results are measured Stage 4 evidence. New full JF and
partial JF+feespec scaling campaigns completed, with runtime `10df4c6e3`
(Stages 1–4 plus geometry and serial cleanup fixes). Both campaigns passed the acceptance gates; their results and historical
comparisons are below.

## Full JF and partial JF+feespec scaling

The existing September 26 [JF baseline](jungfrau_current_scaling.md) and
[mixed-detector baseline](jf_feespec_single_gpu_scaling.md) use pre-user-kernel
runtime `ad8d454d1`. They include automatic JF calibration. The August
[full matrix](jungfrau_single_node_sdf.md) uses `e18cf6bb7` and also calibrates.
Stage 4's input regression covered one GPU with 1/2/4 BDs, not the full matrix.

The new campaigns measure **JF staging without calibration or GpuTask**:
read, parse, and dense raw gathering. A benchmark-only input adapter requests
that gather without introducing an empty callback. Mixed runs additionally retain
the old per-event feespec `raw.hproj` GPU int64 sum and post-loop validation.
Historical versus new rates therefore describe different workloads and revisions;
their ratio must not be called a matched regression result or a batching speedup.

| Campaign | Allocation / topology | Preflights | Timed samples | Job |
|---|---|---:|---:|---|
| Full JF | 1 GPU: 1/2/4/6/8 BDs; 2 GPUs: 2/4/6 BDs; 4 GPUs: 4/8/12 BDs | 22 | 88 | 39369005 |
| JF+feespec | 1 GPU: 1/2/4 BDs | 6 | 24 | 39369006 |

Both use run 387, 10,000 events, batch 20, depth 1, eight KvikIO workers/BD,
1 MiB tasks and bulk targets, one EB, automatic per-GPU peer budgets, CPU
fallback rather than GDS, and private node-local staged prefixes. Cold/warm
residency gates and bulk-read off/on match the prior reports. Two fresh-process
repetitions reverse topology/cache/read-mode order. Explicit DataSource/run
setup is recorded separately; rates use the event loop, including lazy work
inside it. Cache preparation, staging, imports and MPI launch are excluded.

Preflights check three JF raw arrays against the original CPU reference; mixed
preflights also check all 200 feespec arrays. Every timed mixed sample validates
10,000 timestamp-associated sums. Exact timestamp hashes, payload/read counts,
GPU placement, memory budgets and drained resources gate acceptance. No
calibrated-pixel check is claimed for the staging-only workload.

Frozen artifacts under `/sdf/scratch/users/m/monarin/gpu-validation/`:

- `jf-user-kernel-full-scale-20260928-r1` (estimated 3–4 hours after allocation).
- `jf-feespec-user-kernel-scale-20260928-r1` (estimated 60–90 minutes).

Both jobs started on September 28 around 14:46 Pacific: full JF on
`sdfampere035`, mixed on `sdfampere033`. All 22 + 6 preflights and 88 + 24 timed samples passed. See the completed
comparison below for scheduler status and artifact validation.

Each includes source identity/patch, hashes, launcher, native dependency links,
references and per-sample logs. Require exit 0, `CAMPAIGN_COMPLETE`, complete
provenance and final hash verification before accepting a campaign. The
maintained scaling harness passed 26 CPU tests before submission.

<!-- scaling-results-begin -->
## Completed scaling results

Rates below are 10,000 divided by the median of two event-loop durations. Explicit
initialization is separate. All historical/current comparisons retain the workload
difference: previous JF calibration versus current staging only. Rate differences
are descriptive, not matched regression or kernel-scheduling speedups.

### Full JF matrix

Job **39369005** completed on **sdfampere035** in **03:07:01**, exit 0. All **22 preflights**, **88 timed samples**, and **1289 frozen file hashes** passed.

| GPUs | BDs | Cold off | Cold on | Warm off | Warm on |
|---|---|---|---|---|---|
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

Best measured cold median: **360.10 events/s** (**12.08 GB/s**) at 4 GPU(s)/12 BDs, bulk off.

Best measured warm median: **1007.14 events/s** (**33.80 GB/s**) at 4 GPU(s)/4 BDs, bulk off.

Comparison with the September 26 calibration-enabled baseline (`ad8d454d1`):

| GPUs/BDs | Cache | Bulk read | Previous ev/s | Current ev/s | Rate difference |
|---|---|---|---|---|---|
| 1/1 | cold | off | 200.70 | 221.58 | +10.4% |
| 1/1 | cold | on | 192.06 | 211.13 | +9.9% |
| 1/1 | warm | off | 306.46 | 437.29 | +42.7% |
| 1/1 | warm | on | 323.64 | 382.33 | +18.1% |
| 1/4 | cold | off | 302.16 | 341.64 | +13.1% |
| 1/4 | cold | on | 289.00 | 351.64 | +21.7% |
| 1/4 | warm | off | 475.87 | 484.13 | +1.7% |
| 1/4 | warm | on | 498.42 | 489.12 | -1.9% |
| 2/4 | cold | off | 298.18 | 354.06 | +18.7% |
| 2/4 | cold | on | 303.30 | 353.18 | +16.4% |
| 2/4 | warm | off | 618.49 | 689.48 | +11.5% |
| 2/4 | warm | on | 592.33 | 658.26 | +11.1% |
| 4/8 | cold | off | 303.00 | 357.54 | +18.0% |
| 4/8 | cold | on | 301.75 | 354.89 | +17.6% |
| 4/8 | warm | off | 680.58 | 918.04 | +34.9% |
| 4/8 | warm | on | 750.93 | 848.15 | +12.9% |

The [August full matrix](jungfrau_single_node_sdf.md) (`e18cf6bb7`) also
calibrated. This table pairs its reported rates with current bulk-off medians;
it is historical context across different revisions, nodes and workloads.

| GPUs | BDs | August cold | Current cold off | August warm | Current warm off |
|---|---|---|---|---|---|
| 1 | 1 | 174.5 | 221.58 | 339.2 | 437.29 |
| 1 | 2 | 247.5 | 314.45 | 417.3 | 494.16 |
| 1 | 4 | 274.1 | 341.64 | 408.9 | 484.13 |
| 1 | 6 | 272.4 | 355.81 | 445.3 | 456.77 |
| 1 | 8 | 271.6 | 357.91 | 403.1 | 432.26 |
| 2 | 2 | 263.7 | 320.36 | 549.8 | 729.66 |
| 2 | 4 | 276.7 | 354.06 | 635.0 | 689.48 |
| 2 | 6 | 274.2 | 352.20 | 614.8 | 715.47 |
| 4 | 4 | 268.5 | 353.86 | 683.1 | 1007.14 |
| 4 | 8 | 269.5 | 357.54 | 779.4 | 918.04 |
| 4 | 12 | 264.8 | 360.10 | 693.3 | 914.20 |

### Partial JF+feespec matrix

Job **39369006** completed on **sdfampere033** in **00:57:03**, exit 0. All **6 preflights**, **24 timed samples**, and **1289 frozen file hashes** passed.

| GPUs | BDs | Cold off | Cold on | Warm off | Warm on |
|---|---|---|---|---|---|
| 1 | 1 | 205.93 | 188.87 | 377.41 | 336.26 |
| 1 | 2 | 282.74 | 263.74 | 447.31 | 456.40 |
| 1 | 4 | 305.36 | 305.60 | 500.29 | 489.88 |

Best measured cold median: **305.60 events/s** (**10.26 GB/s**) at 1 GPU(s)/4 BDs, bulk on.

Best measured warm median: **500.29 events/s** (**16.79 GB/s**) at 1 GPU(s)/4 BDs, bulk off.

Comparison with the September 26 calibration-enabled baseline (`ad8d454d1`):

| GPUs/BDs | Cache | Bulk read | Previous ev/s | Current ev/s | Rate difference |
|---|---|---|---|---|---|
| 1/1 | cold | off | 198.80 | 205.93 | +3.6% |
| 1/1 | cold | on | 187.93 | 188.87 | +0.5% |
| 1/1 | warm | off | 315.89 | 377.41 | +19.5% |
| 1/1 | warm | on | 291.23 | 336.26 | +15.5% |
| 1/2 | cold | off | 281.84 | 282.74 | +0.3% |
| 1/2 | cold | on | 272.73 | 263.74 | -3.3% |
| 1/2 | warm | off | 342.18 | 447.31 | +30.7% |
| 1/2 | warm | on | 402.51 | 456.40 | +13.4% |
| 1/4 | cold | off | 305.87 | 305.36 | -0.2% |
| 1/4 | cold | on | 313.06 | 305.60 | -2.4% |
| 1/4 | warm | off | 408.40 | 500.29 | +22.5% |
| 1/4 | warm | on | 416.99 | 489.88 | +17.5% |

[Compact evidence](user_kernel_scaling_20260928.json) preserves every sample,
setup/loop/read-wait measurements, cache checks, GPU mapping, resource accounting,
scheduler completion and artifact hashes. Full logs and timestamp arrays remain
in the frozen scratch campaigns.

<!-- scaling-results-end -->

## Kernel scheduling: batch off versus batch on

Here **off** means equivalent user work scheduled once per event in the public
loop, and **on** means a `GpuTask` scheduling the work once per selected execution
subbatch. It does not mean `gpu_fn=None`, which omits that work. It also does not
refer to `gpu_bulk_read`, the independent file-read grouping setting above.
These are compared execution paths, not a new production on/off flag.

The existing completed Stage 4 comparisons already measure this distinction.
They predate the two cleanup fixes; they are not new measurements of `10df4c6e3`.
The [full findings](../user_kernel_stage4_findings_20260928.md) retain methods,
all unfavorable cases, exact frozen revisions and validation history.

| Same-work comparison | Batch size / execution depth | Median paired loop-time change with batching |
|---|---|---:|
| Small input, scalar output | 20 / 1 | −49.7% |
| Small input, scalar output | 20 / 2 | −48.0% |
| Full JF-sized input, scalar output | 20 / 1 | −20.9% |
| Full JF-sized input, scalar output | 20 / 2 | −25.3% |
| Real run-387 I/O, dense input, scalar output | 20 / 2 | −7.45%; −2.55 s per 10,000 events |

The fixture rows use six balanced rounds and actual public event/result delivery,
but immutable GPU inputs exclude DataSource setup and file I/O. They isolate the
scheduling/delivery benefit more closely than a storage-bound throughput run.
Matched preallocated scalar cases still improve about 44–46% for small inputs
and 18–23% for full-frame inputs, so allocation amortization alone does not
explain the gain. [Fixture evidence](user_kernel_stage4_public_20260928.json),
[plot](user_kernel_stage4_public_20260928.svg).

The real-I/O comparison computes `raw.flat[300] + 1` and returns the same
independent NumPy scalar for every event. Six pairs passed identical ordered
timestamp/value checksums. Batching was faster in five pairs, but paired loop
changes span −27.63% to +25.70%. In the slower pair, an extra 8.50 seconds of
KvikIO read waits accounts for nearly all the 8.63-second increase. The median
benefit is useful evidence, not a precise universal speedup. Separate loop
medians are 34.60 s off and 32.55 s on; their difference is distinct from the
median paired saving of 2.55 s. Setup medians are 1.86 / 1.78 s and are excluded
from those loop rates. [Real-I/O evidence](user_kernel_stage4_datasource_20260928.json).

### Evidence that scheduling and delivery are batched

For a diagnostic 60-event fixture processed as three subbatches:

| Count | Per-event reference | Batched pipeline |
|---|---:|---:|
| User allocations | 60 | 3 |
| User kernel submissions | 60 | 3 |
| Output copy groups | 60 | 3 |
| CUDA event creations | 63 | 6 |

The candidate invokes three callbacks; the reference launches kernels inline
rather than invoking a framework callback. Real-DataSource diagnostic runs
observed ten callbacks of 20 for 200 events. The benchmark's copy count tracks
publication groups, not a native CUDA trace. Separate A100 tests wrapped actual
copy calls and verified one payload copy per nonempty group and one terminal
copy event per execution, including mixed, sparse and empty outputs. No Stage 4
Nsight copy trace was collected. CPU row lookup/materialization remains per event.

### Limits and output policy

Batching has a fixed publication cost: the small-input batch-one case was
10–13% slower. Full-frame output can also lose when its group exceeds the default
64 MiB pinned cap. With ordinary host memory, batch-20 delivery was about 23–24%
slower than a per-event reference reusing one host destination; when both paths
allocated fresh host destinations, batching instead improved 1.8–2.2%.

With a 1.5 GiB pinned cap, full-image batch-20 delivery changed by +5.6% at depth
one and −15.1% at depth two. It used 640/1280 MiB pinned capacity, versus the
reference's 32 MiB destination. This is an explicit memory/overlap tradeoff,
not the default memory footprint. [Output-policy evidence](user_kernel_stage4_image_20260928.json).
