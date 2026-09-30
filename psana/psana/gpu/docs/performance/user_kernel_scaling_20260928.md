# User-kernel support: scaling and scheduling comparisons

**Status:** Scheduling results are measured Stage 4 evidence. New full JF and
partial JF+feespec scaling campaigns are running, with runtime `10df4c6e3`
(Stages 1–4 plus geometry and serial cleanup fixes). No new scaling rates are
claimed until the campaigns pass their acceptance gates.

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
`sdfampere035`, mixed on `sdfampere033`. All 22 + 6 preflights passed; timing
collection is in progress.

Each includes source identity/patch, hashes, launcher, native dependency links,
references and per-sample logs. Require exit 0, `CAMPAIGN_COMPLETE`, complete
provenance and final hash verification before accepting a campaign. The
maintained scaling harness passed 26 CPU tests before submission.

<!-- scaling-results-begin -->
## Scaling results awaiting completion

Dependent CPU job **39369900** verifies both campaign exit codes, complete matrices,
median calculations and every frozen hash before filling in the tables here.
Partial or failed campaigns are not accepted performance results. The reporting
script and log are frozen under
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-scaling-report-20260928-r1`.
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
