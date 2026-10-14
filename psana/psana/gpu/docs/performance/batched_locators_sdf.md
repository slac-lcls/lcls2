# Stream-grouped batched field location on B

The subsequent [matched A/B/B+ rerun](batched_locators_abo_sdf.md) measures all
three variants in one allocation: **21.850 / 33.561 / 28.743 seconds**, respectively,
from six clean repetitions each. Use that report for the direct three-way
comparison; this report preserves the original B/B+ allocation and device traces.

Implemented on `codex/psana2-gpu-batched-locators`, based on
`803a70011d18168200927e279cbeaca90568e13f`. This change deliberately stops
before gathering optimization and bulk-read integration. No merge, commit,
or publication is part of this work.

## Implementation

Configure handles are deduplicated with stable output indices and uploaded
once as stream ranges plus numeric Names/field/output records. Each parser slot
owns one `[handle, capacity, 11]` locator allocation. Per-handle views preserve
the existing locator API and window-local dgram indexing; capacity strides are
passed explicitly when a smaller tail reuses a larger allocation.

Each nonempty input with configured handles submits the existing walker,
`init_locators`, and `locate_fields`, followed by one shared locator-ready event.
Initialization and decoding are separate launches. One decode block per dgram
spans only the owning stream's configured handles and actual references. All
metadata stays on the device. Field-offset traversal and atomic duplicate/error
semantics are shared with the original lazy kernel, which remains available for
unregistered handles and retains its own completion event.

The two run-scoped tables are charged as fixed memory. Combined backing is
charged once, per-handle views are not counted again, and lazy storage remains
separately charged. Combined-buffer growth reserves the full new allocation
while the old one is live and rolls back that reservation on allocation failure.
Existing EventPool consumer retirement continues to govern slot reuse.

## Validation and measurement

Artifacts and reproduction commands:
`validation/batched-locators-20260917/README.md`.

The isolated worktree build and actual import paths were verified. The first
A100 run passed 187 focused CPU/GPU checks (six slow cases deselected), including
15 new batched-locator device cases, stream/Names isolation, missing fields,
scalar/array decoding, malformed/duplicate/corrupt/bounds/capacity statuses,
empty input, tails, growth/reuse, and cross-stream/lazy consumption. CPU tests
cover stream tables and allocation rollback. Existing GPU result lifetime and
pixel-level gather/calibration tests also pass.

The four longer `byhand_*` MPI tests passed. The initial main suite had 233
passes, 25 skips, eight deselections, and one pre-existing stdout-sensitive
assertion failure. The same failure was reproduced using frozen B. The small
`test_extract_subset_xtc2.py` correction from `599f856ae` validates the event
count in the subprocess instead of requiring calibration startup to be silent.
This is a test-only correction, independent of parser behavior. The final main
suite passes: **234 passed, 25 skipped, eight deselected**.

Seven timing-harness tests verify nested scopes and AST preservation for both
B and optimized implementations. Hardware throughput is not asserted in pytest.

## Warm comparison

Job **38504647**, `sdfampere004`, A100-SXM4-40GB,
`GPU-3e8528f8-fe59-785e-c2c4-c969ab13acda`, driver 575.57.08. CuPy 13.6.0,
CUDA runtime 12.9, KvikIO 24.08.02 in compatibility mode (CPU fallback, not GDS).
Both native builds use identical `debugoptimized` compiler options; installed
parser source checksums match their respective source trees.

Workload: `mfx101210926` run 387, 10,000 events, Jungfrau 32 segments across
files s005–s009; one SMD0, one EB, one BD, one GPU; batch size 20, pool depth 1,
8 GiB budget, eight KvikIO reader threads, 1 MiB task size, no user D2H.
Each measurement follows a separate 100-event warmup. Clean and instrumented
measurements are repeated three times with reversed variant/mode order on the
second repetition. Nsight captures are separate and excluded from throughput.

The first cache preparation needed three passes (90.4%, 97.1%, then 100%).
Cgroup accounting showed the 450 GiB host-memory limit was almost fully used,
mostly by file cache, despite ample host-wide available memory. No shared source
cache was purged. Every accepted clean sample was 100% resident before and after.

| Variant | Clean times, seconds (repetitions 1–3) | Median seconds | Median events/s | Events/s range |
| --- | --- | ---: | ---: | ---: |
| B | 37.380, 45.331, 40.238 | 40.238 | 248.5 | 220.6–267.5 |
| B + batched location | 38.508, 30.539, 35.452 | 35.452 | 282.1 | 259.7–327.4 |

The optimized median is **4.786 seconds shorter (11.9%)**, or **13.5% higher
throughput**. This is an observed comparison of three clean samples per variant,
not a precise causal speedup estimate: total times vary substantially, and the
instrumented samples show detector setup varying from about one second to
5.7 seconds in both variants. CPU/NUMA placement is not pinned. The host-scope
and kernel-count evidence should be assessed separately from these noisy totals.

All four real-data correctness preflights (B/B+ with and without timing hooks)
match the saved CPU raw/calibrated pixel reference for three events. Every clean
run has exactly 10,000 unique timestamps matching the stage manifest checksum.
Sampled peak device memory is 3,633–3,635 MiB for B and 3,633 MiB for B+; batch size,
pool depth, admission budget, raw reads, gathering, and calibration are unchanged.

## Comparison with the earlier A/B allocation

The Design GPU XTC Parser task's 21.768-second A and 33.451-second B results
come from job **38478311 on sdfampere003**, not this allocation. Both tables
report medians of three clean 10,000-event loops:

| Allocation | Node | A | B | B + batched location |
| --- | --- | ---: | ---: | ---: |
| 38478311 | sdfampere003 | 21.768 s | 33.451 s | not run |
| 38504647 | sdfampere004 | not run | 40.238 s | 35.452 s |

The unchanged B baseline takes **6.787 seconds longer (20.3%)** in the newer
allocation. Its clean sample range widens from **33.142–34.646 seconds** to
**37.380–45.331 seconds**. The newer optimized time therefore must be compared
with the newer B control; these jobs do not establish an A-versus-B+ speedup.

Verified unchanged settings: the stage manifests and calibration snapshot hash;
experiment/run, 10,000 timestamps, five input streams, and payload bytes; batch
20, depth 1, budget 8 GiB; three MPI ranks / one BD / one GPU; eight KvikIO
threads with 1 MiB tasks in CPU-fallback mode; 100-event warmup per measurement;
and the clean timer boundaries, including lazy detector setup and final GPU
synchronization. All clean samples in both jobs were 100% file-cache resident
before and after timing. Both scripts request 48 CPUs, 450 GiB host memory,
and one A100. Both use MPI `--bind-to none` without explicit NUMA placement.

B uses the **same frozen installation directory** in both jobs, at revision
`803a70011d18168200927e279cbeaca90568e13f`. All six B clean runtime records match
exactly, including logged module paths/hashes, CuPy 13.6.0, CUDA runtime 12.9,
and KvikIO 24.08.02. The benchmark worker differs only in its allowed variant
labels; its B execution and timing code are identical.

Observed environment/procedure differences are the node/GPU UUID, node-local
staging directory, and allocation time. Both GPUs are A100-SXM4-40GB with driver
575.57.08. The new script explicitly sets `CUPY_CACHE_DIR` to the validation
directory; the old script leaves it inherited/default and does not log its
resolved value. The new job also runs focused GPU tests before staging and
preflights. Both jobs still execute correctness preflights and per-sample
warmups before clean measurements. There is no evidence assigning the slowdown
to compilation or the cache-directory change. Variant ordering is reversed on
repetition 2 in both jobs, but B occupies the second variant position in the old
A/B sequence and the first in the new B/B+ sequence.

Separate CPU/NVTX B samples help locate the observed change:

| Median host scope | Earlier B | Newer B |
| --- | ---: | ---: |
| Read completion waiting | 15.596 s | 18.221 s |
| Whole slot submission (inclusive) | 14.283 s | 14.747 s |
| Eager field location | 5.179 s | 5.322 s |
| Field access/dependency/gather submission | 7.009 s | 7.283 s |

The newer B setup samples are 0.959, 5.660, and 1.014 seconds, versus 0.945,
1.000, and 0.939 seconds previously. These instrumented samples are separate
from clean throughput and cannot exactly decompose the 6.787-second clean
median difference. The separate B traces preserve identical read-copy counts
and bytes; summed H2D device time increases from 17.741 to 18.941 seconds while
gather/calibration device durations remain nearly unchanged. Those overlapping
device intervals are not additive elapsed-time components.

The evidence establishes a changed allocation and increased read/setup
variability, not a specific hardware cause. Historical CPU affinity, NUMA page
placement, contention, and clock measurements are insufficient to distinguish
those possibilities. A controlled A/B/B+ comparison within one allocation with
recorded CPU/NUMA placement is needed to compare all three implementations.
The cross-job checks and raw values are saved in
`validation/batched-locators-20260917/historical-comparison.json`.

## Host submission measurements

Median elapsed host scopes from three separate CPU/NVTX runs per variant,
covering 9,980 steady-state events / 499 subbatches. The first batch is excluded.
These are submission/waiting scopes, not pure CPU utilization or GPU execution.
Independent medians and nested scopes must not be added as an elapsed-time budget.

| Scope | B, seconds | B + batched location, seconds |
| --- | ---: | ---: |
| Eager field location | 5.322 | 0.593 |
| Whole slot submission (inclusive) | 14.747 | 10.031 |
| Field access/dependency/gather submission | 7.283 | 7.298 |
| Read submission | 0.881 | 0.896 |
| Read completion waiting | 18.221 | 19.230 |
| Dgram metadata preparation/upload | 0.153 | 0.147 |
| Batch construction / walker submission | 0.065 | 0.065 |
| Event views | 0.222 | 0.111 |
| Detector source selection | 0.485 | 0.493 |
| Calibration submission | 0.275 | 0.270 |
| Producer retirement waiting | 0.004 | 0.004 |

The target field-location scope drops **88.9% (4.729 seconds)**. Its per-run
ranges are 5.209–5.390 seconds for B and 0.552–0.618 seconds for B+. The
unchanged gathering path remains the largest host submission scope.

Instrumented median total times are 37.274 seconds (B) and 33.101 seconds (B+).
Startup detector-setup medians are 1.014 and 0.976 seconds, respectively, with
one approximately 5.7-second outlier in each variant. These samples are kept
separate from clean throughput; overlapping I/O and GPU work prevent treating
the scope reduction as an exactly additive end-to-end saving.

## Separate device traces

Both captures completed and exported with Nsight Systems 2025.3.1. The audit
checks the expected counts over all 10,000 events / 500 execution subbatches
(20 events each), independently
of the CPU timing samples above.

| Device operation | B count | B+ count | B GPU seconds | B+ GPU seconds |
| --- | ---: | ---: | ---: | ---: |
| XTC walker | 500 | 500 | 0.012904 | 0.012791 |
| Field decoding | 96,000 | 500 | 0.590496 | 0.007952 |
| Locator status fill / batched initialization | 96,000 | 500 | 0.172677 | 0.003983 |
| Locator-row memset | 96,000 | 0 | 0.185133 | 0 |
| Gather | 320,000 | 320,000 | 1.993051 | 1.995042 |
| Calibration | 10,000 | 10,000 | 1.832849 | 1.832915 |
| Missing-row cleanup | 10,000 | 10,000 | 0.510156 | 0.510267 |

The optimized trace has exactly one initialization and one decoding kernel per
parsed input, with **no lazy single-field launches**. Location plus initialization
uses approximately 0.012 seconds of GPU execution versus 0.948 seconds in B.
These are sums from separate traces, not additive contributions to the clean
host elapsed times. Gathering and calibration counts and device durations remain
essentially unchanged.

| CUDA operation | B calls | B+ calls |
| --- | ---: | ---: |
| Kernel launch | 532,500 | 341,500 |
| Memset | 116,000 | 20,000 |
| Event record / event creation (each) | 97,008 | 1,510 |
| Stream wait on event | 320,500 | 320,500 |
| KvikIO H2D read task | 370,000 | 370,000 |

There are two extra run-scoped H2D transfers totaling **4,656 bytes** for the
stream ranges and handle table. Neither trace has any D2H transfer. The unchanged
320,500 waits and 320,000 gathers are explicitly left for subsequent review.

### Calls per execution subbatch

Dividing the measured kernel/memset counts above by 500 makes the target
reduction explicit. This workload has **192 configured GPU-detector field
handles**. These are execution subbatches, not larger upstream EB batches.

| Operation | B per subbatch | B+ per subbatch |
| --- | ---: | ---: |
| Field-decoding kernel | 192 | 1 |
| Locator status-fill / initialization kernel | 192 | 1 |
| Locator-row memset | 192 | 0 |
| XTC walker kernel | 1 | 1 |
| Total parser kernels (walker + initialization + decoding) | 385 | 3 |
| Gather kernel (unchanged) | 640 | 640 |

`GpuXtcBatchPool.parse()` replaces the CPU loop calling `batch.locate(handle)`
192 times with one `batch._locate_configured(...)` call. That call submits
one initialization kernel and one decoding kernel, then records one shared
locator-ready event instead of one event per handle. The two kernels remain
separate to ensure initialization completes before decoding writes its results.

`build_field_location_tables()` groups handles by XTC input stream and builds
prefix ranges once. The run-scoped pool constructor uploads those ranges and
the handle table alongside the existing Configure tables. `parse()` reuses
the device tables. `locate_fields` launches one block per dgram; the block uses
`stream_handles[stream:stream+2]` to distribute only that stream's handles
across the dgram's actual ShapesData references. No unrelated stream's handles
are decoded and no device metadata is read back to schedule this work.

This achieves constant launch/event submission per nonempty configured input;
it does not remove all O(N) Python bookkeeping. `_locate_configured()` still
loops over handles to construct compatible `DeviceFieldLocators` views sharing
the same ready event. That loop does not parse fields or launch kernels.
Initialization also retains dense handle-by-dgram rows, including absent-stream
rows, to preserve the existing API. Lazy requests for unregistered handles keep
their separate path; no lazy single-field launches occurred in these traces.

Nsight reports possible incomplete CUDA/NVTX collection in both captures. The
expected principal operation counts match exactly, and the warning text is
preserved in each trace summary and `audit.json`. Do not interpret trace coverage
as a precise idle-time or utilization measurement, or use profiled throughput
in place of the clean controls.

## Completion and artifacts

Job **38504647 completed with exit 0**, elapsed **56m27s**. Accepted evidence is
six clean runs, six CPU/NVTX runs, two separate traces, and four successful
CPU-reference preflights. The final scripted audit verifies event identity/counts,
all warm-cache guards, unchanged installed parser hashes, and the expected kernel,
copy, and event counts. `git diff --check` passes.

All generated evidence is retained under
`validation/batched-locators-20260917/job-38504647-warm-ab/`:

- `results.json`, `summary.json`, `audit.json`: samples, medians, validated counts.
- `provenance.json`, `stage.json`, case logs and GPU CSVs: reproducibility evidence.
- `B/O-trace-cpu-nvtx-r0.nsys-rep` and matching `.sqlite` files: native captures.
- `B-trace-summary.json`, `O-trace-summary.json`: device durations, API counts,
  H2D volumes and collector diagnostics.

The validation directory additionally contains build provenance, main/MPI/GPU
suite logs, the isolated harness, and timing-transform tests. Generated artifacts
are ignored by its local `.gitignore`; scripts and this report remain reviewable.
The original handoff and frozen B source/install are unchanged.

## Subsequent review boundary

The bulk branch must later adapt its `allocation_requirements()` and trimming
to this single eager backing allocation plus separate lazy buffers. Its
`InputWindow` must retain the backing and scheduling tables through completion
and deduplicate waits for the shared ready event without dropping separate
lazy or consumer dependencies. Resident and transient windows retain independent
row numbering and raw-data bases. No bulk ownership, admission, resident-input,
read planning, gathering, or calibration algorithms are changed here.
