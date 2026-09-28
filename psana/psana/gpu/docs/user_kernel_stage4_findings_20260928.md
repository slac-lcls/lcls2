# Stage 4: named host delivery

Implemented and reviewed after Stage 3c commit `6291d21ed` passed scheduling and
performance gates. Implementation and correctness review are complete, and all performance
collection and the focused recovery run have finished. Stage 4 remains
uncommitted. Performance is workload-dependent; limitations are recorded below.

## Behavior and ownership

`gpu_fn=GpuTask(...)` now runs through the public serial/MPI event path. Each
nonempty selected execution subbatch invokes the task once. `publish(name,
array, event_indices=None)` groups contiguous rows with a frozen shape, dtype,
byte extent and event mapping. After successful producer completion recording,
`GpuEventManager` queues each nonempty publication group once on a copy stream.
One terminal event covers the execution's copies. Empty arrays preserve metadata
and producer ordering without a payload copy. No publications means no output
staging, copy stream or copy event.

`evt.gpu.get(name)` resolves the exact published name, independent of the number
of GPU detectors. `.on_cpu` waits for its terminal host token and caches an
independent NumPy row, including scalar/empty rows. Device accessors reject task
outputs. Parsed detector input fields remain accessible during normal event
delivery; host-only outputs do not mark the entire event's input state released.
Sparse/reordered rows, multiple names, and disjoint groups with different layouts
under the same name retain their own metadata.

`gpu_d2h_pinned_bytes` is a finite per-BD output-staging cap, **64 MiB by default**.
It counts full page-rounded allocations, including free cached and token-held
blocks, across all names. Direct CuPy pinned allocations bypass its global pinned
pool. The first-fit cache does not evict during a run. If no block fits or no
capacity remains, the whole group copies synchronously into ordinary NumPy
storage. Zero explicitly selects that fallback. Fragmentation can reduce overlap;
there is no wait for a host slot whose release requires delivery of this event.
The cap bounds output staging, not user-retained ordinary CPU results, input
metadata staging, or arbitrary user device allocations. Multiple BDs each have
their own configured cap.

The copy guard is registered on the producer lease **before any copy command**.
It retains every destination, including ordinary fallback buffers, until terminal
completion is proven. If event recording fails, it drains the copy stream. If
that drain fails, occupied EventPool quarantine retains device/input owners and
copy destinations for a safe later retry. The MPI submission path explicitly
drains failures too. Array metadata is revalidated before any transfer, so shape
or dtype mutation after publication fails before copying.

Host tokens contain only destination storage, row metadata and completion state;
they never retain publication records or device arrays. Pinned blocks become
reusable only when transfer completion and all row-token releases are proven.
Run close materializes any still-retained rows into ordinary NumPy storage,
detaches pinned aliases, and releases cached capacity. A group lock serializes
host materialization with close/reuse. Stage 4 allocates no user GPU buffer and
does not charge user results to psana's device budget.

## Review and correctness

Reviewed the successful path, asynchronous destination ownership, source/input
lease ordering, metadata mutation, partial-copy/record/drain failure, close retry,
ignored/retained results, scalar row handling, and CPU-only service imports.
No blocking issue remains in that review.

Frozen source base `6291d21ed` plus patch and manifests:
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage4-20260928-r1`.
Native dependencies are inherited from the accepted Stage 1b installation.
Runtime and test files were checked against the frozen hashes after validation.

- Local unit suite: **466 passed**.
- CPU retry **39323696**: **533 passed**, 135 skipped, seven deselected; byhand
  **four passed**. Full retry completed in 3m51s.
- A100 **39323503**, sdfampere017: **135 passed**, 3m22s allocation time.
- MPI **39323504**, sdfampere033: **6m49s**, exit 0. Existing input/setup checks
  passed. New public-task checks passed for exclusive and hybrid modes: exactly
  13 events, callback batch sizes **5, 5, 3**, exact timestamps/count/pixel/empty
  results, retained host access, and CUDA only on BD ranks.
- Scheduling matrix harness: **four passed** in an isolated dependency layout.

Initial CPU job **39323502** had 532 passes and one shared-memory smalldata
fixture failure: the test server exited before the client attached, leaving
`/oneint` absent. Its byhand suite passed. The full unchanged suite passed on
retry, including that test; logs are retained. Initial local harness invocations
lacked the native package / relative helper layout; the complete isolated helper
layout passed without changing or skipping tests.

New coverage includes actual pinned and ordinary-host copies; one copy per group
and one output completion event; mixed/sparse/scalar/empty layouts; cap pressure,
ignored/retained rows; close materialization; delayed D2H before preallocated
output reuse; copy/record/drain failure and retry; device-owner weak references;
public serial delivery and live parsed-input access. The former Stage 2 setup
test now remains setup-only, while actual callback processing is covered by the
new public-delivery tests.

## Performance campaigns

Input-only job **39323870** used the established 40-sample / eight-preflight
JF run-387 warm/cold protocol, including A/A controls. All A/B pairs completed;
a final baseline-control failure and the successful focused recovery are detailed
below. The Stage 4 matrix labels are explicit in the harness. Root:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage4-regression-20260928-r1`.

Public-loop smoke **39323772** passed in 4m15s on sdfampere001: 144 counted
preflights and 144 one-submission samples, with both runtimes in reversed order.
Those single-submission timings are not accepted performance evidence. Full
six-round performance job **39323933** completed on sdfampere001 in 26m05s.
Both source manifests verified before and after. Root:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage4-public-comparison-20260928-r1`.
The reference uses benchmark-only dense-input injection into `GPUResult`, then
actual `Run.events()`, `on_gpu_view()`, per-event user launches and synchronous
host copies. A real SlotLease joins the view completions before input reuse.
The candidate uses batched `GpuTask`, automatic output D2H, and actual public
`.on_cpu`. Both return independent NumPy results. The fixture source bypasses
DataSource/I/O/transition setup; these timings are public API scheduling costs,
not end-to-end application throughput. Real serial/MPI tests separately cover
DataSource execution.

Matrix: 900-pixel and synthetic `(32,512,1024)` uint16 inputs, N=1/3/20,
depths 1/2, compact scalar outputs, preallocated compact outputs, and full images.
Compact copies use 64 MiB pinned capacity; image copies use zero on both sides
so the fallback policies match. The preallocated reference reuses one device
output after blocking host completion; the candidate uses one buffer per execution
slot. Neither preallocation nor compilation/parse/input upload is timed.
Separate counted/numerical preflights precede immediate warmups; GC resets
outside timing and remains enabled inside the loop. Pool counters after drain
are not peak-device-memory measurements. Counts and copy policy are preserved
alongside the timings.

For N=20, the smoke measured one allocation, launch and output group copy per
subbatch (zero allocations when preallocated), versus 20 launches/copies in the
public event-loop reference. CUDA event creations are two per subbatch on the
batch path (producer and copy completion), versus 21 for the reference (one
producer plus 20 public input-view completions). Both perform one gather.
Reference staging is one 4 KiB pinned buffer for compact output, zero for image
fallback; candidate staging is tracked by its aggregate allocator. The smoke's
`user.output_pinned_peak` counter is candidate-only; use the timed sample's
`output_pinned_after_loop` field for reference capacity, not that zero counter.

The first public-loop pairs showed compact-output batching gains and a slower
full-image fallback versus the reused-host reference. A controlled follow-up
keeps that original case alongside `image_fresh_host` (reference host destination
allocated per event) and `image_pinned` (1.5 GiB cap on both paths, enough for two
640 MiB groups). This separates allocation policy and available staging capacity
from scheduling. Runtime behavior is unchanged; the 64 MiB product default is
unchanged. Fresh-host/pinned smoke **39324364** passed on sdfampere019 in 4m47s
(32 counted checks and 32 one-submission samples, not performance acceptance).

Follow-up **39324659**, sdfampere014, uses six balanced rounds of all three image
policies on the same node, sizes 1/20, depths 1/2, 100 measured submissions per
sample. It started only after smoke success. Root:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage4-image-policies-20260928-r2`.
The follow-up harness also records CPU affinity and correctly initializes the
reference pinned-capacity counter. Its frozen sources remain distinct from the
original campaign. Final results are reported below.

## Completed public-loop comparison

Job **39323933** completed in **26m05s**, exit 0, with both source-manifest checks
passing. All 432 counted/numerical preflights and 432 timed samples passed.
The [compact evidence](performance/user_kernel_stage4_public_20260928.json)
preserves every timed row plus paired summaries and hashes; the
[scaling plot](performance/user_kernel_stage4_public_20260928.svg) shows median
paired loop changes and the six-round ranges (not confidence intervals).

| Case | Depth 1 | Depth 2 |
|---|---:|---:|
| Micro, N=1, scalar output | +10.2% | +11.2% |
| Micro, N=3, scalar output | -16.9% | -17.1% |
| Micro, N=20, scalar output | -49.7% | -48.0% |
| Full-frame input, N=20, scalar output | -20.9% | -25.3% |
| Full-frame image output, N=20, ordinary fallback | +25.6% | +26.4% |

Positive means slower than the per-event public-loop reference. Scalar output
at N=20 costs approximately **57–61 us/event** for the micro fixture and
**218–233 us/event** for full-frame input. Matched preallocated scalar cases
still improve **44–46% / 18–23%**, so gains extend beyond device allocation
amortization. Full-frame scalar N=1 cases improve **6–22%** depending on depth
and allocation policy, while micro N=1 cases are consistently **10–13% slower**.
There is no batching benefit to amortize publication/host-token overhead at N=1.

The ordinary-host image comparison is against a reference that reuses its host
destination. Its slower batched fallback is a real limitation under that policy,
not an input-only regression or a failed numerical test. The controlled
fresh-host/pinned follow-up below attributes much of this cost to allocation
policy and measures the capacity/overlap tradeoff. These results do not justify a blanket claim that
all Stage 4 task-enabled workloads are faster.

## Real DataSource compact-output comparison

A separate warm-cache, one-BD, batch20/depth2 comparison includes real run-387
file I/O and the complete public event loop. Both paths prepare the same dense
input, execute `raw.flat[300]+1`, and deliver an independent NumPy scalar per
event. The reference injects a borrowed dense GPUResult for per-event public
view/kernel/blocking-copy calls; the candidate declares GpuTask and reads its
named host result. Compilation is outside setup/loop timing on both paths.
DataSource/run setup is reported separately; process launch/imports and cache
preparation are outside loop timing. The ordered timestamp/value checksum must match
across variants for all 10,000 events; diagnostic processes also validate raw
pixels and callback batch sizes.

Numerical smoke **39327487** passed in 42 seconds on sdfampere040: both
200-event output checksums matched, three raw pixel checks passed per path,
and callback sizes were ten groups of 20. Six balanced rounds **39327544**
completed on the same node after this success. Root:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage4-datasource-public-20260928-r1`.
Allocation duration: 36m28s.

## Completed image-policy comparison

Job **39324659** completed in **44m00s**, exit 0; manifests verified before/after.
All 144 counted/numerical preflights and 144 timed samples passed.
[Evidence](performance/user_kernel_stage4_image_20260928.json) includes all
paired measurements and the exact frozen protocol. These are full-image fixture
results from one allocation, with both protocols evaluated under each policy.

| Batch20 output policy | Depth 1 | Depth 2 |
|---|---:|---:|
| Ordinary host, reference reuses destination | +23.8% | +23.4% |
| Ordinary host, both allocate fresh destination | -1.8% | -2.2% |
| Pinned, 1.5 GiB cap on both paths | +5.6% | -15.1% |

Positive means slower. Most of the ordinary fallback gap relative to the
reused-host reference disappears when allocation policy is matched. Sufficient
pinned staging plus depth2 allows overlap; additional staging alone at depth1
does not guarantee improvement. The candidate pins 640 MiB at depth1 /
1280 MiB at depth2 for batch20; the reference reuses one 32 MiB destination.
These configured capacities differ from the product's unchanged **64 MiB**
default, under which a batch20 full-image group uses synchronous fallback.

At batch1, ordinary fallback versus fresh allocation is still slower by 4.1% /
11.4% (depth1/2); pinned results are -0.9% / -15.3%. Together with the micro
batch1 overhead, this remains a workload-dependent performance limitation.
No runtime change was made merely to favor one benchmark allocation policy.

## Completed real DataSource comparison

Job **39327544**, sdfampere040, completed in **36m28s**, exit 0. Source
manifests passed before/after; two pixel preflights and all 12 timed samples
passed, with identical ordered timestamp/output checksums for every 10,000-event
run. [Evidence](performance/user_kernel_stage4_datasource_20260928.json) preserves
all samples, resource/cache records, paired setup/loop/read-wait statistics, and
source hashes.

The median paired event-loop change is **-7.45%**, or **-2.55 seconds per 10,000
events**. Batching is faster in five of six pairs. Separate loop medians are
34.60 seconds for per-event scheduling and 32.55 seconds for the batched path;
the difference between medians is not the median paired difference. Median
DataSource/run setup is 1.86 / 1.78 seconds, with a paired median difference of
-0.043 seconds. These are not total process/job runtimes.

| Round | Per-event loop (s) | Batched loop (s) | Paired change |
|---|---:|---:|---:|
| 1 | 34.98 | 33.78 | -3.42% |
| 2 | 33.57 | 42.19 | +25.70% |
| 3 | 34.22 | 26.99 | -21.14% |
| 4 | 34.03 | 30.12 | -11.48% |
| 5 | 43.27 | 31.31 | -27.63% |
| 6 | 37.35 | 37.10 | -0.66% |

The wide range limits the precision of an end-to-end speedup claim. In round2,
the candidate spends 8.50 additional seconds in KvikIO read waits, accounting
for nearly all of its 8.63-second loop increase. A later reference sample also
slows substantially. These counters identify the immediate waiting cost; they
do not prove whether its underlying cause is environment variability or an
overlap effect. No samples were discarded. The fixture scheduling gains are
more consistent than the full-file throughput gains.

## Input campaign failure and focused recovery

Original job **39323870** failed after **1h25m41s**, with **38/40 timed samples**
and all **eight preflights** completed. All 32 Stage3c/Stage4 samples (four
balanced rounds of warm1/2/4BD and cold4BD) completed. The failed next sample
was the unchanged Stage3c `control_b` at warm1BD round4: KvikIO completion raised
`map::at` in its first batch. Three A/A pairs completed; the fourth pair did not.
Those three control pairs are not order-balanced and must be labeled as such.
All logs and samples are retained. The complete 6,212-entry source manifest was
manually reverified after failure; the job's final automatic verification was
not reached.

Focused recovery **39329321** completed on the same sdfampere023 node in
**21m46s**, exit 0, with identical frozen runtimes and cache controls. All four
preflights and eight timed samples passed, with manifests verified before/after.
It covers two balanced rounds of warm1BD A/A and Stage3c/Stage4 A/B. The failed
original campaign remains intact. No runtime fix was applied to this baseline-only
failure, whose underlying cause has not been established.

## Final input-only results and review conclusion

[Input evidence](performance/user_kernel_stage4_input_20260928.json) retains the
original campaign and focused recovery separately. Full per-event timestamps
remain in the hashed raw job results; the compact evidence records their counts
and ordered checksums alongside all timing/resource rows.

| Four-round input A/B case | Median paired loop change | Median paired loop delta | Median paired setup delta |
|---|---:|---:|---:|
| Warm, 1 BD | +0.25% | +0.058 s | -0.087 s |
| Warm, 2 BD | +0.78% | +0.168 s | +0.069 s |
| Warm, 4 BD | +2.12% | +0.471 s | +0.330 s |
| Cold, 4 BD | -1.70% | -0.545 s | +0.047 s |

Each loop processes 10,000 events; setup means DataSource/run setup after
process imports. Original three A/A pairs have median +0.64%, but lack the
fourth reverse-order pair. The balanced two-round recovery has A/A median
**+7.57%** and A/B median **+5.69%** (individual A/B pairs +10.95% and +0.43%).
These controls show substantial identical-code variability. The small original
A/B medians do not establish a repeatable Stage4-specific input slowdown; the
measurement precision cannot exclude small changes. Do not subtract the A/A
median from A/B to claim a corrected speedup.

The original warm4BD round4 candidate is +19.15% slower: +4.03 seconds of loop
time, with +3.86 seconds of maximum-BD read waits. Other warm4BD pairs are
+1.27%, +2.60%, +1.63%. This outlier is retained, as are the cold-cache and
real-file output variations.

No correctness or ownership blocker remains in the Stage4 review. It fulfills
exact named CPU delivery, bounded staging, terminal owner retention, and public
serial/MPI integration. Callback/allocation/kernel scheduling remains once per
selected execution subbatch; D2H is once per contiguous publication group, with
one terminal output completion event per execution.

Performance collection supports batched compact-output scheduling, with the
strongest consistent gains in isolated scheduling measurements. Micro batch1
overhead, ordinary-host full-image allocation costs, and noisy real-file I/O
remain explicit limitations. Stage4 is not a blanket performance improvement
for every output size or staging policy. No further stage was started.
