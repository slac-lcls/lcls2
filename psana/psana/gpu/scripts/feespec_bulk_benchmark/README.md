# Feespec on Weka FFB

## Bulk-on direct group submission comparison

`acceptance.py --study groups` compares baseline and candidate using the
14-sample bulk-on matrix, reporting `issue_group` profile cost. Both builds
include slot selection and pending-file reference accounting; the candidate
adds direct group descriptor/range construction and shared validation.
Full call graphs also show whether `_coalesced_plan` is reached by groups.
Use the same depth-1, batch-100, 8-GiB, eight-worker, 4-MiB target/task settings.
`--study groups-warm` or `--study groups-cold` runs four alternating
control pairs for the selected cache state if confirmation is needed.

Submission diagnostics and native tracing wrap `_submit_read` when present,
falling back to `issue_batch` for older builds. The new hook measures allocation
and actual submission; it excludes group/legacy planning. Historical
`read_submit_s` regions therefore differ across this refactor. Controls still
use native read counters, and end-to-end loop timings remain comparable.

## Bulk-on pending-file cleanup comparison

`acceptance.py --study files` uses the same 14-sample baseline/candidate
matrix and correctness/provenance gates as `slots`, but attributes profiles
to `KvikioGpuReader._prune_files`. Both builds include the committed slot
selection index; only `gpu_kvikio_read.py` differs. Keep depth 1, batch 100,
8 GiB, eight workers, and 4 MiB target/task settings fixed. Controls alone
determine throughput; profiles and paired pipeline diagnostics are separate.
`--study files-warm` supports four additional alternating warm control pairs.

## Bulk-on slot-selection comparison

`acceptance.py --study slots` compares `baseline` and `candidate` build entries,
both with `E-on`, on one allocation. Freeze otherwise identical runtimes with
the same transition-drain fix; only `gpu_input_group.py` should differ.
Use depth 1, batch 100, 8 GiB, eight workers, and 4 MiB target/task settings.

The 14 samples comprise eight cold/warm controls in two reversed-order rounds,
four separate measured-loop profiles, and two warm pipeline diagnostics.
`slots_summary.py` reports control ratios and `plan_slots` attribution, and
requires identical subbatch sizes, selected kernel counts, allocation-reserve
counts, and peak charged bytes in the paired pipeline samples. Correctness,
request/byte, cache, NIC, placement, and provenance checks remain enabled.
Profiles and pipeline timings do not enter the throughput result. Run the
targeted device correctness suite before starting the campaign.

When warm results need confirmation, `--study slots-warm` runs four warm-only
baseline/candidate control pairs, alternating order each round. It reuses the
same frozen builds and validation gates; it has no profile/pipeline samples
and reports those diagnostics as unavailable rather than passing them.

## CPU profiling of group scheduling

`acceptance.py --study profile` uses a `current` build entry with the desired
`depth`, `bulk_target_bytes`, and `task_size`. It runs 16 samples: bulk off/on,
cold/warm, two rounds, each with a control and a separate BD Python profile.
The second round reverses cache, variant, and instrumentation order. Existing
warmup correctness, checksum, request/byte, page residency, NIC, tier, and
source provenance checks remain enabled.

`bench.py --python-profile OUTPUT.pstats` enables cProfile only around the
BD measured event loop, including its normal final device synchronization.
Warmup, cache preparation, and profile serialization are excluded. It cannot
be combined with pipeline counters, native tracing, or full-array diagnostic
mode. The ordinary read counters remain identical in controls and profiles.

`profile_summary.py` saves control medians separately from instrumented
medians, plus self/cumulative times, call counts, and caller edges in
`results.json`. Raw `.pstats` files retain the full call graph. Self times
include blocking native calls on the BD thread; they are not GPU kernel or
KvikIO worker timings. Cumulative times overlap and must not be summed.
Use controls for throughput and profiles to locate follow-up targets; do not
attribute an improvement to instrumentation-induced schedule changes.

## Pool depth comparison

Use `acceptance.py --study depth` with `depth1` and `depth2` build entries
pointing to the same frozen current runtime and setting `depth` to 1 or 2.
The four cases are current bulk off/on at each depth. Batch 100, GPU budget
8 GiB and eight KvikIO workers remain fixed.

The 28-sample matrix starts with four separate warm pipeline diagnostics,
followed by 16 two-round cold/warm controls and eight two-round cold native
traces. Diagnostics count execution subbatches, selected parser/gather/JF
calibration launches and allocation-reservation peak/calls. JF calibration
remains per event; parser and gather launches are batched. Charged memory
includes owned cached backing and excludes context/unowned user allocations.
No diagnostic CUDA synchronization is added. Diagnostic rates do not enter
the throughput comparison.

Trace reports include POSIX-active wall time and its complement within the
BD event loop, alongside the single-file percentage. Intervals without POSIX
reads may contain useful H2D, GPU processing, CPU work, setup or tail drain;
they are not a measurement of GPU idle time. All cache/reference/placement
and provenance gates from Stage 4 remain enabled. `summary.md` is incremental.

## Stage 4 stream-read acceptance

`acceptance.py` compares frozen previous/current E builds with bulk off/on,
two rounds each of cold and warm, on one allocation. It reuses the private
FFB data and CPU references from the quick baseline: 1,000 events, JF plus
feespec, batch 100, depth 1, and 8 GiB. Sixteen untraced timing samples are
followed by four current-build cold traces (off/on, two rounds).

Pass `--directory`, `--references`, `--constants`, `--builds`,
`--fallback-library`, and a new `--output` directory. The builds JSON has
`previous` and `current` entries, each with `prefix`, `python`,
`native_prefix`, and a `hashes` map relative to `prefix`. Freeze the code,
harness and native trace helpers on scratch before submitting. Native
dependencies can be shared by both Python builds and must be hashed too.

Warm preparation reads only the measured prefixes (about 33.6 GB total),
then requires >=99% residency for **each** prefix before and after timing.
Cold requires <=1% before timing and physical NIC RX >=98% of payload.
All samples retain the quick baseline's warmup correctness checks and
measured timestamp/feespec sum validation. Weka SSD placement and frozen
input/code hashes are checked before and after the campaign.

`results.json` and `summary.md` update after every validated sample. The
performance target is current bulk-on median loop time <= current bulk-off
for both caches; trace rates do not enter that verdict. Two rounds provide
a focused comparison, not a statistical or long-run/scaling acceptance.

The group scheduler has overlapping requests. Trace attribution therefore
matches stable file handles and byte ranges, validating complete, nonduplicated
coverage and each owner's issue-to-ready bounds. A native global submission
tag alone cannot identify a queued read's owner. `read_interval_s` is the union
of request windows; `read_interval_sum_s` retains their sum, which can overlap.
The POSIX single-file percentage is computed directly from native intervals.

## Quick cold baseline

`quick.py --directory PRIVATE_FFB_XTC --constants SNAPSHOT --jf-cpu-log CPU_LOG`
reuses the verified six-stream input for 1,000-event cold samples in order
E-off, E-on, E-on, E-off. Batch 100, depth 1, and budget 8 GiB are unchanged.
Each fresh MPI process validates 200 warmup feespec arrays and three JF
raw/calibrated samples before evicting input. The measured loop has no
full-array D2H. It accumulates the existing reader byte/request/wait and
request-to-ready counters with one lightweight wrapper per read window.

SMD-derived extents bound `mincore` checks to the first 1,000 events in each
file. Cold eviction covers the private files; page residency must be <=1%
for each measured prefix. Tier checks require SSD-only Weka backing before
and after the sweep. Every measured run checks event identity and feespec
GPU sums. `--events 2000` supports the longer fallback if 1,000 is unstable.
Results and a four-sample summary are saved in `job-SLURM_JOB_ID/` on scratch.
This mode retains fresh processes and setup per sample; it does not yet reuse
initialized GPU workers across cases.

## Current-code cold and warm comparison

Use `acceptance.py --study current --events 10000` with the usual directory,
references, constants, builds, output, and fallback-library arguments. The
`current` build entry supplies a frozen Python runtime, native prefix, hashes,
and exact `expected_requests` for `E-off` and `E-on`. Prepare measured CPU
references and SMD manifests for the requested event count; warmup remains
200 events. Existing acceptance studies retain their 1,000-event default.

This runs eight controls: bulk off/on with cold/warm node page caches, twice,
reversing both variant and cache order in round two. It validates timestamps,
GPU sum hashes, exact request and payload counts, affinity, runtime provenance,
and SSD placement. No profiles or native-operation traces enter the rates.

The 10,000-event combined inputs occupy about 336 GB. Reserve 700 GiB of host
memory. Warm-prefix preparation uses `numactl --interleave=all` when measured
prefixes total at least 64 GiB; only requested prefixes are read, and timed
workers keep their original policy. Every warm prefix must be at least 99%
resident before and after timing. Cold prefixes must start at most 1%
resident, with physical NIC RX at least 98% of payload during the loop.

## Native fallback trace for the short cold baseline

Add `--fallback-trace` to `quick.py` to run eight cold samples: bulk off/on,
two rounds, each with a control and a trace. Round two reverses variant and
trace/control order. Copy `kvikio_fallback_trace.py`, its C source, and
`summarize_kvikio_fallback.py` from the parent scripts directory into the
scratch campaign. Compile the C source with the command in its header as
`fallback.so`. The required `stage.py`, `page_cache_residency.py`, `memory_state.py`, and
`warm_cache.py` helpers are included in this directory; copy them too.

Tracing is active only during the measured loop and buffers records in RAM.
The tracer observes native POSIX reads, H2D calls, and existing stream waits;
it adds no CUDA synchronization. Each trace must pass byte, operation-count,
triplet, and interval audits. Controls have no tracer preload. Both modes keep
the 200-event correctness warmup, measured-prefix eviction verification,
network traffic check, and before/after Weka SSD-tier check.

`summary.md` reports the numerator and denominator for the single-file
percentage: wall time with exactly one file in POSIX reads divided by wall
time with any POSIX read active. Concurrent calls on the same file count once;
idle gaps are excluded. Files are identified by stable open handles for this
single-chunk workload. Each trace also produces `.audit.json`, `.json`, and
`.bin` artifacts; `control-summary.md` retains the untraced baseline report.

## Combined JF + feespec mode

Submit on SDF with account `lcls:data` and QoS `normal` (the Ampere association
is `lcls:data@ampere`). Bare `lcls` selects the default preemptable account.
Request one A100; whole-node exclusivity is not required for this harness.

Pass `--include-jf --constants /path/to/cpu-calibration.pkl.gz
--jf-cpu-log /path/to/cpu-check.log` to `run.py`. This stages physical streams
0 and 5–9 and runs the same 12-case matrix with batch size 100, depth 1, and
an 8 GiB GPU budget. Reserve sufficient host RAM for the roughly 336 GB input
to remain in the page cache (the submitted job reserves 700 GiB).

A keeps `gpu_det='jungfrau'`, calibrates JF on the GPU, and extracts feespec
on the CPU followed by per-event H2D and GPU sum. E uses
`gpu_det=['jungfrau', 'feespec']`; its bulk switch applies globally to both.
JF calibration executes in all cases; clean runs do not export JF images.
This measures the current combined behavior, including admission changes,
and does not isolate feespec-only coalescing or its incremental cost.

Drop `field`, `segments`, and `values` after the field-view context and before
advancing the event iterator. A Python `with` block leaves those variables
bound; the feespec slice can otherwise retain the entire mixed input buffer
through a later buffer replacement. The maintained consumer explicitly clears
them. See `docs/performance/jf_feespec_admission_failure.md` under `psana/gpu`
for the reproduced failure and allocation accounting.

Validation runs check every feespec array and three JF raw/calibrated arrays
against the existing CPU baseline. The frozen JF calibration snapshot is
shared read-only and hashed before/after. Diagnostic read counts are broken
down by physical file. Warm preparation uses `warm_cache.py` under NUMA
interleave; residency is verified separately for all six files. That helper is included in this directory.

## Feespec-only mode

Proof-of-concept comparison of historical A with per-event CPU field extraction
and H2D, current E bulk off, and current E bulk on. Feespec only; JF is absent.
Current E is frozen at ac87a93b2. No production runtime module is modified.

Workload: mfx101210926 run 387 s000, first 10000 events; hproj is an 8 KiB int32
array. Preserve complete shared-stream datagrams and SMD offsets. A uses the
normal CPU path, including any CPU read coalescing. E uses a process-local
exclusive routing exception for feespec's stream and reads all its bytes once.
Both produce per-event GPU int64 sums. E's public on_gpu_view API includes
locator metadata D2H and consumer lifetime bookkeeping in the timing.

Batch 100, execution depth 1, 8 GiB per-BD GPU budget for E. A has no detector
GPU pipeline in this small-detector-only case; its user loop does cp.asarray
and the identical GPU sum. One BD plus EB/SMD0, one A100, KvikIO CPU fallback
with 8 workers and 1 MiB tasks. Three validation runs copy every array back
for exact CPU-reference checks and collect API read counts. Twelve clean runs
contain no read hooks and no full-array D2H. End-of-loop GPU synchronization
is timed; checksum retrieval, initialization before the loop, and cache
preparation are excluded. Lazy setup inside run.events remains timed.

Two rounds reverse variants and warm/cold order. Warm >=99% page residency;
cold <=1% before the loop plus physical Ethernet receive counters. FFB copies
must have full SSD coverage and no object/remote backing, checked before/after.
Server-side caches are not flushed; node background traffic can contribute
to NIC counters. This is a small-input latency test, not a network saturation test.

Copy all scripts in this directory to a new shared-scratch campaign directory. Provide builds.json with
A and E frozen prefixes and complete file hashes. The controller stages a
private bounded prefix under /sdf/data/lcls/drpsrcf/ffb/users/monarin on the
compute node and writes logs/results/summary to shared scratch.

## Four MiB target/task comparison

`acceptance.py --study size4` runs current bulk off/on at execution depth 1:
two separate warm pipeline diagnostics, eight controls (cold/warm, two rounds),
and four cold native traces. A `size4` build entry specifies `depth: 1`,
`bulk_target_bytes: 4194304`, `task_size: 4194304`, and expected per-variant
request totals. The driver sets `KVIKIO_TASK_SIZE` and verifies the effective
KvikIO value; `bench.py --bulk-target-bytes` forwards the independent
`gpu_bulk_target_bytes` DataSource parameter. Omit that option for historical
builds that lack it. The production bulk target still defaults to 1 MiB.

For the current 1,000-event, batch-100 input, the real SMD preview gives the
same 19 feespec groups and 5,000 singleton JF reads at either bulk target.
Batch boundaries and physical gaps limit feespec grouping first. This run
therefore primarily probes KvikIO task granularity, not larger feespec bulks.
Trace metadata records task size; the native operation-count audit uses it,
with a 1 MiB default for historical trace metadata.
