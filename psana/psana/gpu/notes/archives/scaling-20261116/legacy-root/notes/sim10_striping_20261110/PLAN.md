# Sim10 Lustre read and striping experiment

Status: proposed experiment; no new measurements yet. Scope is a one-node
standalone CPU read benchmark on Perlmutter. The result will be the best tested
stripe count for a specified read pattern and concurrency, not a universal
Lustre optimum or a measured psana speedup.

## Inputs and fixed controls

Source directory:
`/pscratch/sd/m/monarin/psana2-gpu/datasets/mfx101555026-r0010-sim10-100k-jf`.
Read all bytes of `mfx101555026-r0010-s00{5,6,7,8,9}-c000.xtc2`, approximately
313 GiB combined. SMD and generation metadata are excluded from timed reads.
Current `lfs getstripe` inspection: count 1, size 1,048,576 bytes, pool original;
OST indices for s005 through s009 are 273, 106, 76, 91, 92. These observations
are captured in evidence.txt; capture exact lengths and hashes before execution.

Sweep stripe counts **1, 2, 4, 8, 16, 32**, holding stripe size at **1 MiB**,
pool at original, node class, CPU affinity, buffers, request ordering, and read
API fixed. Leave OST selection automatic, recording every file's actual layout,
unique OST coverage, and overlap across files. Stripe count is per file, so
five files at count 1 already offer five OSTs of parallelism in this fixture.

Use one exclusive GPU node on `lcls_g`, `--qos=premium` (reported as
gpu_premium), to retain the node class used in the earlier benchmark. The
reader uses CPUs only. Record actual resources charged, node identity, mount,
kernel/Lustre versions, CPU/NUMA topology and job QoS. Bind each worker to a
distinct physical core; first-touch its reusable buffer on that core. Keep the
mapping fixed within a block and record it. No competing benchmark, staging,
hashing, or profiling runs during timed trials.

## Copies and provenance

Use a new campaign directory under
`/pscratch/sd/m/monarin/psana2-gpu/sim10-striping-20261110/`.
Create six new full copies, including a fresh count-1 copy, with simple fixed
layouts. Set the layout before writing, e.g. an empty destination created with
`lfs setstripe -c 8 -S 1M -p original DEST`, then explicitly stream source bytes
into that inode. Do not use links, reflinks, sparse skipping, or a copy method
that silently replaces the pre-striped inode. Inspect getstripe after writing.
Use bounded copy concurrency on allocated compute resources; record copy time.
Flush and close each output. Verify exact file sizes and independent full-file
SHA256 against the accepted source manifest before admitting a copy to testing.
Hashing is outside timing and followed by the same cache reset as any read.
Original files remain untouched; do not migrate them for this experiment.

The first six copies require about 1.84 TiB extra. Confirmation creates a second
placement for count 1 and up to two finalist counts, giving at most nine copies
or 2.75 TiB extra (plus the existing source). Check scratch quota, free capacity,
and at least 10% headroom before staging. If insufficient, revise staging to
retain the finalist sets and remove only manifested, verified campaign copies;
never reduce validation silently. Keep manifests and reports after cleanup.

## Standalone reader

Implement a small C++ POSIX `pread` harness with threads, independent file
descriptors and reusable aligned buffers. No psana, XTC parsing, calibration,
CUDA, or decompression in the executable. A Python orchestration script can
manage trial manifests and summaries outside timing. Native threads keep the
hot path independent of Python allocation and interpreter scheduling.

Primary pattern: scan all five files concurrently. Worker counts are **5, 20,
40** (1, 4, 8 workers per file). Within each file divide the byte range into
equal contiguous disjoint partitions, aligned to request boundaries; each
worker reads sequentially through its partition. Fix the partition algorithm
for every stripe count. Primary request size is **1 MiB** and each worker has
one blocking request in flight. All bytes are read exactly once, including the
last partial request. Unequal file lengths are real workload imbalance and
remain part of the aggregate elapsed time.

For finalists add total workers **1, 10**: one worker scans the five files in a
fixed file order, while 10 assigns two workers per file. Also run each of the
five files separately with one worker as a diagnostic for single-file benefit.
These diagnose the concurrency dependence; they do not replace the five-file
primary result. Compare threads with processes at 20 workers on count 1 and
the leading finalist using the same partitions and barriers; if execution
model changes ranking, report separate results rather than pooling them.

Preallocate/first-touch buffers and open descriptors before the start barrier.
Time from release of all ready workers until the last read completes with a
monotonic clock; include barrier release skew, exclude setup and cleanup.
Compute throughput = actual successfully returned bytes / elapsed / 2^30.
Track per-worker/file bytes and completion times, CPU time, errors, short reads,
and start skew. Handle EINTR and retry short reads at the remaining offset;
unexpected EOF, other errors or a missing worker invalidate the whole trial.
Use deterministic partition checks and full byte accounting; hashes of copies
are the separate content integrity check, not part of the timed loop.

Before large trials test nondivisible lengths, final tails, unequal files,
worker failure, short reads and accounting on tiny fixtures. Verify every
byte is assigned exactly once, no offsets overlap, and errors cannot yield a
successful throughput record. A short native pilot checks binding, resource
usage, timing, filesystem semantics and cache measurement on the actual node.

## Caching and I/O modes

Primary mode is buffered POSIX I/O, matching the buffered nature of the prior
pipeline, with a **verified cold local page cache before every trial**. Close
other readers, use file-specific POSIX_FADV_DONTNEED, then check residency of
all pages of all participating files using mmap/mincore without touching data.
Require zero resident pages immediately before opening the start barrier, with
the last EOF page counted correctly. Record counts and elapsed verification
time. Bound eviction attempts; if zero residency cannot be established, mark
the buffered trial invalid and diagnose. Do not silently relabel it cold or
use privileged global cache dropping. Use one fixed readahead policy for all
trials (default kernel behavior) and record it; partition count changes the
access pattern intentionally as the concurrency factor.

Lustre client internals and storage/server caches are not proven cold by
mincore. Report this as local-page-cache-cold application throughput, with
backend cache state uncontrolled. Record available per-client filesystem byte
counters before/after, identifying units and scope; if unavailable (as in the
earlier run), report remote bytes unverified. NIC Ethernet counters alone are
not accepted as Lustre traffic evidence.

On count 1 plus finalists, add **O_DIRECT** trials as a separate diagnostic.
Probe filesystem-supported offset/length/buffer alignment, use posix_memalign
buffers and aligned partitions, and reject unsupported or failing direct I/O
instead of silently falling back. Use the aligned prefix of each file, record
excluded EOF tail bytes, and repeat buffered trials on the identical prefixes
for the direct/buffered comparison. Do not mix these with full-file rankings.
O_DIRECT changes readahead/copy behavior and bypasses the ordinary page cache;
it does not prove physical-media-cold reads. If direct mode is unavailable,
retain the valid buffered study with the diagnostic explicitly unavailable.

## Stages and replication

1. Preflight: Iris balance, storage quota, source manifests, source/build IDs,
   harness correctness, staging verification, native pilot and cache control.
2. Screen: six stripe counts x three worker counts x three randomized complete
   blocks = **54 full-dataset trials**, about **16.5 TiB** read. Every block
   contains each combination once, runs serially on the same node, and uses a
   saved random seed/order. Record UTC times and failed trials; never discard
   slow valid observations as outliers. A failed trial can be rerun with its
   original receipt retained and the reason documented.
3. Select count 1 plus up to two non-1 finalists using the geometric mean of
   paired throughput ratios to count 1 across the three worker counts/blocks.
   Also retain any count with a distinct concurrency-specific win that would
   otherwise be hidden by this mean; if that requires more than two finalists,
   narrow the recommendation by workload or explicitly expand the experiment.
4. Confirm: make a second independently allocated copy set for each retained
   count. Verify actual placement differs; if automatic allocation repeats
   the same ordered per-file OST layout, recreate that campaign copy up to two
   times within the resource cap. Keep automatic OST selection. If it still
   repeats, confirmation is incomplete. Record per-file and combined OST-set
   intersection/union fractions and ordered layouts; different allocations
   are placement replicates, not a claim of disjoint OSTs or statistical
   independence. Run both placements, all
   three primary worker counts, and three new randomized complete blocks in a
   second allocation/time window: at most **54 more full-dataset trials**.
   Each confirmation block includes both placements for all candidates. This
   separates repeatability over time from sensitivity to OST placement. Record
   node differences and normalize to count 1 within the same block, mode,
   concurrency and placement generation. It does not isolate OST server load
   causally; two placements still provide limited sampling.
5. Diagnostics: on retained counts, three repeats of the 1/10-worker patterns,
   single-file scans, direct/aligned-buffered comparisons at 20 workers, and
   the process/thread comparison. On count 1 and the leading non-1 count, also
   repeat full-file 20-worker scans with **4 MiB and 16 MiB** requests (three
   blocks). Keep stripe size 1 MiB. If this changes ranking, report the request
   size dependency instead of declaring one overall optimum. Stripe-size
   tuning and multi-node scaling are separate follow-up experiments.

The contiguous partitions are a controlled bulk-read workload, not a faithful
replay of the previous KvikIO task queue. Before claiming a psana recommendation,
add an isolated fixed-offset/request-list replay or a short pipeline validation
as a separate follow-up; the present conclusion is expressly standalone I/O.

## Selection, resource envelope and deliverables

Publish every valid trial, median and range, paired ratios to count 1, and
results separated by worker count, placement, mode, request size and node.
Use confirmation data (not screening winners alone) for the recommendation.
Prefer the **smallest count within 5% of the fastest confirmation median at
each primary worker count**, requiring that conclusion to hold separately for
both placement generations. Report a common count only if one satisfies that
rule across all three primary worker counts; otherwise give a small table of
workload-specific choices. Compute variability within each fixed placement,
worker count and stripe count across time blocks. Variability over 10%
(max/min - 1) triggers three extra blocks for all compared candidates in that
worker-count/placement cell, capped at six confirmation blocks and subject to
the resource cap. If each placement is repeatable but their rankings disagree,
report placement sensitivity; extra time blocks cannot resolve that confound.
A third placement is a separate extension, not included in this budget.
If ambiguity remains,
report a tie or inconclusive result rather than a precise optimum. Three
repeats are descriptive evidence, not a strong statistical confidence claim.

Do not expand beyond count 32 automatically. If throughput still rises strongly
at 32, report the tested boundary; propose a targeted extension instead of
asserting optimality. The 5% tolerance is a declared practical tie rule, not
a statistical significance threshold.

Planning envelope: up to 2.75 TiB extra storage, approximately 33 TiB read for
screening plus confirmation, plus diagnostics, staging, hashes and optional
extra blocks. Diagnostics use just the first placement and add up to 69
full-dataset-equivalent scans (about 21.1 TiB): 18 for 1/10-worker trials,
9 equivalents from 45 single-file trials, 18 direct/buffered pairs' individual
scans, 12 process/thread scans and 12 request-size scans. Direct EOF exclusions
reduce this slightly. At an assumed 2-10 GiB/s, those 108 primary trials alone take
about 0.9-4.7 hours of timed I/O. These are estimates, not promised throughput.
Allow roughly 4-12 single-node wall-hours for the staged campaign initially;
use pilot timings to produce an explicit revised estimate including eviction,
copy/hash cost and diagnostics before the main sweep. Submit bounded stages
with checkpointed manifests, not one oversized job; cap the initial campaign
at 12 node-hours and report incomplete cells if more work is needed.
Priority is preflight/pilot, screening, confirmation (including any extra blocks
required for the decision), then optional diagnostics. Do not start diagnostics
at the expense of confirmation. Incomplete confirmation produces preliminary
observations only, with no stripe-count recommendation. If diagnostics are
omitted, bound the recommendation to the completed primary 1 MiB/threaded cases
and label all untested modes explicitly. Query
Iris before submissions and report actual charged allocation hours separately
from node-hours. Premium on lcls_g remains the user's preference.

Deliver executable/source identity, job scripts, copy manifests/hashes/layouts,
randomization schedule, per-trial JSON/CSV, cache/correctness receipts, Slurm and
Iris accounting, plots of GiB/s versus stripe count by concurrency, and a short
recommendation with repeatability and scope limits. Execution will use another
implementation/review cycle before expensive measurements and review the
measured conclusions afterward. This document review does not approve code or
claim that any proposed test has already passed.

## Reference

Consulted 2026-11-10: [NERSC Lustre striping guidance](https://docs.nersc.gov/performance/io/lustre/).
It explains setting layouts before writing, inspecting them with getstripe,
and the tradeoff between more OST parallelism and excess striping overhead.
The sweep values and decision rules here are experimental design choices.
