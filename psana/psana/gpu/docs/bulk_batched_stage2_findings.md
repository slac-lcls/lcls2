# B++ bulk integration: Stage 2

2026-09-23, branch `codex/psana2-gpu-bulk-batched-integration`.
Stage 1 was committed as merge **25e07ad40**, with parents **786cd16ca** (D+)
and **4c3cdf5a7** (B++). Stage 2 acceptance and pre-commit review are complete.

## Implementation

Canonical gather now supports detector segments spanning resident and transient
input owners. The execution map stores `(owner index, owner-local dgram row)`
for each event/stream pair. A distinct-owner table carries raw address/byte size,
combined-locator address, allocated row capacity, and active dgram count. One
kernel gathers across all owners and writes every output/presence row, including
zeros for missing or rejected fields. Calibration remains per event.

The gather kernel checks owner index, owner-local row against both active count
and capacity, field status/type/rank/size, and raw byte bounds. It uses each
owner's allocated locator stride, not execution length or another owner's
capacity. The fixed canonical routing table must match every owner's configured
handle layout. Gathering does not use `event.batch`, which is absent for events
spanning multiple owners.

The two execution tables share one owned device allocation and one pinned host
source. Admission conservatively reserves 56 bytes per event/stream entry:
16 bytes for its row pair plus 40 bytes for a possible distinct owner. Device
allocations additionally use the established 512-byte pool rounding. Uploads
include only the row pairs and actual owner records. For 20 events and five
streams, the logical allocation is 5,600 bytes; one owner requires a 1,640-byte
upload. Pinned capacity is reported separately. Reuse, full replacement costs,
and trimming use D+'s allocation owners and execution-slot lifetime.

EventPool's existing execution input leases retain all input windows through
queued gathering and failure drains. The map cache stores pointers without
keeping parsed owners across retirement. Standalone detector/gather callers
must keep inputs alive through their stream completion, as before.

Input windows deduplicate ready events by object identity while retaining
distinct walker, configured-location, fallback-locator, and terminal-consumer
events. Each gather waits on distinct configured dependencies across producer
streams. This preserves configured readiness when the locator-wrapper cache is
empty and avoids repeated waits from public access to configured fields.

## Launch-count preservation

The multi-owner GPU tests count actual launch submissions for two executions:

| Residency | Parsed input windows | Walker | Locator init | Grouped locate | Canonical gather |
|---|---:|---:|---:|---:|---:|
| None; two transient owners per execution | 4 | 4 | 4 | 4 | 2 |
| One resident owner and two transient windows | 3 | 3 | 3 | 3 | 2 |
| Both owners resident across both executions | 2 | 2 | 2 | 2 | 2 |

These cases pass for uint16/calibrated and float32 passthrough synthetic inputs,
using independent raw bases, reversed local rows, unequal capacities, missing
streams, tails, and producer streams distinct from the execution stream. They
assert that no per-field locator wrappers are created. A resident input is
walked and located once, and each detector execution still launches one gather.
The original 44-kernel count for one full 20-event JF execution with one input
window remains the expected structure; no new full-JF Nsight trace is claimed.

## Validation

Artifacts: `validation/bulk-integration-stage2-20260923/`.

- CPU unit suite: **323 passed**.
- A100 device suite: **61 passed**, job **38900749**. Includes the original
  allocation/admission tests, grouped locator and gather equivalence, production
  mixed-rate residency fixtures, and seven new multi-owner cases.
- JF bulk-off/on acceptance: **five cases passed**, job **38900749**,
  **2,112 allocation-identity/capacity checkpoints**.
- CUDA memcheck: **12 passed, zero errors**, job **38901323**, using
  `--show-backtrace device`. This includes the seven multi-owner tests and five
  additional bounds/delayed-consumer tests. All 66 distinct GPU cases pass
  across the ordinary device suite and this focused sanitizer suite.
- Main psana: **390 passed, 68 skipped, 10 deselected**; MPI byhand:
  **four passed**, job **38900750**. Test groups overlap.

GPU/JF job **38900749** completed on `sdfampere001` in **5m26s**, exit `0:0`.
Core/MPI job **38900750** completed on `sdfmilan262` in **3m50s**, exit `0:0`.

| JF case | Events | Batch/depth | Budget GiB | D2H chunk | Peak live MiB | Loop-end live MiB |
|---|---:|---|---:|---:|---:|---:|
| Bulk off, all facades retained | 1,003 | 20/1 | 8 | 0 | 3,201.809 | 0 |
| Bulk on, all facades retained | 1,003 | 20/1 | 8 | 0 | 3,201.809 | 0 |
| Bulk off, partial tail | 1,003 | 13/2 | 8 | 7 | 3,970.334 | 0 |
| Bulk on, partial tail | 1,003 | 13/2 | 8 | 7 | 2,305.199 | 0 |
| Bulk on, tight budget, all facades retained | 1,003 | 20/1 | 4 | 0 | 3,201.809 | 0 |

Each case matched all timestamps and 15 raw/calibrated image samples against
the frozen independent CPU reference. Non-D2H cases also checked direct parsed
JF fields. Loop-end checks precede forced GC and diagnostic synchronization,
with selected facades still retained. Data: `mfx101210926`, run 387, streams 5–9,
one BD/A100, KvikIO CPU fallback. These are correctness diagnostics.

The lower bulk-on peak at batch/depth 13/2 reflects one live working set,
whereas bulk off holds two at its peak. Allocation traces show about 640 MiB
of fixed storage plus, per event, 32 MiB input, 32 MiB gathered raw, and 64 MiB
calibrated output. This predicts approximately 3,200 MiB for 20/1, 3,968 MiB
for two 13-event working sets, and 2,304 MiB for one. Metadata and allocator
rounding explain the remaining differences. The existing bulk path flushes
the EventPool and trims caches before starting the next resident input batch;
depth two therefore does not guarantee two working sets overlap. Lower peak
memory is not evidence of higher throughput. Drain/trim cost and overlap are
explicit Stage 4 measurement items, not policy changes in this commit.

Pre-commit review covered pointer-table bounds, readiness ordering, input leases
on partial submission failure, allocation estimates, and cache retirement.
No additional runtime fix was required. Stage 3 will extend the lifecycle and
failure coverage with the integrated parser/gather path.

The injected map-upload failure occurs after asynchronous submission and then
forces an unproven stream drain. The occupied execution slot retains both input
owners and the pinned/device map; successful retry drains and releases them.
Additional device tests exercise invalid owner/row/capacity/raw-byte bounds and
two delayed input consumers. A CPU test checks that deduplication preserves a
distinct lazy-locator event and a distinct consumer event.

No native extensions, residency policy, bulk cache-trimming policy, calibration
algorithm, or detector selection changed. Stage 3 remains the broader failure,
lifecycle, and memory validation gate; Stage 4 remains the clean performance
comparison. No throughput improvement is claimed from these diagnostic runs.

## Sanitizer host-backtrace retention

The first memcheck job **38900850** reported zero device access errors but failed
seven post-trim allocation-lifetime assertions. Diagnostic job **38901057** ran
the same case normally and under memcheck: the normal case passed, while the
sanitized case retained backing. Reference inspection in **38901144** found
completed parser and gather Python frames retaining their array locals.

With host backtrace capture disabled (`--show-backtrace device`), all 12 tests
passed unchanged and memcheck still reported zero errors in **38901323**.
This controlled comparison attributes the extra retention to sanitizer host
backtraces. No production code, lifetime assertion, or budget was relaxed, and
no forced GC was introduced into acceptance. Device error checking and device
backtraces remain enabled. The earlier diagnostic-only GC experiment and failed
logs are preserved in the artifact directory.

The runtime matches the frozen installation; native hashes match Stage 1.
All original acceptance script/test hashes remain unchanged. Source snapshots,
the final source manifest, and sanitizer provenance are saved with the results.
