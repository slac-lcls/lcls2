# Warm B+ versus batched canonical gathering on SDF

Completed job **38561649**, sdfampere004, 2026-09-18: four alternating clean
pairs, one separate host-timed pair, two Nsight captures, and four CPU-reference
preflights. Slurm reports COMPLETED / 0:0, allocation elapsed 48:04. The
provenance and trace-count audits pass. Clean median elapsed time falls
**37.275 → 28.578 s (23.3%)** for 10,000 events.

Initial job 38559942 on sdfampere024 was interrupted by `cudaErrorContained`
in its final frozen-B+ control. The CUDA message names peer-memory access or a
hardware error; its exact cause was not established. Its samples are retained
separately and are not pooled with this completed retry. Compute Sanitizer
subsequently passed all 17 gather tests with zero errors.

The subsequent [same-allocation A/B/B+/gather comparison](four_way_sdf.md)
provides a direct A reference and additional evidence of timing variability.

## Compared implementations and workload

- **O / B+:** `b0c9c3c02`, stream-grouped batched field location with the existing
  per-event/per-segment gather. The validated installation was copied to
  `validation/batched-gather-20260918/install_baseline` before any refresh.
- **G / B+gather:** the working-tree implementation based on that commit, with
  one canonical gather per detector execution subbatch. All changed production
  modules were checked against the installed sources before submission.
- Neither variant includes bulk-read integration. Calibration and result
  delivery retain the existing per-event path. No user D2H is requested.

Both use 10,000 `mfx101210926` run-387 events, Jungfrau 32 segments across
physical files s005-s009, batch size 20, one execution slot, 8 GiB GPU budget,
and one SMD0/EB/BD (three MPI ranks) on one A100 SXM4 40 GB. KvikIO compatibility
is ON: CPU fallback, not GDS; eight reader threads and 1 MiB task size. Versions:
CuPy 13.6.0, CUDA runtime 12.9, KvikIO 24.08.02, driver 575.57.08.

One allocation requests 48 CPUs and 450 GiB host memory. Every rank receives
the same allowed CPU mask, `1-4,9-16,46-57,65-68,73-80,110-121`, for every variant. This
is shared affinity, not exclusive per-rank core or GPU-local memory binding.
Staged payload is 335,571,760,000 bytes, with 10,000 unique event timestamps;
input timestamp SHA256 is
`23f9051ac9f98208310171f734251b0ccd1f40fa048d1748553cbd38cce559d6`.
The same trusted CPU calibration snapshot is used by both variants, SHA256
`c3679e0fb47bebfafe41f36750e88fe1ec0d6b71ef386814fb1eac37dc197ce9`.

Four clean repetitions alternate OG/GO/OG/GO. The first repetition also has
a separate CPU/NVTX pair; the retry reduces supplemental instrumentation while
retaining four clean controls per variant.
Separate Nsight captures follow. Both builds pass CPU pixel-reference
preflights with and without instrumentation before timed measurements.

Each sample follows a 100-event warmup and requires at least 99% file-cache
residency before and after timing. Setup and final GPU synchronization are
inside the unchanged event-loop timer; staging and cache preparation are
outside it. The retry needed three initial preparation passes (90.08%, 96.61%,
100%); all 24 before/after cache checks across its 12 samples were 100%.
Clean throughput, host scopes, and GPU traces are separate measurements and
must not be added together as a wall-time budget.

## Clean elapsed time

All four clean pairs completed in job 38561649 on sdfampere004. No samples were
excluded. Values below are seconds for 10,000 events, including setup and final
GPU synchronization. The order alternates between pairs.

| Repetition | Order | B+ | B+gather |
| --- | --- | ---: | ---: |
| 1 | B+, gather | 38.306 | 34.915 |
| 2 | gather, B+ | 38.289 | 28.535 |
| 3 | B+, gather | 36.261 | 28.621 |
| 4 | gather, B+ | 34.978 | 27.571 |
| **Median** | | **37.275** | **28.578** |

The median falls **8.697 s (23.3%)**; throughput derived from the medians rises
from **268.3 to 349.9 events/s (30.4%)**. Each matched pair favors gathering,
but the magnitudes vary. B+ ranges 34.978–38.306 s and gathering ranges
27.571–34.915 s. Four pairs do not establish a universal speedup across nodes.
These clean samples remain separate from host instrumentation and Nsight.

## Separate host-timing pair on the retry node

One CPU/NVTX sample per variant completed on sdfampere004. Total instrumented
elapsed times were 37.058 s (B+) and 31.025 s (B+gather). These are single
samples, not repeated medians or clean throughput. Steady scopes cover 9,980
events / 499 execution subbatches; setup is reported separately.

| Host scope, seconds | B+ | B+gather |
| --- | ---: | ---: |
| Detector setup | 2.814 | 1.042 |
| Read submission | 1.005 | 0.868 |
| Read completion waiting | 20.897 | 22.377 |
| Whole slot submission (inclusive) | 10.054 | 2.463 |
| Field location | 0.617 | 0.578 |
| Field access/map/dependency/gather submission | 7.216 | 0.071 |
| Detector event selection | 0.491 | 0.504 |
| Calibration submission | 0.277 | 0.242 |
| Producer retirement waiting | 0.005 | 2.777 |

The gather submission scope falls 99.0%, including the new map preparation,
upload and dependency handling. The remaining producer wait reflects GPU work
that still must finish after CPU submissions become shorter. Read waiting and
setup differ too; this pair is not an exact decomposition of clean elapsed time.

## Supplemental host scopes from the interrupted allocation

The initial allocation completed three CPU/NVTX repetitions per variant before
the later CUDA error. These supplemental medians are not combined with the retry. Steady scopes
cover 9,980 events / 499 execution subbatches, excluding the first subbatch.
Setup is a startup scope. These are host elapsed scopes, not GPU durations.

| Scope | B+ | B+gather |
| --- | ---: | ---: |
| Detector setup | 0.964 | 0.947 |
| Read submission | 1.010 | 1.046 |
| Read completion waiting | 16.484 | 15.360 |
| Whole slot submission (inclusive) | 9.715 | 2.367 |
| Field location | 0.569 | 0.555 |
| Field access/map/dependency/gather submission | 7.073 | 0.067 |
| Detector event selection | 0.484 | 0.485 |
| Calibration submission | 0.263 | 0.237 |
| Producer retirement waiting | 0.004 | 2.819 |

The gather scope includes the new row-map preparation/upload and dependency
handling. Its median falls about 99.1%. Producer waiting grows after much less
CPU submission work is queued; the GPU still has to finish gathering and
calibration. Inclusive scopes and independent medians must not be summed.

Instrumented total medians are 29.750 s (B+) and 24.011 s (B+gather).
Read waiting also varies across samples; these numbers do not establish a
pure instrumentation overhead or an exact decomposition of clean elapsed time.

## Nsight call counts and device work

Separate captures cover the same 10,000 events / 500 execution subbatches.
Kernel and driver-launch counts match the expected workload exactly.

| Operation | B+ total | B+gather total | Per 20-event execution, B+ → gather |
| --- | ---: | ---: | ---: |
| Walker | 500 | 500 | 1 → 1 |
| Locator initialization | 500 | 500 | 1 → 1 |
| Locator decoding | 500 | 500 | 1 → 1 |
| Canonical gathering | 320,000 | 500 | **640 → 1** |
| Calibration | 10,000 | 10,000 | 20 → 20 |
| Missing-row cleanup | 10,000 | 10,000 | 20 → 20 |
| **All kernels** | **341,500** | **22,000** | **683 → 44** |
| Explicit target/presence memsets | 20,000 | 0 | 40 → 0 |

Observed `cudaStreamWaitEvent` calls fall 320,500 → 501. The 320,000 per-field
gather waits disappear; 500 existing parser/configuration waits remain, plus
one new fixed-plan dependency. The production gather uses the parser stream,
so it needs no additional locator-ready wait. Cross-stream gathering is covered
by correctness tests and uses one shared locator dependency per execution.
Event-record calls are 1,510 → 1,512; record counts are reported separately
from kernel counts.

KvikIO driver H2D calls remain 370,000. Total H2D copies are
370,511 → 371,012, and bytes are 336,649,567,192 → 336,649,968,216.
The difference is exactly 500 × 800-byte row maps plus one 1,024-byte fixed
routing table: 501 copies / 401,024 bytes. Neither trace contains D2H copies.

| Summed GPU operation duration, seconds | B+ | B+gather |
| --- | ---: | ---: |
| Gather kernels | 1.994 | 1.730 |
| Calibration kernels | 1.833 | 1.803 |
| Missing-row cleanup | 0.511 | 0.511 |
| All H2D copies | 19.676 | 18.968 |

These are profiler measurements, not clean throughput or host scopes. Gather
primarily removes CPU submission work; it still copies the same detector pixels.
GPU activity interval-union coverage is 24.074 → 22.871 s, within first-to-last
operation spans of 34.692 → 26.598 s. Coverage is not SM utilization, and
operation sums must not be added to overlapping host scopes.

Both collectors report “Not all NVTX events might have been collected” and
“Not all CUDA events might have been collected.” The expected kernel counts,
launch calls, reader-copy count, and extra map bytes reconcile; this does not
prove every trace event was captured. Profiler runs also emit UCX unmatched
receive-descriptor warnings at shutdown and exit successfully. Keep these
limitations with the trace evidence.

## Memory and placement audit

The GPU UUID throughout is `GPU-3e8528f8-fe59-785e-c2c4-c969ab13acda`.
Clean and host-instrumented peak device memory is 3,633 MiB for both variants;
profiler peaks are 3,671 MiB (B+) and 3,667 MiB (gather). Sampling every 500 ms
at MiB resolution is not an allocation ledger or a precise transient peak.
Explicit new storage is a 1,024-byte device table and an 800-byte device map
plus an 800-byte pinned host map per occupied execution slot.

The audit checks all 12 samples, four CPU-reference preflights, one GPU identity,
uniform CPU affinity, runtime versions, input count/order/hash/bytes, warm-cache
guards, and unchanged installed-source hashes. At measurement completion, production sources matched
the measured installation; the frozen control matches its recorded B+ base.

## Correctness and implementation review

- 203 focused CPU/GPU cases passed in job 38559897, including 16 initial new
  batched-gather cases; six slow real-data cases were deselected there.
- An additional delayed-consumer/EventPool retirement test passed in job
  38560120, followed by all six slow real-data pixel-exact cases. Those cover
  tails, slot reuse, D2H chunk sizes, and hybrid routing.
- Compute Sanitizer job 38561752 passed all 17 gather tests with zero reported
  errors after the interrupted control run.
- Main suite: 234 passed, 41 skipped, eight deselected. Four longer MPI tests
  passed. Seven benchmark AST/timing tests passed.
- The reference per-segment gather and missing-row cleanup implementations
  remain unchanged. No production edits followed the initial GPU pass.

See [the implementation review and call path](../batched_canonical_gather_review.md)
for storage accounting, ownership, supported scope, test-only harness fixes,
and the separately measured geometry prototype that was not applied.

## Reproduction and artifacts

Harness, correctness launchers, probe, and instructions:
`validation/batched-gather-20260918/`.
The retry evidence is under `job-38561649-warm-ab/`; the interrupted run is
retained under `job-38559942-warm-ab/`. Generated logs, CUDA
cache, installation copy, SQLite exports, and Nsight reports are ignored.
The completed audit can be reproduced with:

```bash
python validation/batched-gather-20260918/audit.py \
  validation/batched-gather-20260918/job-38561649-warm-ab
python validation/batched-gather-20260918/summarize.py \
  validation/batched-gather-20260918/job-38561649-warm-ab
```

No throughput thresholds are asserted. Historical A timings are from other
allocations; A is not remeasured in this job, so this comparison cannot establish
an exact current same-node gap between A and the new gather implementation.
