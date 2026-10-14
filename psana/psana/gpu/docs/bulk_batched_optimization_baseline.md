# GPU XTC parser and bulk read: optimization checkpoint

2026-09-23. Functional integration is complete for the validated JF-only,
ordinary GPUBAT1 event loop, one EB/one BD/one A100, KvikIO CPU fallback.
No missing core parser/bulk integration component or new correctness blocker
was found in this checkpoint review. Performance optimization starts here.

## Code checkpoint

Branch: codex/psana2-gpu-bulk-batched-integration.
Validated runtime: ac87a93b2. This checkpoint adds tests and reports only.

- 786cd16ca: allocation ownership, memory accounting, retirement fixes (D+).
- 25e07ad40: merge preserving D+ and B++ histories.
- ac87a93b2: canonical gather across independent resident/transient input owners.

Frozen integrated runtime hashes and all Stage 3 test hashes were rechecked
and remain unchanged. No runtime edit, default change, or new benchmark was
made during this review. This checkpoint records the Stage 3 tests and reports
before subsequent production optimizations. Unrelated worktree files are
excluded. Scratch also preserves the pre-commit source snapshot.

## Component review

| Piece | Current implementation/evidence |
|---|---|
| GPU XTC parser | Configure-derived numeric schema; GPU XTC walking and field location; no parser-metadata D2H on the canonical path |
| B++ batching | Stream-grouped locator initialization/location, shared readiness, lazy wrappers, one canonical gather per detector execution |
| Bulk I/O | Adjacent range coalescing, logical event/stream identity, file/chunk epochs and read fences; bulk off/on selection |
| Input residency | Independent InputWindow owners, resident/transient composition, parse/location once per window, gather at execution lifetime |
| Multi-owner gather | Owner-specific raw/locator addresses, local rows, capacity/count checks, missing-field zeroing; calibrated uint16 and float32 passthrough tests |
| Device accounting | Owned backing remains charged through aliases, full replacement reservations, admission holds, fixed/parser/reader/output/map costs, pressure rejection before I/O |
| Lifetime and failures | Explicit producer/consumer events; delayed consumers; retry after failed drains; partial submission cleanup; early close/max_events; BeginStep/EndRun |
| D2H | Calibrated-result pipeline with completion-linked slot reuse; tail/chunk and early-exit validation |
| Benchmark baseline | Frozen standalone B++, integrated off/on; one allocation; two warm/cold rounds; separate correctness/profile diagnostics |

## Validation evidence

Stage 2: 323 CPU tests, 61 GPU tests, and 12 focused memcheck cases (zero
errors); 66 distinct GPU cases across those overlapping suites. Five real
JF cases matched all timestamps and 15 raw/calibrated samples each, with
2,112 reconciled allocation checkpoints. Retained-facade cases reached zero
live allocations before diagnostic GC/synchronization. Main psana: 390
passed, 68 skipped, 10 deselected; MPI byhand: four passed.

Stage 3: 323 CPU tests; 28 focused GPU tests, including 13 additional
lifecycle/growth/transition cases; those 13 passed memcheck with zero errors.
No production change was needed. Initial transition-fixture failure and
correction are documented in the Stage 3 report. Suite counts overlap.

Stage 4 job 38904896: completed successfully on sdfampere001 in 1h04m23s.
12 clean timings, three CPU-reference preflights, three full-window pixel/
profile diagnostics; cache, placement, runtime hash, and result audits passed.

## Performance checkpoint

JF mfx101210926 run 387, 10,000 events, batch 20/depth 1, 8 GiB budget,
no automatic/user D2H in clean timing. Rates are events divided by median
elapsed time; each entry has two rounds.

| Variant | Warm seconds | Warm events/s | Cold seconds | Cold events/s |
|---|---:|---:|---:|---:|
| B++ | 27.326 | 366.0 | 70.996 | 140.9 |
| Integrated off | 34.126 | 293.0 | 73.208 | 136.6 |
| Integrated on | 33.598 | 297.6 | 111.328 | 89.8 |

Integrated off is about 3% slower cold versus B++; the paired warm difference
ranges from -2.6% to -33%. Historical B++ native-binary hashes differ from the
integrated build, so these timings compare complete builds and do not isolate
Python integration overhead. Bulk on is consistently about 34% slower cold
than integrated off; warm off/on ordering changes between rounds.

All three variants retain 500 walks, 500 locator initializations, 500 grouped
locations and 500 canonical gathers over 500 executions; zero eager wrappers.
These counters cover those named kernels, not all CUDA launches. Integrated
on reduces reads from 50,000 to 2,885. Both integrated variants reach zero
CuPy used memory at loop end; peak used memory is about 3,201.8 MiB.

D/C+/D+ job 38898399 is a separate, user-closed two-round ownership comparison.
It showed no consistent D+ versus D regression and resolved the old live-memory
excess. Its different allocation must not be pooled with job 38904896.
Job 38904093 was discarded and contributes no results to this checkpoint.

## Next optimization order

1. Use the saved profiles to investigate warm variability and host submission,
   admission, ownership/map construction, read wait, drain, and trim costs.
2. Remeasure integrated allocation turnover. The historical 206 versus 99,507
   allocation-call counts belong to pre-integration D+; they are not counts for
   the integrated runtime and are not necessarily fresh cudaMalloc calls.
3. Preserve useful reader/parser/output capacity across bulk batches. Trim
   selectively for admission pressure; keep every live/retained byte charged.
4. Improve resident-read/compute overlap and avoid unnecessary drains while
   preserving transition barriers and consumer completion requirements.
5. Revisit residency width/depth choices and then remaining calibration/cleanup
   launch batching as separately measured changes.

Keep bulk off/on explicit in benchmarks. The API currently defaults
`gpu_bulk_read=True`; this review does not change it. The performance evidence
supports keeping bulk off available and does not justify claiming a bulk
speedup yet. Performance targets remain open despite functional completion.

## Scope boundaries

This is not acceptance of every psana GPU configuration. True GDS, multi-BD
sharing and mixed/hybrid workloads need their own validation; multi-EB device
coordination remains a known issue. GPU RunParallel.steps() is not implemented
(the events() path handles BeginStep). Bulk intg_det/timestamp filtering and
GPU smd_callback combinations are explicitly rejected. Common-mode calibration,
automatic image results, general user outputs, and a separate pinned-host byte
admission policy remain outside this integration. Automatic D2H covers dense
calibrated results, not arbitrary raw/parser fields.

## Reports and artifact locations

- [Stage 2 findings](bulk_batched_stage2_findings.md).
- [Stage 3 findings](bulk_batched_stage3_findings.md).
- [Integrated performance results](performance/bulk_batched_integration_sdf.md).
- [D+ ownership comparison](performance/dplus_ownership_comparison_sdf.md).
- [Broader known issues](known_issues.md).

Full timing artifacts and frozen builds:
`/sdf/scratch/users/m/monarin/gpu-validation/bpp-integration-performance-20260923/`.
Reviewed source snapshots and manifests:
`/sdf/scratch/users/m/monarin/gpu-validation/parser-bulk-checkpoint-20260923/`.
Stage 2/3 correctness artifacts are under the worktree's
`validation/bulk-integration-stage{2,3}-20260923/` directories.
