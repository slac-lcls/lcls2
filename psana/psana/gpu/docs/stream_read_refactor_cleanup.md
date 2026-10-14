# Stream-read refactor: completion checklist

Last reviewed: **2026-09-25**, after cleanup commit `cc4451b3c`.
The four follow-up items agreed after the 10,000-event comparison are complete.
Correctness and cleanup acceptance passed; throughput improvement has not been
established. Performance work remains deferred.

## Follow-up completion

These item numbers refer to the recent follow-up list, not the original
implementation stages or the campaign later named “Stage 4 performance.”

| Item | Status | Evidence |
|---|---|---|
| 1. Review and commit outstanding runtime, test, benchmark and report changes | Complete | Transition drains/configuration `3ca412d70`; direct group submission/trace attribution `717c7c9a3`; audited harness and 10k report `d4cd86d10`. |
| 2. Broader GPU ownership acceptance | Complete, `4821499c0` | [Ownership report](stream_read_ownership_acceptance.md): 18 new A100 cases for retained public views, 4/8 MiB budgets, depth 1/2, D2H, and independent delayed consumers; 38 tests passed in that campaign. |
| 3. Remove obsolete code after mapping replacement coverage | Complete, `cc4451b3c` | [Cleanup report](stream_read_legacy_cleanup.md): residency policy/orchestration and reader bulk adapter removed; fixtures and timing hooks migrated. Final acceptance: 430 CPU checks (406 unit + 24 harness), 53 A100 tests, and timing-hook installation. |
| 4. Reconcile this checklist with completed optimization, ownership, 10k and cleanup work | Complete | Current status, retained contracts and deferred work are recorded here. Historical reports retain the scope and source identity of their own campaigns. |

## Three CPU-overhead optimizations

All three original profiling candidates are implemented. Their profiled savings
are local costs, not additive predictions of end-to-end speedup.

| Original candidate | Implementation | Evidence |
|---|---|---|
| Repeated raw-slot selection, about 0.60 s per 1k events | Per-planning-call capacity index, `23600736f`; preserves best fit, deterministic ties, busy-slot exclusion, stream credits and replacement reservations | [Slot-selection results](performance/stream_read_slot_index.md): over 93% less profiled selection time. |
| Pending-file ownership scans, about 0.40 s per 1k events | Bulk-on per-file reference counts, `c3357e622`; acquired handles remain live until all associated futures drain | [File-ownership results](performance/stream_read_file_refs.md), including short reads, partial submission failure and out-of-order completion. Applies after each completed group. |
| Legacy-plan reconstruction, about 0.25 s per 1k events | Direct group submission/shared validation, `717c7c9a3`; unused reader adapter and discarded stream-planner reference-plan construction removed in `cc4451b3c` | [Direct-group results](performance/stream_read_direct_group.md): 26–33% less profiled group-submission time; [cleanup coverage](stream_read_legacy_cleanup.md). |

The [original CPU profile](performance/stream_read_cpu_profile.md), job 39067790,
is the historical source of those candidate costs. It predates these changes.

## Longer-run acceptance and performance status

The [10,000-event comparison](performance/stream_read_current_10k.md), job
**39084570**, completed eight audited controls: bulk off/on, cold/warm, two
rounds with reversed order. All event, checksum, sampled pixel, payload,
request-count, cache, placement and provenance checks passed.

| Cache | Bulk off events/s | Bulk on events/s | Bulk-on median loop-time increase |
|---|---:|---:|---:|
| Cold | 134.15 | 127.26 | 5.42% |
| Warm | 304.32 | 245.88 | 23.77% |

Both modes used the same frozen runtime with all three optimizations, one A100,
one BD, batch 100, depth 1, 8 GiB GPU budget, 4 MiB bulk/task sizes and eight
KvikIO workers in CPU-fallback mode. Bulk on reduced requests from 60,000 to
50,182 without improving throughput.

This comparison **predates cleanup `cc4451b3c`**. The cleaned runtime passed
CPU and device correctness acceptance; it has not received a new 10k throughput
comparison. Correctness tests do not establish a performance result for it.
The cause of the remaining bulk-on overhead and historical warm variability
remains unresolved. No throughput win or true-GDS, live-data, or multi-BD scaling
acceptance is claimed.

## Cleanup disposition

Paths are relative to `psana/psana/gpu`, except tests.

| Area | Current disposition |
|---|---|
| Residency ranking and orchestration | Removed: candidate ranking/diagnostics, resident admission fields, `GpuReadSelection`, resident start/close and mixed resident/transient branches. Complete-event admission remains. |
| File-major bulk adapter | `_coalesced_plan` and the bulk branch of `issue_batch` removed. Bulk on uses resolved `issue_group`; bulk off retains per-dgram `issue_batch`. |
| Shared validation | Descriptor and overlap validation shared without constructing a discarded reference plan in the stream planner. Generic `build_read_plan` remains a CPU reference. |
| Residency fixtures | Removed after migration. Group tests retain byte-budget, identity, missing-event, tail, hybrid CPU payload, transition, failure and pixel assertions. The exact mapping is in the cleanup report. |
| Timing and native tracing | Maintained timing hooks use group issue/submit; all 41 patches install. File/range attribution and union wall time support overlapping native reads. Historical result formats and frozen scripts remain preserved. |
| Reader storage and parsed owners | Retained: slot generations, raw-input holds, `InputWindow`, input references and completion leases. These implement the active ownership contract. |
| Parser arenas and detector gathers | Retained: batched multi-base parsing, shared metadata, canonical gather maps, lazy field access and partial-setup failure quarantine/retry. |
| Old untracked policy diagnostics | `compare_admission_priority.py`, `trace_bulk_reads.py` and their launchers remain untouched historical tools requiring the old runtime. They are not current group diagnostics. |
| Scratch artifacts | Frozen installations, tests, traces, scripts and logs retained. No scratch/log deletion is part of this completion. |

## Contracts to preserve

- Reclaim each input group only after all planned uses and consumer completion
  tokens finish. A later timestamp or completed later group cannot release it.
- Carry small-stream credits across EB batches. Split executions at available
  group coverage so an execution cannot hold the credit needed for its own
  missing input. Skip blocked streams where independent work is available.
- Reserve actual rounded allocation growth, including old-plus-new replacement
  peaks, raw/parser/output storage, fixed/cache allocations and live aliases.
  The planner's progress bound is not a runtime peak-memory measurement.
- Keep parser/gather launches batched across independent read groups. Request
  ordering alone does not prove native worker concurrency.
- Drain dependent work before BeginStep calibration replacement and EndRun
  dispatch. Preserve ownership after a failed drain so retry remains safe.
- Keep bulk-off behavior and byte/pixel parity coverage. Public field contexts
  still conservatively retain every input owner for their event.

## Deferred and optional work

| Item | Status |
|---|---|
| Explain bulk-on overhead/warm variability and improve throughput | Deferred by the user. Resume with the maintained audited harness and frozen runtime; rerun cold/warm controls when performance work resumes. |
| Selective field-owner leases | Optional. Current contexts safely retain all event input owners; narrower leases require new source-selection and delayed-consumer tests. |
| Backend-neutral I/O diagnostics | Optional. Startup wording still mentions NVMe and infers causes from compatibility mode; that mode alone does not identify storage or explain why GDS is unavailable. |
| Preview bootstrap simplification | Optional. Preserve the preview's aligned six-stream, c000 metadata limitation; it is not a general live EventBuilder. |
| Detector materialization proposal | Deferred. Keep leased XTC field views; mandatory copies and duplicate Jungfrau gathering are outside this refactor. |

## Historical stage evidence

Original stage numbering is retained in report titles for provenance; none of
these older “remaining work” sections supersedes the current checklist.

- [Stage 1 planner](stream_read_refactor_stage1.md): CPU-only planning and
  real-SMD preview. Review fixed missing fences, isolated source-only test
  loading, and SMD-fd cleanup; 60 tests passed. Replay preserved 5,019 requests
  and 33,566,911,424 bytes across the ten recorded plans.
- [Stage 2 ownership](stream_read_refactor_stage2.md): independent input groups
  and nonblocking completion polling, with CPU and real-CUDA failure/lifetime
  coverage.
- [Stage 3 integration](stream_read_refactor_stage3.md): production group
  scheduling and batched multi-buffer parsing, followed by partial parser-child
  cleanup fixes. Its temporarily retained residency branches are now removed.
- [Stage 4 correctness](stream_read_stage4_correctness.md): controller lifecycle,
  transition drains and retry safety; subsequently extended by the ownership
  and cleanup campaigns above.
- [Stage 4 performance campaign](performance/stream_read_stage4_acceptance.md):
  earlier audited 1k controls/traces. Original “Stage 5 longer-run acceptance”
  is covered by the later 10k campaign for its recorded runtime; performance
  improvement remains deferred.
