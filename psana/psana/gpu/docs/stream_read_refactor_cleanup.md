# Stream-read refactor: review and deferred cleanup

Last reviewed: 2026-09-24, Stage 1, based on `84c49cdce` plus uncommitted
planner/preview changes. This is the persistent cleanup checklist for later
stages. Entries below are migration candidates, not authorization to delete
live code or benchmark evidence now.

## Stage 1 review outcome

No remaining blocker found for the CPU-only request planner and real-SMD
preview. It is not yet a production scheduler or a runtime memory reservation.

Review fixes:

- Missing event entries in `fence_by_event` now raise a descriptive ValueError
  before planning, rather than leaking a KeyError. Added a regression test.
- The source-only unit-test loader uses a private module alias so collection
  does not overwrite `psana.gpu.gpu_stream_read_plan` for other tests.
- The preview stats the BigData file before opening the SMD fd, avoiding a
  leaked descriptor if the data file is missing.
- The planner docstring explicitly distinguishes batch-local dependencies
  from the runtime credit that must survive across EB batches.

Validation: **60 tests passed** (14 Stage 1 and 46 existing read-plan cases).
The original real-data preview remains historical evidence from its recorded
source hash; post-review replay matched all ten complete plans exactly: 5,019 requests
and 33,566,911,424 bytes.

## Required integration checks

1. **Independent, nonblocking reclamation.** `InputWindow._try_retire()` in
   `gpu_input_window.py` synchronizes every completion event after the last
   reference is released. Reusing that call directly in the scheduler would
   block the BD on one slow group. Stage 2 needs a readiness-query/try-reclaim
   path and a collection of pending groups that can be polled independently.
   Keep explicit blocking drains for shutdown and error cleanup. Preserve
   planned uses, all consumer tokens, failure-safe retention and exactly-once
   release; a refcount reaching zero alone does not establish GPU completion.
2. **No ordered-retirement shortcut.** A group's newest timestamp and the
   completion of a later group do not establish its readiness. Event identity
   and delivery ordering are distinct from input-buffer reclamation ordering.
3. **Small-stream credits span batches.** `after_group` is batch-local and
   group identity is `(batch_id, group_id)`. An empty dependency on a new
   batch's first small group does not permit reading it while an earlier
   batch's small group still has planned or active consumers.
4. **Avoid availability deadlocks.** With one outstanding small group, an
   execution batch must not wait for the next small group while holding the
   planned uses that prevent its predecessor from retiring. Split execution
   at available small-group coverage where necessary. Keep batched kernels;
   recheck B++ launch counts after integration rather than assuming them.
5. **Progress bound is not peak memory.** The planner checks one largest raw
   request per stream. Runtime must reserve actual input concurrency, parser
   tables, raw/calibrated results, scratch, retained consumers and allocation
   growth under the shared budget. Never issue every listed group at once.
6. **Plan order is not worker placement.** KvikIO still splits large requests
   and defers sub-threshold reads to future retrieval. Stage 3 must honor
   dependencies while skipping blocked streams, then verify actual overlap
   with native traces. The CPU preview does not prove runtime concurrency.

## Cleanup inventory and removal gates

Paths in this table are relative to `psana/psana/gpu`, except tests.

| Area | Existing code or overlap | Later action and gate |
|---|---|---|
| Residency ranking | `gpu_admission.py`: `_ResidentCandidate`, `_ResidencyDecision`, `_resident_candidates`, `allow_residency`, resident fields in `AdmissionPlan` | Replace after Stage 3 uses bounded stream groups. Preserve minimum-event fit checks, detector/parser cost accounting and execution-budget splitting. Do not delete the entire admission module blindly. |
| Resident orchestration | `gpu_events.py`: `_start_resident_input`, `_close_resident_input`, `_resident_window`, `_resident_streams`, extra input slot at `event_pool.depth` | Remove after group owners supply all event inputs and drain/error tests pass. Remove resident/transient branches in `_submit_gpu`, `_issue_gpu_read`, `_wait_gpu_read` and EB-boundary setup/teardown together. |
| File-major bulk planner | `gpu_kvikio_read.py::_coalesced_plan` and its use of `gpu_read_plan.build_read_plan` | Retire the old production bulk branch after the new scheduler passes acceptance. Preserve file/chunk resolution, transition fences and the per-dgram bulk-off reference path. |
| Shared descriptor validation | `gpu_stream_read_plan.py` currently builds and discards an old `ReadPlan` to reuse validation; imports private `_uint64` | Extract shared validation/types when wiring the new production path. Avoid constructing two plans per runtime batch. Keep duplicate, timestamp, overlap and integer checks. The old planner remains useful for comparison until its callers are audited. |
| Reader storage | `gpu_kvikio_read.py`: `_slot_bufs`, `_input_holds`, `_generations`, single packed `PendingBatch.data_gpu` | Adapt to independent group backing in Stage 2. Preserve generation checks, pending destination/file ownership, full future draining and read-error poisoning. Never drop these protections as residency cleanup. |
| Parsed input owners | `gpu_input_window.py`, `gpu_input.py` references/completion leases | Reuse or adapt; these are not obsolete. Add nonblocking reclamation and retain zero-copy field lifetime protection. |
| Parser/gather implementation | `gpudgram` pools, `gpu_detector.py` canonical gather maps and kernels | Preserve B++ batching, owner mappings, reuse and lazy field access. A new I/O group must not automatically become a separate parser/calibration kernel launch. |
| Residency tests | `tests/gpu/unit/test_gpu_admission.py`, `tests/gpu/integration/test_gpu_residency_device.py`, `tests/gpu/unit/test_gpu_residency.py` and resident cases in `test_core.py`/`test_gpu_retirement.py` | Replace policy-specific ranking expectations with group bounds/fairness tests after migration. Retain byte-budget, missing-event, transition, lifetime, failure and pixel checks. |
| Timing hooks | `scripts/bulk_phase_timing.py`, `summarize_bulk_phases.py` | Update resident method hooks and phase labels when those methods change; retain old frozen campaign scripts for interpreting baseline evidence. |
| I/O diagnostic wording | `gpu_events.py` fallback/GDS startup messages hardcode NVMe and infer causes from compatibility mode | Replace with backend-neutral storage wording when updating I/O diagnostics. Compatibility mode alone does not identify the filesystem, cache state or reason GDS is unavailable; the Weka benchmark already demonstrates why the current message is misleading. |
| Earlier policy diagnostics | Untracked `scripts/compare_admission_priority.py`, `trace_bulk_reads.py`, `run_trace_bulk_reads_{sdf,perlmutter}.sbatch` | They exercise/override the existing residency policy. Decide whether to archive or port after Stage 3. They were intentionally excluded from the latest benchmark commit; they are not evidence of the new scheduler. |
| Deferred materializer proposal | `docs/proposals/detector_materialization_ownership.md` | Keep marked deferred. This refactor retains leased XTC field views; do not accidentally reintroduce mandatory field copies or duplicate JF gathering. |
| Preview bootstrapping | `scripts/preview_stream_read_plan.py --planner-source`, private source-only test loading | Revisit once the new planner is installed normally. Keep the preview's limitation explicit: aligned six-stream, c000 benchmark metadata, not a general live EventBuilder implementation. |
| Scratch duplicates | Frozen benchmark helpers, installations, native traces and Stage 1 source snapshots | Preserve until accepted evidence is durably retained and dependencies are checked. Repository benchmark helpers are now maintained; frozen copies intentionally identify prior results. No bulk scratch/log deletion as part of code cleanup. |

## Stage gates

- Stage 2: independent input ownership and completion polling, with CPU and
  device lifetime/failure checks; production residency can remain temporarily.
- Stage 3: switch the runtime scheduling policy and remove superseded branches
  only after all event-input construction, drains and error paths migrate.
- Stage 4: correctness, retained-view, out-of-order completion, tight-budget,
  transition and early-exit acceptance.
- Stage 5: cold Weka traces/controls with verified eviction, then warm/10k
  acceptance. Finish policy-test and timing-tool cleanup before committing the
  completed migration. Preserve historical benchmark summaries and provenance.
