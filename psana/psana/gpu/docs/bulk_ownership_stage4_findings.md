# Stage 4: JF ownership acceptance

Completed 2026-09-23 on `codex/psana2-gpu-allocation-ownership`, based on bulk
commit `8f94e3c7bc4643309d022cb5c1dd7e85805d3ac4`.

## Decision

**The single-BD JF ownership gate passes. Proceed with the B++/bulk integration
while preserving the Stage 2/3 ownership and failure-drain changes.**

No production fix was required in Stage 4. Runtime sources and native libraries
match the Stage 3 reviewed build. This establishes correctness and memory
ownership before integration; throughput acceptance comes after the merge.

## End-to-end results

One A100 allocation: job **38896070**, node **sdfampere019**, completed with
exit `0:0` in **20m 53s**. One BD, three MPI ranks, JF only, run 387 of
`mfx101210926`, streams 5–9, KvikIO compatibility mode ON. All five cases used
the same staged 10,000-event input and frozen calibration constants.

| Case | Events | Batch / depth | Budget GiB | D2H chunk | Saved facades | Peak live MiB | Loop-end live MiB |
|---|---:|---|---:|---:|---|---:|---:|
| Bulk off, long | 10,000 | 20 / 1 | 8 | 0 | all | 3,201.874 | 0 |
| Bulk on, long | 10,000 | 20 / 1 | 8 | 0 | all | 3,201.874 | 0 |
| Bulk off, partial tail | 1,003 | 13 / 2 | 8 | 7 | third and last | 3,970.476 | 0 |
| Bulk on, partial tail | 1,003 | 13 / 2 | 8 | 7 | third and last | 2,305.267 | 0 |
| Bulk on, tighter budget | 1,003 | 20 / 1 | 4 | 0 | all | 3,201.874 | 0 |

- Every ordered timestamp matched the independent CPU reference.
- Raw and calibrated image hashes matched for **23 samples per long case**
  and **15 per short case**, including batch boundaries, SMD boundaries,
  events 999–1,003, events 4,999–5,001, and the end of the long run.
- **127,206 checkpoints** reconciled actual CuPy allocations with budget
  tokens by identity, requested bytes, physical capacity, and category.
- Committed bytes equaled owned capacities. Committed + held + declared
  admission margin stayed within the configured budget. Pool used equaled
  managed plus independently observed external allocations.
- Both long cases held exactly **206 live allocations and 3,201.874 MiB** at
  every 200-event delivery checkpoint. Saving all facades did not increase
  live device storage.
- All five cases reached zero live allocations at loop end **while the
  selected facades were still retained**, before explicit GC or diagnostic
  synchronization. Retired GPU access was rejected; sampled CPU caches stayed
  readable. An additional GC check did not change the result.

Zero live allocations does not mean zero device residency. CuPy retained free
pool blocks at loop end: about 3,201.874 MiB in the long/tight cases,
3,970.476 MiB for the bulk-off D2H case, and 2,306.392 MiB for bulk-on D2H.
Pool free/total and contextual device-wide usage are recorded separately.

## Finding for the next integration/performance stage

The long runs have identical live-memory plateaus, but very different allocation
turnover:

| Allocation category | Bulk off | Bulk on |
|---|---:|---:|
| Fixed tables/constants | 7 | 7 |
| Reader | 1 | 500 |
| Parser | 195 | 97,500 |
| Detector outputs | 3 | 1,500 |
| **Total allocations, each eventually freed** | **206** | **99,507** |

`GpuEventManager._process_batch()` flushes the execution pool and calls
`_trim_gpu_caches()` before `_start_resident_input()` when admission chooses
resident streams. With these 500 batches, the bulk path therefore repeatedly
recreates all 199 variable allocations. The non-bulk path reuses them.

This is an allocation-reuse and batching candidate for the later performance
work. The counts do not establish a throughput penalty: this trace writes a
record/checkpoint for each allocation, so it adds much more overhead to the
bulk case. Preserve honest ownership accounting when changing cache reuse;
do not recover speed by dropping charges for still-live storage.

## Code and test changes in Stage 4

Production runtime code was unchanged. Added:

- A device test for retained generations through 1→4→2→8 output sizing,
  same-capacity reuse, exact live-pool reconciliation, and budget rejection
  before another allocator call.
- Three device fault-injection cases: fail allocation at each position in a
  three-table fixed upload. Earlier uploads drain and all charges/unused holds
  return correctly.
- Two CPU tests for partial multi-owner acquisition and partial view forking:
  failure on the second owner returns the first child reference and preserves
  the parent lease.
- `validation/ownership-stage4-20260923/`: scalar-only allocation tracing,
  independent CPU reference, configurable JF worker, one-allocation runner,
  automatic result/summary generation, isolated install, and provenance.

Test evidence:

- **312 CPU unit tests passed** in `unit.log` before the two additional tests.
- **9 input-window tests passed** in `partial-ownership.log`, including those
  two additions: **314 distinct CPU tests covered in total**.
- **20 A100 device tests passed** in `job-38896070/device.log`, including the
  four new growth/failure cases.
- Existing failure/drain, delayed-consumer, transition, missing-stream,
  early-close, retained-alias and retry coverage was retained.
- `git diff --check` and Python compilation passed.

The Stage 3 main psana and MPI byhand results remain applicable to the unchanged
core code (379 passed / 23 skipped / 10 deselected; 4 byhand passed). No core
implementation was edited in Stage 4, so these broader suites were not repeated.
The final Stage 3 saved-batch guard was already covered by its reviewed device,
unit, and JF runs; it was included unchanged in this longer campaign.

## Scope and handoff

This completes the agreed longer-run/failure-growth acceptance for the current
JF-only, single-BD integration gate. Multi-BD/IPC, true GDS, hybrid small
detectors, and throughput remain later acceptance scopes. Custom allocators
still need a capacity contract. Pinned-host allocations and non-pool CUDA
storage are outside this per-process device-pool reconciliation.

Continue with the optimized parser/gather integration, preserving allocation
owners, lease retirement guards, full replacement reservations, and failed-work
quarantine. Re-run the same acceptance matrix on the integrated build before
making performance claims. The cache-trim allocation turnover above supplies a
concrete follow-up measurement target.

## Artifacts

Repository-relative paths:

- `validation/ownership-stage4-20260923/job-38896070/summary.md`
- `validation/ownership-stage4-20260923/job-38896070/results.json`
- `validation/ownership-stage4-20260923/job-38896070/provenance.json`
- `validation/ownership-stage4-20260923/job-38896070/cpu-reference.json`
- Per-case `.log` and `.jsonl` files alongside those summaries.
- `validation/ownership-stage4-20260923/build.json` and `source-manifest.json`.
- `validation/ownership-stage4-20260923/README.md` for reproduction.

Installed hashes were checked before and after the job. All recorded runtime
source files also match the worktree. The copied install's standalone
`gpu/scripts/trace_bulk_reads.py` was refreshed from the worktree; this unrelated
diagnostic script is not imported by the acceptance runner. The two extra CPU
tests were added after GPU-job submission and their final hashes are recorded
in `source-manifest.json`; they were tested separately in `partial-ownership.log`.
