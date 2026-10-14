# Stage 2: allocation-backed input/parser accounting

Implemented and measured 2026-09-23 on
`codex/psana2-gpu-allocation-ownership`, based on bulk HEAD
`8f94e3c7bc4643309d022cb5c1dd7e85805d3ac4`. Changes are uncommitted.
The optimized B++ history has not been merged.

## Result

Reader bytes, parser tables, and locator rows now retain one budget charge
until their backing allocation is relinquished. Trimming or replacing a cache
entry no longer returns credit while another view still owns that allocation.

The JF diagnostic independently reconciles these charges with CuPy allocation
hooks at **all 216 checkpoints**, including detached generations. There are
**zero unexplained reader/parser bytes**. The first three raw/calibration
images and all 1,000 timestamps pass in each of the bulk-off/on runs.

This is an intermediate accounting fix. It does not reduce retained memory:
the observed allocation peaks remain **3,201.874 MiB off** and
**8,325.501 MiB on**. Output/fixed-storage migration, retirement access guards,
and reference cleanup remain Stage 3. This branch is not ready for integration
or performance acceptance.

## Code changes

- `gpu_allocation.py`: `owned_empty` attaches an acyclic owner to CuPy
  `UnownedMemory`. The owner retains the original pooled array and its charge.
  Slices, reshapes, MemoryPointer arrays, and the tested DLPack export preserve
  that owner. Its destructor drops the pooled backing before returning credit;
  it submits no CUDA work and performs no synchronization.
- `gpu_budget.py`: allocation tokens distinguish construction rollback from
  eventual destruction. Rollback restores the original admission hold; later
  destruction never credits an unrelated active hold. The scalar-only
  allocation inventory identifies category, requested bytes, and capacity,
  without keeping arrays alive.
- `gpu_kvikio_read.py`: raw allocations use the helper. Growth and trim drop
  cache references; live aliases keep their charges. Cache reports use actual
  backing capacity.
- `gpudgram/batch.py`: parser tables and locator allocations use the same
  mechanism. Admission requirements reserve full rounded replacements, even
  when logical growth fits the same 512-byte size class. Reuse costs zero.

The supported allocator is CuPy's default memory pool. Admission reserves its
512-byte block capacity before allocation and verifies the returned block
size. Custom CUDA allocators are rejected before allocation; supporting them
requires a separate capacity contract. NumPy-backed CPU fixtures use weak
finalizers to exercise the same charge semantics without CUDA.

Allocation ownership does not imply safe overwrite. Existing pending-I/O,
window, and consumer leases remain responsible for completion/reuse. No
residency-policy changes, global synchronization, pool clearing, or normal-path
forced garbage collection were added.

## Validation

Python 3.9, CuPy 13.6, A100, KvikIO CPU fallback. Tests use an isolated copy
of the frozen bulk installation with the Stage 2 Python modules overlaid;
native binaries are unchanged. The JF job verifies the installation hashes
before and after execution. This does not establish true-GDS performance.

| Check | Result |
|---|---|
| Complete GPU CPU-unit directory | 304 passed; 2 strict expected failures |
| Initial CuPy owner prototype, job 38893937 | 5 passed |
| Final reader/parser/device regression set, job 38894285 | 14 passed |
| JF off/on retention diagnostic, job 38894278 | Both passed; 216 checkpoints reconciled |
| Whitespace/diff validation | Passed |

Device coverage includes escaped raw/locator aliases after real parsing and
trim, slice/reshape/pointer/DLPack ownership, full replacement reservations,
wrapper failure rollback, custom-allocator rejection, zero-sized allocations,
exact-admission growth 1→4→2, delayed input consumers, bulk/per-dgram equality,
and mixed-rate resident input reuse. CPU failure injection verifies that all
started read futures drain once and retained failed backing stays charged
after cache removal.

The first broader GPU run (38894243) had 12 passes and two failures in existing
mixed-rate test budgets that omitted pool rounding. The final run adds the
exact rounding cost of one resident and two transient input/parser sets to
those test budgets, preserving the original logical residency plan and its
five two-event slow reads. No production budget was enlarged.

### Two intentionally unresolved tight-budget regressions

`test_tight_budget_preserves_all_events_and_each_input_once[3500/1500]` now
raises `GpuMemoryPressureError`: retired envelopes still retain previous input
generations. Their old success depended on returning credit for live backing.
Both original success assertions remain under strict `xfail`, restricted to
that exception, with an explicit Stage 3 reason. This is a behavior regression
in the intermediate branch, not a completed success gate.

Companion tests verify committed+held never exceeds either limit, pending
reads drain, retained input remains charged, and deleting remaining aliases
releases all tracked reader allocations. Stage 3 must remove the unnecessary
retention and restore the original success tests, without weakening charges.

## JF measurements

Job 38894278, sdfampere002, completed in 1m55s including staging. One SMD0,
one EB, one BD, one A100; run 387, batch 20, depth 1, budget 8 GiB; JF only,
eight KvikIO workers, 1 MiB tasks, automatic D2H disabled. Each fresh MPI
launch retains event-three and final-event GPU facades as in the historical
diagnostic. These are diagnostic runs, not throughput measurements.

| Mode | Checkpoints | Loop-end reader/parser charges, MiB | After dropping final facade | After dropping event-three facade |
|---|---:|---:|---:|---:|
| Bulk off | 58 | 641.814 | 641.814 | 0 |
| Bulk on | 158 | 1,283.628 | 641.814 | 0 |

At the largest matched bulk-on trim checkpoint, the ledger now retains
**1,283.628 MiB** of detached input/parser backing. Total committed is
**1,923.686 MiB**, compared with **640.058 MiB** in Stage 1. Pool used remains
**5,763.687 MiB**. The remaining discrepancy is detached detector output
(3,840 MiB) plus 1,112 bytes of legacy rounding.

Dropping all saved facades returns actual pool-used memory to zero before the
diagnostic GC call. Allocation tokens keep the budget object alive after the
manager dies, so input/parser inventory remains valid then. Legacy aggregate
fixed/output counters can be stale after manager teardown; do not interpret
the whole committed counter at that point as reconciled live storage.

## Stage 3 handoff

1. Extend allocation-backed ownership to output and fixed allocations,
   including failed uploads and owned versus borrowed IPC storage.
2. Close stale facade access, preserve valid child consumers, and detach
   unnecessary batches, bound allocator callbacks, and result references.
3. Fix the iterator/envelope and active-exception retention chains identified
   in Stage 1; restore the two tight-budget success tests.
4. Preserve completion of every consumer and retain failed owners until
   completion is proven. Then run Stage 4 whole-ledger/end-to-end acceptance.

## Evidence

All paths below are relative to `validation/ownership-stage2-20260923/`:

- `unit.log`: complete CPU-unit results.
- `prototype-38893937.log`: initial CuPy ownership representation checks.
- `device-38894243.log`, `device-38894285.log`: initial/final GPU regression runs.
- `job-38894278/summary.md`, `summary.json`: JF summary and reconciliation.
- `job-38894278/{off,on}-third.jsonl`: complete allocation traces.
- `job-38894278/provenance.json`, `build.json`: environment/build/script hashes.
- `source-manifest.json`: final source/test hashes and production install match.
- `README.md`: reproduction commands and validation scope.
