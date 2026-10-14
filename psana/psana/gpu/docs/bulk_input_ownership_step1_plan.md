# GPU bulk-input ownership and accounting: step-1 plan

Planning review, 2026-09-20. No fixes or integration implemented.

Verified target: `codex/psana2-gpu-xtc-parser`, HEAD
`8f94e3c7bc4643309d022cb5c1dd7e85805d3ac4`; tracked files were clean.
Verified optimized worktree HEAD:
`4c3cdf5a791f5c3801b03154bf696579d0b8be92`.
Existing untracked reports, harnesses, and logs were preserved. No applicable
AGENTS.md was found in the target's ancestor directories or psana subtree.
Read the psana2-gpu, psana, and gpu-cuda-python skills, current lifetime/issue
notes, all four requested optimized-branch documents, and the related task.

**Recommendation.** Make charges follow allocation lifetime, and make public
facades follow explicit access lifetime. CUDA completion permits reuse; it does
not prove that Python views have released a backing allocation. Clearing
`InputWindow.batch` is necessary cleanup but is insufficient on its own.
Implement this as a separate step on a future isolated branch based on the
bulk history. Keep residency policy, optimized-branch merging, and performance
work in their agreed later steps.

**1. Confirmed findings and remaining hypotheses**

The latest result files contain 24 primary, 48 mixed-detector, and 24 scaling
timing records, without double-counting continuation parents. The corresponding
directories contain 16 full diagnostic PERF_RESULT records. The saved bulk
diagnostics confirm:

| Case | Ledger + held peak, MiB | CuPy used peak, MiB | CuPy used at loop end, MiB |
|---|---:|---:|---:|
| Primary, batch 20, depth 1, 8 GiB | 3,201.795 | 8,325.500 | 5,123.687 |
| Mixed, batch 1000, depth 2, 8 GiB | 7,197.944 | 13,756.116 | 12,623.714 |

Sources: [primary diagnostic](../../../../validation/perf-acceptance-20260916/job-38419641-primary/D-diagnostic-bs20-pd1-mem8-bd1.log)
and [mixed diagnostic](../../../../validation/perf-acceptance-20260916/job-38419425-bulk/D-diagnostic-bs1000-pd2-mem8-bd1.log),
both line 39; [acceptance report](performance/parser_bulk_acceptance_sdf.md),
lines 264–301 and 384–437. The evidence index also lists repository-relative
artifact paths.

These are sampled used-pool values, so unused pool cache alone cannot explain
the discrepancy. They are not exact allocation peaks or simultaneous per-owner
balances. The harness samples after `reserve`/`hold`, often before the associated
allocation, and at loop end. It retains the last pixel-checked `gpu` facade
(normally event three) and the loop's final `evt`; see `bench.py:80–90,181–207`.
Those reference patterns must be reproduced explicitly.

Confirmed source paths:

- `gpu_input_window.py:93–115`: successful retirement clears the callback but
  keeps `batch`. `InputWindowUse.window` and `InputSlotLease._owners/_uses`
  also survive release.
- `gpu_input.py:125–156`: stream facades hold `owner.batch` directly; a
  single-owner event also holds it. `GpuFieldResult._views` caches raw-array
  slices. Clearing only the window cannot detach these references.
- `gpudgram/batch.py:216` and `parser.py:184`: the parsed batch retains the
  bound method `slot.locator_rows`, which retains that slot and its complete
  buffer cache, beyond just the rows visible in the batch.
- Reader, parser, and detector growth/trim release charges when their cache
  reference is replaced/dropped, without tracking other views:
  `gpu_kvikio_read.py:214–254`, `gpudgram/batch.py:85–102,294–302`,
  `gpu_detector.py:403–441`.
- `EventPool.finish_retire_next/flush` clear several input fields on their
  record, but retained event states, record result dictionaries, and leases
  can still reach arrays (`gpu_stream.py:112–137,243–268`).
- Input field methods check the event lease. Stream access checks the window,
  which can remain alive for later resident executions; direct `.batch`
  attributes and `DeviceFieldLocators.rows_gpu/wait_on` bypass an event lease.
  Ordinary `GPUResult` checks only its initial automatic-D2H flag, and
  `SlotLease` never becomes closed (`context.py:48–79,175–195`).

A read-only CPU probe loaded these source modules with NumPy arrays and weak
references. After closing a window and trimming reader/parser caches, committed
bytes were zero while a retained stream facade kept 192 backing bytes alive.
Clearing the probe object's `window.batch` still left all 192 bytes alive;
dropping the stream facade released them. The same probe retired a result
lease, overwrote its backing array, and observed the old `GPUResult.on_gpu`
return the new value. These establish Python ownership/access defects, not
real CUDA memory measurements. No source was changed by the probe.

Still unproven: how much of each historical multi-GiB gap comes from each
reference chain, temporary generator locals, warmup objects, exception
tracebacks, or allocator effects. Do not equate a matching byte multiple with
causal attribution. The execution-width collapse and lost overlap remain
separate step-3 and profiling questions.

**2. Current allocation and ownership trace**

```text
manager holds future input/parser/output growth before I/O
  -> reader reserves full raw allocation; PendingBatch retains its slice
  -> wait_batch drains futures; KvikioBatchRead retains PendingBatch via closure
  -> parse_window pins reader generation and selects a free parser slot
  -> parser reserves tables/locators; batch retains views and slot allocator
  -> InputWindow owns parsed input; execution/event leases acquire window uses
  -> detector reserves gather/output/presence buffers; result views share them
  -> result-ready and terminal consumer events establish completion
  -> leases close; window releases reader/parser reuse pins
  -> caches remain charged until growth/trim drops their current reference
       BUG: surviving allocation aliases can outlive that charge
```

| Phase | Current charge/release behavior | Step-1 requirement |
|---|---|---|
| Setup | Constants/geometry and Configure uploads reserve fixed bytes. IPC followers borrow constants. | Inventory owned versus borrowed allocations; charge owners once and report borrowed mappings separately. |
| Admission | `_reserve_gpu_subbatch` holds full replacement sizes for every growing array; existing capacity stays committed. `_start_resident_input` additionally holds execution progress. | Preserve pre-I/O holds and full replacement peaks; estimates and allocation helpers must use the same cost convention. |
| I/O | `_ensure_slot_buffer` consumes credit; pending futures and input pins prohibit overwrite. `wait_batch` drains all started futures before returning/raising. | Retain backing and charge through every future and published read/window alias. |
| Parse | `parse_window` acquires a raw pin; `_rows/locator_rows` allocate tables; release callback returns both pools to reusable state. | Separate reuse pins from allocation charges; bind every raw/table/locator view to its allocation owner. |
| Execute/deliver | Windows can serve many executions. Detector slots stay cached; event states/results retain slices. Child input views fork window uses. | Transfer references without duplicating charges; result completion and input completion remain independent. |
| Retirement | Uses transfer terminal events to the window, which waits and invokes release. EventPool waits result consumers. | Close access, drain recorded work, detach internal heavy references, then permit reuse. A still-live allocation remains charged. |
| Growth/trim | Full new size is reserved, then old cache size is released; trim drops free caches. | Removing a cache reference transfers old capacity to a retained category when aliases remain. Credit returns only when backing is relinquished. |
| Failure/close | Partial reads drain; parser sync failure retains `_failed_inputs`; failed execution can occupy an EventPool slot; fixed-upload helper can retain failed allocations. | Keep failed backing charged and unreusable until completion is proven; cleanup/retry must release exactly once. |

Audit setup exceptions too: `GpuXtcBatchPool.__init__` releases aggregate
Configure credit on exception without the explicit drain/retention used by
`gpu_calib._upload_fixed_arrays`. Test partial upload/event-record failure
before claiming that path safe. Also test `GpuEventManager.finish`: its `finally`
marks the manager closed even if an upstream drain failed; preserve reachable
failed owners and a defined retry or terminal-error path.

**3. Proposed invariants and smallest coherent implementation**

1. **One charge per backing allocation.** Introduce a small allocation owner
   and charge token in/next to `gpu_budget.py`, with allocation ID, category,
   device, requested capacity, allocator capacity, and state. Cache references,
   slices, reshapes, locator wrappers, and active consumers share that owner.
   The accounting registry must not itself retain the arrays indefinitely.
2. **Reuse and deallocation are separate.** Completion changes a live buffer
   from busy to reusable; its charge stays. Trim/growth removes the cache's
   reference, while retained aliases keep the old charge. Only last ownership
   release returns allocation credit. Keep strong owner references until CUDA
   completion; no destructor may infer completion or synchronize the device.
3. **Retirement is an access boundary.** Successful window/lease retirement
   detaches batches, bound allocators, cached array views, and consumer lists
   that are no longer needed. Preserve IDs, timestamps, field metadata, and
   independent host caches. Event facades reject fresh storage access after
   their event lease closes even if a resident window remains open elsewhere.
4. **An entered view has its own use.** Existing child uses remain valid until
   their terminal events complete. Acquisition/closure must be atomic. A saved
   unentered context cannot acquire storage after retirement. Once a view's
   context ends, its raw ndarray is no longer a valid reusable-data snapshot;
   if it remains referenced, its underlying allocation must still be charged.
   Plain CuPy arrays cannot retroactively reject indexing; guarded facades can.
5. **Reservations remain transactional.** Reserve old-plus-full-new peak before
   allocation or I/O; rollback only unallocated/failed work. Separate rollback
   into its originating hold from eventual allocation destruction. The current
   `release()` refunds whichever hold is active, and clamps over-release; do
   not reuse that behavior blindly for delayed allocation callbacks. Token
   release must be idempotent and ledger underflow must be observable.

Preferred allocation prototype: pipeline-created arrays retain an explicit
owner through their memory backing, rather than putting a finalizer on an
arbitrary ndarray facade. A possible implementation wraps a pool MemoryPointer
with `UnownedMemory(owner=...)`, retaining the original pointer in an acyclic
owner object. Validate that design with real CuPy before adopting it; retain
the default allocator and avoid a process-global allocator replacement.
Local CuPy 13.6.0 type inspection found no weakref slot on `Memory`,
`PooledMemory`, `MemoryPointer`, or `UnownedMemory`, so a direct
`weakref.finalize(memory, ...)` design cannot be assumed to work.

Apply the shared owner to reader `_ensure_slot_buffer/trim_free_buffers`, parser
`_rows/locator_rows/parse_window/trim_free_buffers`, and detector
`_slot_buffer/trim_slot_buffers`. Include fixed upload ownership, without
double-charging IPC mappings. Preserve every existing failed-input and
failed-execution drain.

Refactor `InputWindow._try_retire`, `InputWindowUse`, `InputSlotLease`,
`GpuEventDgrams`, `GpuStreamDgramView`, `GpuFieldResult`, and
`DeviceFieldLocators` so public access resolves through guarded ownership
instead of retaining unchecked batch references. Give standalone parser callers
an explicit independent lifetime contract; do not silently invalidate them
through a production EventPool lease.

For outputs, add closed-state/access enforcement to `SlotLease/GPUResult` and
detach result references through shared result state used by `GpuEventState`
and EventPool retirement/flush. Since these methods must change, collect all
terminal result consumers and register copy completion on its actual stream;
the existing single-event overwrite is a documented correctness defect.
Keep active view contexts pinned, and preserve automatic-D2H host delivery.

Add allocation-aware snapshots to `GpuEventManager`, including retained and
failed allocations. Correct detector category aggregation while touching this
reporting code; reconcile the optimized branch's corresponding reporting fix
at the later merge.

Keep the existing budget boundary for independent `on_gpu` copies and arbitrary
user allocations: report those separately in diagnostics. Aliases of
pipeline-owned backing are always charged, even if they escaped a context.
Extending admission to all independently allocated user results would be a
separate API/policy change. Preserve the current residency selection algorithm;
more accurate accounting may legitimately reject a previously undercounted plan.

**4. Focused reproductions and tests**

Extend the current GPU unit/integration tests with behavioral assertions, plus
allocation identity and actual pool observations. Existing admission tests
compare the ledger with cache inventories, so both can omit the same aliases.

| Case | Required observation |
|---|---|
| Repeated windows | Fixed-size and 1→4→2 growth sequences, bulk off/on, with no saved facade, final event saved, event-three facade saved as in the harness, and all facades saved. Retired metadata must not retain unnecessary backing; retained allocations remain charged. |
| Facades and views | Independently retain event, stream, batch, locator, field result, detector result, entered/unentered context, slice/reshape, and raw ndarray alias. Guarded access fails after retirement; valid child views and host caches keep working. |
| Delayed consumers | Two streams complete in the opposite order to registration; exercise field/result views, copies, and automatic D2H. No overwrite or charge release before every required completion. |
| Safe reuse | Same capacity reuses the same allocation without a second charge; stale facades cannot observe the next event. A resident input survives several transient executions and preserves its own local rows. |
| Growth | Retain old generations through several replacements. Charge all live old backing plus full new allocations, reject insufficient credit before allocator/I/O, and release each old charge only once. |
| Trim | Trim with pending I/O, live window uses, active execution/output consumers, retired facades, and escaped array aliases. Cache totals may fall while retained totals remain; committed must reflect both. |
| Failure | Fail each allocation, partial read submission/completion, parser/config upload, locate/gather/calibration launch, consumer-event recording, and synchronization. No reuse/credit return on unproven completion; retry is safe; unused holds return exactly once. Include partial multi-owner acquisition. |
| Lifecycle boundaries | BeginStep, EndRun, early iterator close, max_events, missing streams, and tails. Drains preserve event order and cached host delivery without orphaned charges. |

Start with `unit/test_gpu_input_window.py`, `test_gpu_input.py`,
`test_gpu_admission.py`, `test_gpu_bulk_read.py`, `test_gpu_result_lifetime.py`,
and `test_core.py`. Extend `integration/test_gpu_input_window_device.py`,
`test_gpu_admission_device.py`, and `test_gpu_residency_device.py` with real
allocation/lifetime checks. Follow with exact raw/calibration and public field
acceptance, then the main psana and `byhand_*` groups required for a core change.
GPU correctness runs must use an isolated verified install and new artifact
directory. No GPU jobs or implementation tests were launched during this review.

**5. Measurements and step-1 acceptance**

Log allocation IDs and transitions at reserve, allocation success/failure,
replacement, trim, lease close, completion, and actual ownership release.
Record committed and held separately, owner category/state, capacity, window
and execution IDs, consumer counts, and pool used/total/free. Count shared
backing once. Use bounded trace storage or compact transition records so
diagnostics do not become another owner of the arrays.

Track requested bytes versus actual allocator block sizes. Choose and test a
conservative preallocation rounding model for the supported allocator; verify
the returned size and account for rounding explicitly. CuPy rounds allocations,
retains unused blocks, and CUDA context/library allocations exist outside the
pool ([CuPy 13.6 memory documentation](https://docs.cupy.dev/en/v13.6.0/user_guide/memory.html)).

At matched checkpoints reconcile:

```text
committed = fixed owned + cached/live input/parser/output
          + detached but retained backing + failed/quarantined backing
committed + held + declared headroom <= configured budget
pool used = managed pool allocation blocks + separately observed external blocks
pool total = pool used + pool free
device usage = pool residency + non-pool allocations/context/IPC/other activity
```

Use actual charged block capacities, or explicitly include measured rounding
in the headroom reconciliation; never silently compare logical bytes with
rounded pool bytes. Admission holds represent future allocations and are not
part of pool used. Device-wide samples are contextual observations, not the
per-process ownership ledger. Pinned-host storage is a separate category.

Capture setup baseline, post-allocation, post-consumer completion, post-retire,
post-trim, and final owner teardown. Compare no retained facade with the exact
historical retention pattern. Observe before and after explicit reference
deletion; a diagnostic-only GC pass can distinguish cycles, but normal progress
must not depend on forced GC, global synchronization, or pool clearing.

Acceptance requires:

- Every managed live allocation has one charge; every charge has an identified
  allocation or documented temporary reservation. No unexplained positive
  used-memory residual in controlled tests beyond measured external allocations
  and explicitly bounded allocator rounding.
- Fixed-shape repeated windows reach a bounded steady state. Retained metadata
  alone does not accumulate device backing. Valid retained owners have an exact
  explainable cost and apply pressure rather than silently escaping admission.
- Growth peaks, failure quarantine, and reused capacities reconcile at the
  allocation transition, not just at sampled high-water marks.
- After all variable owners/views and work finish and caches are trimmed,
  managed variable live bytes return to zero. Fixed storage persists until
  its owner closes. Free pool retention may remain and is reported separately.
- Correct pixels, lifetime/access behavior, and preserved failure drains pass
  with bulk off and on. No throughput claim is needed for step 1.

Initially reproduce with small fixtures, then short bounded diagnostics of the
primary and mixed retention patterns. The later performance campaign and BD
scaling remain gated on this reconciliation.

**6. Risks and questions for later integration**

- Validate the backing-owner representation across real CuPy views,
  MemoryPointer-based arrays, allocator reuse, and supported interop exports.
  Raw integer pointers cannot convey Python ownership; low-level users need an
  explicit owner/use contract. Avoid cycles and Python-shutdown callbacks that
  perform CUDA work.
- Output-context pinning can expose user-held-view stalls. Report the actual
  owner; do not block the same caller indefinitely waiting for its own open
  context or silently overwrite it. Keep this lifecycle behavior explicit.
- Honest accounting can make current tight-budget plans fail. That is evidence
  for step 3, not permission to loosen charges or change residency priority here.
- The merged gather map must be `(event, stream) -> (owner, local row)`, with
  owner-specific raw bounds, locator base/capacity, and independent lifetimes.
  `event.batch` is already `None` for multiple owners; preserve that distinction.
- Location remains input-window work; gather and its owner/row-map uploads
  remain execution work. Hold every owner and pinned upload source through the
  terminal event, including partial submission failure. Avoid double-counting
  repeated owners in a gather or releasing a resident owner after one execution.
- At merge, consume `configured_locations().ready` even with zero lazy
  wrappers, preserving distinct fallback-locator and consumer dependencies.
  Add combined locator, fallback, device map, fixed routing, and pinned map
  costs. Preserve the rectangular layout initially and all bulk full-size
  reservations and failure drains. Do not substitute older optimized-branch
  lifetime methods wholesale.
- IPC ownership and device-wide context/cache overhead still limit claims about
  multiple BDs. The separate multi-EB GPU placement issue remains outside this
  step. The optimized timing medians come from another allocation and do not
  establish a stable 1.84% wrapper speedup or predict integrated throughput.

**Evidence index (repository-relative paths)**

- `psana/psana/gpu/docs/performance/parser_bulk_acceptance_sdf.md`
- `validation/perf-acceptance-20260916/bench.py:36,80,181`
- `validation/perf-acceptance-20260916/job-38419641-primary/results.json`
- `validation/perf-acceptance-20260916/job-38419425-bulk/results.json`
- `validation/perf-acceptance-20260916/job-38419641-scale/results.json`
- `validation/perf-acceptance-20260916/job-38419641-primary/D-diagnostic-bs20-pd1-mem8-bd1.log:39`
- `validation/perf-acceptance-20260916/job-38419425-bulk/D-diagnostic-bs1000-pd2-mem8-bd1.log:39`
- Optimized worktree `psana/psana/gpu/docs/bulk_batched_integration_plan.md`,
  `batched_pre_bulk_review.md`, `batched_canonical_gather_review.md`, and
  `performance/lazy_locator_wrappers_sdf.md`.
