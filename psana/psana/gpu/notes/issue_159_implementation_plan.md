# Issue 159: alignment-safe GPU reads

Updated 2026-10-19. Status: PR #187 integrated locally; alignment implementation
has not started. This document records the agreed scope, staged implementation,
dependency expectations, and acceptance gates.

## Goal summary

Fix [issue #159](https://github.com/slac-lcls/lcls2/issues/159) while preserving
access to supported packed XTC fields, including unaligned fields. Prevent
misaligned typed GPU reads in dense gathering and direct event-loop field access.
Preserve exact values, field semantics, asynchronous lifetimes, and byte-budget
accounting. Measure overhead on already-safe data with a matched, same-node
staged-data A/B comparison.

Keep existing APIs. Use byte-wise reconstruction in dense gather and an aligned
direct view or cached aligned copy for field access. Do not reject a supported
field solely because its address is unaligned. Normal invalid-data, expired-lease,
and memory-capacity checks remain applicable.

Decisions are recorded in the [revised issue and follow-up](https://github.com/slac-lcls/lcls2/issues/159#issuecomment-6253752732),
following [Seema's review](https://github.com/slac-lcls/lcls2/issues/159#issuecomment-6202352705).

## Branch and integrated baseline

| Item | Identity |
| --- | --- |
| Working branch | `fix/gpu-unaligned-reads-159` |
| Checkout | `/sdf/home/m/monarin/lcls2_worktree/psana2-gpu` |
| Previous base | `c157b490ccb42940632ac83fce2a33e98335e9ce` |
| Previous-base backup | `backup/gpu-unaligned-159-before-pr187-20261019` |
| PR #187 branch | `features/psana2-ipc` |
| Pinned PR #187 head | `4b068a867d9164354ac7b7613d89812f06dc5ee6` |
| Integration merge / initial unfixed A | `90c05965cf34869388cc526a50e7e822b50a6eba` |
| Initial baseline ref | `baseline/gpu-unaligned-159-pr187-20261019` |

The merge preserves both histories and completed without conflicts. Its first
parent is the previous base; its second parent is the pinned PR head. Relative
to PR #187, the merge adds only the existing performance report and JSON evidence.
No PR #187 implementation was edited or repaired during integration. Existing
untracked investigation artifacts are unrelated and must not be swept into commits.

PR #187 was confirmed open and draft at integration. Its ancestry predates the
review-stack reconstruction: `9b32dda19` and `410ebba1c` have identical trees but
different subsequent histories. Avoid a plain rebase that replays the review
stack. Keep alignment commits above the recorded integration base.

## Scope and ownership

- **Dense gather:** test the actual source address; retain aligned typed reads
  and reconstruct unaligned uint16/float32 values from bytes within the existing
  kernel. Keep one thread per pixel and the existing aligned output allocation.
  Preserve bounds/type/shape/presence checks, invalid-row zero filling, and float
  object bits. Add no extra full-image buffer, pass, or launch.
- **Direct field views:** use original packed input when aligned. Otherwise
  safely copy only the requested field into aligned, budgeted device storage,
  then cache the borrowed view for its valid event/storage generation.
- **Actual address:** check pointer plus offset, including sliced input bases;
  offset divisibility alone is insufficient. Do not pad XTC records or round
  pointers past data.
- **Lifetime:** the parent event input lease retains cached copies independently
  of `_views`. Reclamation must wait for all relevant consumers, including child
  contexts overlapping parent retirement. Retain input through copy completion.
- **Streams:** materialize on the requesting context's stream after producer
  readiness. Publish a copy-ready event and wait on it for subsequent access
  from other streams. Preserve consumer completion tracking and failure safety.
- **Budget:** pass the manager's existing budget through event/field construction;
  use `owned_empty(..., budget, 'field-copy')`. Charge actual backing capacity
  once, including allocator rounding, through safe release. Cache hits do not
  add charges. Copies are private per BD, not IPC allocations.
- **Semantics:** retain dtype, original shape, physical segment IDs, scalar shape
  `()`, and omission of absent segments. Missing segments allocate nothing.
- **Copy accessors:** `.on_cpu` copies source bytes directly into aligned host
  storage, then interprets dtype/shape, preserving host independence and caching.
  `.on_gpu` initially uses the safe field path and makes an independent device
  copy; first unaligned access may make two copies. Measure before optimizing.
- **Documentation:** keep `on_gpu_view()` and explain at completion that first
  unaligned access may allocate/copy; it returns borrowed storage, not an
  unconditional zero-copy view of packed input.

Deferred: gathered-raw-buffer reuse; execution-slot ownership in field results;
API renaming/removal; migration of event-loop work into batched tasks; new generic
typed task-field convenience APIs; expanded dense-gather types; optional direct
`.on_gpu` copy optimization. These do not block the correctness fix.

## PR #187 anticipation

Seema owns [PR #187](https://github.com/slac-lcls/lcls2/pull/187). A separate task
summarizes its reviews. Use that task and the PR discussion to track resolution;
do not implement its review fixes in this branch. The following items are
dependency risks and anticipated changes, not new assignments or claims of
hardware reproduction by this task.

| Area | Current interaction and anticipated action |
| --- | --- |
| Placement | #187 selects the actual CUDA device and discovers peers. Allocate field copies on the correct device; update validation harness assumptions about device zero and removed placement helpers. Reverify after dependency updates. |
| Budget sizing | #187 sizes the existing manager budget from discovered peers and shared constants, and adds `set_limit`. Consume that final budget; do not calculate a separate allowance or modify sharing policy. Test private field-copy accounting for both constant owners and importers. |
| Shutdown | The [early-exit review](https://github.com/slac-lcls/lcls2/pull/187#discussion_r4289222022) identifies potential EB-traffic/collective-teardown ordering problems. Anticipate changes to close sequencing. Our copies and consumers must retire correctly without adding peer collectives. Record inherited failures separately and wait for Seema's fix where needed. |
| Scalar/empty constants | Reviews identify [scalar shape](https://github.com/slac-lcls/lcls2/pull/187#discussion_r4289222120) and [zero-byte IPC](https://github.com/slac-lcls/lcls2/pull/187#discussion_r4289222142) concerns. These concern calibration constants, separate from scalar/empty event-field handling in #159. Do not repair them here. |
| Calibration lifetime | The agreed model is one constant set per run. A review asks for refresh, but that conflicts with Mona's prior decision. Do not reintroduce per-step refresh as part of alignment work; follow the finalized #187 contract. |
| Validation | #187 replaces a calibration driver and has a [two-EB coverage review](https://github.com/slac-lcls/lcls2/pull/187#discussion_r4289222182). Use actual observed roles/device IDs and current entry points. A reported test count is not evidence that our integrated baseline passes. |
| Other limitations | The PR reports MIG limitations and a pre-existing multi-run hang. Keep these distinct from #159 and avoid widening this fix to resolve them. |

At the pinned head, `gpu_detector.py`, `gpu_input.py`, `gpu_stream.py`,
`context.py`, `gpu_allocation.py`, and `gpu_task_batch.py` are unchanged from our
previous base. Direct code conflicts should be limited; setup, budgets, device
selection, and teardown still require integrated validation.

### Updating the dependency

1. Record Seema's new exact head and review its delta before integrating it.
2. If history advances normally, integrate the update and retain an explicit
   unfixed baseline containing the same dependency version as the candidate.
3. If history is rewritten, build a fresh integration base and transplant only
   #159 commits; do not replay the old review-stack history or duplicate IPC work.
4. Rerun checks affected by placement, budgets, lifetime, or teardown changes.
5. After #187 merges, prepare the final #159 review against the updated feature
   branch with only alignment changes in the diff. Reconcile merge/squash history
   explicitly and verify the resulting tree.
6. Freeze final A/B revisions on the same stabilized #187 version. Provisional
   measurements on an older dependency must be labeled and cannot substitute
   for final acceptance where dependency changes affect the measured path.

## Revised implementation stages

Each stage keeps its implementation and focused regressions together. Stages
2-5 can proceed on the pinned dependency while Seema works; final integrated
acceptance depends on a sufficiently stable dependency and documented baseline.

| Stage | Work and likely files | Exit gate | Status |
| --- | --- | --- | --- |
| 0. Integrate and record | Preserve prior base, merge pinned #187, record provenance and this plan | Conflict-free merge; runtime matches pinned #187; latest performance documentation retained | Complete |
| 1. Establish baseline | Verify installed imports/environment, run existing focused tests, record inherited failures, audit real field addresses, prepare reusable fixtures and frozen benchmark inputs | Exact baseline/source/environment manifest and honest pass/failure/skip inventory; safe data identified for A | Pending |
| 2. Safe dense gather | `gpu_detector.py`; extend `test_batched_gather.py` / `test_raw_preparation.py` | Bit-exact aligned/unaligned results, unchanged presence/bounds behavior, no extra gather allocations/launches | Pending |
| 3. Copy retention | Minimal parent/child input-lease retention in `gpu_input.py`; focused ownership tests | Slow and open child consumers survive retirement; failed joins retain ownership; charges/storage release safely | Pending |
| 4. Safe field views | `gpu_input.py`, budget propagation through `context.py` / manager event attachment as needed; focused field/device tests | Aligned direct access, lazy cached copies, actual-address checks, cross-stream readiness, scalar/absence semantics, accounting and failure cleanup | Pending |
| 5. Copy accessors | Direct byte D2H for `.on_cpu`; independent `.on_gpu` through safe field resolution | Copies survive retirement, preserve shape/bits, and have the agreed copy/allocation counts | Pending |
| 6. Integrated correctness | Sanitizer, field/gather/raw/ownership/pixel-exact tests, serial and MPI lifecycle coverage on integrated #187, required core groups | Correctness gates pass; inherited dependency/environment blockers are explicitly separated and resolved or accepted by Mona | Pending |
| 7. Performance | Isolated operation measurements and matched staged-data A/B with one common frozen harness | Complete paired samples, correctness/cache/provenance gates, memory/copy counts, and investigation of repeatable safe-path regressions | Pending |
| 8. Documentation and review | API copy caveat, concise evidence report, final dependency reconciliation and fix-only review diff | Reviewable #159 change and reproducible validation; no unsubstantiated completion/performance claims | Pending |

## Regression matrix

- Gather uint16/float32: every address residue, independently shifted bases and
  offsets, mixed segments, exact end bounds, partial CUDA blocks/batches, slot
  reuse, invalid-row zero filling/presence, and bitwise float comparisons including
  signed zero, infinities, and NaN representations.
- Fields: supported numeric types with 1/2/4/8-byte alignment, scalars, arrays,
  ragged segment shapes, empty arrays where supported, and omitted absent segments.
  Check direct pointer identity/no copy, first-copy alignment and exact bytes,
  cached pointer reuse/no recopy, and no stale cache after event/slot replacement.
- Lifetimes: controlled delayed consumers after context exit, open child contexts
  across parent retirement, repeated cleanup, exceptions, and failed completion.
  Assert retained ownership/charges as well as final values; do not depend on
  allocator overwrite happening by chance.
- Streams: delay materialization on one stream while a second acquires its cached
  view; prove copy-ready ordering and completion tracking for both consumers.
- Accessors: host results without intermediate device materialization; independent
  device copies; modification isolation; survival after event retirement; expected
  initial one-/two-copy cases.
- Budget/failures: actual rounded backing charge once, cache reuse, retention until
  consumers finish, release when reclaimable, insufficient capacity, allocation or
  copy/context-entry failure, and no partially initialized cache publication.
- Integrated dependency: shared-constant owner/importer and sharing-disabled paths,
  correct selected device, normal completion and early close. Keep field-copy
  retirement distinct from IPC teardown. Do not mask #187 failures as test passes.

Run Compute Sanitizer on focused real-device cases. Deliberately faulting old
typed-load probes use separate processes. After focused acceptance, run both
required core groups from the repository root in the built environment:

```bash
pytest psana/psana/tests/
pytest psana/psana/tests/byhand_*
```

GPU/data skips are not device acceptance. Record commands, imported runtime paths,
device identity, environment, dependency head, and failures with their provenance.

## Performance plan

Use the [current comparison report](../docs/performance/read_staging.md) and
[staging harness contract](../scripts/jf_scaling/README.md). The October accepted
environment uses official KvikIO 26.10, CuPy 14.0.1, Python 3.11 and CPU fallback;
verify availability and freeze one identical environment for both variants.
This supersedes the earlier local KvikIO-backport planning detail, not the matched
dependency requirement. Adapt any harness device assumptions for #187 identically
in A and B before freezing it.

**A = integrated dependency without alignment changes. B = the same dependency
plus #159 only.** Initially A is `90c05965c`. Update A when the dependency changes;
never compare old pre-IPC code against IPC plus alignment and attribute the entire
difference to alignment. Hash frozen source/native libraries/harness/input files.

### Isolated measurements

- A-aligned versus B-aligned gather for uint16 and float32.
- B-aligned versus B-unaligned gather with identical values/shapes.
- A versus B aligned direct field access; B first unaligned access versus cache hits.
- First unaligned `.on_gpu` two-copy cost and direct `.on_cpu` cost.
- Warm compilation/allocations; separate CUDA kernel duration from host API latency.
  Record copy/allocation/launch counts in separate diagnostics, not timed controls.

### Staged end-to-end campaign

- Exp mfx101210926, run 387; full 32-panel Jungfrau across five streams; 10,000
  events per timed sample; verified private node-local prefixes.
- One exclusive A100 node/allocation, same physical GPU, four BDs plus SMD0/EB;
  batch 20, execution depth 1, eight KvikIO workers per BD, 1 MiB task/group
  targets, CPU fallback. Match placement, affinity and effective budgets.
- Use the explicit dense-gather benchmark adapter with no calibration or full-image
  D2H in timed controls. This workload measures staging, not IPC constant-sharing
  performance; separate integrated correctness tests exercise shared constants.
- Audit consumed addresses to ensure baseline A reads safe data. Run separate
  200-event correctness preflights.
- Bulk off/on x cold/warm x four balanced AB/BA pairs = **32 fresh-process timed
  samples**. Verify each measured prefix is below 1% resident before cold samples
  and above 99% resident for warm samples, including required post-run checks.
- Preserve timestamps, payload/request counts, device/peer placement, budget and
  drained-resource gates. Record physical GPU UUID/PCI identity and runtime paths.
- Report individual samples, median paired loop-time changes and spread, events/s,
  setup separately, and memory high water. Lazy work inside event iteration stays
  timed; staging/cache preparation and explicit setup do not. Profile/sanitizer
  results are separate from throughput. Investigate repeatable safe-data slowdowns;
  no performance thresholds in pytest.

Put frozen installs, compiler caches, staged inputs and large results on scratch,
under a unique `/sdf/scratch/users/m/monarin/gpu-validation/issue159/<run-id>/`.
Use node-local scratch only for temporary staging, preserve evidence on shared
scratch before the allocation ends, and retain concise provenance/results here.

## Integration verification and next action

Completed: pinned-head/status check; backup ref; conflict-free merge; exact parent
verification; runtime comparison against pinned #187; `git diff --check` for the
merge; retention of the prior performance documentation.

Not performed: new runtime tests, CUDA/sanitizer execution, field-address audit,
alignment implementation, or performance runs. Integration success is not runtime
acceptance and does not resolve PR #187 reviews.

Next action: stage 1 baseline/environment verification and inherited-failure
inventory, followed by stage 2 dense-gather implementation. Maintain this status
as evidence accumulates; keep #187 repairs in Seema's task.
