# Issue 159: alignment-safe GPU reads

Updated 2026-10-21. Status: PR #187 comparison completed; A frozen; stage-2
measured overhead accepted by Mona for now; stage-3 focused validation and warm
screen complete; stage-4 safe field views implemented, focused checks passed, and staged screen failed cache preparation. This document records the agreed scope, staged implementation,
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
| Frozen A tag | `freeze/issue159-A-pr187-20261019` |

The [stage-2 record](issue159_stage2_20261019.md) tracks the frozen install and
dependency hashes, exact candidate revisions, device validation and remaining
performance checks.

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
Performance checks now accompany the stages; stage 7 is final confirmation,
not the first opportunity to detect a regression. Keep stages in separate
reviewable commits so a measured change can be isolated.

| Stage | Work and likely files | Exit gate | Status |
| --- | --- | --- | --- |
| 0. Integrate and record | Preserve prior base, merge pinned #187, record provenance and this plan | Conflict-free merge; runtime matches pinned #187; latest performance documentation retained | Complete |
| 1. Establish baseline | Finish focused baseline tests and address audit; freeze microbenchmarks and a short staging screen | Recorded PR #187 comparison plus safe-address evidence and baseline correctness inventory; no additional A/A campaign | Complete: A frozen; aligned baseline and old unaligned fault recorded; real-input address audit passed; inherited failures inventoried |
| 2. Safe dense gather | `gpu_detector.py`; extend gather regressions; benchmark aligned fast path and fallback | Bit-exact results and unchanged invalid-row handling; no extra gather allocations/launches; aligned microbenchmark and short warm staging screen document overhead for acceptance | Implemented; Mona accepted the +0.98%/+1.30% bulk-on cost for now and authorized stage 3; broad-suite failures remain tracked |
| 3. Copy retention | Minimal parent/child input-lease retention in `gpu_input.py`; ownership tests and event-loop timing | Correct delayed/open-child lifetime and failure handling; no copy-specific allocation, event or synchronization when copies are unused; no unexplained event-loop overhead | Implemented: focused checks passed; native isolation repaired; private-cache reset remedy applied after the two cache-gated attempts; screen 41400127 completed all eight warm samples with 100% pre/post residency; focused stage checks complete |
| 4. Safe field views | `gpu_input.py`, budget propagation through `context.py` / manager as needed; direct, first-copy and cached-view measurements | Correct alignment/streams/semantics/accounting; aligned path remains zero-copy; one materialization per cached field/segment; no repeat copy on hits; direct-access overhead checked | Implemented; 399 CPU / 323 CUDA / 305 memcheck checks passed; field overhead measured; [screen 41494246](issue159_stage4_20261021.md) failed its pre-timing cache gate; [remedy follow-up](issue159_cache_fallout_20261022.md): existing method passed on sdfampere036/042; NUMA alternative failed in 41596700; retain existing preparation; performance acceptance open |
| 5. Copy accessors | Direct byte D2H for `.on_cpu`; independent `.on_gpu` through safe field resolution; separate accessor timings | Copies preserve independence and survive retirement; copy counts verified; no device alignment materialization for `.on_cpu`; first unaligned `.on_gpu` two-copy cost recorded separately | Pending |
| 6. Integrated correctness | Sanitizer, field/gather/raw/ownership/pixel-exact tests, serial and MPI lifecycle coverage on integrated #187, required core groups | Correctness gates pass; inherited dependency/environment blockers are explicitly separated and resolved or accepted by Mona | Pending |
| 7. Final performance | Repeat full 32-sample staged campaign against unchanged integrated base; field/API benchmarks separately; incremental comparisons to locate any remaining slowdown | Complete paired samples and correctness/cache/provenance gates; no unexplained repeatable safe-path regression; fallback costs reported separately | Pending |
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

### Measured dependency baseline

The [completed PR #187 comparison](pr187_staging_comparison_20261019.md)
(job 41310953, `sdfampere030`, exit 0) passed four correctness preflights and
32 timed samples. Median paired integration time changes were +1.35% cold/off,
-0.99% cold/on, -1.56% warm/off, and +5.00% warm/on. Warm/on ranged from -8.93%
to +13.81%; cold/off was slower in all four pairs. Preserve these signals
without claiming equivalence or assigning a cause. Peak tracked memory was
1.2518 GiB/BD for both versions at a matched explicit 6 GiB/BD cap.

This campaign compares the dependency with pre-PR #187 code, not the alignment
fix. It establishes a reproducible harness and runtime baseline. It does not
complete the focused alignment tests, address audit, sanitizer or IPC-constant
validation. No numerical regression tolerance has been agreed.

### Performance checks during implementation

1. **Freeze a reference.** Use `90c05965c` as the unchanged integrated reference.
   Per Mona's decision, do not run another A/A campaign. Use balanced repeated
   comparisons and their spread to assess the candidate. Keep environment,
   device, affinity, batch/depth and effective budget fixed.
2. **Measure gather before touching field ownership.** Keep the typed aligned
   path and byte fallback inside the existing kernel. Avoid a preliminary
   scan, repacking of the complete input, extra launch or device-wide barrier.
   Check actual source alignment before typed access and preserve object bits.
   Compare kernel latency for safe aligned uint16/float32 data after warmup;
   measure fallback and mixed-alignment cases separately. If aligned latency
   worsens repeatably, inspect generated loads, registers and branch behavior
   before expanding the change. Profiling runs are separate from timed controls.
3. **Keep field-copy work lazy.** Add only the retention needed for correctness.
   Creating an event or accessing an aligned field must not eagerly allocate
   aligned backing or copy-completion events. Preserve existing producer and
   consumer ordering; use required stream/event dependencies, not a new host
   or device-wide synchronization. Cache materialization and its readiness
   state per valid event/field/segment, and retain it through all consumers.
4. **Measure each field/API path explicitly.** Compare no-field event loops,
   aligned direct views, first unaligned materialization and cached views.
   Record host latency separately from CUDA work and retained bytes. A cache
   hit must not recopy, recharge backing, or re-record materialization readiness;
   required consumer tracking and cross-stream waits still apply. Test small
   scalar fields as well as large arrays so Python/event overhead is visible.
   Keep `.on_cpu` direct byte D2H and retain the agreed independent `.on_gpu`
   semantics; defer the optional two-copy optimization until measurements
   justify a separate change.
5. **Use short screens, then full confirmation.** For stages that
   affect event processing, run warm/bulk-off and warm/bulk-on with two balanced
   A/B pairs each (eight 10k-event samples), preceded by correctness preflights.
   Stage 2 expands this to 16 samples to include the B0/B1 alignment comparison
   specified below; it replaces, rather than duplicates, that stage's screen.
   Retain the same staging/hash/residency gates. This is an early diagnostic,
   not acceptance or proof that cold performance is preserved. The dense-gather
   workload does not exercise field access, so add the focused field/API timing
   cases rather than relying on staging rates alone. Run additional cold checks
   early if read/admission behavior or a measured regression warrants them.
6. **Attribute changes without resetting away regressions.** Compare each
   candidate to the preceding accepted stage to locate new overhead, and to
   the unchanged integrated reference to detect cumulative drift. Rerun both
   sides together; historical rates from separate allocations are context only.
   Record repeatable positive changes even if smaller than the old
   warm/on spread. Mona accepted stage 2's +0.98%/+1.30% warm/bulk-on
   change for now; this specific decision does not waive future comparisons. Use more balanced A/B pairs for ambiguous
   differences. Do not silently accept a blanket 5% slowdown or hide a
   reproducible regression by increasing the tolerated variance.

The goal for already-aligned data is no measurable repeatable regression at
the resolution established by these controls. It is not a promise of zero
instruction overhead. Correctness and lifetime safety remain mandatory; do not
remove required synchronization or budget accounting to improve a timing.
Fallback cost is reported separately because the original unaligned typed
read is not a valid correctness/performance baseline.

### Shared environment and final reference

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

### Stage 2: large unaligned Jungfrau input

Use three cases with identical pixel values, shapes and detector work:

| Case | Runtime | Input addresses | Comparison purpose |
| --- | --- | --- | --- |
| A0 | Integrated baseline `90c05965c` | Aligned | Valid pre-fix reference |
| B0 | Stage-2 gather fix only | Aligned | B0/A0 detects safe-path regression |
| B1 | Same stage-2 fix | Unaligned | B1/B0 measures alignment penalty; B1/A0 measures safe unaligned handling versus the old aligned path |

The old typed-load kernel on unaligned input is not a valid timing reference:
it may fault or produce invalid results. B1/A0 changes both code and alignment;
do not describe that ratio as a same-input before/after comparison. B1/B0
holds the runtime fixed and provides the alignment comparison.

**First isolate the gather.** Use the production gather kernel with full
32-panel uint16 Jungfrau layout (512 x 1024 pixels/panel), representative owner,
locator and row tables, batch 20, and the same aligned output. This is 32 MiB
of raw pixels/event, 640 MiB/batch. Prepare and upload equivalent aligned and
odd-address fixtures before timing; verify `base_pointer + payload_offset`.
Test both base shifts and payload-offset shifts, all unaligned rows as the
worst case, and mixed aligned/unaligned segments. Batch 1 and a partial tail
check sensitivity to launch overhead. Validate exact output bytes and presence,
then warm compilation/allocations and time only the gather with CUDA events
on its stream. Report latency and useful payload throughput with its byte
definition; do not present that number as physical DRAM bandwidth.

**Then use the same staged data through the full pipeline.** Add a frozen
benchmark-only reader adapter, identical for both revisions, that reserves
N+1 bytes and exposes either `[0:N]` (A0/B0) or `[1:N+1]` (B1) as the uint8
read destination. KvikIO reads directly into the selected view: no extra
per-batch device copy, no edited XTC records, no padded output image. Apply it
consistently to input-buffer growth and allocation-requirement accounting;
retain the backing owner, charge its actual rounded capacity, and preserve
buffer reuse and valid bounds. All cases reserve the same capacity.

The current parser uses byte-based load_u16/load_u32/load_u64 helpers. Verify
the complete shifted-input path with CPU-reference pixels and sanitizer before
timing; do not assume that this source observation proves every consumer safe.
Confirm the KvikIO CPU-fallback path accepts the shifted contiguous destination.
If it does not, record the limitation rather than silently adding a repacking
copy that contaminates the comparison. An address audit must establish that
A0/B0 are aligned and that B1 exercises the intended odd uint16 addresses;
record the actual aligned/unaligned row counts. Use mixed-address fixtures
where whole-buffer shifting cannot give every row the intended alignment.

Run all cases on the same physical A100 with four BDs, the same staged prefixes,
batch 20, depth 1, eight read workers, 1 MiB tasks, and 6 GiB/BD. The first
screen uses warm cache and both bulk settings: two balanced pairs for A0/B0
and two for B0/B1 per setting (**16 timed 10k-event samples**). Each pair runs
adjacently, reverses order on its second repetition, and prepares/verifies cache
state before each sample. Every case/bulk combination gets a separate 200-event
correctness preflight. Increase repetitions if the result is ambiguous.

This staged B1/B0 ratio includes effects of shifting the KvikIO destination
and parser input as well as gather. Use the isolated gather timing to distinguish
kernel cost; do not attribute the entire pipeline difference to the byte loads.
First check the gather on one GPU process, then check four-BD contention through
the staged campaign. Report gather latency, loop events/s, read wait, setup,
actual address mix, memory high water, and extra allocation/copy/launch counts.
Keep profiler diagnostics separate from throughput runs.

Stage 2 passes only after correctness checks and investigation of repeatable
aligned-path regressions. Report the large-image fallback cost explicitly;
there is no agreed numerical fallback penalty limit. The final 32-sample safe
data campaign remains required. Expand the shifted-input comparison to cold
cache if the early results indicate a read-path effect or before making broad
end-to-end unaligned-performance claims. Direct field-view copies belong to
later stages and are measured separately.

### Staged end-to-end campaign

- Exp mfx101210926, run 387; full 32-panel Jungfrau across five streams; 10,000
  events per timed sample; verified private node-local prefixes.
- One exclusive A100 node/allocation, same physical GPU, four BDs plus SMD0/EB;
  batch 20, execution depth 1, eight KvikIO workers per BD, 1 MiB task/group
  targets, CPU fallback. Use the measured explicit 6 GiB/BD cap on both sides;
  match placement and affinity. Exercise automatic budgets separately in
  integrated correctness tests, especially after PR #187 updates.
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
Also completed: independent pre/post-PR #187 builds and A/B/A import checks,
19 harness checks, four real-GPU pixel preflights, the 32-sample 4BD/1GPU staging
campaign, final artifact verification and staging cleanup. See the linked
report for the exact workload and evidence.

Stage 2 focused device and selected sanitizer checks passed on the optimized
candidate. The [16-sample four-BD screen](issue159_short_screen_20261019.md)
and real-input address audits are complete. Mona accepted the measured aligned
bulk-on **+0.98% / +1.30%** loop-time change for now and explicitly authorized
stage 3. Keep the measured overhead; further investigation is deferred.
The broad-suite failures remain tracked for integrated correctness and are not
waived by performance acceptance. These campaigns do not resolve PR #187
reviews or validate IPC constant-sharing.

The [stage-3 record](issue159_stage3_20261019.md) documents the implemented copy
retention, passing lifetime/failure checks and host timings. The eight-sample
staged screen against accepted stage 2 failed native-library isolation before
timing in job 41349323. The [clean-build retry](issue159_stage3_rebuild_20261020.md)
passed validation/native/pixel checks in job 41397249, but warm-cache residency
failed before timing. Screen-only retry 41397802 on `sdfampere013` also failed
the warm-cache gate before timing; see the [cache diagnosis](issue159_cache_diagnosis_20261020.md).
The scoped one-time cache reset is applied in
[screen retry 41400127](issue159_stage3_cache_reset_retry_20261020.md), which
completed on `sdfampere030`: all eight warm samples passed with 100% pre/post
residency, 2,084 hashes verified and private-stage cleanup complete. Bulk-off
paired loop changes were -0.03%/+1.79%; bulk-on changes were -1.89%/-8.19%.
The next implementation stage is Stage 4, safe field views; it has not started.
Preserve frozen integrated A for final cumulative
comparison. No additional A/A measurement is planned; keep #187 repairs in
Seema's task.
