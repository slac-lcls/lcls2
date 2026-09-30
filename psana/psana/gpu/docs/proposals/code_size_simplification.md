# GPU branch code-size simplification plan

Status: **DEFERRED; no runtime refactor applied.** Revision 4,
2026-09-26. The joint simplification effort, including further structural
analysis, is deferred by decision after the CPU/GPU comparison. The earlier
steps and LOC forecasts remain below as a reference for possible future work;
they are not an active implementation queue.

This document supersedes revisions 1–3 and merges the Codex
[review](code_size_simplification_review_20260926.md) into a single plan, so
the earlier proposal and review no longer need to be read together.

Revision 1 claimed a 33% line reduction and recommended off-path consolidation
first. **Both were wrong.** The Codex review's six findings were independently
verified against the code and all six hold; five change the plan materially.
Revision 1's numeric target is withdrawn, its two largest "savings" items are
withdrawn, and its ordering is replaced. The corrections are recorded in
[Withdrawn claims](#withdrawn-claims-revision-1) rather than deleted, because
the reasoning errors are reusable.

The per-item LOC forecasts below were added after rechecking revision 2 against
the checkpoint. They are planning ranges, not measured candidate results or
removal quotas. The supported changes are small: setup/logging extraction may
increase total LOC, while bounded deduplication may offset that increase.

## Decision to defer

The measured **9,455 LOC of GPU-specific runtime** versus roughly **5,385 LOC
in selected CPU responsibility groups** does not, by itself, justify a broad
refactor. These are different scopes, not equivalent standalone runtimes.
The GPU path extends the shared CPU framework and adds asynchronous reads,
CUDA completion tracking, reusable-buffer ownership, memory admission, and
D2H delivery. The additional responsibilities explain much of the size gap,
without proving that every current abstraction is necessary.

The validated joint proposal forecasts **7,116 → 7,086–7,150 LOC** in its
affected production files: **30 fewer to 34 more lines**, approximately flat.
Its main potential benefit is organization, while changing these boundaries
still requires ownership, failure-path, device, and performance validation.
We have not identified enough concrete maintenance benefit or safe code
removal to justify that work now.

Prioritize correctness, performance, and operational stability. Keep this
proposal and the comparison as a baseline; revisit a specific area when a bug,
feature, repeated maintenance difficulty, or demonstrated duplication provides
a concrete reason. GPU tests, benchmark tooling, and historical validation
campaigns can receive a separate cleanup review later. Their size is not
evidence of runtime complexity, and no deletion is planned by this decision.

## CPU/GPU responsibility and current LOC comparison

Counts are physical LOC, including comments, docstrings, and blank lines, at
the checkpoint below. CPU groups are selected source scopes; GPU rows form
the complete 24-file GPU-specific runtime inventory. For source details and
ownership differences, see the
[CPU/GPU comparison](../cpu_gpu_complexity_comparison_20260926.md).

| Responsibility | CPU LOC / scope | GPU LOC / scope | Comparison |
|---|---|---|---|
| Event delivery, reads, and scheduling | **589** — `Events` + `EventManager` | **3,192** — manager, reader, execution/read-group scheduling | GPU scope also includes setup, D2H, diagnostics, and admission; CPU calibration is requested separately by user code. |
| XTC parsing and Configure/field access | **1,427** — `dgram.cc` + `container.cc` | **2,121** — `gpudgram/{config,batch,parser}.py` | GPU adds device metadata, batched locators, and parser-arena lifetimes. |
| Native XTC format dependencies | **977** — five selected XtcData files | Included in parser group above; no separate GPU LOC assigned | CPU dependency sample is not the full transitive parser inventory. |
| Detector assembly and calibration | **2,392** — Jungfrau/inherited Python plus calibration wrappers/native code | **1,185** — detector, calibration, CUDA helper | CPU scope includes more calibration variants, common-mode support, and generic detector behavior; GPU still uses CPU support infrastructure. |
| Public fields/results and input lifetime | Distributed through Event, detector APIs, native buffer references, and copies; not separately totaled | **1,452** — input, context, input window | GPU exposes explicit ownership and asynchronous consumer completion. Distributed CPU work is not zero LOC. |
| Aggregate byte accounting | No equivalent aggregate quota in the inspected CPU loop; read chunk limit is part of existing code | **418** — budget and allocation | GPU accounts for owned allocations and admission, beyond read grouping. |
| Transport metadata | Existing XTC/SMD plus shared `PacketFooter` (60 LOC, excluded from CPU subtotal) | **487** — `gpu_batch.py` | GPU descriptor producer changes in shared EventBuilder are outside this GPU runtime total. |
| Constant sharing and device placement | Shared MPI calibration/cache infrastructure; excluded from CPU subtotal | **547** — `gpu_mpi.py` | GPU additionally manages device selection, peer groups, CUDA IPC, and diagnostics. |
| Package exports | Not inventoried separately | **53** — two `__init__.py` files | Included to reconcile the complete GPU inventory. |
| **Selected CPU subtotal / GPU-specific runtime total** | **5,385** | **9,455** | Different boundaries; not a whole-pipeline complexity ratio. |

The separately measured **9,327 LOC of shared framework files** supports both
paths. It is neither the GPU branch's additions nor another CPU-only runtime
total. Additional CPU detector support and parser dependencies also sit beyond
the selected CPU subtotal. Thus “about 9k GPU versus 5k CPU” is useful context
for prioritization, not a claim of matched functionality or minimal code size.

## Source and evidence

- Branch `codex/psana2-gpu-bulk-batched-integration`, checkpoint HEAD
  `31d68655a6f2fb711e2489f89d3149172b46dad3`.
- Worktree `/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline`.
  Tracked files were clean at this recheck; these proposal/review documents,
  `parser_warm_ab_timing_sdf.md`, and unrelated historical artifacts are still
  untracked. The maintained scaling harness and the two Jungfrau reports are
  now committed (see Step 1).
- Frozen campaign `jf-current-scale-20260925-r4`, job **39104724**, complete.
  Its measured runtime commit is
  `ad8d454d10e203d3ef02d9c75069da48a31de182`, plus its recorded
  `source.patch` / `harness.patch`. The newer checkpoint changes only
  benchmark/test/documentation files; production GPU Python is unchanged.
  Current counts below use the checkpoint working files. Do not update either
  frozen campaign in place.
- Mixed-detector retry `jf-feespec-scale-20260926-r2`, job **39178621**:
  **RUNNING at the revision 3 recheck**, with six recorded preflights and five timed
  samples, no complete summary. This remains baseline work; do not treat it as
  candidate acceptance or final performance evidence.
- Handoff: [simplification baseline](../simplification_baseline_20260925.md).
- Review: [code-size review](code_size_simplification_review_20260926.md),
  which also records 406 unit tests passing against the frozen runtime and 26
  GPU modules byte-identical to it.

## Category added LOC versus master

Disjoint, non-`.md`, non-`notes/` added lines from
`git diff --numstat master...HEAD`. This replaces revision 1's overlapping
buckets (which double-counted the benchmark drivers and mixed additions with
additions-plus-deletions):

| Category | Added LOC |
|---|---:|
| GPU runtime modules, excluding the three benchmark drivers | 9,455 |
| GPU tests | 9,317 |
| Historical validation campaigns (`validation/`) | 6,205 |
| GPU scripts and benchmark drivers | 6,078 |
| Integration into `psexp/`, Cython, and `dgram.cc` | 1,515 |
| Debug tools and other changes | 895 |
| **Total** | **33,465** |

Scripts/drivers comprise 4,788 script lines plus 1,290 lines in
`gpu_mpi_perf_compare`, `gpu_performance_benchmark`, and `gpu_mpi_benchmark`.
Debug/other comprises 731 + 164 lines. Tests, validation, and scripts/drivers
account for **21,600 added LOC**. This explains most of the gap between the
roughly 30k branch-addition figure and the 9,455-line GPU runtime inventory.
These support categories are deferred to a separate review; reducing them
would not directly simplify runtime ownership or scheduling.

The earlier 32,569 total remains correct for `ad8d454d1`; the checkpoint adds
896 lines to this historical added-line inventory through harness changes.
These additions are not current production LOC or a simplification target.

Three accounting rules for any future claim in this document:

1. Count one disjoint file inventory; never let a file appear in two buckets.
2. Distinguish *additions* (historical diff), *current lines* (`wc -l`), and
   *net removal* (what a candidate actually deletes). Revision 1 conflated all
   three.
3. Moving a function to a new module, or splitting a test file, changes module
   size and not repository size. Count extracted modules before claiming any
   reduction.

## Validation budget

The completed campaign constrains how any candidate is judged. Rates are
`10000 / median(loop_s)` over two repetitions.

- **Repetition spread is large.** 2 GPU/4 BD warm bulk-off gave **712.12 and
  546.61 events/s** on identical configuration. A single paired run cannot
  settle a regression question in either direction. Use matched repeated
  baseline/candidate runs and inspect variability first.
- **Cold rates cluster near 300 events/s at four or more BDs** (302.16,
  298.18, 303.00). This is *consistent with* a storage limit; it is **not**
  proof of disk saturation, and it does not establish that CPU-overhead
  changes are undetectable cold. Keep matched **cold** checks for any change
  touching I/O or scheduling, and keep single-BD checks alongside multi-GPU
  scaling.
- **Bulk-on is not a universal win**: warm deltas versus off are +5.61%,
  +4.74%, **−4.23%**, +10.34% across the four topologies. Direct evidence for
  keeping both read paths.
- **Throughput is not acceptance.** Mixed-rate stream progress, delayed
  consumers, D2H, transitions, and failure recovery each need dedicated
  checks. The JF-only CPU-fallback campaign does not cover them, and no
  refactor has a known upper bound on its effect.

## Deferred plan — retained for future reconsideration

All steps below are deferred. Their before/after ranges describe the earlier
candidate plan if reopened, not expected changes under the current decision.
The deferral itself changes no production LOC. Validation requirements remain
applicable to any future runtime candidate.

### LOC convention

**Before** is measured physical file LOC (`wc -l`), including comments,
docstrings, and blank lines. **After** is an estimated range for exactly that
scope **plus any new production helper modules**. Tests and documentation are
excluded from production forecasts and must be reported separately in the
implementation diff. No saving is credited for deleting tests, comments, or
historical evidence just to meet a count.

Each item is estimated independently against the checkpoint. In particular,
2a and 2b overlap in `gpu_events.py`: do not add their file totals. The unique
file rollup below counts that file once. Zero-change audit estimates mean no
implementation deletion is currently justified, not that a future audited
proposal could never reduce those files.

### Step 1 — Preserve what the harness depends on (no code change)

The checkpoint already tracks `scripts/jf_scaling/`,
`jungfrau_single_node_sdf.md`, and `jungfrau_current_scaling.md`, along with the
mixed-detector report and cache-repair tests. The preservation task remaining
before implementation is to commit this joint plan, the earlier review, and
the still-untracked `parser_warm_ab_timing_sdf.md` as appropriate; leave
unrelated historical artifacts intact. If reopened, start implementation in a separate
worktree at the exact checkpoint plus the committed plan.

| Item | Counted scope | Before LOC | Expected after LOC | Expected net change |
|---|---|---:|---:|---:|
| 1 | Non-Markdown files in `scripts/jf_scaling/`, including its tests | 731 | 731 | 0 |

Committing an existing report or harness does not reduce its physical LOC.
`jf_scaling` remains dependent on the feespec cache helpers (see [F1](#f1)).

### Step 2 — Small extractions and bounded iteration cleanup

2a–2b are intended as relocations; 2c changes runtime iteration and is not a
pure move. Use the validation requirements below, rather than revision 2's
blanket “CPU tests only” label.

| Item | Counted scope | Before LOC | Expected after LOC | Expected net change | Action and estimate basis |
|---|---|---:|---:|---:|---|
| 2a | `gpu_events.py` + new setup module | 1,532 | 1,540–1,560 | +8 to +28 | Extract `_setup_gpu_pipeline`. Its AST span is 313 lines (`581–893`; 314 including the following blank). Keep a small delegating method and count new imports/module documentation. Preserve setup order, lazy imports, logging, and partial-initialization cleanup. This improves organization; it does not remove the setup work. |
| 2b | `gpu_events.py` + `gpu_mpi.py` + new logging module, excluding the separate 2a change | 2,079 | 2,080–2,095 | +1 to +16 | Move `_GpuMemStats` (82 lines, 83 with decorator) and `_fmt_mib` (3). The 39-line `log_gpu_mem` function has different output/units, debug gating, and exception handling. Count import/helper overhead; share only proven common formatting/sampling, with no speculative saving from merging these reports. |
| 2c | `gpu_batch.py`, including any new iteration helper | 487 | 462–477 | −25 to −10 | Share event-row construction/bounded iteration only where clearer. The six named method bodies total 87 lines, and subbatch read iteration and descriptor lookup already delegate to the parent. Preserve subbatch-local descriptor offsets, original event indices, validity filtering, timestamps, parent ownership, and the subbatch API's rejection of an explicit `event_index`. The available duplication is modest. |

### Step 3 — Bounded refactors, each needing a written boundary map first

| Item | Counted scope | Before LOC | Expected after LOC | Expected net change | Action and estimate basis |
|---|---|---:|---:|---:|---|
| 3a | `gpudgram/batch.py`, including any new helper | 453 | 449–453 | −4 to 0 | Both `parse_window` and `parse_groups` already call `parse`; no three-way parser preamble needs rebuilding. At most extract their repeated free-owner lookup (two five-line blocks). Keep ownership acquisition, group arenas, and failure/drain paths separate. Leave unchanged if a helper does not improve clarity. |
| 3b | `gpu_kvikio_read.py`, including any new helper | 500 | 490–500 | −10 to 0 | Small preparation helpers only; the submit engine is already shared. The group path takes resolved contiguous descriptors, while the per-dgram path resolves files and builds independent ranges. Preserve these differences, distinct slot handling, file-reference counts, and direct submission without sorting/replanning. No common planner is proposed. |
| 3c | `gpudgram/config.py` | 726 | 726 | 0, audit only | Audit `field_handles` (27 lines) and `names_for_id` (8), both used by tests; `field_handles` supports multiple device acceptance cases. No deletion or API-chain flattening is authorized by this estimate. Moving them to test support would not automatically be a repository saving, and removing a public utility requires an explicit compatibility decision. |
| 3d | `psexp/ds_base.py` (951) + `mpi_ds.py` (995) + `run.py` (925) | 2,871 | 2,871 | 0, audit only | No repeated option-validation block was identified across these three files. `ds_base` already owns option validation/routing; MPI setup and serial construction have different roles. Do not invent a consolidation just to reduce this count. Reopen only with specific duplicate code identified. |
| 3e | `debugtools/net_bandwidth.py` (460) + `gpu_slot_overwrite_repro.py` (219) | 679 | 679 | 0, retain diagnostics | Standalone command-line tools do not need production import callers to be useful. The reference search found their own usage examples, not a production dependency. Keep both; their removal would not simplify production pipeline code. |

### Unique production-file rollup and acceptance

The explicit affected production file set is `gpu_events.py`, `gpu_mpi.py`,
`gpu_batch.py`, `gpudgram/batch.py`, `gpu_kvikio_read.py`, `gpudgram/config.py`,
and `psexp/{ds_base,mpi_ds,run}.py`. These currently total **7,116 LOC**.
Adding the independent net-change ranges for 2a–3d gives **7,086–7,150 LOC**
including new helpers: **30 fewer to 34 more lines**, approximately flat.
This is a low-confidence planning envelope, not an implemented result. It
assumes the small helper opportunities survive implementation review; if they
do not, keep the original code and revise the estimate. Step 1 and diagnostics
are outside this production total. All deferred/preserved files remain fixed.

The main benefit of this joint plan is clearer module responsibilities. It
does not yet identify a substantial production code-size reduction. Record
actual before/after counts and explain each structural change as it lands;
accept extractions on maintainability evidence, not a fictitious removal rate.

For each runtime candidate:

- Run the CPU GPU-unit suite, import/setup checks, and affected harness tests.
  Preserve private names where tests or timing instrumentation patch them, or
  migrate those consumers explicitly. Check logging content/units for 2b.
- For 2a, validate real-device setup, budget admission, geometry/calibration,
  and failed/partial initialization. For 2c, validate descriptor/event identity,
  missing rows, subbatch boundaries, both read modes, and pixel parity. For
  3a–3b, validate retained views, delayed consumers, failure cleanup, and
  transition drains. Run the established A100 lifetime/pixel acceptance before
  accepting a combined runtime candidate; CPU mocks alone do not prove these.
- Compare a separately frozen candidate with matched repeated warm runs,
  keeping cold checks for I/O/scheduling changes and single-BD plus shared-GPU
  cases. Keep mixed-rate correctness separate from throughput. Mechanical
  logging relocation alone need not trigger a full scaling campaign, but does
  not exempt the combined candidate from the baseline acceptance requirements.
- If a later approved change touches psana core (`psexp` or native integration),
  also run both `pytest psana/psana/tests/` and
  `pytest psana/psana/tests/byhand_*` in the built environment.

### Deferred — not to be started from this document

“After” here means after the currently scoped plan, so these counts stay fixed.

| Deferred item | Counted scope | Before LOC | Expected after LOC | Reason |
|---|---|---:|---:|---|
| Lease unification | `context.py` + `gpu_input.py` | 1,255 | 1,255 | Different retirement/state machines; ownership mapping and replacement coverage required (F5). |
| Locator collapse | `gpudgram/parser.py` | 942 | 942 | Distinct event-lease, parsed-owner, and retired-array checks (F5). |
| Gather consolidation | `gpu_detector.py` | 897 | 897 | Static routing and slot-owned uploads differ; the dtype template already exists (F5). |
| Feespec harness retirement | Tracked non-Markdown `scripts/feespec_bulk_benchmark/` files, including tests | 2,392 | 2,392 | Required cache dependencies and benchmark capabilities; includes checkpoint cache repairs (F1). |
| Reference planner deletion | Entire `gpu_read_plan.py`, including its 59-line `build_read_plan` | 243 | 243 | Independent test oracle and live shared types/validators (F4). |
| Historical helper deduplication | Five tracked `validation/*/stage.py` (580) plus five `trace_rank.py` (75) | 655 | 655 | Preserve campaign source/provenance (F4). |

## Preserve unchanged

These carry the branch's contracts and are in scope only for preservation.
Counts are whole-file current LOC, not branch additions. All have zero planned
net change; they are excluded from the affected-file rollup above.

| File | Before LOC | Expected after LOC |
|---|---:|---:|
| `eventbuilder.pyx` | 975 | 975 |
| `dgramlite.pyx` | 256 | 256 |
| `src/dgram.cc` | 1,257 | 1,257 |
| `gpu_stream_read_plan.py` | 125 | 125 |
| `gpu_group_schedule.py` | 75 | 75 |
| `gpu_admission.py` | 69 | 69 |
| `gpu_file_epochs.py` | 94 | 94 |
| `gpu_input_group.py` | 267 | 267 |
| `gpu_input_window.py` | 197 | 197 |
| `gpu_budget.py` | 286 | 286 |
| `gpu_allocation.py` | 132 | 132 |
| `gpu_calib.py` | 233 | 233 |
| `gpu_stream.py` | 287 | 287 |

Per-file pending-read reference counts — one of the three measured CPU-overhead
optimizations — live in `gpu_kvikio_read.py:89` (`_pending_file_refs`), **not**
in `InputWindow`. Revision 1 misattributed them.

## Verified findings from the review

Each was checked against the code; all six hold.

### F1
**`jf_scaling` depends on `feespec_bulk_benchmark`.** `scripts/jf_scaling/run.py:13`
imports `common`; `verify()` requires `common.py`, `warm_cache.py`,
`memory_state.py` in the frozen manifest; `cache_preflight()` executes
`feespec_bulk_benchmark/warm_cache.py`. The campaign's `run.sbatch:20`
PYTHONPATH includes **both** directories. The old harness also supplies paired
baseline/candidate studies, separate profiling, and pipeline-counter
comparisons that the scaling driver does not replace.

### F2
**Config accessors are internally chained, not dead.** Verified call sites:
`resolve:687 → resolve_all`, `resolve_all:501 → find_fields`,
`find_field:471 → find_fields`, `find_fields:448 → find_all`,
`find_all:408 → matches`. Four of revision 1's six removal candidates support
APIs it explicitly retained. External-call counts are not dead-code evidence;
reflective dispatch was not the blocker, the explicit chain was.

### F3
**Revision 1's arithmetic did not add up.** Its 22 runtime rows summed to
11,125 → 9,047 (−2,078) while its total row said 11,825 → 9,997 (−1,828); the
combined figure was 32,934 → 20,997, not the advertised 32,569 → ~21,900. Its
`psexp` figure (1,149) was additions **plus** deletions (886 + 263) while its
native figure (629) was additions only, and the 1,290 driver lines appeared in
two buckets.

### F4
**Test-only and historical code has a purpose.** `build_read_plan` is **59
lines** (not 93) and `tests/gpu/unit/test_gpu_direct_group_read.py:35` uses it
as an independent reference for ranges, logical rows, descriptor tables, and
byte totals across 102 groups — deleting it removes a cross-check on the
production planner. On historical copies: the tree has **19** identical
`stage.py` and **6** identical `trace_rank.py`, but only **5 of each are
tracked**. The 6,205-line validation inventory itself is tracked; it was the
copy-count evidence used to justify its proposed reduction that mixed tracked
and untracked files. It does not establish a 4,205-line removable subset.
Campaign provenance hashes these scripts (e.g.
`validation/bulk-integration-stage4-20260923/submission.json` records
`stage.py`, `warm_cache.py`, `memory_state.py`, `bench.py`, …), so rewriting
them weakens replay provenance.

### F5
**The lease, parser, and gather structures are not redundant copies.**
`SlotLease.wait_until_safe_to_reuse` (`context.py:74`) *raises* if views are
open and synchronizes before retiring; `InputSlotLease.acquire_view`
(`gpu_input.py:478`) forks owner uses so a field view can outlive event
delivery, and its retirement (`gpu_input.py:500`) transfers completion events
onto those owners. Owner retirement can defer or wait, and the no-owner path
can synchronize consumers; it is not unconditionally nonblocking. Different
state machines. In the parser,
`DeviceFieldLocators` checks an event lease, `ConfiguredFieldLocations` checks
the parsed owner and exposes shared-arena stride, and `_batch_storage` guards
raw/parser arrays after retirement. For the gather, the C++ template
**already exists** at `_gather_kernel_source()` — the uint16 and float32
gather entry points call it; the third kernel independently zeroes missing
rows — and `_batched_gather_kernel(dtype)` is already dtype-parameterized, so
revision 1's item 18 proposed work that is already done.

### F6
**Revision 1 overstated the performance conclusions.** It wrote
"storage-limited at roughly 300 events/s" and "invisible in the cold case by
construction" where the baseline says "consistent with a storage limit but not
proof." It also asserted that observed variability exceeds "every effect this
survey could produce" without any upper bound on a refactor's effect. The
[Validation budget](#validation-budget) above states the hedged version.

## Withdrawn claims (revision 1)

| Claim | Status |
|---|---|
| 32,569 → ~21,900, a 33% reduction | **Withdrawn.** Arithmetic and scope errors (F3). Revision 3 gives narrow per-item planning ranges, not a removal quota. |
| Off-path consolidation first, "no acceptance rerun needed" | **Withdrawn.** Too broad (F1). A mechanical move needs no throughput campaign, but does need helper tests, the real cache-subprocess check, and a device smoke run of affected modes. |
| `feespec_bulk_benchmark/` superseded, 4,167 → 1,500 | **Withdrawn** (F1). |
| `validation/` 6,205 → 2,000 by deduplication | **Withdrawn** (F4); duplicate-count evidence mixed tracked/untracked files, and historical sources are provenance-hashed. |
| Six config accessors removable, 726 → 450 | **Withdrawn** (F2); two survive as audit candidates. |
| `build_read_plan` removable, 243 → 150 | **Withdrawn** (F4). |
| Lease unification as an early step, 1,255 → 900 | **Withdrawn as an early step** (F5); deferred pending a boundary map. |
| Gather kernel consolidation, 897 → 750 | **Withdrawn** (F5); the template already exists. |
| Per-file pending-read counts in `InputWindow` | **Corrected**: `gpu_kvikio_read.py:89`. |

## Revision 3 validation record

- Measured every current-file count above from file bytes/newlines and checked
  AST spans for setup, logging, iteration, parser entry points, and Configure
  accessors. Recomputed the disjoint added-line inventory at `31d68655a`.
- Checked the checkpoint diff: no production GPU Python changed since
  `ad8d454d1`; all 26 matching non-script GPU Python files remain byte-identical
  to the frozen JF-only runtime. The earlier 406-unit-test result remains
  historical evidence for that runtime; it was not rerun for this document edit.
- Rechecked the proposed call boundaries, test consumers, and tracked-file
  status. Corrected the already-shared parser/iterator work, unsupported
  option-validation deduplication, Step 1 status, and CPU-only validation claim.
- Ran current `feespec_bulk_benchmark/{test_harness,test_cache_repair}.py` and
  `jf_scaling/{test_contract,test_cache_preflight,test_feespec}.py` against the
  verified frozen Python runtime: **59 passed**. These cover the newly committed
  cache repair and mixed-detector harness logic as well as prior helper checks.
- Queried Slurm and read the mixed campaign's partial results: job 39178621 was
  still running, six preflights plus five timed samples were recorded, and no
  complete summary existed. Final mixed-detector rates are not established here.
- No production refactor, GPU acceptance, or new throughput campaign was run.
  Forecast after-LOC must be replaced by measured candidate counts during
  implementation; this recheck validates the scope and assumptions only.

## Related

- [CPU/GPU complexity comparison](../cpu_gpu_complexity_comparison_20260926.md)
  — responsibility scopes, current LOC, shared infrastructure, and lifetimes.
- [Simplification baseline](../simplification_baseline_20260925.md) — handoff,
  provenance, constraints, completed campaign rates.
- [Code-size review](code_size_simplification_review_20260926.md) — the six
  findings and the validation Codex performed.
- [Stream-read refactor cleanup](../stream_read_refactor_cleanup.md) —
  acceptance counts and deferred work.
- [Memory backpressure and results](../memory_backpressure_and_results.md) —
  ownership and lease invariants.
- [Bulk read plan](bulk_read_plan.md) — staged history of the read planner.
