# Task: Plan GPU bulk-input ownership and memory accounting fixes

Work in `/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline` on
`codex/psana2-gpu-xtc-parser`. Verified starting HEAD:
`8f94e3c7bc4643309d022cb5c1dd7e85805d3ac4`.
Check the current branch/status before working. Preserve existing untracked
validation files and logs. This request is **investigation and planning for
step 1 only**: do not implement fixes, merge branches, or launch a performance
campaign yet. Read relevant repository instructions and GPU/psana skills.

## Existing optimized parser work to integrate later

The separate worktree is
`/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-batched-locators`, branch
`codex/psana2-gpu-batched-locators`, committed/pushed through
`4c3cdf5a791f5c3801b03154bf696579d0b8be92`.

- Stream-grouped batched field location: Configure-derived numeric tables are
  uploaded at setup. Each parsed input batch uses one locator initialization
  launch, one grouped field-location launch, and shared readiness. GPU matching
  loops over handles belonging to the input stream. This removes the CPU loop
  that submitted decoding work independently for every configured handle.
- Batched canonical gathering: one gather launch per supported detector across
  the existing execution subbatch, replacing per-event/per-segment submissions.
  Calibration and cleanup remain per event. The measured JF case reduced
  gathers from 640 to 1 and total launches from 683 to 44 per subbatch.
- Lazy Python locator wrappers: public `batch.locate(handle)` creates/caches a
  wrapper on demand; canonical gathering uses combined locator storage directly
  and creates zero eager per-handle wrappers (previously 192).
- This does not redesign XTC walking as a single field-discovery pass. It also
  does not yet support canonical gathering across multiple bulk input owners.

Latest same-allocation warm comparison: job 38676923, sdfampere034, 10,000
events, six repetitions each: median A 22.464 s, B+gather 25.288 s, B+gather
zero wrappers 24.824 s. Wrapper results overlap; do not promise a stable 1.84%
speedup. These timings are not the older bulk-report allocation.

Read these files in the optimized worktree:

- `psana/psana/gpu/docs/bulk_batched_integration_plan.md` (local proposed plan,
  not included in the optimized HEAD above).
- `psana/psana/gpu/docs/batched_pre_bulk_review.md` (mandatory merge adaptations).
- `psana/psana/gpu/docs/batched_canonical_gather_review.md`.
- `psana/psana/gpu/docs/performance/lazy_locator_wrappers_sdf.md`.

## Agreed four-step integration sequence

1. **Fix ownership and memory accounting on the bulk base first.** Admission
   needs trustworthy costs for live allocations and replacement peaks.
2. **Merge the optimized branch normally, preserving both histories.** Extend
   the gather map to `(event, stream) -> (owner, local row)` with owner-specific
   pointers/capacity/bounds. Keep parsing/location at input-window lifetime and
   gathering at execution lifetime. Preserve bulk failure drains and full-size
   replacement reservations. Readiness must not depend on lazy wrappers being
   instantiated. Validate bulk-off behavior and bulk-on correctness.
3. **Fix residency/admission using integrated allocation costs.** Reserve useful
   execution width/depth before whole-stream residency. Keep this separate from
   step 1 and the merge; consider bounded resident windows later if necessary.
4. **Benchmark in one allocation:** frozen optimized zero-wrapper baseline,
   integrated bulk off/on, with A as reference. Require correct pixels,
   explained/bounded live memory, preserved batching, and repeatable throughput
   benefit. Keep bulk-off available. Scale BD ranks after memory correctness.

Implementation should eventually use an isolated integration branch based on
the bulk history, preserving both validated heads. Do not create/switch that
branch or start integration during this planning task.

## Step 1: evidence and investigation request

Read the local bulk report in the target worktree:
`psana/psana/gpu/docs/performance/parser_bulk_acceptance_sdf.md`, and its
`validation/perf-acceptance-20260916/` artifacts. They may be untracked; do not
delete or overwrite them. The related task is **Design GPU XTC parser**,
thread `01a06d1e-038a-7151-b61e-e86cd3f27204`, host
`remote-ssh-discovered:sdfiana`, if task-reading tools are available.

The completed report had 96 timing samples and 16 diagnostics; correctness
passed but performance acceptance failed:

- Primary full diagnostics: bulk ledger peak 3,201.8 MiB versus sampled CuPy
  **used** peak 8,325.5 MiB; about 5,123.7 MiB remained used at loop end.
- Mixed batch-1000/8-GiB diagnostics: ledger 7,197.9 MiB versus CuPy used
  13,756.1 MiB. This is not solely unused allocator cache.
- JF+epix residency reduced execution width from 26 to 2 (39 to 500 executions
  per 1,000-event batch); warm throughput fell 238.6 to 87.4 events/s. This
  admission-policy problem is step 3, not a reason to expand step 1.
- Primary execution size stayed 20 and reads fell 50,000 to 2,885, yet cold
  throughput worsened. Drain/trim/read overlap needs later profiling separately.

Trace reader/parser slots, `InputWindow`/`InputWindowUse`, GPU event and stream
facades, locator views, detector output buffers, and their budget charges.
Inspect `InputWindow._try_retire`: it clears the release callback but retains
`self.batch`. Inspect reader/parser/detector trim paths: dropping a cached
reference and releasing its ledger charge does not establish that all other
views have released the underlying allocation. These are hypotheses to verify,
not a proven explanation for every excess byte.

Distinguish CUDA-safe reuse, Python reference lifetime, actual CuPy allocations,
unused pool retention, ledger/held bytes, and sampled device memory. Retained
valid views must remain accounted for; retired facades should reject access.
Avoid proposing global synchronization or allocator-pool clearing as the normal
solution. Preserve failure-drain and full-replacement allocation protections.

Return a concise summary and a concrete step-1 plan containing:

1. Confirmed findings versus hypotheses, with source/log references.
2. An ownership/call-path trace showing when allocation charges are acquired,
   retained, transferred, and released, including trim/growth/failure paths.
3. Proposed lifetime invariants and the smallest coherent implementation scope,
   identifying affected methods and public access behavior.
4. Focused reproductions/tests for repeated windows, retained facades/views,
   delayed consumers, safe reuse, growth, trimming, and asynchronous failures.
5. Diagnostic measurements and acceptance criteria that establish accounted
   live memory without conflating used memory with allocator cache or context.
6. Open questions and risks for the later multi-owner gather integration.

Stop after reporting the plan and summary for review. No production changes,
merges, commits, or pushes are requested in this task yet.
