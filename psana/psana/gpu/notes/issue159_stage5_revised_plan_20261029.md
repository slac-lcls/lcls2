# Issue159: revised Stage 5 and completion plan — October 29

**Decision for review: finish #159 as an alignment correctness fix. Stage 5
becomes an existing-accessor compatibility check, not an accessor optimization
stage.** Batched tasks remain the primary interface for routine detector
processing. Existing per-event field APIs remain available and safe while their
long-term role is considered separately.

Mona approved this direction and requested Claude's review before proceeding.
This PR changes planning/review documentation only. It does not implement Stage 5,
change APIs, add tests, launch jobs, or claim that the remaining work is complete.
After this plan is accepted, execute the bounded check below in a separate PR
into `fix/gpu-unaligned-reads-159` if changes are needed.

## Starting point and what changes

PR246 merged into the main fix branch at
`019855bc12f94b6daaadce04dbe2ee6ef38b869b`. Its merged tree equals reviewed head
`c182dc758`; production runtime remains the reviewed `405f868ed` implementation.
Both final PR187 and PR208 are integrated. Stages 1–4 and PR246's technical gates
passed, including accepted batched cold A/B and per-event field measurements.
This is not authorization to repeat those campaigns or to reopen their accepted
performance disposition without new evidence.

| Area | Previous plan | Revised plan |
| --- | --- | --- |
| Field `.on_cpu` | Add direct byte D2H without an intermediate aligned device copy | Keep current safe resolution/copy path; verify exact data, host independence, caching and retirement behavior |
| Field `.on_gpu` | Retain independent copies and add copy-count coverage/timings | Verify the existing copy semantics/counts; defer direct single-copy optimization and optional performance expansion |
| `on_gpu_view()` | Safe direct view or cached aligned materialization | Keep implemented behavior and its naming/copy caveat; preserve lifetime/budget correctness |
| API design | Existing properties assumed as continuing implementation targets | Inventory consumers and decide support/deprecation/explicit methods separately in #257; no removal or rename in #159 |
| Final performance | Another campaign scheduled as a stage | Reuse accepted evidence when runtime/measurement inputs are unchanged; rerun affected measurements only for a material code/dependency change, failure or unresolved concern |

The old requirement that `.on_cpu` must avoid all device alignment
materialization is explicitly withdrawn from #159. Direct byte D2H remains a
possible later optimization, not a safety requirement while the current safe
field path is used. This is a scope reduction, not a claim that it is implemented.

## Scope: field accessors only

This decision applies to `GpuFieldResult` returned by detector `field(...)`.
It does not automatically change calibrated-result accessors, batched outputs,
`GpuTask`, or other classes with `.on_cpu`/`.on_gpu` names.

The source remains the original packed GPU input bytes. Field access does not
route through dense gathered raw output. For unaligned fields, a separate aligned
field copy is created lazily and retained on the event's root lease. Current
copy accessors call that safe view path before producing independent output.

Per-event access can serve small metadata, occasional CPU inspection or unusual
snapshot needs. Documentation should direct full detector processing on each
event to the batched task/output path. State costs in concrete terms: device
allocation/copy for `.on_gpu`, host transfer and synchronization for `.on_cpu`,
and possible first-access materialization for `on_gpu_view()`.

## Bounded Stage 5 compatibility check

First inventory existing tests and actual call sites. Record what is already
covered and which assertions are missing; do not duplicate the full field-view
matrix merely to increase counts. Repository callers are a starting point, not
proof that no external consumers exist. Useful files to inspect:

- `gpu_input.py`: `GpuFieldResult.on_cpu`, `.on_gpu`, `.on_gpu_view`, `_slot_views`,
  `_materialize` and root/child lease lifetime code.
- `tests/gpu/unit/test_gpu_input.py`, `test_gpu_field_views.py`,
  `test_gpu_input_copy_lifetime.py`, `test_gpu_retirement.py`.
- `tests/gpu/integration/test_gpu_field_views_device.py`,
  `test_gpu_input_copy_device.py` and existing related result-lifetime tests.
- User docs `gpu/docs/gpu_xtc_parser.md` and `gpu/docs/user_kernels.md`.

The required behavior, using representative aligned/unaligned fields:

1. Preserve exact bytes, dtype, shape (including scalar), segment mapping and
   absence/empty semantics. Check independently shifted bases and offsets where
   that coverage is not already established for the accessor path.
2. CPU output must not alias or modify the GPU input or aligned field cache.
   Repeated `.on_cpu` accesses return its existing host cache; do not redefine
   this as a new host snapshot on every call. A previously obtained host result
   remains usable after event retirement. Distinguish cached access after
   retirement from attempting a first device read after retirement.
3. Each `.on_gpu` call returns independent GPU output. Mutating one result must
   not alter input, the aligned field cache, or another result. Preserve the
   existing stream/readiness contract; test survival after safe event retirement
   without inventing an implicit cross-stream guarantee for an owned array.
4. Verify logical copies outside timing, not exact CuPy implementation details:

   | Nonempty field state | `.on_cpu` first access | `.on_gpu` call |
   | --- | --- | --- |
   | Input aligned | One device-to-host copy | One independent device copy |
   | Input unaligned, no materialization yet | One device alignment copy, then one device-to-host copy | One device alignment copy, then one independent device copy |
   | Input unaligned, aligned copy already cached | One device-to-host copy | One independent device copy |

   Repeated `.on_cpu` after its host cache exists adds no transfer; repeated
   `.on_gpu` still creates a new independent result. Empty/absent fields must not
   force nonempty alignment materialization. Do not require the backend to use a
   particular memcpy API or kernel for the independent copy.
5. Preserve producer/copy/consumer ordering, root/child lifetimes, exact budget
   charging and safe cleanup on failures. Existing #256 limitations are accepted:
   copied fields may block host retirement and accumulate across a slot. This
   check does not introduce trimming, admission redesign or deferred-copy queues.
   Distinguish framework-retained copy charges from independently held outputs;
   do not demand zero charge while a charged alias legitimately remains live.
6. Add only missing meaningful CPU/device regressions and update usage guidance.
   If the audit finds a correctness defect, report and narrowly fix it with
   appropriate checks before calling Stage 5 complete. Do not hide a defect behind
   the API's lower priority or the optimization deferral.

Exit: coverage inventory plus passing relevant checks establish existing
compatibility. User docs explain the costs and preferred batched path. No new
accessor, rename, removal, direct D2H optimization or automatic memory-pressure
recovery is required. Runtime may remain byte-identical to the reviewed baseline.

## Complete #159 after Stage 5

**Stage 6 — reconcile integrated correctness.** Map each remaining requirement to
existing exact-revision evidence or an explicit missing check. PR246 already
covers CPU/CUDA/memcheck, MPI ownership/reporting, and real pixel checks; do not
silently treat that as every planned final check. Preserve the implementation
plan's required core groups (`psana/psana/tests/` and `psana/psana/tests/byhand_*`)
and focused real-device sanitizer coverage. Run uncovered or affected checks in
the proper environment; report skips, missing data and inherited failures
explicitly. A skip is not device acceptance. No blanket rerun is needed solely
because a plan/doc changed, and no required unrun check is silently waived.

**Stage 7 — evidence reuse with change triggers.** Cold job 42255812 (A 8cf05e86a /
B 405f868ed) remains the accepted cumulative batched input/gather comparison,
batch 20, 4BD/1GPU, mean loop increases 1.32% off / 1.95% on. It excludes user
GpuTask/calibration work. Field jobs 60467642 (Perlmutter) and 42263681 (SDF)
measure per-event access and retirement; keep site results separate. Reuse this
evidence if runtime remains identical and no new concern invalidates it.

A runtime correction requires correctness/sanitizer checks for the affected
path and a fresh candidate identity/build provenance. Measure affected accessor
costs if that path changes; changes to gather, common retirement, budget/admission
or dependency behavior require an impact-based decision on the batched screen.
If a new full screen is justified, retain the accepted same-node/separate-build,
private cold-staging and balanced four-pair procedure with mean/sample SD. Do not
launch another A/A or unconditional cold campaign. No new performance threshold
or unmeasured end-to-end claim is introduced.

**Stage 8 — final review and target PR.** Verify the then-current
`features/psana2-gpu` target, inspect any intervening changes, and show a cumulative
alignment-specific diff with provenance. Prepare the final fix-branch PR with
completed checks, API costs and explicit follow-ups. Use a separate integration
PR if later upstream changes materially affect this work. #159 closes only after
that final PR is accepted and merged; merging the plan or PR246 does not close it.

## Deferred work, with ownership boundaries

- [#256](https://github.com/slac-lcls/lcls2/issues/256): host wait for retained
  field copies, slot-wide accumulation and potential-copy admission planning.
- [#257](https://github.com/slac-lcls/lcls2/issues/257): consumer/use-case inventory
  and long-term policy for per-event field copy APIs. Direct byte D2H and optional
  single-copy GPU optimization are deferred alongside that policy; no particular
  API migration or performance implementation is selected here.
- [#247](https://github.com/slac-lcls/lcls2/issues/247): inherited host calibration
  scalar-shape transport. Separate from scalar field correctness in #159.

## Prompt for Claude: review this plan before execution

Review this documentation PR against base merge `019855bc1`, not the already
completed Stages 1–4/PR246 review. Record the exact head reviewed. Consult
`issue_159_implementation_plan.md` and this note; inspect current accessors and
existing tests as needed to assess the plan. This is an independent plan review:
no implementation, benchmark launch, merge or modification of the author worktree.

Please check:

1. Does compatibility-only Stage 5 still close the unaligned-read safety problem
   for every currently supported field accessor, without mistaking an optimization
   for correctness or ignoring a genuine safety gap?
2. Are CPU caching, GPU output independence, copy counts and retirement/stream
   semantics accurately described? Are test additions bounded to real gaps?
3. Is the distinction between fields, dense gather, batched tasks and other result
   APIs clear? Are there remaining active promises of direct D2H or API expansion?
4. Is reusing completed performance evidence appropriate with the stated change
   triggers? Are required final correctness checks preserved and unmet checks
   still visible?
5. Are #256/#257/#247 sufficiently separated, with no implicit API removal or
   redesign, and is the final target-PR/issue-closing sequence clear?

Report blockers, nonblocking suggestions and whether the revised Stage 5 plan is
ready to execute. Mona will request the next step after reviewing your response.
