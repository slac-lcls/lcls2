# Review prompt: issue159 Stages 1–4, then PR246

Review the alignment fix for issue159 in slac-lcls/lcls2. This is an independent
code review, not an implementation task. Stages 1–4 were committed directly to
the fix branch; reviewing only PR246 would miss those changes. Review both
scopes below. Do not merge PRs, advance Stage 5, launch duplicate GPU/performance
jobs, or edit the author's worktree. Use a separate worktree if needed. Report
findings in your response; posting GitHub comments is not required.

## Exact scope and commands

Repository: https://github.com/slac-lcls/lcls2
Main fix branch: `fix/gpu-unaligned-reads-159`, review pin `87bce00cb`.
Integration PR: https://github.com/slac-lcls/lcls2/pull/246
Integration branch: `integrate/issue159-pr187-pr208`.
Runtime integration merge: `31c4c2f214de14628820cfdfdf309b74367620e6`.
PR head at handoff: `f2b3c2d4e`; subsequent validation-only notes/harness updates
may arrive. Record the exact head you review and inspect any later delta.

1. Review cumulative Stages 1–4 against the original integrated unfixed base:
   `git diff 90c05965c 87bce00cb -- psana/psana/gpu psana/psana/tests/gpu`
   Stage 1 froze the baseline and evidence; it is not another runtime feature.
   Stage 2 final gather revision: `7f29f0dc7` (review `90c05965c..7f29f0dc7`).
   Stage 3 lease retention: `f425e601e` (review `7f29f0dc7..f425e601e`).
   Stage 4 safe views: `ef0a453d4` (review `f425e601e..ef0a453d4`).
   Later CUDA fixture repairs `196e595e8` and `ef313f20e` are included at
   `87bce00cb`; review them too. The cumulative diff is authoritative.
2. Review PR246 as a dependency integration into the fix branch:
   `git diff 87bce00cb 31c4c2f2 -- psana/psana/gpu psana/psana/psexp psana/psana/tests/gpu`
   Its second parent is finalized target `8cf05e86a` (PR187 + PR208).
   Review budget, placement, teardown, and reporting interactions with our fix.
3. Verify the final alignment production delta against the updated target:
   `git diff 8cf05e86a 31c4c2f2 -- psana/psana/gpu/gpu_detector.py psana/psana/gpu/gpu_input.py psana/psana/gpu/gpu_stream.py`
   This should preserve the original three-file alignment production patch.
   Do not treat the already integrated old PR187 snapshot as new alignment work.

## Agreed behavior and review focus

Trace KvikIO input -> dense gather or event field -> context/consumer -> lease
retirement, including actual base address plus byte offset, not offset alone.
Dense uint16/float32 gather keeps aligned typed loads and reconstructs unaligned
values byte-wise, preserving object bits, bounds, presence, and invalid rows.
Public field views use a direct aligned view or one lazy cached aligned copy.
Supported unaligned fields must not be rejected merely for misalignment.

Audit scalar shape (), empty/absent segments, supported 1/2/4/8-byte types,
physical segment order, shifted bases, exact tails, and slot reuse. The aligned
path must not acquire eager copy buffers/events or device-wide synchronization.
Audit parent/child lease retention, producer/copy readiness, delayed consumers
on multiple streams, open children at retirement, exception cleanup, cache
publication failure, and safe release. Check actual allocation capacity is
charged once to the existing manager budget and released only when safe.

For PR246 specifically, check owner/importer/private budget paths, per-device
limits, early-close EB draining before IPC teardown, scalar/empty constants,
and actual selected GPU. Private field copies appear in PR208's retained
residual, not its borrowed imported-constant category. Confirm this is coherent
with the allocation ledger rather than treating any retained bytes as a leak.

Stage 5 is pending: direct byte D2H for on_cpu and explicit copy-count/independence
coverage for on_gpu. Do not report that planned optimization as a completed
requirement of Stages 1–4. Still report existing correctness defects. Keep the
on_gpu_view naming caveat: first unaligned access may allocate/copy; document
at completion. No API redesign, gathered-raw reuse, per-step constant refresh,
or optional on_gpu copy optimization is part of this integration.

## Evidence and limitations

Read `psana/psana/gpu/notes/issue_159_implementation_plan.md`,
`issue159_target_merge_review_20261028.md`, `issue159_stage2_20261019.md`,
`issue159_stage3_rebuild_20261020.md`, `issue159_stage4_20261021.md`, and
`issue159_stage4_cold_20261022.md`. Older reports are historical: their final
"next stage" statements may be superseded by the implementation plan.

The integrated trial passed a fresh native build, 990 CPU tests and three
CPU-backed accounting probes; these are not CUDA/MPI acceptance. Earlier
Stage 4 passed 323 CUDA and 305 selected memcheck checks on the old dependency.
Fresh integrated CUDA/MPI validation is being run by the author in parallel;
consult `issue159_parallel_validation_20261028.md` for its scope/status. Do not
assume pending tests passed or require unrelated known #209/MIG #155 work here.
Do not reopen the separate accepted PR187 startup investigation. Performance
comparisons must use matched dependencies, same-node staged cold data, separate
builds, and report rate/loop means and sample SD; old rates are context only.

Return findings ordered by severity with file:line, concrete trigger, ownership
or behavioral consequence, and a focused test/fix suggestion. Distinguish code
bugs from missing test evidence and optional improvements. Finish with separate
verdicts for (a) Stages 1–4 and (b) PR246 integration, plus any checks still
needed before Stage 5. If no blockers are found, say so and identify residual
risks rather than inventing findings.
