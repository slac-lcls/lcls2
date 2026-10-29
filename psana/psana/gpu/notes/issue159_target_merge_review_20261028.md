# Issue159: integrating merged PR187 and PR208

Reviewed October 28, 2026. **Recommendation: merge the current target into the
fix branch now, before implementing Stage 5.** No merge conflict or CPU-level
integration blocker was found. This review leaves the active fix branch's
runtime unchanged; the merge/build/tests were performed in a detached scratch
worktree. GPU/MPI acceptance of the combined tree remains a follow-up gate.

## Exact revisions and merge verification

- Current fix branch before review: `e5569ff3566cc6b5861c802653e6080c0e0087b0`.
- Target: `origin/features/psana2-gpu`, `8cf05e86a7511f84a24f82b9c9720bc2308c471d`.
- [PR187](https://github.com/slac-lcls/lcls2/pull/187), GPU peer discovery and
  CUDA-IPC calibration constants: merged October 27 at `c1ab81fa2`;
  approved head `6687ca5b7059792d3fc57b2b1603f565acd25bac`.
- [PR208](https://github.com/slac-lcls/lcls2/pull/208), owned/imported constant
  reporting: merged October 27 at `8cf05e86a`;
  approved head `f2e6143a9c1058c09e71a33b41d9106be426ae50`.
- Our older PR187 pin `4b068a867` is an ancestor of the merged PR187 head.
  Git has two merge bases (`c157b490c` and `4b068a867`) because we already
  integrated the older PR branch. A normal merge handles both histories.
- `git merge-tree --write-tree` and an actual `--no-commit --no-ff` merge in
  the isolated worktree both succeeded without manual resolution. Trial tree:
  `88d0fd3ff4a8de45f8a0da7e27f7fd1a5da8f14b`.
- Against the updated target, the production delta is exactly our three files:
  `gpu_detector.py`, `gpu_input.py`, `gpu_stream.py`. That patch is byte-for-byte
  identical to the original integrated-base → current-fix production patch.
  No alignment change was lost or duplicated. No native source/build-system
  change is introduced by this dependency update.

Do not rebase the review-stack history, cherry-pick the entire PR187 again, or
mix new alignment work into the dependency merge commit.

## Changes relevant to our plan

| Change since our old pin | Interaction with issue159 | Review decision |
| --- | --- | --- |
| PR187 computes capacity per physical device and broadcasts a job-wide budget verdict | Field copies use the manager's existing budget; automatic owner/importer limits can differ | Compatible; test field charges under owner/importer/private modes and keep the same budget object |
| PR187 closes/drains event/EB iterators before collective constant teardown | Our lease/copy completion must survive early exit and finish before resource destruction | Correct dependency ordering; retain focused early-close and delayed-consumer CUDA/MPI coverage |
| PR187 preserves scalar constants and handles zero-byte constants without IPC handles | Separate from scalar/absent event fields in issue159 | Removes anticipated dependency defects; no field-view redesign |
| PR187 exposes read-only `Run.gpu_device` | Validation can verify physical GPU identity using PCI bus ID | Use actual selected device; retain non-default/reversed-mask checks |
| PR187 documents one constant snapshot per run | Event-field cache lifetime remains per lease/event | Agrees with our scope; no per-step constant refresh |
| PR208 separates owned constants, imported `borrowed`, and residual `retained` | Our private `field-copy` allocations remain charged and are not imported constants | Compatible; do not count field views in `borrowed` or subtract imported bytes from committed memory |

Neither PR modifies our gather, field-view, lease-retention or copy-accessor
implementation files. Stage 5's direct-byte `.on_cpu()` design and independent
`.on_gpu()` behavior remain unchanged.

One reporting caveat deserves an explicit note: PR208 has no top-level
`field-copy` display category. A live aligned materialization therefore appears
in the **retained residual**, with its identity visible in the allocation ledger
as `field-copy`. That is expected while the lease/cache owns it, not sufficient
evidence of a leak. On retirement the charge must disappear. Adding a new log
category is optional follow-up work, not a reason to delay this merge.

## Verification executed for this review

A fresh independent Meson Release build/install of the trial tree passed,
using the existing Python 3.11/KvikIO 26.10 dependency environment. Production
Python/native origins were checked against that install; no mixed imports were
found among 146 production modules.

- **990 GPU CPU-unit tests passed** in 14.21 seconds, CUDA hidden, including
  our alignment/lifetime tests and both PRs' new tests.
- **Three additional CPU-backed integration probes passed** in 0.17 seconds:
  IPC-owner-shaped accounting, importer-shaped accounting, and private constants,
  each combined with the actual public field materialization/cache/retirement
  path. They verify one private field charge, exact bytes, no recopy on a second
  view, unchanged limits/held/committed values during snapshot, separate constant
  ownership/import figures, and disappearance of field charges on retirement.
- These probes use injected CPU arrays and representative constant byte counts;
  they do not perform CUDA IPC, asynchronous GPU completion or MPI teardown.
  No new GPU job or performance campaign was launched for this review.

Artifact root:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159-target-merge-review-20261028-r1`

`trial/source` contains the uncommitted detached merge; `trial/build` and
`trial/install` are independent. Build/test logs, XML, import-origin audits,
review metadata and the extra probe are retained under `logs/` and `metadata/`.
[Compact evidence](evidence/issue159_target_merge_review_20261028.json).

## Integration sequence and performance treatment

1. Merge target `8cf05e86a` into `fix/gpu-unaligned-reads-159` as a dedicated
   dependency commit. The trial proves this exact source combination merges
   cleanly. If target moves again, review its additional delta first.
2. Keep original frozen A `90c05965c` and all historical artifacts unchanged.
   Establish a new unfixed reference at the updated target for final issue159
   acceptance. Target tree `7f8ddb3d7a32fc7cd4dda6561c761bb3c9eede85` exactly
   equals the other task's C integration `701f5ebb1`; its independent install
   can be reused after manifest/dependency verification, without another A/A
   timing campaign or relabeling prior results.
3. Run focused CUDA/sanitizer checks for actual-address gathering and cached
   views, then MPI checks combining field retention with early close and shared
   constants. Exercise owner/importer and sharing-disabled budgets, scalar/empty
   constants, and selected-device correctness. CPU tests passing is not this gate.
4. Proceed to Stage 5, keeping its changes separate. For incremental timing,
   compare post-merge Stage 4 against post-merge Stage 5. For final timing,
   compare the updated unfixed target against the complete alignment fix.
5. Keep same-node cold comparisons, separate native installations, four balanced
   pairs per bulk setting, rate/loop sample SD, and recorded SSD/NUMA topology.
   Report field/API and unaligned-fallback costs separately from dense raw.

The other task's Ada campaign reports approximately +10.17% total-loop cost
for PR187's four-BD calibration, concentrated before the first event; PR208's
incremental four-BD calibration comparison is +0.51% with 1.58-point sample SD.
Those are existing measurements on the dependency, not results from this trial
and not evidence that issue159 caused a slowdown. The October 28 closeout note
records the user's decision to move on after the PRs merged and cancel the
pending Ampere startup investigation. Do not reopen that campaign here.
Startup versus remaining-loop performance is not fully resolved by that closure;
keep both arms on the same merged dependency and preserve timing definitions.

Known multi-run issue #209 and deferred discovery/MIG scope under #155 remain
separate from issue159. No new blocker was found that justifies completing the
alignment implementation against the older, already superseded dependency.

Waiting until after Stage 5 would make us repeat integration checks and risk
confounding accessor costs with changed budget/teardown behavior. Merging now
makes later validation apply to the actual target branch.

## Integration PR handoff — October 28

Following the user's approval, the reviewed target was merged on
`integrate/issue159-pr187-pr208`, based on fix-branch commit `87bce00cb`.
Merge commit `31c4c2f214de14628820cfdfdf309b74367620e6` has parents
`87bce00cb465b1b1766e631d07227521222caba5` and
`8cf05e86a7511f84a24f82b9c9720bc2308c471d`. No conflicts or manual production
edits were needed. Its tree differs from the tested trial tree only in the three
review/plan/evidence note files added by `87bce00cb`; production, tests, and build
inputs match exactly. The existing 990-unit/three-probe results therefore apply
to the same code; they were not rerun or relabeled as a new hardware campaign.

Worktree:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159-pr187-pr208-pr-20261028/source`

The PR targets `fix/gpu-unaligned-reads-159`, which remains unchanged. Stage 5
waits for independent agent review and acceptance of this integration. Reviewer
focus: shared owner/importer/private budget interactions, early-exit ordering
with retained field consumers, and PR208's retained-versus-borrowed reporting.
Focused CUDA/MPI validation remains pending before Stage 5; the source-equivalent
CPU checks do not establish GPU lifetime or collective teardown correctness.
Keep the original frozen A unchanged and use the updated target for subsequent
matched acceptance, as specified above.
