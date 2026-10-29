# Issue159: personal-fork workflow and Claude handoff

Mona authorized moving intermediate development and review to her personal fork
on October 29, 2026. This reduces upstream PR traffic while retaining reviewable
stages. The existing upstream PR260 was merged first at
`3feab5b2443bd65195628c017c2b57fb15afd6aa`. Its final head was `7d112827a`,
including the preserved calibration traceback requested by Claude.

## October 29 update: group work into milestones

Mona requested fewer, larger PRs. Group related implementation, tests,
documentation and review fixes around a meaningful outcome. Do not create a
separate PR for each stage or small correction. Intermediate review stays in
this fork; the completed alignment fix now proceeds as one upstream milestone.
This supersedes the original per-stage instructions in the historical prompt
below. Stage 7 merged as personal-fork PR1 at `86a1eba3f`; see the
[final review map](issue159_final_review_20261029.md).

## Repository routing

| Purpose | Repository / branch |
| --- | --- |
| Intermediate development and stage review PRs | `monarin/lcls2` |
| Integration branch in the fork | `fix/gpu-unaligned-reads-159` |
| Final milestone contribution | `slac-lcls/lcls2`, base `features/psana2-gpu` |
| Existing issue and historical reviews | Upstream [issue159](https://github.com/slac-lcls/lcls2/issues/159), [PR246](https://github.com/slac-lcls/lcls2/pull/246), [PR258](https://github.com/slac-lcls/lcls2/pull/258), [PR259](https://github.com/slac-lcls/lcls2/pull/259), [PR260](https://github.com/slac-lcls/lcls2/pull/260) |

The existing personal fork was reused. The merged integration branch was copied
without rebasing, squashing or force-pushing, preserving validation commit IDs.
Historical PR discussions remain upstream. Future PR numbers are repository
specific; always include the repository or full URL when referring to them.

An independent checkout is ready at:

```text
/sdf/scratch/users/m/monarin/gpu-validation/issue159-personal-fork-20261029/source
```

Its remotes are `origin = git@github.com:monarin/lcls2.git` and
`upstream = git@github.com:slac-lcls/lcls2.git`. Git's default push destination
and the GitHub CLI default repository are the personal fork. This checkout has
its own Git configuration and object storage; the existing shared checkout's
remotes and worktrees were not renamed or reset. Generated builds, frozen
validation inputs and run artifacts remain on scratch at their recorded paths.

## Completed branch cleanup

After migration, Mona authorized cleanup of the agreed redundant branches.
Eleven branch names in the old shared local repository and eleven upstream
branches were deleted, including the redundant upstream fix branch. The active
`monarin/lcls2:fix/gpu-unaligned-reads-159` branch remains the integration branch.
Four archives with unique history were first preserved and verified as tags in
`monarin/lcls2` under `archive/issue159-cleanup-20261029/`.

The [cleanup record](evidence/issue159_branch_cleanup_20261029.json) lists every
deleted name and original commit, plus archive tags and retained worktrees.
The four completed step worktrees are now detached at their original commits;
no worktree directory, frozen build or validation artifact was removed. The old
main fix worktree's uncommitted changes remain untouched. Open documentation
PR branches and other contributors' branches remain. Use the personal fork for
future work; the upstream fix branch no longer exists.

## Prompt for Claude

Continue as the reviewer for issue159, using the new personal-fork workflow.
Mona has authorized the migration. Intermediate stage PRs must now be opened,
reviewed and merged **inside `monarin/lcls2`**, targeting its
`fix/gpu-unaligned-reads-159` branch. Do not open intermediate review PRs against
`slac-lcls/lcls2`. Reserve upstream PRs for completed major milestones; the final
alignment fix targets upstream `features/psana2-gpu`.

PR260 is already merged upstream at `3feab5b24`. Your review found no blockers.
Codex addressed your evidence-location request in `7d112827a`: the preserved
attempt01 byhand log contains a NUL byte; `rg -a` reveals captured calibration
HTTP stderr, and the log hash matches the previously committed evidence. The
[Stage 6 report](issue159_stage6_correctness_20261029.md) contains the excerpt.
There is no remaining PR260 merge/review task.

Start future work from this independent checkout, or create your own review
worktree from it:

```bash
cd /sdf/scratch/users/m/monarin/gpu-validation/issue159-personal-fork-20261029/source
git status --short
git remote -v
gh repo set-default --view
git fetch origin
```

Confirm `origin` and the CLI default are `monarin/lcls2`. Your older review
worktree belongs to the shared upstream checkout and can remain a historical
reference; do not assume its `origin` has changed. Do not reset or clean another
agent's work. Use a detached review worktree when checking a PR while the main
checkout is in use, and record the exact head you review.

Always select the repository explicitly, even with the correct default:

```bash
# Substitute the personal-fork PR number supplied by Mona or Codex.
gh pr view NUMBER --repo monarin/lcls2
gh pr diff NUMBER --repo monarin/lcls2
```

For a new stage PR, use `--repo monarin/lcls2` and
`--base fix/gpu-unaligned-reads-159`. Post reviews on that fork's PR, identifying
yourself as Claude when using Mona's account. Keep significant stages in
separate PRs; request explicit authorization before merging unless Mona has
already authorized that specific merge. Do not post routine stage updates to
upstream just to mirror the fork discussions. Keep existing upstream issue
links qualified so fork issue numbers are not confused with upstream ones.

Stages 1–6 are complete and merged. Stage 6 passed 1,646 core tests (five
non-GPU skips), five byhand/MPI tests and five real-Jungfrau cases, with source
hashes/origin audits verified. Production runtime is unchanged by Stages 5–6.
See the [implementation plan](issue_159_implementation_plan.md) and
[revised completion sequence](issue159_stage5_revised_plan_20261029.md).

Stage 7 is complete: personal-fork PR1 merged at `86a1eba3f` after Claude's
review and the two wording corrections. Stage 8 verified the unchanged upstream
target and opened the complete alignment milestone as
[upstream PR262](https://github.com/slac-lcls/lcls2/pull/262). Claude's final
review found no blockers; its documentation follow-ups remain on that PR.
The remaining step is Mona's authorization to merge PR262, followed by manually
closing issue159. Deferred retirement/admission work is upstream #256, field API
policy is #257, and host scalar transport is #247. No new GPU campaign or
intermediate PR is required for these documentation corrections.
