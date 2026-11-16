# Campaign consolidation implementation

The user selected **results/psana2-gpu-scaling** on November 16, 2026.
This is the current local branch and intended upstream branch name in
slac-lcls/lcls2. It supersedes results/psana2-scaling wherever that name
appears in the previously reviewed organization proposal and review records.
Those historical records remain unchanged.

Base: 1995752058bb9aaa3f16ccd86c22e84153257538
(features/psana2-gpu, including PR298).

Completed: establish the clean feature-derived local branch, retain the
reviewed design and its evidence, and apply the user-selected branch name.
The worktree remains at
/global/u2/m/monarin/lcls2_worktree/benchmarks/psana2-scaling-results.
The directory name does not determine the Git branch name.

Next implementation steps from the agreed plan:

1. Selectively import the PR187/208 and PR299 documentary packages, preserving
   frozen bytes and PR299 archive paths.
2. Migrate the old fork scaling catalog and its full dependency closure into
   the reviewed GPU layout, with a path/hash ledger.
3. Validate archive integrity, summaries, links, test/lint isolation and
   packaging; run the independent implementation review loop.
4. Publish the verified branch to slac-lcls/lcls2 and verify remote contents.
5. Preserve original heads with archive tags before retiring the three source
   branches; recheck source heads and open-PR dependencies before deletion.

The local setup step performs no campaign migration, runtime changes,
remote publishing, tagging, branch deletion or benchmark submission.
