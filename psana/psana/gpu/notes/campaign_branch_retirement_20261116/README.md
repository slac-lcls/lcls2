# Archive and retire the source campaign branches

The consolidated home is `slac-lcls/lcls2:results/psana2-gpu-scaling`.
This operation preserves full source commit history while removing the three
superseded branch names. Tags are historical source snapshots, not measurements
of the consolidated branch head or new scientific acceptance.

| Source repository and branch | Archive tag | Commit |
| --- | --- | --- |
| slac-lcls/lcls2: docs/gpu-performance-pr187-pr208-20261028 | archive/campaigns/pr187-pr208-20261028 | 66072235d3cd18c002a13c581af82e111a418758 |
| slac-lcls/lcls2: docs/gpu-performance-pr299-perlmutter-20261115 | archive/campaigns/pr299-perlmutter-20261115 | e8d69a7e6604e1d459ad04c633de50a56b46ac9a |
| monarin/lcls2: results/psana-scaling | archive/campaigns/psana-scaling-20261116 | 9674e3e803354b65c86e42d616af50b23d2c6e3a |

The two documentary PRs are already closed as superseded; PR299's results link
was updated while its implementation remains open. See the
[PR cleanup receipt](../campaign_link_cleanup_20261116/execution.json).

Local inventory covered Git repositories registered under the known Perlmutter
lcls2 worktree roots and the primary lcls2 checkout. Two independent Git object
stores were found. Only results/psana-scaling has a local branch and attached
source worktree; the two documentary branches have remote-tracking refs in the
benchmark repository. There were no matching refs in the primary lcls2 repository.
No claim is made about checkouts on other hosts such as SDF or a laptop.

[plan.json](plan.json) pins the source refs, tags and allowed local removal.
[retire.py](retire.py) supports read-only --preflight and guarded --execute modes.
All tags are published/verified before source refs are removed. It checks open
PRs via paginated repository API listings (both head and base) and expected heads
again at execution. Remote deletes use exact SHA leases; local deletions compare
expected object IDs. The single obsolete source worktree must be clean, including
ignored files. No broad local cleanup is performed.

To recover a source branch later, fetch its named tag from the original repository,
then create a new local branch at that tag's peeled commit. The old branch name
can be restored explicitly if needed; routine reporting uses the consolidated
results branch. Tag object IDs and verified execution outcomes are recorded in the
[execution receipt](execution.json). See [status](status.json) and [scope](scope.md).

## Completed November 16, 2026

All three annotated tags were published and verified before deleting any source
branch. The final readback confirmed all three remote branches absent and all
archive tag objects and peeled commits intact. Local target branch/tracking refs
are absent in both inventoried Perlmutter Git stores. The clean obsolete
benchmarks/psana-scaling-results worktree was removed without force; the canonical
results worktree remains. Pushes removed the documentary remote-tracking refs;
the local scaling branch was deleted with its expected object ID.

Claude [round 1 passed](review-01.json) commit
575f818f6d6f6e21470388acf4721ba2426dc519 with no blocking findings. Its one optional
hardening suggestion and the execution constraint are recorded in the
[response](response-01.md). This closure adds evidence only.

Published source snapshots:

- [PR187/208 archive](https://github.com/slac-lcls/lcls2/tree/archive/campaigns/pr187-pr208-20261028)
- [PR299 archive](https://github.com/slac-lcls/lcls2/tree/archive/campaigns/pr299-perlmutter-20261115)
- [Fork scaling archive](https://github.com/monarin/lcls2/tree/archive/campaigns/psana-scaling-20261116)

The maintained regression template and dataset qualification remain a separate
planned step; no runtime, dataset or benchmark job changed during retirement.
