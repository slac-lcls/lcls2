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
results branch. Tag object IDs and verified execution outcomes are recorded at
closure. See [status](status.json) and [scope](scope.md).
