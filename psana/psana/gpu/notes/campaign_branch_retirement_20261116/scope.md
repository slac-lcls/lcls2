# Source branch retirement scope

User explicitly requested archive-tagging and retiring the three source branches,
including local and remote cleanup. Continue the session's implement-review-loop
skill (maximum three rounds) before the guarded retirement action.
Base results commit: 3e927ee1b005c4a04d8c10ee11817fc2c9bc9f99.

Preserve each exact source head with a published annotated tag in its original
repository. Verify all tag objects and peeled commits before any source deletion.
Recheck open PR head/base dependencies without relying only on search indexing.
Use explicit expected-SHA leases on remote deletions and compare-and-delete for
local refs. Remove only the named clean source worktree, refusing tracked,
untracked or ignored content and refusing unexpected HEAD/branch identity.
Retain active results/feature/research worktrees and historical detached reviews.
No broad prune, git clean, recursive removal, force worktree removal, runtime
change, new benchmark job, issue/PR mutation or history rewrite.

Run preflight, then independent read-only Claude review of plan.json and retire.py.
Only the coordinator executes after passing review, captures step receipts,
verifies final local/remote refs and publishes closure on the results branch.
Do not repeat a partially executed action blindly; inspect execution.json first.
