# Clean master-based PR

User approved creating a clean branch from current upstream master, carrying the
final callback implementation and regressions, adding the required changelog,
rerunning validation and the implementation/review loop, and opening a PR.
Existing experiment/audit branch is retained and linked instead of importing its
large notes directory into the PR. No merge or bypass of review rules authorized.

Base: 9ab5c193b4277e44e3e8ef9563d830c8943fede3.
Branch: fix/smd-callback-persistence, isolated /tmp/smd-callback-pr/worktree.
Remote: origin (slac-lcls/lcls2), same repository as prior work. Source final
delta: 71378658a..100dded91 restricted to psana; applied cleanly, preserving newer
master MPI layout checks. Clarify the prior low-priority docstring finding about
bounded output versus unbounded pending transitions. No greenlet dependency.

Build current-master native code into a separate install. Run required core and
byhand tests, including all callback regressions. Repeat Mikhail's exact stage-1
command against current master and the fixed build with protected local outputs,
readonly detector naming and blocked calibration writes. Compare default
failure, 10000 workaround, fixed default, timestamps and full gate arrays.

Independent Claude review, maximum three rounds, restricted read-only checkout at
the tested implementation SHA. Existing external-review authorization persists.
Create PR using the repository template, fill its number into the changelog,
and inspect GitHub Actions. Link #284 only as deferred follow-up, not as fixed.
Retain exact test/review evidence on the existing audit branch. Runtime checks
after any substantive repair must cover affected paths; document environmental
or upstream failures separately from change-induced failures.
