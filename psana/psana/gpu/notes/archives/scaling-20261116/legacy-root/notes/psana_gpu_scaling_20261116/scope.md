# Review contract

Request: select the best measured one-GPU psana staging configuration; plan and
prepare 2/3/4-GPU node scaling; use the implementation/review loop.

Scope: only notes/psana_gpu_scaling_20261116. Fork the accepted stripe harness,
retain the exact PR298 framework build/data pins, add fixed four BDs per GPU,
a feasible controlled CPU policy plus a legacy control, per-device memory and
physical-NIC counters, pilot gating, 12-case randomized strong scaling, raw
receipt auditing and reporting. Preserve all previous evidence. No framework
changes, rebuild, job submission, publication, or automatic repeat attempts.

Acceptance: executable preparation/launch/audit/report scripts; explicit
comparison limitations; offline positive and corruption tests; pinned native
CPU unit tests; Python/shell syntax; readable hardware counter smoke check;
independent review on a detached exact-commit worktree. GPU allocation tests
are intentionally deferred to the prepared pilot, not falsely marked passed.
Default maximum three Claude review rounds, 15 minutes each, Read/Glob/Grep only.

Subsequent session authorization: Mona approved scoped external Claude review and
qualification submission after review/checks pass (authorization.md). The reviewer
remains read-only and must not submit anything.
