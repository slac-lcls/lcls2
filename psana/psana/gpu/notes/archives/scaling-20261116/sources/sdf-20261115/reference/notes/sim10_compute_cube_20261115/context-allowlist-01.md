# Independent allowlist amendment review, round 1 of at most 3

The user requested preparing an allowlist instead of waiting only for 036.
Preparation only: no submission, commits, branches, rebuilding, core changes,
large data operations, cleanup or GPU workload. Review the exact supplied
uncommitted manifest, not Git HEAD. Restricted Read/Glob/Grep only; no delegation,
edits, shell or nested review loops. Missing required evidence is not approval.

## Scope

Inspect `node_selection.py`, changes to `prepare.py` and `launch.py`,
`test_node_selection.py`, `node-allowlist.json`, `ALLOWLIST.md`, README and scope.
Inspect unchanged `run.sbatch`, `plan.py`, `runner.py` and historical CPU/hardware
helpers as needed to trace allocation, environment and all result validation.
Host validation is recorded in `checks-allowlist-01.json`; independent raw logs,
inventory copies and the old three-file baseline are in the allowed evidence
directory supplied to this invocation. The source subset and read-only
dependencies are frozen by the exact manifest hash.

Prior implementation attempt02 had independent PASS in `review-02.json`.
The original manifest is in `manifest-02.json`. Its scientific workload,
40 MPI cases, GPU/BD mappings, counts, controls, Weka input, resource request,
runtime and timing budgets must remain unchanged. Do not infer that unchanged
hardware acceptance has executed: CUDA/MPI/full-shape checks remain allocation
gates. The inherited native `psana/meson.build` mismatch remains explicitly
accepted for preparation in review-02; inspect that prior disposition and the
host check before affirming that this scheduling amendment preserves it.

## Risks to challenge

- `--nodelist=a,b` is not an OR choice. Exactly one exclusive node must be
  requested using the frozen partition complement as exclusions. Membership
  changes must invalidate command generation; post-submission drift must fail
  closed at runtime before staging, not execute on an unreviewed host.
- Node selection is candidate qualification, not availability, a reservation
  or identical performance. Inspect October26 raw inventory coverage and
  conservative rejection of malformed/uncollected/Gen3/other-NIC nodes.
- Runtime must use the allocated hostname for environment, MPI rank checks,
  retention and hardware records. Reject outside-allowlist and multi-node jobs.
  Retain existing whole-node exclusivity/sole-occupancy and UUID/core gates.
- Live CPU/GPU/NIC/driver identity must match that candidate before staging.
  SSD models and auxiliary NIC models differ; no NVMe input or cross-node rate
  pooling is authorized. Local capacity and Weka NIC traffic gates remain.
- Inventory references and normalized profiles must be hash-backed, verified
  and included in the reviewed manifest. No mutable latest-file substitution.
- Legacy single-node manifests remain usable through their original frozen
  launchers. New schema must not silently apply an old review to changed code.

IMPL-R2-001 (reference walltime) and IMPL-R2-002 (header-index setup walltime)
remain deferred operational risks. Do not label them fixed. No guarantee the
24-hour campaign completes; the 1980-second whole-full-case timeout is unchanged.
Use new stable IDs ALLOW-R1-* for amendment findings, preserving old finding IDs
when relevant. Report blockers, nonblocking suggestions and evidence limits.
