# Responses to round 1

Reviewed manifest: 0492701e59828404a84f9d29a4c37b6a9d4054e994e322ad7676d748d431c45e.
Verdict: changes_requested. Findings retained at shared scratch
`sim10-storage-gpu-20261114/review/round01/review.json`.

| ID | Response for independent re-review |
| --- | --- |
| SG-001-walltime-budget | Addressed with measured-budget evidence and deadline guards. Retain 3h, rather than extrapolating the predecessor's unused 1.5h reservation: actual predecessor elapsed 34m43s, matching loops 61.18/62.30s. Freeze runs.csv/accounting.psv. Explicit planning budget 9540s (2h39m), 21m margin, bounded stages and 900s abnormal case ceiling. Before phases, require ceiling+300s remaining against Slurm end time. Refusal writes partial_not_accepted plus cleanup receipts and exits nonzero; no complete/audit pass. Ask reviewer to resolve, not assume closed. |
| SG-002-staging-copy-uninterruptible | Replace monolithic copy and hash with 8MiB blocks, stop-event/deadline checks for both passes. Set stop on signal and worker exception before the pool's wait; join all workers before cleanup. Document unavoidable blocked kernel-I/O/SIGKILL caveat and owned-stage recovery; never use wait=False with live writers. |
| SG-003-cleanup-exception-masking | Independent per-medium try/except cleanup receipts; continue other media after failure. Preserve original exception, otherwise fail explicitly on incomplete cleanup. |
| SG-004-hardware-constants-provenance | Freeze historical 036 hardware.json; README reproduces exact model and PCI sysfs strings from c4:00.0/c5:00.0. No fresh hardware claim. |
| SG-005-scheduler-eligibility-unresolved | Refresh exact node, reservations, own account association/QOS and test-only output in host-validation02/scheduler.json. Submit only after pass; no alternate node/resources. If forecast remains unexplained report it, not immediate eligibility. |
| SG-006-adapter-executes-checkout-example | README explicitly identifies hash-frozen checkout example import as harness provenance, unused by StagingTask; no claim it came from PR298. |

Validation: initial test command lacked the basetemp parent, causing 11 setup
errors (44 tests passed); retained as junit.xml. Created the parent, reran with
the new partial-run test: **56 passed / no failures or skips**, 0.47s,
junit-fixed.xml. All CPU-only; no claim of hardware acceptance. No core edits,
commits, branches, runtime rebuilds, or job submissions. Re-freeze as attempt02.
