# Allowlist review disposition

Independent Claude review passed on attempt03, manifest
`03bcd8121c9919cea0623761c0006add3c57c6658e5aa1b9fff71951a3ab4d0d`.
No unresolved blockers. Frozen implementation files remain unchanged.

- ALLOW-R1-001: accepted documentation clarification. `SIM10_EXPECTED_NODE` is
  informational, not consumed by the runner. Allocation admission and the
  per-rank hostname checks in `validate_result` provide enforcement.
- ALLOW-R1-002: accepted documentation clarification. RAID/SSD/Gen4 identity is
  preparation-time only; live local capacity/filesystem are gated and full
  hardware is recorded. SSD differences affect export/whole-job timing, not a
  claim of equivalent NVMe performance. Stronger identity gates are deferred.
- ALLOW-R1-003: regression-test hardening deferred, not declared fixed. The
  coordinator's `allowlist-preservation-check.json` additionally replays actual
  November14 hardware through the frozen gate successfully. Final command
  verification re-executes schema-2 verification and real inventory profiles.
  These checks do not replace future real-format regression fixtures.
- ALLOW-R1-004: retained operational risk. Frozen command generation refreshes
  membership; queued hardware drift still fails closed. No gates relaxed.
- IMPL-R2-001 and IMPL-R2-002: retained, unmeasured reference/index timing risks.
  No workload or timeout changes and no completed-campaign guarantee.

The raw reviewer summary mentions 20 node-selection tests; actual discovery is
19 new tests plus the previous 65, totaling 84 passing host tests. The supplied
host-native.json is a fresh eight-check host run, not a new CUDA/MPI acceptance.
This clarification does not modify the raw independent review.

Only maintained handoff documentation and review receipts change after review;
the frozen snapshot is not rewritten. No job has been submitted.
