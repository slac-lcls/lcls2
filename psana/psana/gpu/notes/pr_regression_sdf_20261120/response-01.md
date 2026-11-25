# Response to SDF round 1

- SDF-R1-01: Recorded final cleanup job, five per-case receipts, cancelled IDs,
  sentinel progression/accounting, indexed-artifact hash and source association.
  No target scheduler steps survived. No independent process-list scan was made;
  this limit is explicit, rather than claiming an observation we did not collect.
- SDF-R1-02: Added explicit unexecuted maintained-entry-point limitations to the
  validation record, README, handoff and SDF guide. Cleanup is now qualified only
  for its disposable fault tests; actual psana runtime/workload remains pending.
- SDF-R1-03: Added SDF-specific singleton runtime capture command and exact-one-GPU
  requirement; marked capture unvalidated on SDF. Removed generic MPICH-only prose.
- SDF-R1-04: SDF prepare now requires sdf_rank/native_worker/native_contract/native_site
  artifacts; added an offline test that records preparation blockers and rejects execution.
- SDF-R1-05: CPU policy error is site-specific. Documented mpiexec front-end hash
  coverage and incomplete PRRTE/PMIx transitive pinning as a provenance limitation.

Native launch and cancellation code is unchanged. Only preflight artifact
requirements and diagnostic text changed in implementation after the native tests.
