# Field-copy memory reporting

This PR262 follow-up implements Seema's optional reporting suggestion.
`field_copies` is now a separate memory snapshot, ownership-log and high-water
category. It sums charged `capacity` values of `field-copy` ledger entries;
it does not sum logical field sizes. A 1-byte field charged a 512-byte block
therefore reports 512 bytes. `retained` subtracts this known category once and
continues to expose unattributed budget-charged memory.

For example, with 1,000 MiB of identified buffers/constants, 40 MiB of field
copies and 8 MiB of other charged storage, committed remains 1,048 MiB. The
old report put 48 MiB in `retained`; the new report shows `field_copies=40 MiB`
and `retained=8 MiB`. A nonzero residual is diagnostic, not proof of a leak.

No allocation, accounting mutation, admission, copy, synchronization or lease
retirement changes. Copies remain charged through safe retirement and final
alias release. The snapshot reads the same existing ledger as constants;
imported constants remain `borrowed`, outside the charged residual calculation.
The MPI validation driver's serialized snapshot and expected category delta
are updated to this reporting contract.

## Validation and evidence reuse

- Focused CPU reporting tests: **22 passed**. Coverage includes owner/importer/
  private constants, multiple rounded allocations, remaining unknown storage,
  live/released copies and aliases, budget non-mutation, zero/no-budget states,
  current logs, persistent high-water values and MPI snapshot serialization.
- GPU CPU-unit suite: **1,016 passed**, with single-process MPI initialization.
  This is CPU-unit coverage, not a new multi-rank or device run.
- Source-origin audits load the current GPU modules over the unchanged native
  install. The initial full-suite collection lacked the test-package overlay;
  after that harness correction, serial mode exposed an MPI-symbol expectation.
  Single-process MPI mode passed without changing or skipping tests.
- A structural comparison against `ab53cb976` confirms only `_GpuMemStats`,
  `_snapshot_memory`, `log_memory` and `log_high_water` changed in production.
  Other GPU sources and the rest of `gpu_events.py` are unchanged.

The current reporting runtime is **not byte-identical** to measured B
`405f868ed`. Earlier Stage 6/7 records remain historical evidence. Their gather,
copy, lifetime and scheduling paths are unchanged, so their device correctness
and cold A/B results are reused with this explicit diagnostic-only qualification.
No new GPU/MPI/performance run was made. The updated MPI driver was tested only
through its CPU snapshot helper in this step. Deferred retirement/admission
work (#256) remains unchanged.

Artifacts: `/sdf/scratch/users/m/monarin/gpu-validation/issue159-field-report-20261105/`.
The scoped independent review runs against an exact implementation commit;
its final disposition is recorded with the validation evidence on PR262.
