# Stage 3a: aligned batch inputs

Implemented and validated on CPU and A100, with MPI input/setup acceptance.
This is the input
boundary from the [batch scheduling amendment](proposals/user_kernel_batched_scheduling_20260927.md).
Single callback invocation, batched scratch/publication, and launch-count
performance acceptance remain Stage 3b/3c. Public task execution remains gated.

## Implementation

`EventPool.submit()` selects eligible events once before task preparation.
Selection retains GPU event order, original batch-event indices and timestamps,
omits events without GPU descriptors, rejects duplicate selected timestamps,
and excludes unselected/max-events tail rows. Existing input-window and parser
lifetimes remain authoritative.

`DenseInputPreparer.prepare_batch(aligned=True)` preserves all selected rows
across every requested detector. A detector absent from an event, or from the
whole subbatch, receives zero data and false presence. Missing/rejected device
fields keep the existing gather validation. Each declared dense input needs one
gather for a nonempty selection, with no repair gather. The manager configures
gather routing at setup so an entirely absent detector needs no source lookup.
Default preparation still compacts absent sources and performs no added
source-presence scan.

`gpu_task_batch.BatchInputContext` provides:

- Immutable host timestamp/index tuples and contiguous uint64 device versions.
- Common-row dense input and presence arrays, shared constants, canonical
  physical segment IDs, and host run/batch/step-generation metadata.
- Generic field descriptor arrays `(events, canonical segments, 8)` containing
  raw pointer/bytes, configured locator-table pointer/row, expected type/rank,
  element size and host source-presence flag. Rows support independent input
  windows and capacities. Consumers must still validate device locator status,
  type/rank and byte extent before reading payloads.

All requested generic field tables and identities share one pinned host buffer,
one budgeted device allocation and one asynchronous H2D upload per nonempty
subbatch. There is no metadata D2H or metadata kernel. An empty selection has no
gathers, metadata allocation or metadata upload. Prepared context methods expire
when the slot releases storage; Stage 3b will make callback-facing context
methods expire at callback return while owners remain leased.

The context is currently internal at `record.batch_inputs`. The existing
per-event dispatcher temporarily consumes aligned rows through an adapter that
preserves its absent-input `None` behavior. Stage 3b removes that invocation
loop and adapter; this checkpoint alone does not reduce user launch counts.

## Admission and ownership review

Both per-dgram and grouped-read admission count a row for every declared dense
input on every eligible event, even if that detector has no source. They also
reserve metadata capacity with CuPy allocator rounding before reads. Selection
may reduce the actual allocation; pre-selection admission is conservative.

`EventPool` receives the manager's existing budget. Metadata device storage is
charged as `task-metadata`; physical pinned allocation bytes appear in memory
reporting. The upload uses the exact logical extent, excluding pinned-pool
padding; device admission separately accounts for device allocator rounding.
The pool retains a metadata upload owner before starting H2D. Its device backing
and pinned source survive upload failure and unproven stream completion in the
existing quarantine. Input-window leases protect raw and locator pointers.
Successful retirement/drain closes the context and releases metadata references.
User allocations remain user-owned and are not included in this input budget.

## Validation

Local GPU unit suite: **440 passed** with `PS_PARALLEL=mpi`. An initial local
run with `PS_PARALLEL=none` failed the existing MPI import test; rerunning with
the required MPI environment passed. Syntax and whitespace checks passed.

Frozen final runtime/tests:
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage3a-20260927-r3`.
Native binaries inherit the validated Stage 1b installation; no native code
changed. The manifest records packaged source hashes and the base commit;
the source patch records tracked changes. New module/tests are included in the
frozen snapshot and manifest. Home had 13 GiB free and shared scratch 81 GiB
free before preparation. All generated outputs use scratch.

| Job | Coverage | Status |
|---|---|---|
| 39316831 | Full CPU suite and byhand tests, r3 | 507 passed / 126 skipped / 7 deselected; byhand 4 passed; 4m19s, sdfmilan105 |
| 39316832 | Full A100 integration suite, including slow tests, r3 | 126 passed; 3m11s, sdfampere015 |
| 39316690 | Four-rank MPI exclusive/hybrid input and task setup, r2 | Passed, 4m11s, sdfampere016 |

The [compact validation record](user_kernel_stage3a_validation_20260927.json)
preserves source/evidence hashes, scheduler outcomes, and failed-attempt context.
All accepted jobs completed with exit 0. Each MPI input mode delivered the
13-event reference exactly; task setup kept CUDA on the two BD ranks only.

The preceding r1 jobs 39316635/636/637 were cancelled in favor of the reviewed
snapshot, which avoids adding a presence scan on input-only preparation.

R2 GPU job 39316689 finished with 123 passed and three new-test failures.
Two exposed pinned-pool rounding: a 624-byte metadata request yielded a
1024-byte pinned buffer, whose entire capacity was incorrectly used for device
allocation/upload. The fix explicitly bounds the host view to the requested
word count and reports physical pinned capacity separately. A CPU test proves
rounded pinned blocks cannot enlarge the admitted device request. The third
failure was a fault-injection fixture omitting batch_id=7; it failed input
identity validation before reaching the intended upload failure. The corrected
test must actually reach the upload and retain its source through retry.

R2 CPU main tests reported 505 passed, one failure, 126 skipped, seven
deselected. The failure was the existing shared-memory smalldata test: the
server exited and its client timed out before producing the `oneint` dataset.
R2 byhand tests passed all four cases. R3 reran the full suite and all byhand
tests successfully; the earlier failure is preserved in the evidence record.
MPI r2 covers unchanged input-only and task-setup code; it does not execute
the metadata upload modified in r3.

New device cases cover aligned multi-detector data, an entirely absent detector,
empty selection, selected tails, upload/gather counts, independent input-window
field pointers consumed by a single batched kernel, context expiry and metadata
upload failure with unproven completion followed by successful retry. Existing
producer failure/borrowed-input tests remain part of the full suite. MPI checks
cover setup and input delivery, not the still-gated public callback path.
