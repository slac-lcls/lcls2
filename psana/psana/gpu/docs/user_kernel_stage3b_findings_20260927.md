# Stage 3b: one callback per execution subbatch

Implemented and validated on CPU and A100, with MPI input/setup acceptance.
Public task event
processing remains gated until Stage 4. Stage 3c performance acceptance has not
started; the cancelled per-event regression campaign has not been restarted.

## Stage 3a review

Reviewed input selection and shared row identity, missing-detector zero fill,
the configured locator-table pointer/row bounds, independent input windows,
logical metadata bytes versus allocator capacity, pre-read admission in both
read modes, and owner retention across upload/stream failures. No remaining
Stage 3a runtime blocker was found. The pinned-pool extent issue was already
fixed and covered by the accepted Stage 3a tests.

One validation gap was corrected: the independent-window test launched its
checking kernel after EventPool had recorded producer completion. Its final
stream synchronization made that test safe, but it did not prove that the
producer completion token covered the checking kernel. The test now launches
and publishes the whole result inside the actual callback.

The temporary per-event input adapter and host source-presence tuple were
removed with the Stage 3b invocation change. The batch input lifetime remains
owned by the slot; a callback-scoped wrapper now expires immediately on return
or exception without dropping retained input, scratch, or publication owners.

## Implemented contract

`GpuTask(function, inputs, calibconst)` now invokes `function(batch, stream)`
exactly once for each nonempty selected execution subbatch. Empty selections
invoke nothing. There is one current-stream scope and one producer-completion
event after the callback. No framework loop invokes user code per event.

The callback exposes Stage 3a's aligned dense inputs/presence, generic field
tables, host/device timestamps and original event indices, canonical segment
IDs, shared constants, and run/batch/step-generation metadata. Selection and
input gathers happen before invocation. Returned values remain ignored.

`keepalive()` retains an entire scratch allocation. A user kernel must process
the leading event dimension to obtain batching; psana does not fuse arbitrary
per-event code or allocate user scratch automatically.

`publish(name, array, event_indices=None)` registers one contiguous device array
with a leading result-row dimension. By default its rows match all selected
events. Explicit host integer indices address selected batch rows, allowing
reordered/sparse groups. Per-event scalars use `(N,)`; empty per-event arrays
use `(N, 0, ...)`. An aggregate can be attached to one event with `(1, ...)`
and one explicit index. GPU index arrays are rejected without implicit D2H.

Multiple groups can publish the same name on disjoint events, including groups
with different shapes/dtypes. Duplicate `(event, name)` outputs, repeated or
out-of-range indices, missing/mismatched leading axes, reserved names, wrong
devices, host arrays, unsupported dtypes, noncontiguous arrays and invalid byte
extents are rejected before registration. Boolean/float indices are rejected.

`PublicationBatch` retains the actual array, shape, dtype, bytes, shared
producer/consumer lease, selected row indices and timestamps. The slot holds
these groups in `publication_batches`; Stage 4 can copy each contiguous group
under its byte cap. `publications_by_ts` contains lightweight host row records
referencing the groups. Registration creates no per-event device arrays, CUDA
events, kernels or D2H copies. Explicit row-array access makes a borrowed view.

Callback/record failures drain before release. Unproven completion keeps the
slot, batch metadata, publications, user scratch and borrowed input leases in
the existing quarantine. Consumer completion protects the entire backing
allocation even when only one row is consumed. These remain internal producer
records; automatic D2H and public delivery are not implemented here.

## Validation

Local GPU unit suite: **447 passed**. Syntax/whitespace checks passed.
Full frozen snapshot:
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage3b-20260927-r1`.
Native binaries inherit the verified Stage 1b installation; only Python sources
and tests changed. Source hashes and the tracked patch are saved with the run.

| Job | Coverage | Status |
|---|---|---|
| 39317358 | Initial full CPU suite and byhand tests | Main: 513 passed, one shared-memory failure; byhand 4 passed |
| 39317524 | Full CPU/byhand retry in isolated scratch working directory | Main: 514 passed / 128 skipped / 7 deselected; byhand 4 passed; sdfmilan009 |
| 39317359 | Full A100 integration, including slow tests | 128 passed; 3m10s, sdfampere004 |
| 39317360 | Four-rank MPI exclusive/hybrid inputs and task setup | Passed; 4m15s, sdfampere020 |
| 39317376 | Migrated callback harness correctness smoke | Passed; 11s, sdfampere026 |

CPU retry snapshot:
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage3b-20260927-r2`.
Its implementation/tests are identical to r1. It runs in its own scratch work
directory so tests that generate relative filenames do not share those files.
The initial main-suite failure was the existing shared-memory smalldata case:
the server exited/client timed out and the output lacked `oneint`, matching the
intermittent failure observed before Stage 3b. No shared-memory code or tests
were changed or skipped for acceptance.

All final acceptance jobs completed with exit 0. Each MPI input mode delivered
the exact 13-event reference; task setup kept CUDA on the two BD ranks. The
[compact validation record](user_kernel_stage3b_validation_20260927.json)
preserves scheduler outcomes, source/evidence hashes and the failed attempt.
Both frozen package manifests were fully verified; current implementation,
tests and callback harness match both snapshots.

Device tests exercise callback sizes 20, 3 and 1 at pool depths 1/2, checking
one scratch allocation, one user kernel and one producer completion event per
submission, with exact values after reuse. Other coverage includes missing
inputs/tails, zero invocation for empty selection, constants and generic fields,
sparse/mixed publications, scalar/empty rows, borrowed backing with a consumer,
context expiry, delayed scratch-only kernels, callback/completion-record
failures and unproven completion followed by retry.

The maintained callback harness now allocates/launches per batch and checks one
callback per submission. Its smoke run uses two repetitions of one submission,
for correctness only; it is not throughput evidence. Matched per-event versus
batch performance measurements and the input-only controls remain Stage 3c.
MPI checks cover input delivery and setup, not the still-gated public callback.
