# Stage 3 correction: batched user scheduling

Status: required by the user on September 27. Stage 3a inputs are implemented
and validated; see [findings](../user_kernel_stage3a_findings_20260927.md).
Stage 3b single invocation/publication is also implemented and validated; see
[findings](../user_kernel_stage3b_findings_20260927.md). Stage 3c is accepted after the on-demand metadata correction; scheduling
counts and matched performance gates passed. See the [campaign](../user_kernel_stage3c_findings_20260927.md).
This amendment supersedes the per-event callback contract and the deferral of
batch callbacks in the original proposal, handoff, and implementation stages.
Complete these Stage 3 corrections before Stage 4 public delivery.

## Current implementation and required boundary

At `c64fcb2ba`, `gpu_events.py` submits each memory-bounded execution subbatch
to `EventPool.submit()`. `gpu_stream.py` prepares dense inputs once, then
`gpu_producer.dispatch_task()` loops over selected events and calls the user
once for each. The scratch benchmark allocates and launches once per event.
Moving these calls before public delivery did not reduce their count.

The corrected callable is `function(batch, stream)`, invoked once for each
nonempty selected execution subbatch. This is not necessarily an entire EB
communication batch, a read group, or a delivery/join group. Existing byte
admission and slot ownership determine execution boundaries. No batching may
cross a step/constant generation. No invocation occurs for an empty selection.
For E selected events and a user algorithm with K batched kernels, the target
is one callback and K launches, rather than E callbacks and K*E launches.
User code must actually use the event dimension; a hidden event loop is not a fix.

## 3a. Batch context and aligned inputs

Source: `gpu_task.py`, `gpu_producer.py`, `gpu_detector.py`, `gpu_stream.py`.

- Keep `GpuTask(function, inputs, calibconst)` host-only; replace the internal
  callable contract rather than adding a legacy per-event mode. Public task
  execution remains guarded until Stage 4.
- Build the eligible ordered selection once, before task input preparation.
  Carry timestamps, original batch-event indices, batch ID, run, and step
  generation explicitly. Preserve max-events tails and duplicate rejection.
- Expose dense input arrays with shape `(N, segments, rows, columns)` and
  device presence masks `(N, segments)`. All requested detectors use the same
  N selected rows. Currently `DenseInputPreparer.prepare_batch()` independently
  omits events without detector sources; add an explicit aligned preparation
  path for tasks. Missing detector/segment rows are zero with presence false,
  including a detector absent from the entire selected subbatch. Preserve
  existing input-only behavior for other callers.
- Keep one gather per requested dense input per execution subbatch. Update
  framework byte estimates/admission for aligned rows and mapping storage;
  do not compact and then issue a second GPU gather to repair alignment.
- Generic field access returns batch descriptors referencing existing locator
  tables, with event/segment-to-window/row mappings and explicit absence.
  Support independent input-window bases. Prepare any required device mapping
  in one bulk upload on first device-metadata request inside the callback;
  dense-only callbacks need no metadata upload. No locator D2H, per-event
  metadata kernels, or parser re-launches.
- Constants and canonical segment IDs remain shared read-only values. Specify
  which identity/mapping arrays are host or device data; kernels needing an
  identity array receive one bulk-prepared device representation.

Gate: exact identities, aligned multi-detector rows, absent detectors, sparse
segments, tails, and generic descriptors validated on CPU and A100. Existing
parser/gather launch structure and admission bounds preserved.

## 3b. Single invocation, batched scratch, and publication ownership

Source: `gpu_producer.py`, `gpu_stream.py`, producer unit/device tests.

- Replace the per-event invocation loop with one batch context, one stream
  context entry, and one user call. Record producer completion after all work
  submitted by that callback; retain the existing slot/input leases.
- `batch.keepalive(array)` retains a whole scratch allocation. The example
  allocates one `(N, ...)` scratch array and launches one kernel covering its
  event dimension. Psana does not introduce managed user scratch arenas.
- Define `batch.publish(name, array, event_indices=None)`: the leading axis
  contains result rows; default mapping is all N selected events. Explicit
  host-known indices address batch-context rows and permit sparse publication.
  Per-event scalars are `(M,)`; empty per-event arrays can be `(M, 0, ...)`.
  An aggregate can be explicitly attached to one selected event using M=1.
  Validate row count, bounds, duplicates, device, dtype, contiguity and bytes.
- Permit multiple disjoint row groups for a name when per-event shapes/dtypes
  differ; reject duplicate `(event, name)` publications. Obtain shapes from
  actual arrays, with no advance output schema. Keep one owner/lease per
  published backing array plus row metadata, without per-event device copies
  or CUDA events. Any host row bookkeeping must not submit GPU work.
- Keep callback-context expiry, borrowed-input protection, failure drains,
  and retention when stream completion cannot be proven. A batch backing
  allocation remains protected until its last consumer completes.

Gate: exactly one callback for N>0 and zero for N=0; the scratch example has
one allocation and one user launch for N events. Validate exact outputs,
conditional publications, shared backing, delayed kernels, callback/record/
sync failures, and retry without early reuse on A100.

## 3c. Update measurements and accept the corrected Stage 3

Source: `scripts/stage1_regression/callback_cost.py`, producer tests, reports.

- Compare no task, empty batch callback, batch scratch+kernel, and batch
  publication+kernel. Include batch 1, batch 20, uneven tails, and depths 1/2.
- Add an explicit per-event submission reference doing the same numerical
  work. Measure internal producer submission now, and compare actual public
  user-loop scheduling once Stage 4 makes that path usable. Identify reference
  allocation/copy policy and include matched preallocated cases if needed to
  distinguish batching from allocation savings.
- Count callbacks, scratch allocations, user launches, framework launches,
  completion events and copies in separate correctness preflights. Do not
  include diagnostic wrappers in timed measurements.
- Report microseconds per subbatch and per event, submission and drain time,
  total loop time and retained memory. Use equivalent data, outputs and final
  synchronization. Check expected amortization as N grows; do not assume the
  measured per-event 6.3 us simply becomes 6.3 us per subbatch.
- Run required CPU/main/byhand, A100 and MPI acceptance, then restart frozen
  input-only Stage 2/corrected-Stage 3 controls and representative task-enabled
  performance comparisons. Passing the no-task comparison alone is not batch
  scheduling acceptance. Retain the >5% repeatable input-only slowdown
  investigation threshold and interpret timing with paired/A/A noise.

Gate: correctness and lifetime suites pass; launch counts prove batching;
matched measurements demonstrate its practical effect before advancing.

## Stage 4 and Stage 5 follow-through

Stage 4 consumes batched publication records, copies contiguous result groups
under the aggregate pinned byte cap, and exposes the corresponding host row
on each public event. Preserve scalar/empty/sparse/mixed-shape semantics and
completion-based lifetime. Avoid one D2H submission per event for a contiguous
batch result; split only where layout or byte admission requires it. Public
delivery must never trigger user kernel submission.

Stage 5 calibration and azimuthal integration examples operate on the event
dimension in each kernel, with batched scratch/output allocations. Count each
algorithm's launches per subbatch; do not wrap the old event callback in a loop.

## Superseded performance campaign

At user request, regression job 39315680 was cancelled on sdfampere033 after
16m04s. `sacct` reports CANCELLED for the allocation, batch, and extern steps.
Preserve its frozen inputs and any partial artifacts as incomplete evidence;
they do not accept Stage 3. Job 39312035's completed callback timings remain
historical measurements of the per-event implementation. No runtime changes
were included in the original planning amendment; Stage 3a is tracked separately.
