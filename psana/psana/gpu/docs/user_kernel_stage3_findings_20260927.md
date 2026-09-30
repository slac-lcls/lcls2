# User-kernel Stage 3: producer dispatch and owner retention

Stage 2 performance was accepted in `070472794` after the balanced A/A and A/B
controls; see the [performance report](performance/user_kernel_stage2_regression_20260927.md).
This change implements the internal Stage 3 producer boundary. Public
`DataSource(gpu_fn=task)` event processing still raises an explicit Stage 4
delivery error, as required by the [implementation plan](proposals/user_kernel_implementation_stages_20260926.md).
Publication registration is implemented; automatic host delivery is not.

## Implemented behavior

`EventPool.submit()` prepares dense inputs once per subbatch, then invokes
`task.function(context, slot_stream)` once per selected event with GPU dgrams.
Selection uses the delivery envelopes before invoking callbacks, so a read that
covers a `max_events` tail does not cause extra calls. Events without a requested
detector still invoke the callback when another GPU source is present. Dense
row lookup uses original event index and timestamp, preserving identity across
missing-detector row compaction.

The supplied stream is current for each invocation. The producer completion
event is recorded after the callback loop. Return values are ignored; no-output
and scratch-only callbacks have no publications or task-output copies.

`ProducerContext` carries `timestamp`, `batch_event_index`, `batch_id`, integer
`run`, and `step_generation` (zero before the first BeginStep, incremented after
each host BeginStep dispatch). Its methods implement the proposal:

- `input` and `present` borrow prepared arrays, or return `None` for absent
  source dgrams. `segment_ids` returns configured physical IDs in dense row order.
- `field` returns canonical per-segment descriptors containing raw storage,
  locator rows/index, Configure type/rank, and element size. Raw/locator pointer
  properties are host-known. It reuses configured locations without per-field
  parsing, locator D2H, or host shape discovery. Missing fields inside a present
  dgram remain device locator statuses for native consumers to validate.
- `calibconst` reads the already staged declared value. The execution retains
  that exact array generation; it never uploads constants in the callback.
- `keepalive` immediately retains user owners. `publish` immediately retains a
  CuPy array and copies shape/dtype/byte metadata without inspecting contents.
  Scalars and empty arrays are valid. Host arrays, wrong-device arrays,
  unsupported dtypes, noncontiguous storage, duplicate names, and collisions
  with detector names, `det.raw`, or configured `det.alg.field` names fail.

Context methods expire on return or exception. Borrowed storage remains
read-only by caller obligation, including native pointer access. Work must use
the supplied stream; user allocations must be registered before launching work.
Psana does not allocate or budget those user allocations.

## Ownership and failure review

Each occupied slot retains preparation owners, requested constant arrays,
registered scratch, publication arrays, and input-window leases. A shared
producer/result lease retires before input leases, so an output alias of a raw,
locator, or prepared buffer cannot outlive its reusable backing when a terminal
consumer is registered. Slot reuse/clearing waits for every terminal consumer.

Callback or completion-record failures drain the producer stream before owner
release. Failed synchronization preserves the occupied slot and roots the pool
in a quarantine set, preventing garbage collection of unproven work if the
manager is dropped. Successful retry/close drains and removes that root.
Consumer failure leaves the slot occupied and retryable as well. This also
preserves registered scratch when nothing was published.

There is no new scheduler, private callback stream, calibration kernel, or
publication D2H path in Stage 3. KvikIO reads and payload copies are unchanged.
Future Stage 4 copies must wait on each publication's producer lease and register
their terminal completion before retirement; an array reference alone is not a
backing-storage lease.

## Validation

All acceptance jobs completed with exit 0. Final frozen snapshot:
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage3-20260927-r3`.
The [compact validation record](user_kernel_stage3_validation_20260927.json)
preserves job outcomes, final source hashes, manifest hashes, and log hashes.

| Job | Node | Checks | Result |
|---|---|---|---|
| 39304476 | sdfmilan001 | `pytest psana/tests/` and `pytest psana/tests/byhand_*` | 502 passed / 121 skipped / 7 deselected; byhand 4 passed |
| 39304477 | sdfampere010 | Full GPU integration suite, including slow tests | 121 passed |
| 39304313 | sdfampere010 | Four-rank MPI exclusive/hybrid input delivery and task setup | Both modes: 13 events with exact reference digests; SMD0/EB without CuPy, two BDs staging only requested gain |

The CPU and A100 jobs use the final code. The MPI regression uses the r2 snapshot; its MPI/task
setup code is identical. The r3 changes only move the current-stream context
inside each callback invocation and add a completion-record failure test.
Earlier r2 CPU/byhand and A100 jobs 39304311/39304312 also passed (502/4/120 tests).
The r1 snapshot was prepared but not submitted. A local focused unit run passed
435 tests before the final quarantine and per-invocation stream refinements;
the final full CPU suite covers those sources too.

Snapshots inherit the verified Stage 1b native installation; no native sources
changed. `manifest.json` pins the source base, packaged file hashes, and native
origin; `source.patch` records tracked edits, and new sources are included in the
frozen package and manifest. Logs, compiler caches, and generated test artifacts
are on scratch. Home had 13 GiB free and scratch 81 GiB free at preparation.

New real-device coverage exercises selected tails and missing GPU/detector
inputs, multiple dense inputs, requested constant aliases, sparse segment IDs,
native locator consumers, independent input bases, unchanged gather counts,
scalar/empty publication metadata, delayed scratch-only kernels, callback and
completion-record exceptions, failed synchronization plus garbage collection,
and retryable downstream completion for a borrowed publication. The public
Stage 4 guard remains covered by serial/MPI task setup tests.

Full public callback delivery and callback-path performance are not claimed by
this stage. No Stage 3 correctness blocker remains. Stage 4 remains the next
integration boundary.
