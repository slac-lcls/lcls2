# Stage 3: retirement cleanup and complete allocation accounting

Implemented and measured 2026-09-23 on
`codex/psana2-gpu-allocation-ownership`, based on bulk HEAD
`8f94e3c7bc4643309d022cb5c1dd7e85805d3ac4`. Stage 2 and Stage 3 changes remain
uncommitted. B++ has not been merged.

## Findings

The six JF retention cases now have the same **3,201.874 MiB live-allocation
peak** with bulk reads off and on. The previous bulk-on peak was
**8,325.501 MiB**. The **5,123.627 MiB** reduction comes from releasing retired
generations before their replacements, while retaining charges for any actual
escaped ndarray aliases.

All **648 checkpoints** reconcile the entire allocation ledger exactly with
independent CuPy allocation/free hooks. Input, parser, detector output, and
fixed storage are covered; there are **zero unexplained pool-used bytes** in
these controlled runs. Live pool usage reaches **zero at loop end** in every
case, including saved event-three and final-event facades, before forced GC
or reference deletion. CuPy pool-total/free cache remains a separate quantity.

The final review added an event-scoped guard for saved `.batch` handles while
resident input remains alive. That reviewed source passes the complete CPU
GPU-unit and A100 device suites and repeats both retained-facade JF cases.

## Code changes

### Allocation accounting

- `gpu_allocation.py`, `gpu_budget.py`: setup uploads use the Stage 2
  allocation owner, including rounded pool capacity and full preallocation
  holds. A failed stream drain retains submitted device arrays and their host
  sources in an explicit quarantine. `drain_failed_allocations()` retries
  completion before releasing them. Budget mutations and scalar snapshots are
  protected by a reentrant lock, including allocation destruction from another
  thread.
- `gpu_calib.py`, `gpudgram/config.py`, `gpudgram/batch.py`: calibration,
  geometry, and Configure tables use allocation-backed uploads. Setup drains
  its own upload stream before returning; no device-wide synchronization is
  introduced. Configure event-record failure therefore cannot refund active
  upload storage.
- `gpu_detector.py`: calibrated output, raw gather buffers, and presence masks
  retain charges through aliases and cache replacement/trim. Admission reserves
  the full rounded replacement. Geometry replacement drops its old reference
  without manually refunding live backing.
- `gpu_events.py`: memory snapshots include committed, held, retained, failed,
  borrowed, allocation inventory, and pool-used bytes. Detector high-water
  values aggregate all detectors. Failed storage is a diagnostic subset of
  retained storage, not an additional summand. IPC followers report borrowed
  constants separately and do not charge them again.

### Access and completion lifetime

- `context.py`: result leases collect all terminal consumer events. A result
  view acquires its lease on context entry; an unused saved context cannot
  acquire after retirement. Open contexts prevent slot reuse and report an
  explicit retryable retirement error. D2D copies register completion on their
  actual CuPy stream. Event-record failure drains that same stream; if the
  drain also fails, the view stays acquired for retry. Retired results/states
  detach device backing while independent host caches remain accessible.
- `gpu_input.py`: stream, batch, and locator facades enforce the event lease,
  even while a resident window remains live for another event. Retired field
  results clear cached device views. Valid entered field contexts retain child
  input uses through their own terminal completion. Saved public batch handles
  cannot bypass the event access boundary.
- `gpu_input_window.py`, `gpudgram/parser.py`: completed windows detach batches,
  readiness events, locator arrays, Configure references, and bound slot
  allocator callbacks. Escaped raw ndarray aliases continue to own their
  allocation charges independently.
- `gpu_stream.py`: retirement clears record device arrays while preserving
  result keys needed for automatic-D2H host handoff. Failed completion leaves
  the execution slot occupied for retry.

### Iterator and failure cleanup

- `gpu_events.py`: submission clears obsolete pending-read locals after the
  input window owns the read. `finish()` marks the manager closed only after
  all drains succeed, preserving retry after failure.
- `psexp/events.py`: the next batch source is advanced outside the exhausted
  event iterator's `StopIteration` handler. This removes the measured retained
  exception/traceback chain.
- `psexp/mpi_ds.py`: the iterator drops its delivered envelope before advancing
  and avoids `enumerate`'s cached tuple. Its `finally` drains the GPU manager
  on explicit iterator close as well as normal completion.

The original two tight-budget success tests now pass at **1,500 and 3,500
bytes**. Their Stage 2 strict expected-failure markers have been removed.
No production budget was increased and residency selection was not changed.

## Validation

Python 3.9, CuPy 13.6; isolated copies of the frozen bulk installation with
Python source overlays. Native binaries are unchanged. Each JF runner verifies
installation hashes before and after execution. GPU runs use A100s and KvikIO
CPU fallback.

| Validation | Result / evidence |
|---|---|
| Complete GPU CPU-unit directory, reviewed source | **312 passed**, `unit-reviewed.log` |
| Real-GPU regression suite, reviewed source | **16 passed**, job **38895287** |
| Main psana test group | **379 passed, 23 skipped, 10 deselected**, job **38895065** |
| Longer MPI `byhand_*` group | **4 passed**, job **38895125** |
| Six-case JF allocation audit | **6 passed, 648 reconciled checkpoints**, job **38895063** |
| Reviewed-source retained-facade JF audit | **2 passed, 216 reconciled checkpoints**, job **38895288** |
| Fixed-upload quarantine retry | **2 passed**, `fixed-upload-retry.log` |

The core/MPI groups used the final iterator implementation; their installed
core files match the reviewed source. The subsequent saved-batch guard changes
only GPU field access, which was rerun through both full GPU test suites and
the retained-facade JF cases. Test counts overlap across suites and should not
be added together.

The first MPI launch used one Slurm task with sixteen CPUs; PRRTE saw only one
MPI slot. It failed before launching the MPI cases. The successful rerun uses
eight Slurm task slots. Initial device-test failures were assertion updates
for fixed charges now appearing in the inventory and CuPy's `Event.done`
property; those logs are preserved.

Focused coverage includes multiple output-consumer streams, actual copy-stream
registration, unentered/entered contexts, stale output/field/batch/locator
access, retained output/fixed aliases, full replacement peaks, fixed-upload
rollback/quarantine/retry, failed record/drain retry, manager-close retry,
iterator-close drain, and the exception-handler retention regression. Existing
parser failures, partial reads, delayed input consumers, mixed-rate residency,
transitions, missing streams, and tails continue to pass.

## JF comparison

One SMD0, one EB, one BD, one A100; run 387, 1,000 events per fresh MPI launch,
batch 20, depth 1, budget 8 GiB; JF only, eight KvikIO workers, 1 MiB tasks,
automatic D2H disabled. Each case checks every timestamp and the first three
raw/calibration images against the frozen CPU reference.

| Mode / retained facades | Stage 2 live peak, MiB | Stage 3 live peak, MiB | Stage 3 loop-end used, MiB |
|---|---:|---:|---:|
| Bulk off / event three and final | 3,201.874 | 3,201.874 | 0 |
| Bulk on / event three and final | 8,325.501 | 3,201.874 | 0 |

The Stage 3 no-retention and final-only cases produce the same peak and zero
loop-end live bytes. Diagnostics count every pool allocation/free event;
these peaks are not periodic samples alone. Setup uploads are drained on their
own stream; normal event progress requires neither forced GC nor pool clearing.

## Remaining Stage 4 scope

These are correctness and memory diagnostics, not throughput benchmarks or
final B++ integration acceptance. Keep the optimized-branch merge gated on
the agreed Stage 4 end-to-end acceptance.

- Longer runs and broader failure/growth combinations still need acceptance.
- Multi-BD/IPC lifetime and placement, true GDS, and non-pool CUDA memory are
  not validated by this one-BD campaign. Existing IPC exchange is unchanged;
  this stage distinguishes owned allocations from borrowed mappings.
- Independent user copies/allocations remain outside the pipeline budget.
  An escaped ndarray alias of pipeline backing stays charged but is not a
  reusable-data snapshot; use it only within its registered context.
- The supported allocator remains the default CuPy memory pool. Custom
  allocators require a capacity contract. Pinned-host memory is separately
  reported and still has no independent byte-admission policy.

## Artifacts

Under `validation/ownership-stage3-20260923/`:

- `job-38895063/summary.md`, `summary.json`, six `.jsonl` traces and logs.
- `job-38895288/summary.md`, `summary.json`, reviewed-source traces and logs.
- `build.json`, `build-reviewed.json`, job `provenance.json` files.
- `source-manifest.json`, `tracked-changes.patch`, CPU/device/core/MPI logs.
- `README.md`, launchers, `allocation_trace.py`, and `summarize.py`.
