# Detector materialization and early reader-buffer reuse

Status: deferred on 2026-09-15. The experimental Stage 3A code and tests were
reverted before production integration. Stage 3 remains the implementation in
commit `a5f07ee38`; review and bulk-read work continue from that baseline.
The design below is retained for a possible future refactor, not a prerequisite
for completing the current bulk-read stages.

## Deferral decision and current behavior

The scope of a general materializer, independent detector owners, preparation
streams, and independently recycled reader pools is larger than warranted by
the immediate memory benefit for small slow-detector groups. Retaining source
buffers in a bounded pool applies backpressure rather than accumulating a new
allocation for every batch. Making cached buffers reusable earlier does not
itself reduce allocated VRAM; savings require fewer/smaller allocations.

Illustrative sizing using the recorded run-51 raw shape `(32, 512, 1024)`:
uint16 Jungfrau payload is 32 MiB/event. Two reader buffers holding two events
each retain about 128 MiB of Jungfrau source payload, plus XTC metadata and
parser storage. One retained 1,000-event fast input adds 3.9 MiB if its dgrams
are 4 KiB each (an assumption, not a measured fast-detector size). If one
64 MiB reader buffer eventually provides sufficient overlap instead of two,
the source-capacity saving is about 64 MiB; detector inputs/outputs still need
their own storage. Revisit with actual byte counts, parser capacity, budgets,
and evidence that source reservations limit throughput or admission.

Current Stage 3 keeps reader XTC bytes and parser tables under `InputWindow`
references. Detector raw/calibrated slot arrays remain under EventPool result
leases. For a registered user consumer in the ordinary delivery path, EventPool
waits for detector-result completion before releasing the event's input-window
reference and recycling the execution slot. Parsed-field view contexts hold
their own input references and completion tokens. This is conservative source
retention, not release at gather completion. Arbitrary escaped pointers or
unregistered CUDA streams are not protected by this contract.

The existing `SlotLease` for detector results still stores one terminal event;
separate zero-copy consumers of the same result on multiple streams are not
safe unless their work is covered by one registered completion dependency.
`InputSlotLease` for parsed fields already tracks multiple events. This known
result-lease limitation is not fixed by reverting or retaining Stage 3A.

There are currently two field-consumption paths to revisit:

- Jungfrau's supported calibration adapter automatically gathers raw pixels
  into execution-slot storage, then calibrates them in `GPUDetector.process_batch()`.
  Supported precalibrated dense adapters likewise gather into their output.
- Generic field access in `GpuFieldResult` reads parser-located XTC views;
  `on_gpu` explicitly copies those views when requested. This path serves
  arbitrary fields, including Jungfrau fields and detectors without calibration
  adapters. There is no common automatic materialization of every selected field.

If this proposal is revived, unify those paths for detectors selected through
`gpu_det` and `hybrid_det`, preserving all supported event fields. Detector-specific
calibration should consume the shared materialized representation. The proposed
Stages 3A-3D below describe that future work only; the current implementation
continues to permit leased XTC field views.

The deferred design would refine [the bulk-read plan](bulk_read_plan.md). In particular, long-lived
fast data should reside in detector-owned buffers after materialization, rather
than keeping original XTC bytes and parser tables alive through execution.
The stream and reuse requirements below are part of the design, not optional
performance notes. Existing stage acceptance remains evidence for the code
tested at that stage, not for this proposed ownership model.

## Ownership contract

The mandatory psana input pipeline is:

```text
read XTC -> parse -> copy into detector-owned raw inputs -> source reusable
                            |
                            +-> detector readiness -> calibration / user kernels
```

Psana schedules this materialization automatically for every admitted dgram of
the configured GPU detectors. It must not depend on a user field-access request,
a manual copy, a user kernel submission, or advancement of the user's event loop.
Here, detector-owned raw inputs include supported fields and the independent
metadata needed to interpret them, including for detectors without calibration.
Field access exposes this prepared detector storage; it never initiates a lazy
copy from XTC. Consumers wait for detector readiness before reading it.

Completion of all required copies ends the source generation's lifetime, once
its I/O and parser dependencies are also complete. The allocation is then
reusable. Detector raw inputs have a separate lifetime governed by planned and
active consumers, including calibration and user kernels, and all of their
completion dependencies. User references retain detector storage, not the
reader's XTC allocation or parser tables.

| Storage | Owner | Becomes reusable when |
| --- | --- | --- |
| Original XTC device bytes (`data_gpu`) | Reader input allocation, reserved by an `InputWindow` | I/O, parsing, and every required gather using this generation have completed |
| Per-window parser records and locator tables | Parser allocation pool, reserved by the window | All internal operations reading those tables have completed |
| Materialized detector fields/raw data and interpretation metadata | Independent detector-data owner | No planned or active consumer remains, and every submitted consumer has completed |
| Calibration, image, and task outputs | Independent output owner | Their last consumers, including asynchronous D2H, have completed |
| Execution scratch | Execution slot | All operations using that scratch have completed |

Run-scoped Configure tables are shared metadata, not per-window parser storage.
An execution slot holds references to its inputs and outputs; its retirement
does not automatically make every referenced allocation reusable.

Only internal parser and gather operations may read original XTC storage.
User-facing field access and user kernels must never receive pointers or views
into original XTC bytes or per-window parser tables. The reader writes those
bytes initially and may overwrite them only after the reservation is released.

One shared materializer copies each detector's fields and segments to storage
owned by that detector.
Materialization preserves batch/event identity, timestamp, detector/segment
mapping, shapes, types, and presence/damage information needed downstream.
Any retained metadata must be independent of recyclable parser tables.
Subsequent access must not lazily return to the original XTC buffer.

All supported user-accessible fields must have a materialization path, including
detectors without a calibration adapter. An unsupported field must be reported
explicitly; retaining an implicit original-XTC view is not a fallback. Exact
field selection and representation should be reviewed before implementing the
accessor change, so existing field access is not silently narrowed.

## One materialization path for all GPU detectors

Every configured GPU detector uses the same materialization entry point,
allocation/ownership contract, copy-obligation tracking, and readiness protocol.
Configure field handles and parser locators describe the copies. Scheduling
must not branch on detector name/type or on calibration-adapter availability
to choose a different raw-input path. The field set includes all supported
user-accessible fields, not just the field needed by calibration.

Jungfrau uses this same path. Move or generalize its existing gather machinery
into the shared materializer and remove XTC gathering from detector processing
when the new path is connected. Do not leave a Jungfrau legacy-copy branch or
copy its raw pixels twice. Detector adapters receive materialized data and
independent metadata only; detector-specific calibration and interpretation
remain downstream of materialization.

Different field sizes, ranks, and layouts may require different low-level copy
kernels or destination layouts within the shared implementation. Such selection
is based on field metadata, with the same ownership and completion rules for
every detector. Preserve payload bytes and types rather than converting all
fields into a Jungfrau-shaped uint16 or float32 array. Support scalar, array,
variable-shaped, and multi-segment fields within the existing supported field
contract. An unsupported representation produces an explicit error rather than
selecting a legacy XTC-access path.

## Dgram consumption and completion

Every admitted dgram is materialized for its configured detector consumers,
even when no user kernel is scheduled for that event. A false conditional
predicate may skip compute, but does not skip this materialization policy.

Track these logical states for each dgram:

```text
UNREAD -> COPIES_IN_FLIGHT -> CONSUMED
```

A dgram is consumed only after every required detector/field/segment copy has
completed on the GPU. Enqueueing or starting a copy does not count as consumed.
One dgram may serve multiple detectors, so a single bit requires an underlying
completion count or an equivalent grouped completion record.

The BD scheduler establishes the complete set of required copies and seals it
against further additions before allowing the dgram to become consumed.
Each completion is accounted exactly once. Explicitly handle absent fields and
empty descriptors; they must not leave an obligation that can never complete.

Record CUDA events after groups of gathers. The scheduler polls completion
without blocking during normal scheduling and marks the covered obligations
complete. A CUDA event per dgram is unnecessary. When a group's copies span
multiple streams, its completion must cover every participating stream.

Release a window when its consumption plan is sealed,
`consumed_count == n_dgrams`, and no I/O or parser operation remains outstanding.
Calibration completion, user-kernel completion, and event-loop advancement are
not release conditions for original XTC storage. Those consumers use separate
detector-owned storage.

The first implementation can release raw bytes and parser storage together at
this boundary. Separate pools may release them independently if their actual
dependencies are tracked. A partially consumed window remains reserved; its
completed dgrams do not imply that arbitrary portions can be overwritten.

## Persistent allocation pools

Reader-buffer IDs are independent of execution-slot IDs. The scheduler reserves
any available reader buffer with suitable capacity, using this lifecycle:

```text
AVAILABLE -> READING -> PARSING -> COPYING -> AVAILABLE
```

Reuse keeps the GPU allocation and replaces its contents with the next read.
It is not normally a free/reallocate operation. Parser allocations likewise
retain capacity and refill their tables. Allocate initially or grow an
available allocation within the budget when necessary; never resize storage
that still has a live reservation.

Increment a generation ID on each reservation. Read receipts, gather completion
records, and release operations identify that generation, so an obsolete
notification cannot release or access a replacement read.

For example, reader buffer A gathers batch X into detector-owned storage for
execution slot 0. Once those gathers finish, A can read batch Y and gather for
slot 1 while slot 0 still computes X. Neither execution owns buffer A.

## Stream ordering and overlap

Use an explicit input-preparation CUDA stream, separate from execution-slot
streams. Parsing and gathering run on the preparation stream; calibration and
user work run on execution streams after waiting for detector-data readiness.
One preparation stream is sufficient for the initial design; additional input
streams are a later scheduling choice.

```text
Reader I/O:       read completion
                        |
Preparation:     [H2D if required] -> parse -> gather group -> gather_done
                                                                |
Execution:                               wait(gather_done) -> calibrate -> task

Scheduler:       all gathers for window complete -> return source to free pool
Reader I/O:                                      -> next read into same buffer
```

KvikIO file I/O is not assumed to be a CUDA-stream operation. Its futures and
the backend's visibility contract establish when parsing may begin. If a
separate H2D transfer is required, its completion must also precede parsing.
Do not substitute a CUDA event for outstanding file-I/O completion.

Record `gather_done` immediately after the last source-reading operation in
its group, before calibration or task work. Execution waits on the appropriate
group's event; reclaiming an entire reader buffer waits for every group that
uses it. The scheduler must continue polling input completions while execution
slots are busy, rather than waiting for an execution-retirement path to poll.

Avoid device-wide or default-stream synchronization in the steady-state path.
Separate streams permit overlap; actual overlap and throughput must be measured
for the I/O backend, device, memory pressure, and workload.

## Fast and slow detector example

For an EB batch containing 1,000 fast dgrams and 10 Jungfrau dgrams:

1. Bulk-read the contiguous fast dgrams when the read and destination capacities
   fit the budget. Gather their required data into retained fast-detector storage.
2. Release the fast original XTC buffer and parser tables after all fast gathers
   complete. Retain the detector data for declared future consumers.
3. Read and materialize Jungfrau in budget-sized groups. Recycle each source
   window after its gathers complete, independently of Jungfrau computation.
4. A task using fast events 0 and 1 to decide whether to integrate Jungfrau event
   1 acquires references to those detector products before submission. It waits
   for their readiness and records completion after the conditional work.
5. Keep those references until the task completes, including on the false branch.
   Additional task streams must join or supply their own completion dependencies.
   Other planned consumers can retain the data longer; integration output has
   its own lifetime.

If all 1,000 fast events share one physical detector allocation, retaining two
events initially protects that whole allocation. Per-event references do not
imply independently recyclable storage. Do not store retained fast data in
execution scratch that another slot submission can overwrite.

This changes no GPUBAT1 layout, file coalescing rules, or `batch_size` semantics.
Logical event identity remains independent of compact detector-array indices.

## Admission, failure, and shutdown

Account for original XTC capacity, parser tables, materialized detector data,
outputs, scratch, constants, and transfer staging together. Reusable cached
allocations remain charged until actually relinquished. Count aliased storage
once and include pinned host staging in the corresponding host budget.

Reserve destination capacity before launching gathers. Admission must provide
a bounded path to consume every dgram: reserve all destinations for a retained
fast group, or guarantee progress through smaller slow groups whose destinations
can retire. Do not fill the budget with source windows and leave no room to
materialize or execute them. Bound planned retention and result queues; apply
backpressure when there is no safe capacity. Dynamic field sizes need a declared
bound or staged admission with reserved progress capacity.

On failure, never turn incomplete copies into successful consumption. Drain
submitted I/O and GPU work before releasing storage. If completion cannot be
proved, quarantine the affected generation and propagate the error. Early exit
may abandon unsubmitted obligations through an explicit cancellation path after
preventing new submissions; normal successful processing materializes every
admitted dgram. Preserve existing transition/calibration fences and drain once
on shutdown.

## Current implementation and review checkpoints

At documentation time, the Stage 3 worktree has independent `InputWindow`
references and parser reservations, but reader selection still follows execution
slot IDs. `EventPool.submit()` attaches input lifetime to `result_ready` after
detector processing and to delivered event field leases. Detector
`process_batch()` interleaves gather and calibration on an execution stream;
field access can still read original XTC storage. These require changes.

Preserve the validated Stage 3 implementation as a baseline, then implement the
following stages sequentially, with a separate commit and review gate for each.
Stages 3A-3D refine Stage 3 of the bulk-read plan. The new materializer is wired
into production for all GPU detectors together in Stage 3B; Stage 3A does not
introduce an alternative production path.

### Stage 3A: Shared field representation, ownership, and copy contract

Define a common detector-data owner and materialization interface in proposed
`gpu/gpu_detector_data.py` and `gpu/gpu_materialize.py`. Derive field coverage
from Configure-backed bindings independently of calibration support. Describe
destination offsets, actual shapes/types, physical segments, event identity,
presence, and damage using metadata that survives parser reuse. Define
destination size estimates, alias accounting, planned/active references, and
all-consumer completion tracking. Review dense and variable-size layouts before
implementing accessors; raw data ownership must not depend on execution IDs.

Review gate: inventory existing supported field types and access semantics;
exercise the same contract with Jungfrau, a fast detector without calibration,
and scalar/variable-shape/multi-segment fixtures. Verify independent metadata
and multiple-consumer retention. No production behavior change yet.

### Stage 3B: Shared automatic materializer and all-detector migration

Implement the common copy path from locator-described XTC fields into owned
destinations. Generalize/move reusable gather code from `gpu/gpu_detector.py`;
support the full field contract rather than only its current 2/4-byte pixel
gathers. Reserve destinations before copying and preserve types and values.

Wire this path into `gpu/gpu_stream.py` for every configured GPU detector,
including those without calibration adapters. Change `gpu/gpu_input.py` field
access to materialized storage. Change detector processing to consume that same
storage and remove its XTC reads/gathers and execution-slot-owned raw input
allocation. Keep calibration-specific processing downstream. Initially keep
source retention conservative while this common data path is validated.

Review gate: field/pixel parity for Jungfrau and other detector/field fixtures;
automatic copying with no user access or kernels; no duplicate Jungfrau raw
gather; no remaining adapter/accessor path into XTC or parser tables. Missing
fields, variable shapes, and unsupported representations must have explicit
behavior. Both detector classes must exercise the same materializer entry point.

### Stage 3C: Shared preparation and copy-completion release

Run parsing and materialization on an explicit preparation stream, with execution
waiting for detector-data readiness. Implement sealed copy obligations and
grouped completion polling in `gpu/gpu_input_window.py`, `gpu/gpudgram/batch.py`,
`gpu/gpu_stream.py`, and `gpu/gpu_events.py`. Release XTC/parser reservations
after all required copies and preceding I/O/parser work complete. User and
execution references retain detector products only. Apply identical release
rules to every detector, including events with no scheduled kernel.

Review gate: delayed gathers block source release; delayed computation does not.
Cover multi-detector dgrams, empty/absent fields, and multiple streams. Existing
reader selection may remain until Stage 3D; eligibility for reuse is established
here independently of execution retirement.

### Stage 3D: Independent source pools and scheduling progress

Select available reader/parser allocations independently of execution slots in
`gpu/gpu_kvikio_read.py`, `gpu/gpudgram/batch.py`, and `gpu/gpu_events.py`.
Protect reservations with generations and poll input completion while execution
slots are busy. Integrate minimum safe admission with `gpu/gpu_budget.py`:
reserve destinations/progress capacity, charge cached allocations, and handle
dynamic sizes without exhausting the capacity needed to consume admitted input.

Review gate: repeatedly reuse one source allocation across different execution
slots while earlier detector data remains in use. Exercise stale generations,
tight budgets, partial failures, early exit, and transition fences for the common
path. No detector-specific source retirement rules are permitted.

### Remaining bulk-read stages

Stage 4 completes admission/accounting for source pools, retained detector data,
outputs, scratch, staging, and growth peaks. Stage 5 retains materialized fast
data across smaller slow execution groups, using the same materializer for both;
source grouping is a byte-capacity decision, not a separate detector copy path.
Stage 6 proves conditional multi-event consumers and independent output lifetime,
including all completion tokens. Stage 7 runs full acceptance and measures
overlap/performance separately from correctness. The GPUBAT1 ABI, file coalescing
rules, and `batch_size` semantics remain unchanged.

Acceptance must include pixel/field comparisons against CPU data, multiple
consumers of one dgram, automatic copying without user field access, manual copy,
or kernel requests, delayed gathers blocking reuse, and delayed user compute
allowing source reuse while preserving detector
data. Exercise repeated source-buffer reuse across different execution slots,
stale generations, tight budgets, failure/early exit, and transition fences.
All consumer completion tokens must be honored for retained detector/output
owners; the existing single-token result-lease limitation cannot be relied on
for this contract. Record actual fallback/GDS mode and use profiling to verify
overlap; do not infer a performance gain from correctness tests.
