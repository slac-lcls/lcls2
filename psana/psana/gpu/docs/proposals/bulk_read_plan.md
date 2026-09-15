# GPU bulk-read implementation plan

Status: Stages 1 and 2 accepted. Stage 3 implementation and validation complete;
ready for review. Bulk reads are the default. Stages 4-7 remain proposed.
Source baseline: `803a70011` on `codex/psana2-gpu-xtc-parser`. The planner is connected to the reader by default; the existing
execution-subbatch schedule is preserved.

Design refinement: [detector materialization ownership](detector_materialization_ownership.md)
defines proposed Stages 3A-3D before Stage 4. It supersedes the retained-XTC
ownership and residency design below: all GPU detectors, including Jungfrau,
use one automatic materialization path, and long-lived fast data resides in
detector-owned storage. The existing Stage 3 evidence applies to the validated
baseline only; the refinement is not yet implemented. Stages 4-7 must use the
revised ownership contract and review gates in that proposal.

## Stage 1 review evidence

Implementation: `gpu/gpu_read_plan.py`. Focused coverage:
`tests/gpu/unit/test_gpu_read_plan.py` (both relative to `psana/psana`).

`build_read_plan(descriptors, capacity_bytes=..., max_read_bytes=..., batch_id=...,
input_window_id=...)` accepts immutable `ResolvedDgram` records containing
caller-resolved `ResolvedFile(path, chunk_id)` identities. It returns immutable
physical ranges and logical rows retaining their source descriptors. The plan
carries the batch/window identity; each logical tuple position is its future
parser-row index. This is an internal CPU API, not a GPUBAT1 change.

For four 256-byte fast dgrams at offsets 4096, 4352, 4608, 4864 and two
illustrative 1024-byte Jungfrau dgrams at offsets 8192, 9216:

| File | File offset | Read bytes | Window device offset |
| --- | --- | --- | --- |
| Fast | 4096 | 1024 | 0 |
| Jungfrau | 8192 | 2048 | 1024 |

The six logical rows retain the order F0, F1, J1, F2, F3, J3 and receive
device offsets `[0, 256, 1024, 512, 768, 2048]`. The plan requires 3072 bytes
and two pread submissions. Setting `max_read_bytes=1024` produces three reads
without splitting a dgram. These are synthetic sizes, not measured Jungfrau
frame sizes. Stage 1 packs one window; separate fast/slow owners arrive later.

Validation: 46 planner cases passed, including simulated byte-for-byte reads
and standalone execution with only Python's standard library. The complete
CPU GPU-unit suite passed: 211 cases in 5.23 seconds. Python-only installation
was refreshed, and the imported planner was verified byte-identical to source.
No GPU allocation, real file I/O, or performance claim is part of Stage 1.

Reproduce after activating this checkout's runtime:

```bash
unset PS_PARALLEL SIT_PSDM_OFFSITE
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws
python -m pytest -q psana/psana/tests/gpu/unit
```

The existing MPI look-ahead unit test requires the normal MPI import mode;
an initial run with `PS_PARALLEL=none` failed that test before the corrected
211-case pass. Site MUNGE diagnostics and existing amitypes deprecations were
also emitted; the final pytest exit status was zero.

Stage 1 was committed as `c71eac4d2`. Stage 2 started after user review.

## Stage 2 review evidence

`DataSource(..., gpu_det=...)` uses the adjacent-range reader in existing slots
by default, as does `hybrid_det`. `gpu_bulk_read=False` retains per-dgram reads
for debugging and comparison. The BD's
`GpuFileEpochs` snapshots run-start file identity and applies ordered SMD
Enable/chunkinfo transitions before CPU EventManager processing. Immutable
event/stream mappings survive later mutations to CPU file handles. Replayed
history does not rewind chunk state; every transition fences coalescing.

`GpuEventManager._setup_gpu_pipeline()` orchestrates run setup, including the
shared budget, detector adapters, parser/execution slots, input I/O, and D2H
pipelines. Its `_setup_input_io()` helper creates the BD-owned reader and
run-scoped file resolver after the budget and slots exist.

`KvikioGpuReader` now owns pending destinations, ranges, futures, and file
handles through I/O completion. On submission or completion failure it drains
all started futures exactly once, preserves the first error and its cause,
and refuses further submissions. File caching uses resolved file identity
and prunes obsolete handles only after pending reads release them. This I/O
cleanup also applies to the temporary per-dgram comparison path.

The read plan supplies allocation size and logical device offsets. Existing
slot capacity plus available tracked budget bounds the raw-input allowance.
There is no new public per-request size knob and no gap over-read. Broader
fixed-allocation accounting and resident-fast scheduling remain Stages 4-5.

`io_stats()` retains `total_bytes` (bytes from fully validated physical reads),
`total_ns` (wait-only time), and the existing bandwidth calculation. New fields
are `total_requests`, `requested_bytes`, `useful_bytes` (fully successful
batches), `issue_to_complete_ns` (summed submission-to-completion wall time),
and `bulk_read`. Concurrent batch durations may overlap; their sum is not an
end-to-end bandwidth denominator.

CPU validation: all 234 GPU unit cases passed. New cases include fault
injection, immutable file resolution, and the tracked chunking fixture with
exclusive GPU routing and events on both sides of a chunk change in one EB
packet. Device validation job `58335140` passed all 14 GPU integration cases
in 467.54 seconds on an NVIDIA A100-SXM4-40GB, including raw-byte/field
comparisons and exclusive/hybrid Jungfrau pixel checks. KvikIO compatibility
mode was True and GDS was unavailable: this validates fallback I/O, not GDS
or a throughput improvement. This run used explicit bulk-read selection before
the default changed. Logs and the submitted script are under
`validation/bulk-read-stage2/` (generated, not committed).

After changing the default, all 235 CPU GPU-unit cases passed in 5.92 seconds,
including DataSource default propagation, CPU-only special batching, and a
no-argument reader producing two physical reads for six logical dgrams. The
exclusive/hybrid GPU acceptance cases now omit the selector; their previous
explicit-True run above exercised the same coalescing implementation.

Final Stage 2 recheck after the setup refactor: 235 CPU cases passed in
5.32 seconds. Perlmutter job `58338308` passed all 14 GPU integration cases
in 466.33 seconds on nid001524 (A100-SXM4-40GB), including default-mode
exclusive and hybrid pixel-exact cases. KvikIO compatibility mode was True;
GDS was unavailable. Installed runtime modules were byte-identical to source.
Evidence: `validation/bulk-read-stage2/recheck-unit.log` and
`recheck-58338308.log`. Stage 2 acceptance is complete.

## Stage 3 review evidence

Stage 2 was rechecked and committed as `6f4171c92` before Stage 3 began.

`InputWindow` in `gpu_input_window.py` owns a parsed batch and its descriptor
identity, with explicit references for planned uses, execution, and event
consumers. Closing a window stops new root acquisitions. An existing reference
can split into child references (for example, a field-view context); actual
retirement starts only after every reference is released. CUDA dependencies
transfer to the owner and must complete before its backing storage is reusable.
There is no garbage-collection release policy.

The KvikIO read receipt pins its raw buffer by generation. A retained input
blocks overwrite, and an obsolete receipt cannot pin replacement bytes.
`GpuXtcBatchPool.parse_window()` leases free parser storage independently of
execution-slot IDs and shares the run's Configure tables. Parser/submission
errors retain both raw and parser storage if CUDA completion cannot be proved.

`GpuEventDgrams.from_windows()` composes stream views using BD-local batch
identity, original event index, stream ID, and timestamp. Each stream keeps its
own raw base and locator row; no payload concatenation is needed. EventPool
accepts these independent windows and reserves execution/event references.
The default path still creates one input window per existing execution
subbatch. Independent residency admission and scheduling remain Stages 4-5.
Field-view contexts and independent field copies reserve their own input uses;
the separate detector-result lease fan-out issue remains open.

CPU validation: 243 tests passed in 5.28 seconds. New coverage retains fast
input through repeated slow execution retirement, validates composed identities,
blocks acquisition during retirement, waits for delayed/failing consumers,
rejects obsolete raw reads, and retains storage on parser/execution failure.
GPU acceptance includes a delayed CUDA consumer plus stable raw bytes and
locator addresses through slow input/parser/execution reuse, followed by the
existing pixel-exact suite. Initial GPU job `58338947` passed all 14 existing
cases but the new test used unavailable CuPy `Event.query()`. The test now uses
the installed `Event.done` property and explicitly orders initialization before
its delayed kernel. Final-source job `58339082` completed with exit code 0:0
on nid002441: all 15 GPU cases passed in 465.80 seconds on an A100-SXM4-40GB.
This includes the corrected retained-input/locator test and all pixel-exact
cases. KvikIO compatibility mode was True; GDS was unavailable. Source hashes
still match `validation/bulk-read-stage3/runtime.sha256`. The result log is
`validation/bulk-read-stage3/final-58339082.log`. Stage 3 is ready for review;
its implementation remains uncommitted, and Stage 4 has not started.

## Deferred cleanup

- Remove `gpu_bulk_read` and the per-descriptor submission branch after this
  integration. Always use the planner; an isolated descriptor naturally becomes
  a single-descriptor physical range. Keep both paths unchanged for now.
- Remove tests whose purpose is comparing the legacy and planned read paths
  when removing that branch. Use CPU-referenced pixel-exact/raw/field checks
  as the end-to-end correctness oracle. Retain planner bounds/request-count,
  chunk-transition, fault-injection, and ownership tests; those cover behavior
  beyond successful pixel output.

## Target behavior

For one BD receiving an EventBuilder batch with 1,000 fast-stream dgrams and
10 slow-stream dgrams, read all 1,000 fast dgrams together when contiguous and
affordable. Read and process the slow detector, such as Jungfrau, in smaller
groups determined by the remaining memory capacity. Reuse the fast input across
those groups without copying or rereading it.

`batch_size` remains the target number of aligned L1Accept timestamps sent to
one BD. It does not become a per-detector count. Keep these units separate:

| Unit | Purpose |
| --- | --- |
| EB batch | Communication and logical event identity |
| Physical read range | One psana-level KvikIO request within one file |
| Input window | Resident input bytes and their parser tables |
| Execution subbatch | Events processed together within a working-memory allowance |
| Result retention / D2H | Output lifetime and delivery, potentially longer than execution |

Read size does not dictate kernel launch size. One KvikIO request may itself be
split into lower-level tasks by KvikIO; report psana request counts accurately.

## Baseline and scope

The completed Perlmutter validation covers 176 passing pytest cases and a
four-GPU smoke check using CPU fallback. Preserve that numerical baseline.

Today `GpuEventManager._split_subbatches()` selects common event ranges before
`KvikioGpuReader.issue_batch()` reads every selected stream into a slot-owned
buffer. `GpuXtcBatchPool` also indexes parser storage by execution slot.
Consequently, small slow-detector capacity fragments fast reads too.

Implement in separate reviewable changes. Start with adjacent-only ranges;
then add independent input windows to deliver the full mixed-rate use case.
Keep Smd0, GPUBAT1 field layout, detector routing, parser validation, and public
event ordering intact. CPU planning uses descriptors and SMD transition
metadata, never detector payload parsing or a GPU metadata round trip.

Defer the external per-event result-consumer API redesign, user-task C ABI,
integrating-detector GPU support, callback routing, multi-EB GPU coordination,
and parser parallelism experiments. New internal input ownership must still
wait for every consumer. The known external result-lease fan-out limitation
remains documented; this work does not claim to fix it.

## 1. Resolve descriptors and plan physical ranges

Introduce a CPU-only planner, proposed `gpu_read_plan.py`, with immutable
records. Names below are implementation suggestions, not new public APIs.

| Record | Required information |
| --- | --- |
| ResolvedDgram | Batch identity, event index, timestamp, stream ID, file/chunk identity, file offset, dgram size |
| ReadRange | File identity, file offset, read size, destination offset, input-window ID |
| LogicalDgram | Original identity, input-window ID, parser-row index, device offset, dgram size |
| ReadPlan | Ranges, logical mappings, useful/fetched bytes, allocation capacity, request count |

Within one input window:

1. Resolve immutable file identities before submitting any I/O.
2. Group nonempty dgrams by resolved file, then order by file offset.
3. Merge exactly adjacent ranges, subject to the admitted window/range cap.
4. Preserve the original logical event/stream rows independently of physical
   ordering. Rebase a dgram at file offset `f` in a range starting at `r` and
   destination `b` to device offset `b + (f - r)`.
5. Walk only logical dgrams. Gaps, padding, and other file bytes are never
   interpreted as selected events.

Do not merge across files, chunks, transition fences, or resident windows.
Do not split a dgram between windows in the first implementation. Validate
bounds/overflow and reject unexpected overlapping descriptors. Preserve the
existing semantics of missing and zero-sized descriptors.

Multiple detector consumers of a stream share its descriptor and storage.
Do not broaden current routing: exclusive `gpu_det` requires a sole normal
detector owner per stream; hybrid routing supports shared streams and retains
its intentional CPU I/O.

### File and chunk resolution

GPUBAT1 currently has stream IDs and file-relative offsets, not per-descriptor
filenames. `EventManager` discovers chunk changes from Enable/chunkinfo SMD
metadata and updates `DgramManager`. Before Stage 2, the GPU reader cached
CuFile objects by stream ID; it now keys them by resolved file identity.
GPU pre-issue can precede CPU EventManager processing.

Build an ordered file-epoch map from the paired SMD/transition packet before
GPU pre-issue. Map descriptors to the active file at their event position;
do not use the final mutable `dm.xtc_files` state for the entire batch. Share
metadata resolution with the CPU path where possible, without requiring CPU
bigdata reads. Include exclusive streams and replayed step history.

Key CuFile caching by resolved file identity. A pending read owns a handle
reference even if the stream advances to another chunk. Resolve and test
chunk identity before enabling the feature on chunked input; never guess from
an offset reset. If existing envelope metadata proves insufficient, document
the precise case before proposing a transport change.

## 2. Execute coalesced reads safely in existing slots

Replace the per-dgram submission loop with per-range submissions. Maintain
the existing logical descriptor-table interface for the parser. Compute
allocation size from the physical plan, never from the last logical row.

`PendingBatch` owns the plan, destination allocation, referenced handles, and
every submitted future from the first successful submission. Its states are:
planned -> submitted -> I/O complete -> parser ready -> draining -> reusable.

On partial submission, short read, or future failure, stop new submissions,
drain every outstanding future, and preserve the original error with range
and affected logical-event context. Do not parse incomplete bytes, reuse the
destination, or close handles prematurely. Apply the same cleanup to early
iteration exit and parser/processing errors. CUDA completion errors must not
be treated as permission to reuse storage.

This stage is useful independently, but only merges within the existing common
subbatch. It does not yet satisfy the full 1,000-fast-plus-small-slow-window goal.

## 3. Separate resident input ownership from execution slots

Introduce an `InputWindow` owner containing raw bytes, logical rows, parser
tables/locators, readiness events, memory reservations, and completion state.
Execution slots retain detector scratch/output storage and reference one or
more input windows. Configure device tables remain run-scoped and shared.

Refactor the parser allocation interface to accept input-window storage rather
than assuming execution-slot identity. Parse a resident window once, retain
its locator tables, and reuse them across execution subbatches. Keep the
existing walker and locator formats; each parsed window has one base buffer.

Extend `GpuEventDgrams` construction to assemble an event from stream views
belonging to different parsed windows. Each `GpuStreamDgramView` already has
its own batch/base pointer. Replace the single-batch contiguous-row assumption
with explicit mappings keyed by batch identity, original event index, and
stream ID. Do not concatenate or copy full fast and slow payloads to rebuild
one monolithic buffer. Keep missing streams, physical segment identities,
and detector ordering unchanged.

An input window is reusable only when:

- No planned future execution still needs it.
- No CPU-side submission or live execution reference can add another reader.
- Its I/O, parser work, and all registered CUDA consumers have completed.

Use explicit scheduled-use references and completion events. Python garbage
collection and generator advancement are not release signals. Prevent new
consumer registration once retirement begins. Retiring a slow execution slot
must not reset fast parser tables or release the shared fast owner.

Bridge existing per-event parsed-input access to these owners using the
InputSlotLease completion mechanism. Preserve current public yield behavior;
do not attach the shared owner's lifetime to just one event's result lease.

## 4. Plan resident streams and execution windows together

Initially admit one EB batch at a time for shared residency; bound slow
execution concurrency by the existing execution-slot depth. Do not prefetch
another full fast batch while the current batch is resident in the first
version. Further overlap can be added after measuring the memory cost.

Choose residency using descriptor bytes and parser costs, not detector names
or presumed rates. A deterministic first policy is:

1. Determine the minimum executable event working set and reserve progress
   capacity, including required detector outputs and scratch.
2. Consider complete stream inputs in ascending resident-byte cost, with
   stream ID breaking ties. Admit a complete stream only if all remaining
   executions retain a feasible working set. Apply a configured range cap.
3. Build ordered execution ranges from actual descriptor presence and detector
   bindings. Include all streams required by each detector event. Do not charge
   every detector for an event where all its sources are absent.
4. Group nonresident stream reads inside each execution range; resident stream
   references reuse existing windows. Include fast-only events in execution
   and delivery, even when no slow dgram is present.
5. If the full fast input cannot coexist with the working set, shorten its
   resident window. Reduce overlap before rejecting work. If one minimum
   complete event cannot fit, fail before issuing reads with a byte breakdown.

For the target example, if all fast input fits alongside two Jungfrau events,
the fast read occurs once and the ten Jungfrau events execute in five groups.
Exact event-range boundaries follow timestamps/presence, not an assumed fixed
fast-to-slow ratio. With two slow groups in flight, their combined memory must
fit; otherwise run one at a time.

The first implementation continues using existing detector adapters internally.
It may process fast data in execution-range slices even though it reads and
parses all fast data once. A future fast-only user task may process the full
resident window in one invocation; that is a distinct execution choice.

### Memory accounting and forward progress

Admission includes resident fetched bytes and parser tables, active transient
inputs, detector gather/output buffers, fixed calibration/geometry/configuration,
retained results, allocation-growth peaks, and a stated allocator margin.
Track host descriptor/plan storage and outstanding I/O bounds as well.

Record existing fixed allocations once in the accounting used by this mode,
respecting CUDA-IPC ownership; do not subtract them twice. Reserve before new
allocations and roll back failed reservations. For reused capacity, charge
only additional capacity. Distinguish making a window reusable from freeing its
allocation: cached live capacity remains charged until actually relinquished.
CuPy pool statistics alone are not the ledger.

Bound result retention and include existing D2H allocations in reporting;
do not accumulate all slow calibrated images while waiting to publish a batch.
Drain/deliver completed executions to release capacity. The full pinned-host
API redesign remains separate, but this mode must not create an unbounded
new host queue. Unknown dynamic processing sizes require an adapter-declared
bound or bounded staged admission after parsing before output allocation.

## 5. Internal consumer boundary and transition handling

Provide a narrow internal execution hook with device input views, stable event
IDs, an execution stream, and declared output/scratch requirements. Initially
the existing detector processing supplies the consumer. Use a small test kernel
to verify that a future user task can consume fast and slow views internally.
Do not finalize the proposed public task API or C ABI as part of bulk reads.

The task enqueues work on the supplied stream; psana records completion after
submission. Additional streams must join that stream or provide dependencies.
Inputs are borrowed through GPU completion. Output ownership belongs to a
separate managed owner, so releasing an input does not invalidate outputs.
Match detectors by explicit event identity. Exposure-interval associations and
reductions require declared semantics and are not inferred from slow rates.

### Conditional angular integration example

For target Jungfrau event 1, explicitly declare fast events 0 and 1 as decision
inputs and Jungfrau event 1 as the integration input. These are original batch
event identities, not positions in compact per-detector arrays. The producer
must define which fast fields and which Jungfrau product (raw or calibrated)
the task consumes. This example stays within one EB batch; cross-batch history
would require a separately bounded retention policy.

Before launching the decision, acquire references to all declared input owners
and reserve the integration scratch/output capacity. On the task stream, wait
for the required input-read/parser/detector readiness events, evaluate the fast
predicate, then enqueue angular integration guarded by that device predicate.
Record `task_done` after the final task work. The predicate and any scratch
storage also remain valid until their last consumer completes.

The default GPU-gated implementation reads Jungfrau and skips integration work
when the predicate is false. It does not require a predicate readback to the
CPU. Skipping Jungfrau I/O itself would require an earlier decision followed by
CPU scheduling and is a separate optimization. Publish an explicit skipped
status when the predicate is false; do not expose uninitialized or stale output.

Attach `task_done` to the fast-input and Jungfrau-product owners. Neither fast
event 0 or 1 nor the Jungfrau event-1 input may be overwritten until that event
completes, even if the task's host function has returned. The false branch also
records completion so that skipped tasks release their references normally.
Errors follow the same outstanding-work drain rules as other executions.

Logical event references do not imply independently recyclable physical bytes.
If fast events 0 through 999 share one resident allocation, the first version
protects that whole allocation while this task runs. Jungfrau event 1 likewise
protects its containing input/output slot. After `task_done`, drop this task's
references, but retain any owner still needed by another planned or active
consumer. Integration output has its own downstream lifetime.

Acceptance includes true/false predicates, explicit event-index mapping,
delayed integration while other work finishes, and attempted slow-slot reuse.
Verify all three input views remain unchanged through completion and that the
fast owner survives if another task still needs it. Keep this internal test
contract independent of the eventual public task ABI.

Keep current public event delivery ordered. External result access retains its
existing contract and limitations. All-exclusive, hybrid, and missing-stream
events must preserve CPU/GPU event identity.

BeginStep drains dependent work before replacing constants. EndRun and early
exit drain exactly once. File-changing transitions fence read planning and
resolve the next file epoch; ordinary transitions do not force unrelated GPU
work to synchronize. No physical read spans a file epoch or calibration fence.

## Implementation sequence and acceptance gates

Implement the following stages sequentially, with a review checkpoint after
each stage. When implementation is authorized, keep each stage in its own
commit (or small clearly identified commit series), run its focused checks,
and present the diff, changed behavior, test evidence, and remaining limits
before starting the next stage. This plan itself does not start implementation.

### Stage 1: Pure CPU range planner

Add `gpu_read_plan.py` and focused unit coverage. Accept already-resolved file
identities and logical descriptors; return adjacent physical ranges, rebased
logical offsets, request counts, and capacity requirements. Keep it independent
of CuPy, KvikIO, MPI, and runtime scheduling.

Review gate: inspect exact tables for the four-fast/two-Jungfrau example;
verify six logical descriptors become two ranges when capacity allows. Cover
gaps, interleaved streams, file boundaries, missing/empty descriptors, caps,
and integer/overlap validation. No production-path behavior changes yet.

### Stage 2: File resolution and coalesced I/O in existing slots

Integrate the planner in `gpu_kvikio_read.py`. Resolve per-event file epochs
from ordered SMD metadata before pre-issue; retain immutable file identities
and handle references. Submit one pread per physical range, allocate from the
plan's capacity, and drain all started operations on error. Enable adjacent
reads by default, retain a per-dgram comparison selector, and add
useful/fetched-byte and request-count reporting. Touch
`gpu_events.py`, SMD helpers, and `gpudgram/batch.py` only as required by this
existing-slot integration.

Review gate: byte/field/output parity against per-dgram reads on a GPU;
chunk changes inside and between envelopes, including exclusive streams;
partial submission and short-read cleanup. The common subbatch schedule is
preserved, so full-fast residency is not delivered by this stage.

### Stage 3: Input ownership independent of execution slots

Introduce InputWindow ownership for bytes and parser tables. Refactor
`gpudgram/batch.py`, `gpu_input.py`, and `gpu_stream.py` to reference independent
input owners and compose event views across them. Share Configure tables and
retain all original event/stream identities. Add planned-use references,
readiness dependencies, and completion-aware retirement. Initially retain
the current execution grouping to isolate this refactor.

Review gate: existing field/numerical behavior is unchanged; a device test
holds one fast window while repeatedly recycling slow execution storage.
Delay an early consumer and verify neither fast bytes nor locators change.
Confirm retirement cannot race new references and that event-facing input
leases protect the correct owner. External result-lease fan-out remains open.

### Stage 4: Admission and capacity accounting

Add a deterministic admission planner in `gpu_events.py`/`gpu_budget.py`,
with detector estimates based on actual source presence. Account for resident
inputs, parser tables, slow working sets, fixed allocations, retained outputs,
and allocation-growth peaks. Integrate fixed owners only as needed for correct
accounting in this mode. Preserve CUDA-IPC ownership and reservation rollback.

Review gate: budget fixtures show full-fast admission when feasible, smaller
fast windows or less overlap under pressure, and early rejection of an
oversized minimum event. Capacity reuse stays charged correctly; active and
future dependencies cannot consume the capacity reserved for forward progress.
Run these checks before enabling independent resident scheduling.

### Stage 5: Full-fast residency with slow execution windows

Replace the common pre-read split in `GpuEventManager._process_batch()` for
the bulk-read path. Admit complete affordable stream inputs, parse them once,
and reference them from ordered execution windows containing transient slow
inputs. Initially allow one resident EB batch. Continue using existing
detector adapters and public event delivery; fast read grouping may exceed
fast processing grouping. Preserve transition fences and bounded output
delivery during slow-slot reuse.

Review gate: the 1,000-fast/10-slow fixture performs one contiguous fast read
and five two-event slow reads under the specified capacity, with no duplicate
reads or missing/duplicated events. Validate numerical parity, missing streams,
hybrid routing, tails, tight budgets, BeginStep, EndRun, and early exit.
This stage delivers the principal bulk-read use case.

### Stage 6: Internal conditional-consumer proof

Add the narrow internal execution hook and a test consumer for the declared
fast-event-0/1 -> Jungfrau-event-1 decision. Use a small deterministic GPU
reduction as the integration stand-in and compare it to a CPU reference;
a production angular-integration algorithm and public task ABI are separate
work. Acquire all three input dependencies before task submission and record
completion after predicate-guarded execution on the supplied stream.

Review gate: test true/false decisions, delayed completion, another task
sharing fast storage, and attempted slow-slot reuse. Prove all inputs remain
valid until task completion and that output retention is independent. Report
the false branch as skipped and release references normally. No user event
loop is needed to launch this internal test consumer.

### Stage 7: Perlmutter acceptance, measurement, and rollout review

Run the existing CPU/fast-CUDA/run-51 pixel-exact/MPI smoke suites plus the
new mixed-rate and ownership coverage. Benchmark per-dgram reads, coalescing
within common slots, and full-fast residency at matched memory limits. Record
request counts, useful/fetched bytes, total I/O latency, CPU cost, throughput,
and peak device/host memory. Preserve commands and environment evidence.

Review gate: produce a correctness and performance report, explicitly label
fallback versus demonstrated true GDS, and update architecture/handoff notes.
Record unavailable GDS validation separately; only close issue-register items
actually fixed.

Per user direction, adjacent-only merging within existing slots is the default
starting in Stage 2. `gpu_bulk_read=False` retains per-dgram reads for debugging
and comparison. Later mixed-rate residency stages retain their own acceptance
gates. Keep tuning controls internal until measurements justify exposing them.

### Tests

- CPU planner: contiguous/gapped inputs, interleaved streams, non-monotonic
  physical order, empty/missing data, chunk changes within and between batches,
  caps, integer bounds, overlapping descriptors, and multiple detector consumers.
- Reader fault injection: partial submission, first/middle/last short read,
  exceptions with later futures still running, close during pending work, and
  allocation rollback. Assert every started operation is drained exactly once.
- Device tests: compare unmerged and coalesced raw bytes, field locators,
  timestamps, and detector outputs. Preserve malformed-XTC and shape errors.
- Mixed-rate fixture: 1,000 small fast dgrams and 10 large slow dgrams with a
  budget forcing at most two slow events per execution. Require one fast
  request when contiguous and under the cap, five adjacent slow requests when
  feasible, no duplicate reads, all events delivered once, and identical outputs.
  Test memory pressure reducing the fast window and/or execution concurrency.
- Ownership tests: delay an early consumer while later work completes; prove
  fast bytes and locators remain unchanged across slow-slot reuse. Exercise
  multiple internal streams, cancellation, retained output, tails, and BeginStep.
- Repeat the existing fast CUDA, run-51 pixel-exact, and MPI smoke acceptance.
  Add real-device transition stress. Do not reinterpret fallback success as GDS.

### Measurements and later extensions

Compare per-dgram, adjacent coalescing, and shared fast residency using the
same data, routing, batch size, GPU assignment, and memory limits. Record
useful/fetched bytes, psana request count, submission overhead, issue-to-complete
latency, parser/processing time, end-to-end throughput, and peak device/host
memory. Separate compilation/warmup and cache effects. Measure small-detector
and Jungfrau-heavy workloads separately; keep performance thresholds out of
pytest. Validate actual fallback/GDS runtime modes separately.

After the adjacent-only design passes, consider bounded-gap merging with
explicit maximum gap, over-read ratio, and range-byte limits. Charge physical
fetched bytes and padding. Additional resident EB batches, independent full
fast-task execution, generalized host handoff, and the public user-task API
are follow-up changes, not hidden requirements of the first release.

## Completion criterion

A BD can bulk-read an affordable complete fast stream from its EB batch,
reuse it across budget-sized slow executions, preserve all event identities
and numerical outputs, and finish or fail without premature storage reuse or
unbounded retention. Document measured request reduction and memory behavior,
update the architecture/handoff, and close only issues actually fixed in the
issue register.
