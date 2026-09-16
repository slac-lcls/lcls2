# GPU bulk-read implementation plan

Status: Stages 1 and 2 accepted. Stage 3 is committed as `a5f07ee38`.
Stage 4 is committed as `763a8df1b`. Stage 5 is committed as `694b9ff2b`.
Bulk reads are the default. Stages 6-7 remain proposed.
Source baseline: `803a70011` on `codex/psana2-gpu-xtc-parser`. The planner is
connected to the reader by default; Stage 5 adds independent resident inputs
while retaining ordered execution subbatches.

Deferred alternative: [detector materialization ownership](detector_materialization_ownership.md)
retains the proposed shared automatic-copy and early reader-reuse design for
future review. Experimental Stage 3A was reverted; Stages 3A-3D are not current
prerequisites. Continue Stage 3 review and Stages 4-7 using the retained-XTC
ownership described below. The proposal records the deferral rationale and the
current difference between automatic Jungfrau gathering and generic field access.

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
slot capacity and available tracked budget bound the raw-input allowance.
There is no new public per-request size knob and no gap over-read. Stage 4
adds fixed-allocation and pre-I/O growth accounting; resident-fast scheduling
remains Stage 5.

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
`validation/bulk-read-stage3/final-58339082.log`. Stage 3 was ready for review
and committed as `a5f07ee38` before Stage 4 started.

After reverting experimental Stage 3A, production sources again matched that
Stage 3 baseline. Reader/input-window references and detector result leases
retain their existing conservative consumer-completion behavior. The separate
result-lease fan-out limitation remains open.

Revert verification: all 243 CPU GPU-unit tests passed in 4.84 seconds. The
Stage 3 runtime hashes still match the sources tested by GPU job `58339082`.
Refreshed the installed package and removed the two stale Stage 3A modules
left by the incremental installer; neither module is importable. A CPU
control-flow probe confirmed that the reader pin and Jungfrau raw execution
slot are both held when joining a registered user-result consumer, and become
reusable only after that join. This does not extend the single-token result
lease to unsupported multi-stream fan-out.

## Stage 4 implementation and review

Stage 4 preserves the Stage 3 source and detector lifetimes. It adds admission
before reads; it does not materialize arbitrary fields or release reader storage
at gather completion. At the Stage 4 checkpoint, production still used common
execution subbatches; full-stream residency decisions were CPU planning output
only. Stage 5 connects those decisions to the input schedule as described below.

Call path:

1. `_setup_gpu_pipeline()` passes the shared budget into calibration preparation
   and geometry uploads. `_upload_fixed_arrays()` reserves before transfer and
   rolls back failed uploads only after completion is proved. IPC followers
   still skip constant allocation; `_compute_subbatch_budget()` uses charged
   fixed bytes once and leaves 10% allocator/runtime headroom, without a floor.
2. `_split_subbatches()` calls `_event_memory()` and `plan_admission()` in
   `gpu_admission.py`. Costs use actual source presence, all canonical dense
   detector rows, presence masks, raw bytes, and parser tables. A minimum event
   exceeding the total variable allowance fails before I/O with a breakdown.
3. `_issue_gpu_read()` first calls `_reserve_gpu_subbatch()`. Reader, parser,
   and detector `allocation_requirements()` report required/existing capacities
   for the chosen storage. `allocation_growth_bytes()` reserves all required
   new/replacement arrays while existing capacity remains charged. This covers
   old+new peaks even when old event views survive several replacements.
4. `_GpuBudget.hold()` keeps that credit unavailable to unrelated allocations.
   While the hold is active around reading and `_submit_gpu()`, actual allocations
   convert held bytes to committed bytes. Replaced old buffers return credit to
   the hold; final unused credit is returned after submission. Read, parser,
   submission, and early-exit failures close unused reservations.
5. If current retained/cached capacity prevents admission,
   `_retire_issue_and_yield()` drains remaining supported execution consumers,
   then `_trim_gpu_caches()` relinquishes unowned variable buffers and retries.
   Reader I/O/input pins and parser owners prevent trimming reserved storage.
   Detector slot buffers are trimmed only after EventPool is empty. Actual
   concurrency can reduce under pressure without changing event identity.

The follow-up residency-priority refactor (Stage 1, `4a26bc640`) considers complete streams
in ascending mean size of present, nonempty XTC dgrams. Equal means prefer the
smaller total source+parser footprint, then stream ID. This replaces Stage 5's
original total-footprint-first order so a sparse large-dgram stream does not
automatically outrank a frequent small-dgram stream. Missing events and empty
descriptors do not dilute the mean; all supplied descriptors still incur parser
cost. All-empty streams remain execution-scoped. Means are compared exactly.

The policy reserves minimum execution capacity before accepting residency, can
reduce overlap to admit the first affordable candidate, and otherwise builds
smaller ordered ranges. Full resident footprints, not mean dgram sizes, enter
the fit checks. It does not inspect detector names or assume a fixed fast/slow
ratio. Multiple detectors sharing one stream charge the source once and their
actual detector working sets separately. This ranking change does not alter
read coalescing, parser behavior, input leases, or the GPUBAT1 ABI.

Stage 2 records each attempted residency decision in
`AdmissionPlan.residency_decisions`: the exact candidate statistics, previously
admitted bytes, remaining maximum event working set, concurrency before/after,
capacity, and admission result. The optional field defaults to an empty tuple
when no candidates are considered. The trace consumes these recorded decisions;
it does not reimplement the ranking or fit checks. Each `CANDIDATE` line shows
the mean nonempty dgram size, input/parser footprint, `ADMIT` or `SKIP`, and the
actual `previous_resident + candidate + inflight * working <= capacity` test
(or `>` for a rejected candidate). Skipping residency keeps that stream's input
execution-scoped; it does not discard its events.

CPU scheduling and real-device tests now include frequent small dgrams whose
total input and input+parser footprint both exceed a sparse large-dgram stream.
They check one resident input across five two-slow-event execution groups,
exact read requests and event/field values, and retirement. CPU cases additionally
cover delayed consumers, failure/early exit, missing streams, and partial tails;
the device test rechecks resident bytes after all execution-slot reuse.
Stage 3 remains matched-policy acceptance/measurement work; request-savings
scoring and partial resident windows remain outside this refactor.

Stage 1 validation on SDF: 288 GPU CPU-only unit cases passed (13 new priority
cases), and all four byhand MPI cases passed. The full psana suite was not
clean: 350 passed, 14 skipped, 10 deselected, and two failures. The existing
subset-export test expects stdout to contain only the event count, but calibration
startup also prints there. The chunked bulk-read test fails when another module
sets `PS_SMD_N_EVENTS=1` during collection (`test_smalldata_callback.py`); it
passes alone and fails identically with that variable set. Neither failure
requires the new residency ranking. Those test-isolation/output issues were
not changed as part of this policy refactor. Logs are
`/tmp/gpu_priority_stage1_{unit,psana,byhand,chunk_isolated,chunk_env1}.log`
on the SDF login host. Device-policy comparison remains a later-stage gate.

Follow-up Stage 2 validation on SDF: 298 CPU-only GPU unit cases passed.
GPU job `38389330` passed ten fast integration cases (eight slow cases
deselected), including the new larger-total-footprint/smaller-dgram case.
Run-51 traces at 1.5 GiB (`38389330`) and 1 GiB (`38389359`) each delivered
the same 30 timestamps, shapes, dtypes, and detector sums. At 1 GiB, epix alone
was resident and all five Jungfrau streams remained execution-scoped, with
explicit failed-fit decisions. Sampled ledger commitments stayed below their
limits (1355.420/1536 MiB and 906.722/1024 MiB). These runs used CPU fallback,
not true GDS. The detailed trace record is in `../bulk_read_trace.md`; the trace
driver and launchers remain untracked diagnostic scripts, excluded from commits.

The Stage 2 full psana rerun reported 362 passed, 15 skipped, ten deselected,
and three failures: the same two Stage 1 issues above, plus the CPU smalldata
shared-memory path exiting without events and then finding no `/oneint`
dataset. The latter also reproduced with the main and byhand suites run
sequentially, so a shared-output collision alone does not explain it. No
smalldata/shared-memory code was changed; this failure remains unresolved.
The sequential byhand suite reported three passed and one failure in
`byhand_mpi.py::Test::test_mpi`, again at the shared-memory `/oneint` check.
Logs are `/tmp/gpu_priority_stage2_{unit,psana_sequential,byhand_sequential}.log`.

Independent test-fix commit `599f856ae` scopes the callback tests' MPI environment to each
test with cleanup, explicitly give the cross-chunk test a 1000-event SMD
window, and assert the subset event count inside its subprocess rather than
comparing diagnostic stdout. The subprocess uses the current Python executable.
All 34 affected cases passed with prepared fixture data, even with
`PS_SMD_N_EVENTS=1` inherited. Import/cleanup checks confirmed no environment
leak and restoration of both absent and existing settings. Sequential full
reruns passed: 365 main-suite cases (15 skipped, ten deselected) and all four
byhand MPI cases. The earlier shared-memory failure did not reproduce; no
shared-memory production fix was made. Logs are
`/tmp/psana_test_isolation_{focused_with_fixture,main,byhand}.log`.

Remaining residency-priority Stage 3 work is acceptance and measurement, not
another admission-policy implementation:

- Compare total-footprint-first and mean-dgram-first policies with identical
  data, event ranges, budgets, batch sizes, execution depths, and I/O mode.
  Include the mixed-rate case where frequent small dgrams have the larger
  total footprint, plus an equal-rate control and constrained-budget cases.
- Record per-stream admission and pread counts, request sizes, execution
  ranges/concurrency, memory high-water, and warmed event-loop throughput.
  Distinguish psana pread submissions from KvikIO's internal chunking. Current
  correctness traces are not an old/new performance comparison.
- Run the remaining slow/pixel-exact GPU acceptance cases and preserve the
  commands, runtime revisions, and results in a comparison report. Identify
  CPU-fallback versus verified GDS runs explicitly; current SDF evidence is
  fallback only. True-GDS performance remains a separate environment-dependent
  validation item, not a prerequisite for implementing the ranking.

Partial resident windows, request-savings scoring, changes to coalescing, and
changes to lease/backpressure ownership remain out of scope for this refactor.

The ledger charges cached pipeline-owned buffers until relinquished. Retained
slot results remain charged while their consumer leases are live. Independent
user copies and custom allocations are outside this ledger; the stated margin
does not make those unbounded allocations safe. Pinned host buffers, host
descriptor/plan objects, and KvikIO staging are not charged to the device ledger;
the current one-EB-batch/one-preissued-read schedule bounds descriptor and I/O
work, while existing D2H pool/chunk counts bound host staging. A general host
byte quota and future user-task output admission remain separate work.

CPU coverage includes full-fast admission with five two-slow-event ranges,
presence-aware costs, parser overhead, reduced overlap, minimum-event rejection,
growth peaks, cached capacity, protected trimming, failed uploads, unused-hold
rollback, and IPC-follower accounting. The device test exercises real
read/parse/gather under exact admission quotas across growth and reuse, compares
raw fields with CPU data, and rejects insufficient capacity before issuing I/O.
Perlmutter validation is recorded under `validation/bulk-read-stage4/`.

CPU validation: all 263 GPU unit cases passed in 4.38 seconds, including an
occupied-slot check that distinguishes EventPool capacity from live executions
and parser trimming that preserves owned rows and the fixed-allocation charge.
Installed runtime sources are checked against this worktree before GPU tests;
the frozen runtime manifest is `validation/bulk-read-stage4/runtime.sha256`.

GPU validation: job `58376353` passed all 15 existing GPU integration cases,
including the run-51 Jungfrau pixel-exact cases, in a 477.68-second suite run.
The new admission case passed read/parse/gather, growth, and reuse assertions
but exposed an incorrect budget attribute in parser cache trimming. After
correcting that attribute and adding the CPU regression, focused job `58377320`
passed the admission case in 7.81 seconds (Slurm exit 0). The final runtime
diff from the full run is confined to that trimming attribute correction;
`runtime-58376353.sha256` preserves the earlier manifest. These are two runs,
not a claim of a single all-green full suite. Both used A100 GPUs with KvikIO
compatibility mode True and GDS unavailable. Logs and both submission scripts
are in `validation/bulk-read-stage4/`. No GDS or throughput claim is made.

Stage 4 was reviewed and committed as `763a8df1b` before starting Stage 5.
Experimental Stage 3A automatic materialization remains reverted.

## Stage 5 implementation and review

The default bulk-read path now enables the admission planner's residency
decisions. Affordable complete stream inputs are read and parsed once for an
EB batch, then combined with transient input windows for ordered executions.
Detector processing still uses the existing execution slots and adapters.
The GPUBAT1 ABI and public event/field/result interfaces are unchanged.

Call path:

1. Setup gives the reader and parser `n_gpu_streams + 1` lazy input slots.
   Execution/detector slot counts remain `n_gpu_streams`. The extra reader slot
   holds the resident input; the parser selects any free slot independently.
2. `_process_batch()` calls `_split_subbatches(..., allow_residency=True)` for
   bulk reads. When a complete input fits, it drains the previous executions
   and trims free caches before `_start_resident_input()`. This initial policy
   bounds residency to one EB batch and favors predictable capacity over cache
   retention across resident batches; throughput tradeoffs remain to measure.
3. `GpuReadSelection.from_view()` filters descriptors by physical stream while
   preserving original event indices, timestamps, file offsets, and identities.
   It is an input view, not another serialized ABI. Execution views still
   describe all streams needed for their event ranges.
4. `_start_resident_input()` holds resident buffer growth plus the largest
   planned execution's working bytes before I/O. It reads into the extra input
   slot and creates one parsed InputWindow. The owner stays open for subsequent
   planned execution acquisitions; the unused admission credit is returned.
5. For each execution, `_issue_gpu_read()` selects only nonresident descriptors.
   `_input_allocation_requirements()` charges these input/parser requirements;
   detector requirements still count all present detectors in the execution.
   An execution with only resident inputs issues no new read.
6. `_submit_gpu()` parses transient input on `EventPool.next_stream`, then
   submits the full execution view with resident and transient InputWindows.
   Existing event/stream composition and leases bind every detector and event
   to the correct owner. The transient window closes to new acquisitions after
   submission; existing leases keep its storage until consumers complete.
7. Executions are delivered and retired in event order. Before leaving the EB
   batch, the pool drains and `_close_resident_input()` closes the resident
   window. Existing input references and CUDA completion events govern actual
   release. Generator close, pending-read failure, and manager shutdown also
   close the resident owner; no early reader-reuse/materialization path is added.

If no complete stream fits, production uses the existing common-subbatch
schedule. The temporary `gpu_bulk_read=False` path remains unchanged in scope.
Both residency and fallback retain the Stage 4 pressure-driven drain/trim/retry.
File epochs and transition fences still apply to every physical range.

New tests exercise the production manager, reader, and execution ownership:
the 1,000-fast/10-slow case produces one contiguous fast read and five two-slow
reads; other CPU cases cover smaller budgets, absent streams, partial tails,
resident-only executions, BeginStep/EndRun, hybrid CPU envelopes, max_events,
early generator close, and failed transient I/O. The device fixture uses real
XTC fields plus a valid opaque XTC sibling to enlarge the slow input; it checks
field values and dense raw/calibrated results while enforcing the same six-read
schedule. This is synthetic mixed-rate data, not a throughput benchmark.

CPU validation: all 275 GPU unit cases passed in 3.30 seconds. Validation
commands, source hashes, and logs are under `validation/bulk-read-stage5/`.
GPU validation job `58391075` passed all 17 integration cases in 465.26 seconds
with Slurm exit 0 on an NVIDIA A100-SXM4-40GB. This includes the six-request
mixed-rate fixture, input lifetime checks, and every run-51 Jungfrau pixel-exact
case (exclusive/hybrid, slot reuse, tails, and D2H chunk variations). KvikIO
compatibility mode was True and GDS was unavailable. Installed runtime sources
matched this worktree, and the final files match `runtime.sha256`.

An earlier attempt, `58390777`, passed ten cases before a calibration-server
connection reset interrupted setup of the next case. The successful retry used
unchanged runtime sources. This stage establishes correctness and request
grouping; throughput and true-GDS measurements remain Stage 7 work.

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
2. Consider complete stream inputs in ascending mean present, nonempty dgram
   size, with full resident-byte cost and then stream ID breaking ties. Admit
   a complete stream only if its full input+parser footprint leaves feasible
   execution working sets. Physical read ranges still obey any configured cap.
3. Build ordered execution ranges from actual descriptor presence and detector
   bindings. Include all streams required by each detector event. Do not charge
   every detector for an event where all its sources are absent.
4. Group nonresident stream reads inside each execution range; resident stream
   references reuse existing windows. Include fast-only events in execution
   and delivery, even when no slow dgram is present.
5. If a complete stream cannot coexist with the working set, keep its reads
   execution-scoped. Partial resident windows are a future extension. Reduce
   overlap when allowed by admission. If one minimum complete event cannot fit,
   fail before issuing reads with a byte breakdown.

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
