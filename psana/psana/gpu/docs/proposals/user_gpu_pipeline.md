# User GPU kernel support

**Status:** Proposed; the APIs below are not implemented.
**Review baseline:** `d63f45d27`,
`codex/psana2-gpu-bulk-batched-integration`, 2026-09-26, after merging master.
**Scope:** Producer-side user kernel submission, input/constant preparation,
named output publication, and asynchronous CPU delivery. No runtime changes
are part of this document update.

This is the canonical proposal for the next task. It supersedes the earlier
version at this path that required declared scratch/output arenas. It captures
the uncommitted September 17 draft from the stale
`codex/psana2-gpu-user-callback` worktree and the user's September 26
clarification. Implementation must start from the current branch, not that old
worktree. The [dated handoff](user_kernel_integration_handoff_20260926.md)
records merge validation and source history.

The recovered source was
`/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-user-callback/psana/psana/gpu/docs/proposals/user_gpu_pipeline.md`.
Its stale implementation claims have been reconciled below; the old worktree
has not been changed and is no longer needed to read this proposal.

## Recommendation

Add one user callback inside psana's existing BD execution loop, after input
reads, GPU XTC parsing, and any requested dense gather. An explicitly configured
callback replaces automatic detector calibration. The callback queues any CUDA
or CuPy algorithm on the supplied stream and publishes named device outputs.
Psana records completion and starts asynchronous D2H; the public event loop
only consumes results.

Detector-plane raw data and requested values from `det.calibconst` are staged,
read-only inputs to the user's kernel. Calibration, mask construction, inverse
gain, pedestal-plus-offset, and thresholding are user computations. Declaring
`calibconst` requests uploads, not invocation of the built-in calibration recipe.

Users allocate their own scratch and output memory. Psana does not size arenas,
allocate user device buffers, enforce their memory budget, or recover their OOM.
It must still retain registered owners until asynchronous consumers finish.
Allocation policy and asynchronous lifetime are separate responsibilities.

`GpuTask` declares input selectors and calibration-dictionary keys, not input
sizes or output shapes. Psana resolves input metadata; supported dense adapters
provide layouts, while generic runtime-shaped fields remain device descriptors.
Output metadata is obtained when the user publishes an actual array/view.
Native publication still needs pointer, device, dtype, shape/byte extent, and
an allocation owner; this is registration metadata, not advance allocation sizing.

Start with one callback, one supplied stream, and host-delivered outputs. A
callback can launch calibration, thresholding, reductions, or several other
kernels in sequence. No task graph, registry, task list, managed scratch API,
CUDA Graph interface, or user event-loop scheduling is needed.

## Current implementation and required changes

The runtime below was inspected at `d63f45d27`. It has independent read groups,
batched parser tables, and explicit input/result ownership. The old whole-stream
resident/transient selection policy has been removed; do not reintroduce it.
The callback APIs remain proposed.

| Layer | Implemented behavior | Consequence for injection |
|---|---|---|
| EB transport | Coherent CPU/GPUBAT1/step batches; descriptors carry event identity and file ranges | Keep transport unchanged; no user functions or device pointers in MPI packets |
| Routing | `gpu_det` owns whole streams exclusively; `hybrid_det` duplicates whole-stream reads | Keep routing explicit; no CPU/GPU partial detector merge is provided |
| File resolution and bulk reads | `GpuFileEpochs` resolves immutable file/chunk identities; adjacent ranges coalesce within file and transition boundaries | Keep physical I/O independent of callback invocation; do not read once per callback |
| Input groups | `GroupReadSchedule` and `InputGroupPool` manage bounded adjacent stream reads and independently reusable input allocations; small groups can remain live across executions | One callback event can reference several raw bases and shared parser arenas; never tie input ownership to one execution slot |
| GPU parser | Configure tables plus device walker/locators determine field addresses, types, shapes, status | Bind inputs once from Configure; consume device locator rows in the producer |
| Dense adapter | `GPUDetector.process_batch()` gathers fixed-shape arrays and immediately calibrates/passes through | Separate requested input preparation from automatic calibration |
| Injection point | `EventPool.submit()` composes `GpuEventDgrams`, calls detector processing, then records `result_ready` | Dispatch callback here, before the completion event |
| Host delivery | `_submit_gpu()` immediately schedules `_D2hPipeline`; `on_cpu` resolves its token | Preserve this placement and generalize transfer metadata |

Bulk-on uses resolved stream groups; bulk-off retains per-dgram submission.
Admission reserves psana-owned reader, parser, and detector allocation growth
before I/O. Group lifetime depends on planned uses, active holds, and CUDA
completion; parser arenas can be shared across groups. Preserve independent
group reclamation and byte accounting while adding callback consumers.

Sources: [orchestration](../../gpu_events.py), [execution](../../gpu_stream.py),
[admission](../../gpu_admission.py), [stream planning](../../gpu_stream_read_plan.py),
[group scheduling](../../gpu_group_schedule.py), [group ownership](../../gpu_input_group.py),
[input windows](../../gpu_input_window.py), and
[stream-read cleanup](../stream_read_legacy_cleanup.md).

### Gaps that matter before implementing the callback

1. **Generic public field access is not a producer input API.**
   `GpuFieldResult._locator_row()` waits for the locator and copies its row to
   the CPU to construct shaped CuPy views. Calling that accessor for every task
   would introduce a metadata round trip. Use device locators directly or the
   existing device gather pattern. The GPU walker/locator itself has no such
   D2H requirement. See [gpu_input.py](../../gpu_input.py).
2. **Automatic D2H is image-specific.** `_D2hPipeline._init()` assumes rank 3;
   `_PinnedSlot` and row offsets assume float32. Merely adding a named output
   would mishandle uint8 masks or scalar counts. Use shape, dtype, and byte size
   per named output. See [gpu_events.py](../../gpu_events.py).
3. **Multiple consumer completions are already supported.** `SlotLease` and
   `InputSlotLease` collect terminal events and protect open views. Preserve
   this behavior; no lease redesign is implied. Host-only task outputs and
   one supplied callback stream are proposed v1 scope choices, not repairs for
   a single-consumer limitation. See [context.py](../../context.py).
4. **Current delivery already waits for the outgoing producer.**
   `EventPool.begin_retire_next()` and `flush()` synchronize its stream before
   yielding. This proposal removes scheduling from the user's loop, but does
   not claim CPU consumption starts before that event's kernels finish. Other
   slots and D2H can overlap. Removing retirement waits is separate work.
5. **Callback side effects require an exact event selection.** Current
   `_process_batch()` limits CPU envelopes with `max_events` but can still
   submit the complete GPU subbatch to detector processing. Dispatch callbacks
   only for selected delivery identities, never for the discarded tail.
   Missing fields must not shift output-to-event associations.

No performance claim is made here; working KvikIO does not prove GDS is active.

The deferred [detector-materialization proposal](detector_materialization_ownership.md)
is not a prerequisite. Keep current leased input groups and parser descriptors;
separate supported dense input preparation from calibration without requiring
every field to move into a new detector-owned pool first.

## Minimal user contract

Proposed spelling:

```python
# New API: illustrative, not runnable against the current branch.
def userfunc(evt, stream):
    ...  # queue GPU work; publish zero or more results; return None

task = GpuTask(
    function=userfunc,
    inputs=["jungfrau.raw"],
    calibconst=[
        ("jungfrau", "pedestals"),
        ("jungfrau", "pixel_gain"),
        ("jungfrau", "pixel_offset"),
        ("jungfrau", "pixel_status"),
    ],
)
ds = DataSource(..., gpu_det="jungfrau", gpu_fn=task)

for run in ds.runs():
    for evt in run.events():
        process_cpu_data(evt)
        threshold = evt.gpu.get("jungfrau_threshold").on_cpu
```

`GpuTask` is a proposed public psana configuration type. The import location and
exact constructor validation will be finalized during implementation; this
example is not executable against the baseline.

`gpu_fn=None` preserves today's built-in processing. Supplying one task replaces
that processing for the selected GPU path: prepare declared inputs/constants,
then invoke the task. Do not run hidden calibration first or allocate the old
calibrated-output slots. Parser-only inputs do not require a Jungfrau adapter.
In `inputs`, a dense adapter name such as `"jungfrau.raw"` requests preparation;
a `(detector, algorithm, field)` tuple requests device field descriptors only.
All selected detectors use this replacement rule; mixed automatic/task pipelines
can wait. Existing public parser fields remain available under their current
lease rules.

The callback runs once for each selected L1Accept with GPU input descriptors,
inside its BD process. No callbacks for transitions or events having no selected
GPU descriptors; those events still reach the CPU loop without task outputs.
A missing requested detector may be reported as absent to the callback, allowing
it to skip or publish a result from another input. Multiple BD ranks have
independent callback instances. Host callback invocations are serial within a BD,
but submitted GPU work from different slots may overlap.

Scheduling belongs to the **batch/subbatch producer**, before any corresponding
public Event is yielded. The proposed first callback granularity is one event
within that submission, with one producer-completion event after the callback
loop. That does not require per-event I/O or per-event host synchronization.
A true batch callback is a later optimization, not a reason to move dispatch
into `run.events()` or `.on_cpu`.

```text
psana, for each admitted GPU subbatch:
  read groups -> parse -> prepare requested inputs/constants
  for each selected L1Accept:
    userfunc(producer_evt, slot_stream)   # enqueue only; publish outputs
  record producer_done on slot_stream
  enqueue named D2H copies after producer_done
  attach host tokens to the matching CPU events
  deliver through the existing event/retirement path

user event loop:
  process CPU data -> request a named result -> wait on its host token if needed
```

`evt` is a small producer context, not the public Python Event and not a device
object. Its metadata includes timestamp, run, step generation, batch identity,
and original event index. The callback is a host function that launches kernels;
it is not itself a `__global__` kernel. It must return after enqueueing work,
without waiting for kernels or copies. Psana runs it with `stream` current for
CuPy and also passes that stream explicitly. Native launchers use `stream.ptr`.

The callback takes exactly `evt` and `stream`. `GpuTask` declares the callable
and the inputs/constants psana must prepare; it has no `params` field. Algorithm
settings such as thresholds belong to user code, for example function locals,
module configuration, or user-owned callable state. Psana invokes
`task.function(evt, slot_stream)`, where `slot_stream` is its CUDA stream for the
current execution slot. Use the supplied stream on every invocation rather than
caching one across slots.

Setup validates input/constant selectors without constructing GPU objects on
SMD0 or EB ranks. Importable callables and dependency selectors are configuration;
CUDA state is created only on the assigned BD GPU. Compile/setup before steady
state where convenient, or lazily on the first call. There is no background
producer thread in v1: look-ahead is bounded by the existing execution ring,
and a stalled public loop eventually stalls submission too.

### Context operations

| Operation | Meaning |
|---|---|
| `evt.input(name)` | Borrow a read-only prepared dense input; `None` if the detector has no source dgram |
| `evt.present(name)` | Borrow a read-only device presence mask in the input's canonical segment order |
| `evt.field(det, alg, field)` | Read-only device field descriptors and payloads per configured physical segment; no locator D2H |
| `evt.calibconst(det, key)` | Borrow the read-only device value uploaded from a declared calibration-dictionary key |
| `evt.segment_ids(det)` | Host-known physical segment IDs in dense-input row order, for mapping into calibration arrays |
| `evt.keepalive(*owners)` | Retain user allocation owners through terminal completion; does not allocate storage |
| `evt.publish(name, array)` | Register a contiguous device array and its owner for automatic host delivery |

All borrowed storage is read-only: raw payloads, prepared inputs, presence masks,
locator tables, and calibration constants. Callbacks and their native kernels
must not write through these arrays, pointers, or aliases. Constants can be read
by overlapping execution slots, and input storage remains available to other
consumers, including public field access. An algorithm needing in-place changes
must first copy into user-owned storage on the supplied stream, registering that
owner before the copy. Read-only access is a caller obligation; v1 does not
promise to enforce it for arbitrary CuPy or native pointer writes.

Publish accepts supported numeric dtypes, including uint8 masks and uint32
counts, with host-known shapes including scalar shape `()`. It validates device,
contiguity, dtype, byte size, and name uniqueness within this event. It copies
metadata and retains the owner immediately; it does not read array contents.
For the first implementation, each name has one fixed shape/dtype per run;
subsequent mismatches fail clearly. Device-dependent variable-length results use
a fixed-capacity array plus a device count, published separately.

Output names are exact user-chosen keys. No unqualified aliases are inferred.
Duplicate publications or collisions with reserved input/result names fail.
No output is implicitly published for intermediates. Publishing before launching
the producer is valid: D2H is queued only after successful callback return and
the recorded producer event. Returning without publication produces no task
result; `get(name)` then raises `KeyError`.

### Inputs without a hidden synchronization

For the Jungfrau prototype, `evt.input("jungfrau.raw")` is the current canonical
uint16 gather, separated from calibration. Psana owns and budgets this input
buffer and its device presence mask. It zero-fills absent rows. Calibration
arrays retain their dictionary layout; `evt.segment_ids(det)` identifies the
input rows so the user can select the corresponding calibration segments.
The supported detector adapter must document the calibration segment-axis
mapping, including sparse IDs, rather than assume row number equals segment ID.
The helper retains today's fixed-shape adapter scope;
it is not a promise that every detector field is a dense image.

Presence means the field passed the current parser and dense-gather checks;
it does not mean the segment is free of all XTC damage. The gather requires
`STATUS_FOUND` and matching type, rank, and byte size, with valid bounds. The
current parser rejects the `Corrupted` damage bit, but other damage flags do not
necessarily prevent a field from being found and gathered. V1 preserves this
behavior: absent or rejected fields remain zero-filled with `present=0`, while
an otherwise valid field carrying another damage flag can have `present=1`.
The example excludes absent or rejected fields, not every damaged segment.
A broader damage-filtering policy would require explicit device damage access
and a defined rejection mask; that is outside this first implementation.

The general path exposes a list of host-known bindings containing physical
segment ID, raw base pointer/size, locator-table device pointer, row index, and
Configure type/rank. Psana inserts stream waits for every contributing input
window. User kernels resolve offset, runtime shape, byte count, and status on
the GPU. Different bindings may have different raw bases. A missing source dgram
is host-known; a missing/malformed field inside an existing dgram is represented
by locator status. Consumers must check status and bounds before dereferencing.
A reusable device helper can encapsulate that validation.

Arbitrary ragged fields therefore stay descriptor-based in v1. Do not obtain
host shapes by calling the current public `on_gpu_view` accessor internally.
Additional dense adapters can be added explicitly as needed, without changing
the callback contract.

`GpuTask.calibconst` declares exact `(detector, key)` lookups into the current
run's `dsparms.calibconst[detector]`, also exposed as `det.calibconst`. Each entry
contains `(value, metadata)`. Upload only the requested numeric array value;
metadata remains on the host. Preserve the value's dtype, shape, and element
ordering, using a contiguous staging copy if needed. Do not flatten gain planes,
reorder calibration segments, cast to float32, fold in offsets or masks, or invent
derived keys. In particular, `pedestals` means the stored pedestal array and
`gain_mask` is not a synthetic selector. Future supported value types can extend
the same lookup without changing these meanings.

Validate detector/key existence and supported upload dtype/layout at run setup,
before submitting event work. Missing keys, unsupported values, and undeclared
access fail clearly. The example assumes all four listed keys exist; an algorithm
that permits missing offsets must omit that request and implement its own
zero-offset behavior. An empty declaration performs no calibration-array upload
or derived calibration preparation. Existing CPU calibration loading and
distribution can still populate the source dictionary for normal psana users.

The producer's `evt.calibconst(det, key)` retrieves the already staged device
array; it never fetches calibration data or starts an upload in the event hot
path. Psana owns and budgets these uploads. For v1 each BD owns its requested
device copies; the legacy two-buffer CUDA IPC scheme is bypassed. Retain per-GPU
BD-count budgeting and account for every BD's copies. General sharing by declared
key is a later optimization requiring peer-safe refresh and ownership.
Fetch pointers each invocation; drain local users before any replacement at a
run/step boundary, then make the new inputs ready before invoking callbacks.

## CuPy example with user-owned memory

All context methods are proposed. The calibration and threshold launchers below
represent user-written CuPy RawKernel wrappers or a compiled CUDA extension.
The calibration launcher consumes the original dictionary values and implements
gain-state selection, segment mapping, offset handling, and a stated pixel-status
mask policy itself. It is not the existing two-prepared-array psana launcher.

```python
def userfunc(evt, stream):
    cut = 5.0  # Algorithm configuration belongs to user code.
    raw = evt.input("jungfrau.raw")
    if raw is None:
        return
    present = evt.present("jungfrau.raw")
    peds = evt.calibconst("jungfrau", "pedestals")
    gain = evt.calibconst("jungfrau", "pixel_gain")
    offset = evt.calibconst("jungfrau", "pixel_offset")
    status = evt.calibconst("jungfrau", "pixel_status")
    segment_ids = evt.segment_ids("jungfrau")

    # User allocation policy; psana neither allocates nor budgets these arrays.
    with stream:
        calib = cp.empty(raw.shape, dtype=cp.float32)
        threshold = cp.empty(raw.shape, dtype=cp.uint8)
        nhits = cp.empty((), dtype=cp.uint32)

        # Register ownership BEFORE launches, including for error cleanup.
        evt.keepalive(calib)
        evt.publish("jungfrau_threshold", threshold)
        evt.publish("jungfrau_nhits", nhits)

        user_jungfrau_calib(raw, peds, gain, offset, status,
                           segment_ids=segment_ids, out=calib, stream=stream)
        # Writes every mask element and initializes/reduces nhits, excluding
        # segments with present=0 (absent or rejected fields). Presence does
        # not exclude every XTC damage flag; see the input contract above.
        threshold_and_count(calib, present, cut,
                            out=threshold, count=nhits, stream=stream)
```

Per-event allocation is allowed for the prototype. Users can later pool their
own allocations, but may not overwrite an earlier output merely because the
next callback started. Earlier D2H may still read it, even on the same compute
stream. Retaining a Python reference prevents deallocation, not mutation or
explicit `cudaFree`. User reuse must be ordered after all users of that buffer;
v1 provides no automatic scratch/output pool or reuse callback. Fresh owned
arrays are the simplest first experiment.

Persistent accumulators need additional ordering: callbacks execute serially
on the host, but kernels from different slots can overlap. Use independently
owned per-slot partial accumulators, or explicitly order updates using the
supplied streams and completion events. Publishing a view of mutable persistent
state does not snapshot it; publish an independently owned snapshot or ensure
that no update can race its D2H. Do not mutate published output until its final
consumer has finished.

Any temporary used asynchronously must have an owner registered before its
first launch, or be held by user state for the entire run and drain. This also
covers extension-owned allocations wrapped with a proper lifetime owner.
Owners registered through `publish` are retained automatically. All work uses
the supplied stream in v1; private-stream fan-out is deferred.

## Completion, D2H, and lifetime

```text
independent read groups -> device parse/locate -> input readiness
                                                       |
slot stream:     wait inputs -> gather -> user kernels -> producer_done
                                                              |
copy stream:                       wait(producer_done) -> D2H -> host_done
                                                                     |
CPU:                        process_cpu_data(evt) -> on_cpu waits here
```

The read path currently waits on KvikIO futures on the host before parsing;
this diagram does not imply every read is CUDA-stream ordered. Preserve the
existing submission/wait boundary, coalescing, and group input reuse.

One producer event per execution subbatch is sufficient initially. Psana then
queues copies for **every published output**, using a copy stream that waits on
that event. Publication implies host delivery in this prototype, independent of
the legacy calibrated-result `gpu_d2h_chunk_size=0` default. An implementation
must make that policy explicit rather than silently falling back to lazy copies.
A copy completion event may cover several outputs. Each output retains its own
name, shape, dtype, event identity, and host slice.

`on_cpu` waits only for its host token and returns an independent, cached NumPy
array, including any necessary pinned-to-ordinary-host copy. It never invokes
the task and does not initiate the normal D2H transfer. Copy readiness at the
moment of access is not guaranteed. Pinned destinations and asynchronous copies
follow the [CuPy transfer contract](https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.get)
and [CUDA stream ordering](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html).

| Resource | Allocation owner | Required lifetime |
|---|---|---|
| Raw XTC, parser rows, prepared inputs | Psana | All execution and exposed input consumers complete |
| Constants | Psana | All dependent work drains before step replacement |
| User scratch and outputs | User | Registered references held until producer and output D2H complete |
| Pinned output destination | Psana | Copy completes and host token releases its slice |
| Returned NumPy array | CPU consumer | Independent of device and pinned-slot reuse |

For simplicity, retain all registered user owners until the submission's final
D2H completes (or producer completion if nothing was published). More precise
scratch release is an optional optimization. InputWindow references and owner
uses continue to protect every contributing group and shared parser arena;
callback return is never input release.
Context handles are call-scoped, while their referenced device storage remains
valid for the queued work. No public GPU view/copy of task outputs is promised
in v1; reject those accessors clearly instead of exposing recycled pointers.

Use one finite pinned-staging byte cap per BD, shared across all output names
and D2H pipelines. Establish the cap before the first publication and document
its value and configuration in the implementation. Account for the full allocated
capacity of every staging buffer, including free cached buffers and buffers held
by transfers or host tokens. Reserve capacity against this aggregate cap before
creating or growing any pool; a new output name does not receive a separate
budget. Fixed slot counts per name alone do not bound total pinned memory.
Return byte credits only when the underlying pinned allocation is released;
making a slot reusable does not reduce its allocated capacity.

If no suitable slot is available, or allocating one would exceed the aggregate
cap, use the existing safe synchronous materialization fallback, generalized to
all output dtypes/shapes. Copy into ordinary host memory without allocating
additional pinned staging beyond the cap; overlap is lost under pressure. Do
not block waiting for a host slot whose release requires yielding the event
currently being withheld. A result larger than the cap takes the same fallback.
Retained CPU results can consume user host memory; this cap bounds psana's pinned
staging, not ordinary NumPy results. A pinned buffer is reusable only after BOTH
transfer completion and host-token release, including ignored outputs and early
close.

User allocations remain outside `gpu_memory_budget_gb`. Users must leave device
headroom or reduce the psana budget themselves; the existing margin is not a
reservation for arbitrary kernels. OOM or callback failure aborts with task and
event context. Psana does not retry the algorithm or resize user memory. It drains
already queued work before dropping registered owners; if completion cannot be
proved, it retains owners and marks the execution failed. No partial task output
from the failed submission is delivered. The same drain discipline applies to early iterator close,
BeginStep, EndRun, and end-of-input.

## Native CUDA and Seema's proposal

Seema's GPU Task C ABI proposal is available from local Git object
`831f6c4e7:psana/psana/gpu/docs/gpu_task_c_abi_design.md`.
The earlier combined proposal remains in history at `192333e26`.

| Proposal element | First implementation decision |
|---|---|
| Host callback once per event on explicit CUDA stream | Keep |
| Psana event identity, constants, completion, named results | Keep |
| Pure C vtable, ABI version/struct size, optional C++ wrapper | Good follow-up for a direct shared-library loader |
| Declared scratch/output arenas and automatic byte admission | Defer; user owns allocations and OOM |
| `buffer(name)` obtains psana-allocated memory | Replace with registration of user-owned output views |
| Ordered task list | Defer; compose kernels inside one callback |
| Automatic pinned D2H | Keep; generalize beyond images |
| Static native state, no teardown | Permit user state, but require owners through drain; do not require leaks |

For the smallest trial, a Python callback can call a user's compiled extension
with raw pointers, dimensions, constants, output pointers, and the supplied
`cudaStream_t`. The extension only launches kernels; arrays and native owner
capsules are registered using the same `keepalive`/`publish` path. This supports
ordinary CUDA immediately without first implementing a dynamic library ABI.

```cpp
// Pseudocode for a user library called by its Python adapter.
// No psana binary ABI required; arguments and launch dimensions abbreviated.
void enqueue_analysis(const uint16_t* raw,
                      const CalibrationViews& constants, // user-defined views
                      const uint8_t* present,
                      float* calib, uint8_t* threshold, uint32_t* nhits,
                      const Layout& layout, float cut, cudaStream_t stream)
{
    user_jungfrau_calib<<<grid, block, 0, stream>>>(/* raw, constants, calib, layout */);
    threshold_and_count<<<grid2, block2, 0, stream>>>(/* calib, present, outputs */);
    // Check launch errors in the binding; return without synchronization.
}
```

A direct `int userfunc(void* tc, void* stream)` adapter can follow with
the same semantics. Its minimum operations are input/field lookup, constants,
identity, error reporting, and publication of an
owned device view. A pointer/shape alone is insufficient: the adapter must keep
the allocation owner alive through drain. Native task state can be retained for
the whole run, or a submission owner capsule can release its allocations after
completion. Algorithm settings remain in user-owned state; the adapter does not
require a psana-managed parameter dictionary. Do not freeze a new ABI until that
ownership bridge has one working example. C++ exceptions must not cross the C
entry point.

Three assertions in the older proposal need correction when revisiting it:
serial host calls do not prevent cross-slot GPU state races; asynchronous copies
can outlive the callback that queued them; and automatic D2H does not guarantee
a cache hit by the time `on_cpu` is called. Also, this branch routes all detector
segments to the GPU and provides explicit whole-stream mirroring; it does not
merge arbitrary CPU/GPU raw partials before callbacks.

## Implementation stages

Implement in this order. Each stage has an exit check; later stages must not
restore implicit calibration dependencies removed by Stage 1. The public
callback mode is complete only after delivery and lifecycle validation pass.
`gpu_fn=None` keeps the existing default behavior. Share routing, reads, parser,
input ownership, and delivery machinery; avoid a second independent scheduler.

### Stage 1 — Separate staged inputs from built-in calibration

Establish this boundary first:

```text
psana: read/parse -> stage requested detector-plane raw and calibconst values
user:  userfunc(evt, stream) -> calibration/mask/other kernels -> publish outputs
psana: completion -> bounded D2H -> public Event results
```

Extract a raw-input preparation component from `GPUDetector.process_batch()`.
It owns only the requested gather/unpack buffers, presence information, and
layout bindings. It requires no `peds_gpu`, `gmask_gpu`, calibrated-output buffer,
or calibration kernel. Raw retains gain bits needed by the user's algorithm.
Derive fixed detector-plane layout from supported detector/Configure information;
Names type/rank alone does not supply every runtime shape. Validate device locator
shapes against that layout. Fail unsupported layouts explicitly, without fetching
pedestals for their shape or introducing per-event locator D2H.

The following dependencies were rechecked in the current source and belong in
this separation, beyond removal of the calibration launch:

| Current dependency | Callback-mode change |
|---|---|
| `_setup_gpu_pipeline()` derives dense shape from `calibconst['pedestals']` and gates support on calibration adapters | Establish raw layout independently; a detector with no calibration dictionary can still provide raw input |
| `_compute_calib_constants_cpu()` / `prep_calib_constants()` select pedestals/gain/offset, invoke `_mask()` or status fallback, combine values, flatten/cast, and upload two arrays | Remove this entire recipe from automatic callback setup; Stage 2 uploads only declared dictionary values |
| `GPUDetector.process_batch()` allocates `_calib_slot_bufs`, launches calibration and zeros missing calibrated rows | Separate gather/presence; any calibrated output and its missing-data treatment belong to user code |
| `GPUDetector` assumes raw uint16 versus float32 passthrough and rejects `cmpars` | Bind the requested field/layout explicitly; do not label pre-calibrated `fex` as raw or impose the legacy kernel's algorithm limitations on callbacks |
| `_setup_jungfrau_shared_calib()` and `_setup_jungfrau_shared_caches()` eagerly build CPU derived arrays during MPI run initialization | Exclude callback-only detector work using a consistent target list on all participating ranks; preserve needed CPU/hybrid consumers and collective ordering |
| `_setup_gpu_geometry()` plus `setup_geometry[_from_arrays]()` prepare/upload image scatter indices | Skip these for callback detector-plane inputs; image assembly is not a prerequisite for calibration |
| `_make_gpu_event_manager()` and `share_calib_between_gpu_peers()` expect the fixed `peds_gpu`/`gmask_gpu` pair | Bypass legacy leader/follower allocation suppression and handle exchange in callback mode; retain device assignment and per-GPU BD-count budgeting |
| `_dispatch_transition()` and `GPUDetector.beginstep()` recompute/update the same two arrays | Replace callback-mode refresh with the declared input store and drain discipline in Stage 2 |
| Detector byte estimates, allocation reservations, trimming, and memory statistics include calibrated outputs/geometry/two constants | Account for reader/parser, staged raw/presence, and actual requested constant allocations; omit removed legacy buffers |
| `optimal_kernel_batch_size()` derives the automatic batch choice from the calibration launch | Use a documented callback-independent default or explicit batch setting; retain byte-based subbatch admission |
| `EventPool.submit()` manufactures `.calib`/`.raw` keys, `_D2hPipeline` targets `.calib`, and `GpuEventState.get()` qualifies bare names | Separate input bindings from results; Stage 4 delivers only published names with exact lookup |

Keep CPU source calibration loading/distribution and parser/routing infrastructure
available. The cleanup removes derived work forced by GPU calibration, not the
source dictionary or normal CPU detector APIs. In mixed CPU/GPU jobs, do not
disable CPU processing or skip a shared-memory collective on only some ranks.

Main files: [gpu_events.py](../../gpu_events.py),
[gpu_detector.py](../../gpu_detector.py), [gpu_calib.py](../../gpu_calib.py),
[gpu_stream.py](../../gpu_stream.py), [gpu_mpi.py](../../gpu_mpi.py),
[mpi_ds.py](../../../psexp/mpi_ds.py), and [context.py](../../context.py).

**Exit check:** a raw-only preparation test with empty calibration requests and
no available pedestals succeeds. Instrument/disable legacy preparation, mask,
geometry, and kernel entry points so the test fails if any is called. Verify no
legacy output/constant allocations and unchanged default calibration results.

### Stage 2 — Stage requested calibration-dictionary values

Add `GpuTask(function, inputs, calibconst)` and `DataSource(gpu_fn=...)` plumbing
through `DataSourceBase`/`DsParms`, serial GPU setup, and MPI BD setup. Validate
declarations on the host; create CUDA resources only on assigned GPU processes.
Reject callback configuration when no supported GPU event path is active.

Resolve each `(detector, key)` against the current run dictionary. Upload supported
numeric NumPy arrays, including zero-dimensional arrays, preserving dtype, shape,
and values; reject object/text or unsupported dtype/layout values explicitly.
Deduplicate identical requests. Keep host metadata and source owners as needed
through upload completion. Return read-only device values through the context;
the accessor performs neither database I/O nor lazy H2D. No synthetic `gain_mask`,
automatic mask computation, inverse gain, offset folding, or hidden float32 cast.

Reserve actual upload bytes before allocation. Initially allocate one requested
copy per BD and charge it to that BD's quota; do not leave followers without
constants after bypassing the old IPC path. Expose dense-input segment identity
and document how it maps to the unchanged calibration layout. Establish an upload
completion dependency before any callback uses the arrays.

Retain the run's constants until their source changes or the run ends. At a
boundary requiring replacement, drain dependent work, resolve the new source
snapshot after the relevant host transition update, and establish readiness again.
Do not assume BeginStep itself implies a new calibration DB fetch. Empty requests
remain empty through setup and transitions. Generic CUDA IPC sharing can follow
later with explicit peer drains and allocation-owner accounting.

**Exit check:** request only `pixel_gain` while other keys are unavailable; only
that array uploads. Verify empty requests, missing/unsupported keys, original dtype
and shape, sparse segment mappings, multi-run validation, changed constants, and
several BDs on one GPU. No callback-time constant upload is permitted.

### Stage 3 — Invoke the user callback and track asynchronous ownership

Implement the producer context and `userfunc(evt, stream)` dispatch in
`EventPool.submit()` after requested input preparation. Call once per selected
event with GPU descriptors, regardless of how many requested detectors it has.
Restrict dispatch to delivery identities before calling user code, including
`max_events` tails; preserve original event indexes with missing detectors.
The stream is the current execution slot's stream and is made current for CuPy.

Implement read-only input/constant access, segment identity, `keepalive()`, and
`publish()` registration. No `params` field, private-stream execution, implicit
calibration, or user launch from `.on_cpu`. Record producer completion after all
callbacks in the subbatch. Retain user owners immediately, including during
exceptions; connect input windows, prepared raw, constants, and outputs to their
actual completion dependencies. A published view of borrowed storage must also
retain its backing lease through D2H, not merely a Python reference.

**Exit check:** a no-output callback receives the correct inputs and stream once
per eligible event. Delayed kernels, scratch-only work, multiple detectors, tail
selection, and callback exceptions cannot release or overwrite live storage.

### Stage 4 — Deliver exactly the published results to the CPU

Generalize D2H and host tokens to named contiguous arrays with fixed per-name
dtype/shape for the run, including scalar counts and uint8 masks. Publication
triggers host staging after producer completion regardless of the legacy
`gpu_d2h_chunk_size=0` default. Callback result lookup uses the exact published key;
`jungfrau_threshold` must not become `jungfrau.jungfrau_threshold` under the old
single-detector alias rule. Input names are not automatically published outputs.

Enforce the aggregate per-BD pinned byte cap described above before creating or
growing any named pool. Use the safe ordinary-host synchronous fallback for
unavailable slots or capacity. Attach host tokens by event identity and make
`.on_cpu` return an independent cached NumPy result. Reject device accessors on
callback outputs. Retain all submission owners through its last output copy;
copy failures drain already queued transfers before releasing owners.

**Exit check:** publish mask/count, no outputs, multiple names, and newly appearing
names. Verify exact lookup with one and multiple detectors, dtype/shape errors,
ignored outputs, retained events, oversized outputs, and delayed D2H during slot
reuse. Pinned capacity stays bounded and delivery never waits on itself.

### Stage 5 — Validate the first user kernels

Deliver a Jungfrau callback that consumes requested dictionary arrays and produces
calibrated data, a threshold mask, and a count using user-owned output memory.
Compare with the CPU/reference algorithm using identical offset, gain, mask, and
common-mode settings; the built-in GPU kernel does not implement every CPU option.
Demonstrate CuPy and a compiled native launcher using the same staging contract.

Use Jungfrau threshold/mask/count as the first end-to-end acceptance case.
Then add conditional no-output, bounded peak-list plus device-count, and
per-slot accumulation cases. A device-only decision must not require a blocking
host read to decide output allocation; use fixed capacity and a device count.

**Exit check:** declared inputs/constants have the expected values and layout;
published outputs match the reference, including empty and conditional cases;
no kernels are launched from the public event loop.

### Follow-up detector coverage — EpixUHR

The recovered draft also proposed EpixUHR. Retain that design context as a
follow-up, not as a prerequisite for the first Jungfrau callback milestone.
Confirm its current CPU adapter and Configure dependencies before implementation.

For EpixUHR, name the supported variant explicitly. The implemented CPU calibration
here is `epixuhr3x2` (`UtilsEpixUHR.calib_v02`); older `epixuhr.py` has a placeholder
`calib()`. Add its raw adapter with the existing ASIC unpack/orientation and gain
bits preserved. Match detector-plane layout, including missing/sparse segments,
without depending on pedestal shape.

EpixUHR3x2 additionally needs Configure `gainAsic` / `gainCSVAsic` values to select
the calibration gain planes. Before claiming support, define a minimal declared
Configure-input selector/accessor alongside `calibconst`, validate its fields,
and stage the requested values with the same ownership/readiness rules. These
are a separate source, not fabricated calibration keys. The user algorithm builds
gain maps and masks. Do not silently invoke CPU `Storage_epixuhr_v01` to recreate
the calibration preprocessing being removed.

Sources: [UtilsEpixUHR.py](../../../detector/UtilsEpixUHR.py),
[epixuhr3x2.py](../../../detector/epixuhr3x2.py), and
[UtilsJungfrau.py](../../../detector/UtilsJungfrau.py).

**Exit check:** detector-plane raw matches each supported CPU adapter; user
calibration matches the stated reference across gain states, segment order, and
mask policies. For EpixUHR, hold raw/calibration arrays fixed and vary gain
configuration to prove that the required third input is honored.

### Stage 6 — Close lifecycle and integration acceptance

Exercise the full callback path through serial and MPI entry points, exclusive
and mirrored routing, bulk and per-dgram reads, independently owned read groups,
and multiple BDs sharing a GPU. Verify BeginStep, EndRun, end-of-input, early
iterator close, `max_events`, callback/launch/copy failures, and memory pressure.
Drain exactly once where required and retain owners if CUDA completion cannot
be established. Check that callback-mode setup and transitions never re-enter
legacy calibration or geometry setup. Document supported selectors, variants,
layout mappings, pinned limit/default, and CPU-only compatibility behavior.

Run focused tests in `tests/gpu/unit` and GPU integration/reference tests in
`tests/gpu/integration`. Implementation changes touching psana core also require
both `pytest psana/psana/tests/` and `pytest psana/psana/tests/byhand_*` in the built
environment. Record environment blockers explicitly. Performance measurements
remain separate and require verification of the actual KvikIO/GDS mode.

**Exit check:** all acceptance cases below pass with the runtime and datasets
recorded; documentation examples reflect the implemented API. This stage is
correctness acceptance, not a throughput claim.

Acceptance for that implementation:

- Default no-callback calibration remains pixel-exact; injected calibration is
  not run twice. Changing the threshold changes both published outputs.
- In callback mode, an empty constant declaration performs no calibration
  preparation/upload. Requesting only `pedestals` uploads the original pedestal
  array without offsets or masks; requesting only `pixel_gain` uploads original
  gain values without inversion or other uploads. Verify selective setup,
  boundary refresh, per-BD ownership, and budget accounting; undeclared access
  and unsupported selectors fail clearly.
- Multiple dtypes, scalar output, no output, missing input/segments, absent or
  invalid locator status, partial subbatches, and `max_events` preserve identity.
- Cover `Corrupted` damage rejection and an otherwise valid field with a
  non-`Corrupted` damage flag. The latter remains present and is eligible for
  counting under the documented v1 policy; do not silently equate presence with
  damage-free data.
- Demonstrate an algorithm modifying a registered user-owned copy. Verify the
  borrowed inputs and constants remain unchanged across overlapping slots and
  subsequent public field access; arbitrary native writes are not enforced.
- Delay kernels and D2H while reusing execution slots; outputs and registered
  scratch remain valid. Cover several raw bases, retained small groups, shared
  parser arenas, out-of-order group completion, and existing external consumers.
- Public CPU processing contains no task launches; generic descriptor access
  adds no per-event locator D2H. Do not assert throughput in correctness tests.
- Force callback/launch/copy failures, ignored outputs, pinned-slot pressure,
  early iterator close, and step/run boundaries; verify drain before release.
- Publish distinct output names on successive events and outputs larger than
  the pinned cap. Include free cached buffers and retained host tokens in the
  accounting; total pinned-staging capacity never exceeds the per-BD cap, and
  fallback results remain correct without waiting for CPU delivery to free space.
- Compare bulk and per-dgram modes, exclusive and mirrored routing, and normal
  multi-BD placement. Leave performance characterization to the separate task.

This document is a design review and handoff, not a tested implementation.

## Remaining implementation decisions

- Public import/export spelling, validation errors, and callable setup/teardown
  conventions; a closure or callable object can hold algorithm parameters.
- The finite aggregate pinned-memory cap, its default and configuration. Its
  accounting/fallback contract above is required; the numerical default is open.
- Exact supported dense layouts and calibration segment mappings. Generic
  runtime-shaped fields use device descriptors until an explicit adapter exists.
- A native ownership bridge and versioned direct-library ABI, if needed beyond
  a Python callback invoking a compiled extension.
- A safe output-reuse notification, batch callback, optional output device
  access, and generic constant sharing are follow-ups requiring explicit design.

These choices must not reverse the core model: internal producer dispatch,
dependency-only task declarations, user-owned memory, named publication, and
psana-managed completion/D2H. The dated handoff records baseline test evidence;
no test result there constitutes acceptance of these proposed APIs.
