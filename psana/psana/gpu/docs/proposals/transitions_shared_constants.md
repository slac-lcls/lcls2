# Design 1: GPU step scopes and run-scoped shared constants

Design for Mona and Seema, originally 2026-10-07; direction revised 2026-10-15 after the meeting; subbatch/reduction contract clarified 2026-10-16. Covers [#153](https://github.com/slac-lcls/lcls2/issues/153) and [#168](https://github.com/slac-lcls/lcls2/issues/168). This is a design document, not runtime implementation. Implementation tasks and the remaining API/IPC deployment details are not assigned or approved by this update.

## Meeting decision and scope

**Use Design 1: attach immutable step/scope identity to each execution subbatch and permit different steps in different in-flight slots of one BD.** BDs sharing a GPU also advance independently. The October 7 recommendation for local step-boundary drains and the October 9 preference for Design 2 are superseded. BeginStep/EndStep alone do not require a slot-pool flush, device-wide synchronization, or peer barrier.

**GPU EnvStore is deferred. SlowUpdate values and step payload values are both ignored on GPU; CPU handling still retains and interprets them.** GPU steps are indices/scopes, not copies of scan/configuration/environment values. Do not extract, retain, upload, or expose those values as GPU task context. Original dgrams and packet backing may remain owned for ordered CPU delivery; this is not GPU EnvStore storage. GPU kernels requiring changing detector configuration or environment inputs are outside this phase.

Among ordinary in-run controls, Enable carrying new `chunkinfo` is the payload the GPU reader must interpret to resolve the next chunk/file. Configure still supplies initial decoding metadata, and BeginRun/EndRun govern run resources. `calibconst` is initialized once per run, reused unchanged across all steps, and never refreshed/versioned at BeginStep. Sharing identical requested constants remains separate run-scoped work under #168.

Keep the current SMD0 → EB → BD transport and external `GpuTask` API. An ordered controller per BD separates GPU scheduling look-ahead from CPU-visible transition application. Each submission keeps its scope identity, file mappings, and resource leases until its consumers complete. Public CPU events and steps remain ordered even while different scopes execute concurrently.

CPU BDs can be at different steps because each process has its own mutable host state and advances its own iterator. GPU slots are streams inside one BD, not separate processes. Immutable per-submission scope identity provides the intended independence without pretending slots have private EnvStores. Both designs already allow different BDs to advance independently; Design 1 adds overlap across steps within a BD.

The original reconstruction stack through #151 has merged into `features/psana2-gpu`; this proposal remains a separate documentation PR. Transition correctness can ship with local run constants before the shared backend. The earlier per-GPU constant-helper proposal remains an option for review, not a consequence of selecting Design 1.

The October 16 discussion confirms **one step per execution subbatch and per occupied slot**, while different slots may hold different steps. Mixed-step subbatches were considered and are not the baseline: they preserve execution boundaries but complicate per-step/bin reductions and can increase live accumulator storage. Scheduling stays at subbatch granularity. A physical slot is reusable across steps after its consumers finish; it is not permanently assigned to a step.

## Current workflow and API

The October 15 source inspection used `features/psana2-gpu` at `410ebba1c`; the original October 7 baseline was `9b32dda19b46807f7a4db62fb2351ba87b359e47`. Issue #153 and this design now record the revised meeting scope. The review findings linked in [#153](https://github.com/slac-lcls/lcls2/issues/153) and [#168](https://github.com/slac-lcls/lcls2/issues/168) also reference reviewed stack commit `e2cdad6e18fd9263d7adfeed0ce58d5ad074e070`. These snapshots identify evidence, not an instruction to rebase or modify the stack.

```mermaid
flowchart LR
    S[SMD0 small data and recipient history] --> E[EB aligned events and BD history]
    E --> P[One message containing ordered SMD plus GPUBAT1]
    P --> B[CPU BD admission and scheduling]
    B --> R[KvikIO reads into GPU memory]
    R --> X[GPU parse and declared input preparation]
    X --> K[External GpuTask callback]
    K --> D[Publication D2H and Event delivery]
```

SMD0 sends normal small-data chunks to EB ranks. SMD0 and EB each keep transition history per recipient and prepend missing history. New transitions keep their position among L1 events. EB produces a CPU/SMD packet, a GPUBAT1 packet with L1 event identities and big-data descriptors, and a local `step_batch`. EB uses `step_batch` to maintain history; it does **not** transmit it as an independently ordered transition channel. MPI transmits the history-prefixed SMD packet alongside GPUBAT1. Serial processing obtains corresponding batches from the local reader/EB path. See [node.py](../../../psexp/node.py) and [eventbuilder.pyx](../../../eventbuilder.pyx).

A CPU BD still submits I/O, parses inputs, invokes user work on the assigned stream, copies publications, and delivers events. GPU kernels do not perform MPI or open files. Framework ownership distinguishes input windows, execution slots, result copies, and retained host results. Read grouping and execution batching are independent.

The preserved [external API](../user_kernels.md) is:

```python
from psana.gpu import GpuTask

def analysis(batch, stream):
    raw = batch.input("jungfrau.raw")
    pedestals = batch.calibconst("jungfrau", "pedestals")
    # Submit work on stream, retaining scratch and publishing output.
    # batch.keepalive(scratch)
    # batch.publish("result", output)

task = GpuTask(function=analysis,
               inputs=["jungfrau.raw"],
               calibconst=[("jungfrau", "pedestals")])

# Existing result interface:
# evt.gpu.get("result").on_cpu
```

The declaration is host-only. A callback handles one nonempty selected execution subbatch. `batch.run`, `batch.batch_id`, `batch.step_generation`, timestamps, and original event indices describe that invocation. Borrowed inputs and constants are read-only and must not be cached for later execution. All callback work uses the supplied stream; public parsed-input consumers have their existing completion-registration API. Publication storage remains alive through terminal D2H. Retained `.on_cpu` results are independent of slot reuse.

`RequestedConstants` currently validates requested native numeric NumPy arrays, unwraps `(array, metadata)` values, snapshots host bytes, and uploads one admitted copy per BD. It preserves shape, dtype, and values, including scalar and empty arrays. The current manager also invokes a BeginStep refresh comparison; the revised design removes that per-step action. Resolve requested constants once during run setup and retain them for the run. Sharing must preserve array fidelity and must not reintroduce built-in calibration, gain inversion, or fixed Jungfrau array formats. Current dense segment bindings are sorted Configure IDs; this design preserves the current `segment_ids` API and original constant axes, without introducing a new segment-order convention.

The relevant defects and gaps are:

- The fast GPU EB appends BeginStep and then stops. A real packet is `[L1 A, L1 B, BeginStep 2]`; later step-2 L1 events are in a subsequent packet. `_handle_steps()` runs before this packet’s L1 submission, so A/B can get step 2’s state. Draining previously submitted work does not fix unsubmitted A/B.
- Serial GPU processing consumes transitions internally; MPI GPU `steps()` returns immediately. Both need a common stream that can support event and step views.
- `_iter_step_events()` uses service zero for empty records, colliding with ClearReadout. Filtering and `first_service()` compound the loss; file-read coalescing also drops its fence.
- SlowUpdate already reaches the host environment store. Preserve that ordered CPU path; GPU EnvStore and GPU interpretation of SlowUpdate/step values are deferred. The independent CPU timestamp-lookup concern recorded in #153 is separate follow-up work, not a GPU implementation gate.
- The removed calibration IPC path was specialized to prepared Jungfrau arrays. Restoring its old leader formula or float32 assumptions would conflict with the current generic requested-constant API.

## Review findings as acceptance criteria

Source inspection and isolated probes establish the problems; they are not end-to-end validation of this design. Narrowing GPU payload support does not resolve the ordering, iteration, framing, or ownership defects.

| Finding | Required behavior | Acceptance evidence |
| --- | --- | --- |
| [GPU-148-01](https://github.com/slac-lcls/lcls2/pull/148#discussion_r4200896653), #153 P1 | Assign preceding events to the old scope; preserve CPU-visible transition order while scheduling ahead | Real EB trailing BeginStep → planner → callback; delayed old-scope work overlaps new scope without mislabeling events/results |
| [GPU-140-03](https://github.com/slac-lcls/lcls2/pull/140#discussion_r4159309391), #153 | Serial/MPI event and step iteration with CPU metadata handling and GPU results | Two or more steps, empty steps, GPU-only/mixed events, ordinary MPI distribution and history catch-up |
| [GPU-141-02](https://github.com/slac-lcls/lcls2/pull/141#discussion_r4160862286), #153 | Distinguish ClearReadout from empty GPU-only placeholders; preserve CPU dispatch and coalescing fences | Service zero survives; malformed framing and sentinel 13 fail; no fake transition from an empty record |
| [GPU-149-04](https://github.com/slac-lcls/lcls2/pull/149#discussion_r4201235524), #153 | One stream owner; step completion differs from terminal run cleanup | Retained iterators, explicit close, next step, exceptions, terminate, repeated cleanup, bounded MPI shutdown |
| Enable/chunk handling, #153 | Bind every pending read to the correct immutable per-stream file identity and descriptor offset | Rollover inside a step, old reads delayed, replayed/trailing Enable, multiple streams, bulk and non-bulk paths |
| Deferred GPU EnvStore scope | No GPU extraction/storage/upload/API for SlowUpdate or step values; CPU retains its functionality | CPU dispatch once in order; GPU look-ahead does not advance public CPU configuration early; no GPU payload dependency |
| [GPU-147-01](https://github.com/slac-lcls/lcls2/pull/147#discussion_r4199547812), #168 P2 | Share identical requested run constants behind the existing API | Two/four BDs per GPU allocate approximately one copy per identical object; no follower upload; exact values |
| #168 topology and lifetime | Correct actual GPU peers; all imports close before export free | Multiple nodes/GPUs/EB groups, different selectors/content, delayed peers/consumers, failed setup, early exit and repeated close |

The P1 on #153 comes from ordering, not a new P1 classification for every row. #156 was consolidated into #153, not fixed. Coordinate decoding work with [#154](https://github.com/slac-lcls/lcls2/issues/154), peer discovery with [#155](https://github.com/slac-lcls/lcls2/issues/155), and partial-setup rollback with [#169](https://github.com/slac-lcls/lcls2/issues/169).

## Proposed responsibilities

| Component | Responsibility | Owns |
| --- | --- | --- |
| SMD0/EB transport | Existing ordered records, recipient history, coherent SMD/GPUBAT1 packets | Send buffers and history until completion |
| Ordered packet planner in BD | Produce L1 ranges, scope markers and file mappings without mutating public CPU state | Packet backing, stable event/scope identities and CPU transition references |
| GPU scheduler | Submit byte-bounded subbatches carrying immutable scope/file identity; overlap scopes | Execution/input/result leases and per-scope completion accounting |
| CPU delivery controller | Apply CPU transitions exactly once at delivery order; expose events/steps | Public cursor, CPU transition dgrams, iterator lifecycle |
| RequestedConstants facade/local backend | Resolve declared arrays during run setup; retain immutable run constants | Run manifest, local allocation or shared backend leases |
| Shared constant owner/backend | Deduplicate identical run content, admit/upload/export storage, coordinate release | Physical allocations, catalog, budget and readiness |
| Import owner in BD | Retain local mappings through aliases and submitted work | Mapping leases, aliases and completion tokens |

These are logical responsibilities, not a requirement for separate threads or new classes with these names. The controller must establish independent scheduling and delivery cursors even if one class implements both. Shared storage owns constants only; BDs retain event reads, science kernels and delivery.

## Ordered planning and Design 1 execution

### Preserve the coherent packet order

Use SMD record order as the authority: missing history → L1 range → transition → next L1 range. For each L1 or validated all-empty GPU-only placeholder, consume the corresponding GPUBAT1 event-table row. Preserve `(packet identity, original batch_event_index, timestamp)` through every slice. Validate counts, timestamp agreement where real CPU dgrams exist, and descriptor bounds. Never move all transitions before all events.

Every L1 in the current split path has a GPUBAT1 event row, including zero-GPU-contribution rows. Such events remain deliverable even if no task callback is selected. An all-empty record is not itself evidence of either a transition or an L1: establish its event meaning from the coherent packet/table correspondence or reject the packet. CPU-only batches without GPUBAT1 use their real dgrams.

The transition-only decoder counts present dgrams, takes the first present service, validates packet sizes before skipping all-empty records, and yields service zero. The ordered planner must additionally preserve original empty-record positions for L1 correlation; it cannot use the transition decoder's filtered positions as event indices. Remove service truth-value filtering from the manager, bulk-read selection, file resolver and shared service validation.

Preserve control positions in the ordered plan. Execution splitting combines the existing byte-budget ranges with required BeginStep/EndStep boundaries: a subbatch must not cross a step boundary. It carries one immutable run/step identity, original event indices and resolved read-file mappings. Read-group/coalescing fences are separate from execution boundaries; a CPU-only SlowUpdate does not by itself require another GPU execution submission. Preserve read correctness and ownership when composing these plans. No scan/step/SlowUpdate payload values are captured for GPU work. Initial decoder tables and unchanged run constants are retained independently of scope.

### Separate scheduling from CPU-visible delivery

The scheduling cursor may read ahead across steps and admit new-scope work while old-scope kernels/results are in flight. Before advancing its scope, it must assign all preceding L1 rows to the previous scope. It must not mutate a submitted scope label or read a mutable current-step value later during execution.

The CPU delivery cursor processes the original records in order and alone dispatches their CPU state updates through `Run._handle_transition()` (or an equivalent single owner). It offers preceding public events before applying the following CPU transition. `run.events()` hides controls; `run.steps()` uses ordered scope boundaries. Neither adapter repeats the CPU EnvStore update. Scheduling read-ahead must not call the live CPU handler early, and must not rely on that handler to resolve GPU chunk files.

Store only the identity/lifecycle metadata needed for GPU scopes: run identity, logical source-step identity, and a local generation where useful. Existing `batch.step_generation` can remain a local progression value; it is not automatically a cross-BD step key. Define replay handling from source transition identity/order and header metadata, without reading step payload values. A new source BeginStep establishes a scope for subsequent events; receiving that same source transition again through catch-up history preserves its scope identity and does not advance the local step generation. Preserve unseen empty BeginStep/EndStep pairs as empty public steps, and assign events by SMD record order. Validate these behaviors at the EB/BD interface before freezing the identity encoding; timestamp equality alone must not be assumed to prove replay. Enable replay additionally validates its required chunkinfo. Original payload dgrams belong to CPU delivery, not a GPU state snapshot.

Synthetic steps generated upstream by SMDReader (`PS_FAKESTEP_FLAG=1`) arrive at EB and BD as ordinary Disable/EndStep/BeginStep/Enable records. The GPU path applies the same transition rules regardless of their origin; it does not inspect this flag, detect synthetic steps, or implement a separate fake-step path. Running the existing fake-step mode is an optional integration check, not a separate feature or approval gate for this design. The earlier phrase "fake-step/timestamp-jump cases" did not define an additional input contract: no timestamp-jump recovery mechanism is requested here. Any proposed timestamp-based scope key must state its uniqueness and replay assumptions and validate them against the supported EB/BD transition stream.

### Overlap and completion

Different BDs and different execution slots within a BD may process different scopes concurrently. BeginStep/EndStep alone cause no blanket drain. Scheduling remains bounded by byte budgets, slot availability and actual resource dependencies; the design permits overlap but does not promise simultaneous hardware execution or a speedup.

Track pending events, results and registered consumers by scope. Public child-step completion finishes its relevant work and preserves ordered delivery, while later-scope work may already be admitted. It must not flush unrelated later slots as a side effect. A shared mutable accumulator/workspace requires scope-specific storage or explicit dependencies. No new implicit cross-step accumulator behavior is promised.

Retain input windows, execution buffers, publication/D2H storage, scope bookkeeping and file owners until their consumers finish. Callback return, advancing a generator, and changing the current scope are not release signals. EndRun and terminal cleanup drain all owned work; ordinary step boundaries only coordinate the work needed for that scope's completion.

### One step per subbatch: reductions and memory

For an XPP-style scan where each step is one output bin, a homogeneous subbatch lets a reduction task produce one partial sum for one bin. A step may span many subbatches and BDs; its final sum still combines those partials. Public `step.events()` yields only that step's assigned events. An all-BD step sum requires an explicit reduction, separate from local iteration. This is a task use case, not a new framework reduction API or GPU extraction of scan values: the GPU uses scope identity, while CPU retains the physical bin/scan values.

```text
slot 0: [A, B, C]  scope 1 -> one partial sum for bin 1
slot 1: [D, E]     scope 2 -> one partial sum for bin 2
```

Different slots may overlap different steps; splitting at a step boundary does not drain the other slots. Several slots may also contribute to the same step. Track completion by scope so a bin is final only after its boundary is known and all relevant contributions complete. Do not reset a single mutable accumulator while earlier kernels can still write to it; use isolated partials/scope storage or explicit dependencies.

The alternative considered was one mixed-step execution such as `[A, B, C, D, E]` with event step IDs `[1, 1, 1, 2, 2]`. Event metadata alone does not require per-event scheduling, so this alternative can preserve the original byte-budget subbatch boundaries. However, a step/bin reduction must distinguish its contributions: one undifferentiated sum cannot be separated afterward. It needs grouped partial sums or additional processing, event-aligned scope metadata, and completion/ownership across several scopes in one execution. The current scalar `batch.step_generation` would also need a mixed-scope API contract.

If one bin's accumulator occupies `B` bytes and a mixed subbatch represents `K` bins, retaining all its bin partials can require about `K * B` bytes rather than `B`, plus reduction scratch and metadata. This is a conditional capacity estimate, not an inevitable allocation or a measured result: implementations can trade additional passes or staged output for lower simultaneous storage. The concern is strongest for image-sized sums; scalar sums have a much smaller payload cost. Fixed run constants do not multiply with the number of bins.

One-step subbatches simplify each execution, but do not imply one accumulator for the entire BD. Multiple slots, active steps, pending partials and retained outputs may still keep several sums alive. Account for task scratch and retained reduction outputs in the applicable memory budget and lifetime contract; slot count alone is not the memory bound. This design does not claim to police arbitrary unregistered user allocations. The tradeoff is potentially smaller subbatches and extra submissions at step boundaries, especially for short steps. Measure that cost before considering mixed-step execution as a later optimization.

### Scheduling cost contract

- Discover step boundaries during the existing ordered packet traversal; attach one scope reference per execution subbatch. No per-event or per-field scheduling is introduced for step assignment.
- Combine scope boundaries with existing admission ranges once per coherent batch. Reuse decoded descriptors and read-plan metadata; do not rebuild the full read plan for each step/subbatch or rescan field descriptors to determine scope.
- Submit and register execution completion at subbatch granularity. Associate its existing completion/consumer tokens with the scope; do not add per-event polling. Existing per-event delivery and input/result ownership remain necessary and are not additional scope scheduling.
- Keep physical read-group planning separate from execution grouping. Enable/file fences still control read identity/coalescing; CPU-only value updates add no GPU value-processing task or mandatory execution split.
- Measure additional submissions, CPU planning/dispatch time and accumulator high-water memory for long and short steps. Scope bookkeeping must not scale with the number of requested fields. Existing budget/admission costs are not claimed to disappear.

### Transition responsibilities

| Transition or boundary | GPU responsibility | CPU behavior / limits |
| --- | --- | --- |
| Configure / BeginRun setup | Build initial parser metadata/bindings and initialize immutable run constants before callbacks | Preserve existing host configuration/calibration setup |
| BeginStep | Start a scope for subsequent subbatches; ignore payload values; no constant refresh or mandatory slot drain | Apply existing scan/configuration updates at the CPU delivery cursor |
| EndStep | Close scope admission, track completion and finish its public iterator; keep pipeline available | Preserve ordered step boundaries |
| SlowUpdate | Ignore payload values on GPU; no GPU EnvStore, upload or mandatory drain | Existing CPU EPICS/EnvStore handling remains |
| Enable with new chunkinfo | Resolve affected streams' new files and retain old handles for old reads | Preserve CPU file-switch behavior at its own cursor |
| Disable | Recognize boundary; no inferred EndStep or constant change | May surround a chunk rollover within one step |
| ClearReadout (0) / Reset (1) | Recognize nonempty records; no invented file switch, payload interpretation or event deletion | Preserve CPU dispatch; additional GPU reset semantics need an explicit contract |
| EndRun | Stop admission; finish outstanding reads, producers, copies and consumers before release | Apply ordered terminal host handling |
| EOF / event limit / explicit close | Same idempotent terminal cleanup without inventing an EndRun dgram | Preserve documented local/global stop distinction |
| Unconfigure / unexpected Configure | Retire configuration resources after users finish; reject unsupported active-run decoder/layout changes | Normal DataSource setup must establish the next configured run |

ClearReadout is 0; `L1Accept_EndOfBatch` is 11 and L1Accept is 12. Both events satisfy `TransitionId.isEvent()`; end-of-batch is not EndStep. `NumberOf == 13` is an invalid record sentinel. Missing dgrams remain distinct from valid service zero. Preserve the existing conservative coalescing fence at real non-L1 controls; that is not a CUDA fence or a GPU EnvStore operation. Relaxing coalescing fences is separate work.

Initial scope is the existing experiment/run serial and MPI GPU path. New shmem/DRP modes and GPU support for changing detector layouts/configuration values are separate. GPU EnvStore and GPU SlowUpdate/step-value APIs are explicitly deferred. The CPU timestamp-lookup concern documented in #153 should be addressed separately; this work must preserve CPU dispatch order without claiming that defect is fixed.

### Enable, file identity and offsets

Use Enable/chunkinfo to resolve immutable per-stream `(filename, chunk identity)` mappings in record order, including leading history and trailing controls in transition-only packets. Only new chunkinfo changes the mapping; an Enable without a file change does not reset the step. No subbatch/read may consult a mutable latest-file handle after its file identity has been bound. Apply this guarantee to both bulk and non-bulk paths. Current `GpuFileEpochs` is a useful starting point; the non-bulk path currently derives identity from mutable CPU `DgramManager` state and needs equivalent ordered resolution.

KvikIO performs explicit `pread(..., file_offset=...)` calls. Descriptor offsets are offsets within the selected chunk file. “Reset the counting offset” does not mean replace offsets with zero: start a new file/coalescing interval and use the new chunk's supplied offsets, without accumulating the previous file's length. Preserve event/step indices and old pending reads. Open the new file on demand, retain each old handle through its pending reads, and retire it only when safe. File rollover and step rollover are independent.

## Run-scoped constants and ownership

Initialize host `calibconst` once during BeginRun setup. Resolve declared requested arrays for GPU use during run setup, then reuse the same immutable run manifest across every step. BeginStep, EndStep, SlowUpdate, Enable and Disable do not refresh, replace or version constants. Remove the manager's existing BeginStep refresh action. A new run initializes its own constants; do not introduce per-step IPC acquisition/barriers.

Keep scope identity, run constant content identity and physical allocation generation separate. Equal run constants can share storage even when BDs are at different steps. Different selectors/content must receive distinct objects or an explicit application-policy error; step counters never authorize substituting another BD's values. Allocation generations remain necessary for stale-handle/release protection, not as calibration versions per step.

Canonical requested host snapshots preserve shape, native dtype and C-order bytes, including NaN payloads and signed zero. Validate all selectors before allocation. Use a digest for lookup plus exact metadata/byte comparison, or an equivalent trusted immutable source identity. Host snapshots are taken during run setup; concurrent mutation and per-step changes are outside the run-constant contract.

Deduplicate per selector/content object so overlapping declarations share their intersection without uploading unrequested constants. Preserve original constant axes and the current segment-binding API. Scalars remain shape `()`; zero-byte arrays need metadata but no exported allocation. Empty declarations create no constant CUDA state or service dependency.

Run setup is transactional: acquire the entire manifest before first event admission. On partial failure, roll back completed grants and retain pending owners until their completion is known; do not start processing with a partial manifest. Never overwrite a shared allocation in place. Local storage and shared storage obey the same run lifetime.

Each execution retains the run constants through terminal consumers, including publication D2H when an output aliases them. Imported CuPy views retain their mapping owner. Freeing requires:

```text
no pending acquire/import grants
no active run-manifest or execution references
no array aliases retaining the mapping/exported storage
all submitted consumers completed successfully
every importer acknowledged its mapping closed
```

Only the exporting owner frees storage and releases the physical charge. BDs close imports and release references. A recreated physical allocation gets a fresh generation even if content is identical; delayed releases cannot affect its replacement. Defensive alias retention prevents premature free, but does not authorize caching borrowed arrays for later work or unregistered streams/raw pointers. Report retained aliases and budget charges; timeouts never authorize unsafe reclamation.

## CUDA IPC storage backend

CUDA IPC is useful for large, identical, read-only requested arrays shared by processes using one physical GPU. This proposal limits the backend to immutable requested run constants. Sharing other state would require a separate design. It does not share Python dictionaries, envstore mutation, scan callbacks, parser slot state, task outputs, MPI order, or mutable user scratch. Host shared calibration and geometry caches remain separate mechanisms.

Use dedicated IPC-compatible allocations for shared objects, with metadata containing base allocation size, array byte offset, shape, dtype, and byte length. Initial allocations use offset zero and contiguous arrays; validate `offset + length <= allocation size` and alignment before constructing a view. Do not export arbitrary CuPy pool suballocations or assume the current `owned_empty()` allocator is an IPC allocator.

NVIDIA requires export from the allocation base, gives no cross-process pointer-address guarantee, and requires importers to close mappings before the exporter frees memory. Wrap the imported local address with a CuPy memory owner that retains its mapping lease. These requirements come from the [CUDA IPC memory API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html); the allocation-wrapper design above is proposed.

Initially finish the owner’s upload stream before publishing `READY` plus memory handles. A follower imports only after READY and skips local device allocation/upload entirely. This avoids a second distributed event lifecycle. A later optimization may publish an IPC readiness event and have each consuming stream wait on it. Such events require interprocess and disable-timing flags, and their lifetime must be retained too. See [CUDA event management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html). A memory handle alone proves neither readiness nor completion.

Capability checks use the actual deployment’s driver/runtime/CuPy combination. Serial or one-BD groups use the local backend. Proposed `auto` policy chooses shared storage only for a verified supported group; an unsupported topology uses a fully budgeted local fallback with a diagnostic. An explicit shared-required mode fails at startup. Runtime corruption, upload failure, or lost owner is a coordinated failure, not silent fallback to uncharged duplicates. Supported MIG sharing needs a device test; otherwise use the explicit unsupported policy.

## Peer coordination and progress

### Discover peers before event processing

Use #155’s topology work as a dependency. All parent-communicator members participate in initialization splits; subsequent discovery includes only GPU BDs. Respect launcher visibility, identify the selected device by full UUID and MIG instance where applicable, and group by node/shared-memory domain plus actual device identity across EB groups. A process-local device ordinal or `bd_rank - 1` arithmetic is not a peer identity. CPU-only/SMD0/EB roles need not create CUDA contexts.

Cache group membership, capacities, protocol version, and run-instance namespace before data delivery. Validate explicit per-BD budgets against aggregate device capacity. Log rank, role, node, actual device, group, backend, and budget. No new collective belongs in lazy manager creation or a BeginStep callback.

### Give the storage owner independent progress

Earlier shared-backend proposal, still subject to review: one job-scoped helper per active shared GPU, elected/launched during coordinated startup and reachable through node-local control sockets. With run-only constants, evaluate its process/context cost against a simpler run-scoped exporter with safe readiness and shutdown; Design 1 does not select a helper deployment. It owns only requested constant allocations and their catalog. BDs keep MPI transport and event work. The helper has an explicit CUDA context and no MPI membership; launch via a fresh executable/process environment, never fork an initialized CUDA process. Account for its host process and device-context overhead. Its lifetime belongs to the job/service supervisor, not the lifetime of the BD that launched it.

This choice ensures a BD blocked in arbitrary user analysis or waiting on its EB cannot stall another BD’s constant acquire/release. Requests do not require the helper to enter the data-processing loop. Deployment must provide a supported launch/supervision method within the Slurm allocation; agree that method before assigning the shared backend. If helpers are unacceptable, an elected BD owner is an alternative only after proving independent progress, including GIL, MPI-thread support if used, blocking receive, user-code stalls, and early BD exit. Polling only at step boundaries does not establish safe startup/release progress. No per-step constant acquire/refresh protocol is required.

Use bounded, versioned node-local control messages. No new Python object transport is needed on the GPUBAT1 hot path. Transfer canonical host bytes to the service through bounded staging/shared host storage during run-manifest acquisition; do not assume a raw pointer in one BD is valid in the helper. Keep that storage alive through comparison/upload and include its host-memory cost in diagnostics.

| Message or state | Required meaning |
| --- | --- |
| `ACQUIRE(request, manifest/object identity)` | Reserve a reference before returning any usable handle; includes run, group, requester, and request generation |
| `PREPARING` | Object has one creator/upload; matching concurrent requests wait on it rather than create duplicate device copies |
| `READY(grant, allocation generation, metadata, handle)` | Upload complete and import reservation registered; publication is atomic |
| `IMPORTED(grant)` / `IMPORT_FAILED(grant)` | Establish mapping ownership or roll back the pending grant; partial multi-object acquisition is tracked |
| `RELEASE(grant)` | Requester has dropped manifest/execution/alias references, completed CUDA users, and actually closed the import; this is the close acknowledgement |
| `STOP_ADMISSION` / `LEAVE` | Stop new references from a client; leave only after outstanding grants are resolved or explicitly quarantined |
| `ERROR(request, phase)` | Preserve original cause, stop dependent work, and enter rollback or coordinated abort |

Messages are idempotent using request/grant/allocation generations. Closing and acquiring race through the owner catalog: a new grant either pins existing storage before retirement or waits for a new allocation; it never receives a handle being freed. Do not free based on a disconnected socket, a step watermark, a timeout, or a count of currently executing kernels. Pending imports count too.

No all-peer BeginStep handshake is needed. Each BD acquires its run manifest before processing and retains it across steps. A BD finishing early releases its own grants; storage remains alive for slower peers and retained consumers. Run startup/shutdown coordination is distinct from step synchronization. Content deduplication and fresh allocation generations remain useful, but there is no history of per-step calibration versions to retain.

### Budget physical storage once

Let `D` be managed device capacity after explicit allowance for contexts, libraries, and external allocations. Reserve a shared-constant allowance `C` and per-BD working quotas `W_i` such that:

```text
C + sum(W_i) <= D
shared_live_bytes + shared_pending_bytes <= C
local_working_live_i + local_working_pending_i <= W_i
```

Charge shared allocation capacity once to the group ledger, including all distinct live run objects and quarantined objects. Imported bytes are observability (`borrowed`), not physical bytes added again to each BD’s committed total. Exporter ownership does not consume an arbitrary extra chunk of that BD’s work quota. Local fallback copies remain physical allocations and are fully charged. Pinned/host snapshots, retained host results, and user scratch need separate reporting; this policy cannot enforce a limit on arbitrary user allocations.

Use a fixed shared allowance first, with a minimum viable working allocation for every BD. Reserve requested objects for run setup before creating them. Reduce execution overlap and trim only unreferenced caches under pressure. If required constants and minimum working allocations cannot fit, return a clear budget error after safe rollback; do not wait indefinitely for a slow peer, invent a step barrier, or free live objects. An optional bounded retry may retire already-completed references, but pressure cannot create a wait cycle.

Example only: four BDs requesting the same 1 GiB need roughly 1 GiB of shared constant payload instead of 4 GiB. Different requested values require distinct storage; partially overlapping declarations share only identical objects. There is no second copy solely because a BD advances to another step. Allocation overhead and helper contexts are additional. These are capacity calculations, not measured throughput claims.

## Iterator ownership and cleanup

One run stream controller owns the manager, transport adapter, pending deliveries, and close state. `run.events()` and `run.steps()` are alternate views, not independent readers. Reject simultaneous iteration modes and multiple live child event readers. Give the outer and child iterators explicit idempotent `close()` support and compatibility with `contextlib.closing`.

| User action | Proposed semantics |
| --- | --- |
| Exhaust a step normally | Deliver its local events, apply CPU EndStep in order, finish relevant scope consumers, invalidate the child; keep the run and later in-flight scopes open |
| Close a child early, or advance outer iterator with an unfinished child | Invalidate that child, consume the remainder to its boundary, process transitions and retire results, discard public delivery; then allow next step |
| Remainder includes unsubmitted L1 work | Initially run normal task submission and discard its delivery, preserving callback side effects and one scheduling rule. Skipping computation is a later opt-in contract |
| Bare `break` with retained iterator | No implicit close. Advancing the outer iterator invokes remainder handling; otherwise use explicit close/terminate |
| Close outer event/step iterator | Terminal local stream close; stop new local work, drain owned work, and drain the existing MPI transport protocol as necessary |
| Stop the MPI run globally | Call existing `run.terminate()` and close the owning iterator; do not assume local close is a collective stop request |
| Raise in user loop body | `with closing(...)` unwinds the owner; use `run.terminate()` when a global stop is intended. Bare loop-body exceptions cannot magically close a retained generator |
| Fatal callback/pipeline error | Preserve existing MPI global error protection; attempt bounded safe rollback, then abort rather than leave peers waiting |
| Use a retained iterator after child/owner close | Clear closed/stale-iterator error; no access to released storage |

The CPU delivery iterator retains the next boundary marker so a next BeginStep or EndRun cannot be swallowed by child remainder processing. This delivery lookahead is separate from GPU scheduling, which may already have admitted later scopes within the memory budget. Empty BeginStep/EndStep pairs produce an empty Step. Do not synthesize a science step for L1 records outside explicit step boundaries; preserve CPU event/step behavior. MPI steps describe locally assigned events plus the transition history each rank receives, not a promise that every rank handles every L1. End-of-stream history delivery must preserve empty steps under normal routing.

Proposed usage after implementation:

```python
from contextlib import closing

with closing(run.steps()) as steps:
    for step in steps:
        with closing(step.events()) as events:
            for evt in events:
                analyze(evt)
                if stop_this_step(evt):
                    break  # child close consumes/discards the remainder
        if stop_run():
            run.terminate()  # request coordinated MPI stop
            break           # outer close performs local cleanup
```

Normal terminal cleanup follows this order: stop admission; unwind active delivery scopes; finish submitted reads and retain file/input owners; complete producer, D2H, and registered input consumers; preserve retained host result rows; release parser/input/slot resources; drop run-manifest references; close eligible IPC mappings and acknowledge releases; release pinned staging; finish outstanding EB communication. The service frees exports only after the ownership predicate is true and stops after all clients have left. Repeating any phase is safe.

Partial initialization uses staged rollback records from the first reservation onward. Upload failure retains source/destination until completion is known. Import failure closes every successful partial import and acknowledges each grant; it must not call a full manager close on a half-built object. If CUDA completion or a peer’s import closure cannot be proven, quarantine storage and its charge, report phase/rank/object, and use bounded job failure. No graceful owner failover is proposed: a lost exporter invalidates the shared group and triggers MPI/job abort. Timeout means failed coordination, never permission to free potentially live memory.

Escaped aliases are an exceptional retention case: local run close can become logically terminal while the mapping/service remains pinned. Diagnostics must distinguish logically closed from fully reclaimed. At final job shutdown, unresolved aliases or failed completion require a reported failure/quarantine path, rather than falsely claiming a clean zero-allocation exit. The external API continues to prohibit using such aliases for new work.

## Execution timelines

### Trailing BeginStep and overlapping slots

```text
EB packet 1: [L1 A, L1 B, BeginStep 2]    packet 2: [L1 C, L1 D, ...]

GPU planner: bind A/B to scope 1 -> discover scope 2 -> bind C/D to scope 2
slot 0:      A/B execute under scope 1 ----------------------> done
slot 1:                                C/D under scope 2 ----------> done
CPU delivery: yield A, B -> apply CPU BeginStep 2 -> yield C, D
constants:   same run manifest throughout ------------------------------>
```

The planner never applies step-2 payload values to CPU state ahead of A/B delivery. GPU kernels consume no step payload values. Scope bookkeeping and buffer leases outlive their consumers. A leading catch-up BeginStep establishes scope before its first event; known replay does not increment the generation twice. Empty steps remain visible to the public step iterator.

### Uneven BDs sharing a GPU

```text
BD A slot 0: scope 1 --------------------> done
BD A slot 1:         scope 2 ----------------------> done
BD B slot 0: scope 1 --------------------------------------> done
constants:  one shared immutable run object; independent BD grants
```

Neither BD waits for a peer to reach a step boundary. A faster BD can finish its run locally while the shared owner retains allocations for the slower BD. Final export free follows all import-close acknowledgements and consumer completions, not a step watermark.

### Ignored GPU values and independent chunk rollover

```text
records:      [A, SlowUpdate U, B, ClearReadout, Enable(chunk 2), C]
GPU scope:    scope 1 ------------------------------------------>
GPU values:   no SlowUpdate/step payload extraction or retention
CPU delivery: A -> apply U -> B -> controls -> C
read file:    chunk 1 --------------------------| chunk 2
read offsets: descriptor offsets in chunk 1     | offsets in chunk 2
constants:    same run manifest -------------------------------->
```

ClearReadout remains a valid control and conservative coalescing fence without changing files. SlowUpdate remains available through existing CPU handling; this diagram does not claim to repair its independent timestamp-lookup issue. Pending chunk-1 reads retain their original handles even after chunk-2 reads are admitted.

### Child close and terminal cleanup

Closing a child consumes/discards its remainder through the proper boundary and finishes relevant scope work; later scopes may already be running. Closing the outer owner stops further admission and drains all owned reads, producers, D2H and registered consumers, then releases local buffers and run-constant imports. `run.terminate()` requests the existing coordinated MPI stop. The shared owner retains exports until every peer grant closes. None of these releases is inferred solely from a Python generator advancing or being collected.

## Alternatives and tradeoffs

| Choice | Benefit | Cost / status |
| --- | --- | --- |
| Design 1: one step per execution subbatch | One bin partial per reduction invocation; cross-step overlap across slots/BDs | Current direction; step boundaries can create smaller subbatches; retain bounded per-scope ownership |
| Mixed-step subbatch with per-event scope | Preserve original execution boundaries; still permits subbatch-level scheduling | Not the baseline; grouped reductions and multi-scope completion/API; potentially multiple image-sized bin partials per execution, or extra passes to limit storage |
| Design 2: local step-boundary drains | Simpler scope completion | Superseded baseline; blocks next-step admission in that BD; still permits independent BDs |
| Full GPU scan/configuration/EnvStore snapshots | Could support kernels consuming changing payload values | Deferred; SlowUpdate and step values remain CPU-side |
| Ordered planner over SMD + GPUBAT1 | Preserve current transport | Must validate placeholder/event correspondence and avoid duplicate CPU dispatch |
| Separate transition-only EB packets / new ABI | More explicit wire boundaries | Defer; broader history/framing work is unnecessary for current design |
| Run-scoped local constants | Simple baseline and serial/single-BD backend | Duplicates VRAM; does not complete #168 for shared groups |
| Run-scoped IPC with independent helper | Owner survives a client stall/exit | Extra process/context; deployment choice remains open |
| Elected BD exporter | Potentially simpler run-scoped deployment | Must retain exports and prove startup/shutdown progress after local event completion |
| Mutable shared constants plus all-peer step barrier | No benefit under fixed run constants | Reject; unnecessary step coupling and deadlock risk |
| Restore old Jungfrau IPC unchanged | Prototype history | Reject generic dtype/selector/topology/ownership assumptions |

## Staged implementation tasks

The selected direction is Design 1 with GPU EnvStore deferred. Remaining interfaces and IPC deployment details still need review; tasks have no assignees.

| Stage | Work | Exit criterion |
| --- | --- | --- |
| 0 Refine contracts | Scope identity/replay, scheduling versus delivery cursors, step completion/close semantics | Concrete API and acceptance cases; no reopening of deferred GPU payload-value scope |
| 1 Ordered planning and files | Planner, ClearReadout decoder/filter/validation, per-scope ranges, immutable file resolution for bulk/non-bulk reads | Real EB trailing BeginStep labels correctly; Enable rollover and service zero tests pass; CPU state not advanced by planning |
| 2 Design 1 scheduler and public steps | Per-submission scope, bounded cross-step look-ahead, completion accounting, event/step adapters and close ownership | Multiple scopes demonstrably in flight within one BD; serial/MPI steps and ordered CPU delivery work; early-exit matrix passes |
| 3 Run constants and peer/budget contract | Remove BeginStep refresh, retain run manifest, coordinate #155 topology and aggregate budgets | Same constants across all steps; actual GPU groups and fully charged local fallback |
| 4 IPC backend | Choose exporter/progress mechanism, implement readiness, import leases, physical-once ledger and partial rollback | Two/four BDs share generic numeric arrays without follower uploads; differing content isolated |
| 5 Integrated lifecycle | Connect run constants, overlapping scopes and teardown; coordinate #169 | Delayed BD/consumer, partial failure, owner/client exit, pressure and retained aliases handled safely within bounded tests |
| 6 Acceptance and rollout | Documentation, diagnostics, synthetic and available real-data checks | Required correctness/core checks pass; evidence reviewed before default enablement |

Stages 1–2 can ship with local constants; IPC must not block #153's step/transition correctness. Run-only constants simplify #168 but do not eliminate topology, readiness, ownership, budgeting or teardown requirements. Separate implementation PRs should remain independently reviewable.

### Verification matrix

| Area | Required cases and assertions |
| --- | --- |
| Ordered planning | Real fast EB append/break; new BeginStep creates a scope; same-source history replay preserves identity/generation; leading/trailing history and empty steps; transition-only packets; GPU-only placeholders and zero-GPU rows; malformed framing rejected |
| Scope overlap | Every execution subbatch has one scope; two or more occupied slots may retain different scopes; two BDs progress independently; delayed old scope; byte splits preserve identity; no blanket step flush |
| Step/bin reductions | Compare per-step sums with explicit expected values for several subbatches per step, concurrent same/different-step slots, short/empty steps and retained partials; no cross-bin contamination or premature accumulator reset/free; account for scratch/output storage |
| Scheduling cost | Scope discovery reuses packet traversal; no per-field scheduling or repeated full-plan construction per scope/subbatch; submissions change only for required execution boundaries; measure short-step fragmentation and memory high water |
| CPU/GPU division | GPU ignores SlowUpdate and step payload values; no GPU EnvStore/upload/context API; CPU dispatch once in order; GPU look-ahead does not mutate public CPU configuration early |
| File rollover | Bulk/non-bulk; new chunk offsets; multiple streams; Enable without change; leading/trailing/replayed chunkinfo; old handle retained until pending reads complete; rollover independent of step |
| Constants | Run setup once; unchanged manifest across steps/controls; next run setup; no per-step refresh or acquisition barrier |
| ClearReadout | Service 0 present, empty record absent; leading/trailing missing streams; malformed sizes fail; CPU dispatch and coalescing fence; Enable behavior unchanged; sentinel 13 rejected |
| Public delivery | Serial/MPI events/steps, empty steps, GPU-only/hybrid/mixed events, no task, sparse results, retained CPU results; step completion does not flush later scopes |
| Iterator lifecycle | Child/outer close, outer advance, retained stale iterators, bare break, user exception, terminate, repeated close, EndRun/EOF/max-events, outstanding EB receive |
| Sharing | 2/4 BDs per GPU, actual devices across nodes/EB groups, visibility/MIG capability or explicit fallback, same/different selectors/content, no duplicate uploads |
| Array fidelity | Numeric/complex dtype matrix, scalar/empty arrays, noncontiguous host normalization, shape and exact-byte identity, unchanged constant axes |
| Ownership/failure | Delayed reads/producers/D2H/input consumers, early peer exit, aliases through close, failed upload/import, stale grant messages, lost exporter, insufficient budgets; no premature free/double charge |
| Progress | One BD stalled in user code while another progresses or finishes; owner stays available through final release; no step collective; bounded failure handling |

Instrument scope identity/admission/completion and allocation/upload/import/open/close/free counts. Demonstrate actual overlap by holding old-scope work pending while later-scope work is admitted, rather than inferring it from labels. Verify every export free follows close acknowledgements and CUDA completions; distinguish shared physical bytes from imported borrowed bytes and per-BD working bytes. Include retained/quarantined and host/pinned memory in diagnostics.

Implementation validation must run focused CPU, GPU, serial and MPI tests, then both core groups in the built environment:

```bash
pytest psana/psana/tests/
pytest psana/psana/tests/byhand_*
```

Use synthetic arrays/dgrams first; real IPC tests need supported hardware and deployment versions. Performance follows correctness: record topology, batch size, slot depth, scope overlap, backend/budgets, GDS/fallback, memory high water, transition/delivery waits and throughput. No throughput improvement is assumed. This documentation update needs link, Markdown/whitespace and applicable pre-commit checks, not runtime tests.

## Remaining decisions for Mona and Seema

| Decision | Starting point |
| --- | --- |
| Scope identity and public API | Existing local `batch.step_generation` plus replay-safe source scope identity; validate new BeginStep, same-source history replay and empty steps in SMD record order; no origin-specific handling or step payload parsing |
| Scheduling/delivery implementation | Independent cursors and bounded look-ahead; ordered CPU state updates and per-scope result completion |
| Iterator remainder semantics | Child close/outer advance consumes and discards remainder while preserving callbacks; outer close terminal locally; explicit global terminate |
| Stateful user work | One step per execution; isolated partial sums/scope storage or declared dependencies; define combining and retaining multiple subbatch contributions without assuming one accumulator per BD |
| IPC exporter/progress | Reassess helper versus elected exporter for run-only acquisition and safe peer cleanup; no step refresh service |
| Capacity/support policy | Fixed shared reserve plus validated BD quotas; capability-tested sharing, explicit required mode/fallback and diagnostics |
| Exceptional retention | Retain aliases and charges, report incomplete reclamation; no unsafe exporter failover/free |

Design 1, fixed run constants, and deferred GPU EnvStore are the current direction. GPU ignores both SlowUpdate values and step payload values; CPU retains them. Remaining implementation/API choices do not reinstate those deferred features. This document does not close #153 or #168.
