# psana2 GPU Architecture Overview

**Status:** Current on this branch.

## Scope

The implemented path is a CPU-orchestrated, GPU-accelerated psana2 pipeline.
Smd0 and EventBuilder retain their normal roles. EventBuilder creates a
coherent CPU batch and GPU descriptor batch for the same event range. A CPU
BigData (BD) process submits KvikIO/cuFile reads, GPU XTC parsing, detector
and optional dense input preparation on its assigned GPU.

The runtime performs no calibration, geometry preparation, or automatic output
D2H. Calibration algorithms belong in explicit user code; the producer callback
API is still under development. The internal `DenseInputPreparer` supports
validated Jungfrau raw panels, independently of calibration constants.

## User-facing routing

```python
from psana import DataSource

ds = DataSource(
    exp="mfx100848724",
    run=51,
    gpu_det="jungfrau",
    batch_size=5,
    n_gpu_streams=2,
)

run = next(ds.runs())
for evt in run.events():
    fields = evt.gpu.detector("jungfrau").field("raw", "raw")
    raw_by_segment = fields.on_cpu  # explicit input copy for inspection
```

`gpu_det` gives the GPU path exclusive ownership of every selected detector
stream. `hybrid_det` mirrors complete selected streams through both CPU and GPU
paths when a selected detector shares a stream with CPU consumers. Mirroring
therefore trades compatibility for duplicate bigdata I/O. A detector cannot be
selected by both modes, and exclusive and mirrored stream sets cannot overlap.

GPU reads coalesce adjacent ranges by default inside the existing subbatch/slot.
No extra DataSource argument is needed. `gpu_bulk_read=False` retains per-dgram
reads for debugging and comparison. The BD resolves file/chunk identities
from ordered SMD transitions before submitting reads, then coalesces ranges
within each file and transition interval. GPUBAT1 and logical event order stay
unchanged; parser offsets are rebased into the physical read layout.

This mode requires ordinary GPU batch routing; `intg_det`, timestamp filtering,
and `smd_callback` do not supply the required supported packet path. It can
retain adjacent input groups across executions when the byte budget permits;
see [input ownership](bulk_ownership_stage4_findings.md).

## End-to-end flow

```text
Smd0
  -> normal SMD chunks

EventBuilder
  -> align events and transitions
  -> build the CPU-readable SMD batch
  -> build the GPUBAT1 descriptor batch for GPU-routed streams
  -> send one coherent BatchEnvelope to a BD worker

BD / GpuEventManager
  -> issue KvikIO reads into reusable input storage
  -> construct CPU events for CPU-routed streams
  -> parse XTC and locate configured fields on the GPU
  -> retain parsed inputs and record completion
  -> join CPU and GPU state by timestamp
  -> yield normal Event objects
```

CUDA kernels do not open files. A CPU process submits KvikIO reads; true GDS
places the result directly in GPU memory, while unsupported storage uses
KvikIO's CPU fallback.

## Batch and parser contracts

EventBuilder sends a versioned, byte-oriented GPUBAT1 message containing event
identity and bigdata read descriptors. The hot MPI path does not pickle GPU
objects. Timestamp and batch event index keep its events aligned with the CPU
batch.

Configure dgrams are compiled once per run into `GpuStreamConfigTable`. The
numeric tables are uploaded once and shared by all parser slots. For each read
dgram, the GPU walker validates XTC, matches ShapesData to Configure Names, and
produces field locator rows containing type, rank, shape, device offset, byte
count, and status.

`DenseInputPreparer` consumes those locators when explicitly selected internally;
it does not infer a fixed payload offset or stride. Event fields are available through:

```python
field = evt.gpu.detector("jungfrau").field("raw", "frame_cnt")
values_by_segment = field.on_cpu
```

See [GPU XTC parser](gpu_xtc_parser.md) for the table and ownership contracts.

## Segment identity

Physical segment IDs come from Configure. Runtime field locators preserve the
stream and segment ownership needed to map each event into the detector's
canonical output rows. Detector adapters validate Configure membership and do
not use dictionary sorting as a substitute for physical segment identity.

General parser fields remain keyed by physical segment ID because arbitrary
fields may be ragged or have different shapes. Dense stacking, padding,
and shape validation are input-adapter policy. Calibration and image assembly
belong to user algorithms.

## Execution and ownership

`GpuEventManager` is run-scoped and owns:

- Configure-derived routing and parser tables.
- KvikIO readers and optional dense input preparers.
- The per-BD budget for framework-owned input storage.
- `EventPool`, whose reusable slots each own a non-blocking CUDA stream.

Execution slots retain prepared input views and execution completion state. They hold references to `InputWindow` owners for raw bytes and parser
rows. An input window can serve multiple executions and cannot be recycled
until planned uses, event consumers, and CUDA work have finished. The current
scheduler admits affordable complete stream inputs for one EB batch, reads and
parses them once, and combines them with transient inputs for ordered execution
subbatches. Reader/parser pools have one extra lazy slot for resident input;
execution and detector slot counts are unchanged. When no complete input fits,
the scheduler uses common input/execution subbatches.

Before issuing each read, admission reserves reader/parser/prepared-input growth,
including replacement peaks. Configure allocations are charged once per BD.
Cached buffers retain their charge. Pressure drains consumers before trimming
free storage. A 10% margin covers runtime/allocator overhead; user allocations
are outside this ledger. No calibrated output or geometry storage is reserved.

Per-event `GpuEventState` objects expose that event's results and input bindings;
they do not own the manager. Field-view contexts and field copies reserve input
references before accessing raw storage.

The intended lifetime rule is:

> A reusable slot cannot be overwritten until every terminal GPU or D2H
> consumer of its contents has completed.

Advancing the Python generator is not proof of CUDA completion. See
[Memory backpressure and results](memory_backpressure_and_results.md). Parsed
input leases and generic result leases retain every registered consumer event.

## Transitions

- `BeginStep` drains prior work before dispatching the host transition. No
  built-in GPU constant recipe or refresh runs.
- `EndRun` drains pending GPU input executions exactly once.
- KvikIO handles and run-scoped GPU resources are closed when iteration exits.
- Intermediate transitions do not introduce unnecessary full-pipeline drains.

## MPI placement

GPU selection happens on BD ranks before CuPy import. Non-BD ranks hide CUDA
devices so Smd0 and EventBuilder do not allocate GPU state. Multiple BD
processes may share one physical GPU. Framework input storage is process-owned;
fixed-pair calibration CUDA IPC has been removed. GPU-exclusive detectors are
excluded consistently from derived CPU caches on every shared-memory rank; CPU
and hybrid consumers retain their normal caches and source dictionaries.

Multi-EventBuilder GPU placement is an open design issue. The architecture
needs a node-wide GPU ownership and budgeting model across EventBuilder groups;
this overview does not define `PS_EB_NODES=1` as the intended solution.

## Current limitations

- GPU `RunParallel.steps()` does not yield steps; use `run.events()` for the
  implemented transition path.
- True GDS depends on filesystem, driver, and cuFile runtime support.
- GPU budgets are per BD process; coordinated admission across processes that
  share one device remains future work.
- User-defined GPU stages inside the producer pipeline are proposed, not
  implemented. See [User GPU pipeline](proposals/user_gpu_pipeline.md).

The maintained issue list, including correctness risks and validation needs,
is [Known problems and limitations](known_issues.md).

## Active implementation

| File | Responsibility |
|---|---|
| `eventbuilder.pyx` | CPU/GPU stream split and GPUBAT1 construction |
| `gpu_batch.py` | GPUBAT1 views and byte-bounded subbatches |
| `gpu_events.py` | Run-scoped input orchestration, event delivery, and transitions |
| `gpu_kvikio_read.py` | Per-slot asynchronous bigdata reads |
| `gpudgram/` | Configure tables, device XTC walk, and field locators |
| `gpu_input.py` | Event input views, detector bindings, and input leases |
| `gpu_detector.py` | Dense input preparation and batched gather |
| `gpu_stream.py` | Reusable execution slots and retirement |
| `context.py` | `GpuEventState`, `GPUResult`, and result access modes |
| `gpu_budget.py` | Per-BD accounting for explicitly tracked device allocations |
| `gpu_admission.py` | Presence-aware execution sizing and stream-residency admission |
| `gpu_mpi.py` | Device assignment and per-GPU BD counts |
