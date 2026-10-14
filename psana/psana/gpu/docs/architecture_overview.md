# psana2 GPU Architecture Overview

**Status:** Current on this branch.

## Scope

The implemented path is a CPU-orchestrated, GPU-accelerated psana2 pipeline.
Smd0 and EventBuilder retain their normal roles. EventBuilder creates a
coherent CPU batch and GPU descriptor batch for the same event range. A CPU
BigData (BD) process submits KvikIO/cuFile reads, GPU XTC parsing, detector
gather, and detector processing on its assigned GPU.

Jungfrau is the current calibrated detector implementation. The XTC parser and
field-access contracts are detector-independent. Uncompressed raw Jungfrau and
pre-calibrated float32 passthrough data are supported; common-mode correction
is not yet implemented on the GPU path.

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
    calib = evt.gpu.get("calib")
    calib_gpu = calib.on_gpu
    # calib_cpu = calib.on_cpu
```

`gpu_det` gives the GPU path exclusive ownership of every selected detector
stream. `hybrid_det` mirrors complete selected streams through both CPU and GPU
paths when a selected detector shares a stream with CPU consumers. Mirroring
therefore trades compatibility for duplicate bigdata I/O. A detector cannot be
selected by both modes, and exclusive and mirrored stream sets cannot overlap.

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
  -> issue KvikIO reads into an EventPool slot
  -> construct CPU events for CPU-routed streams
  -> parse XTC and locate configured fields on the GPU
  -> gather detector fields into canonical segment order
  -> calibrate or pass through detector data
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

`GPUDetector` consumes those locators; it does not infer a fixed raw payload
offset or segment stride. General event fields are also available through:

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
calibration, and image assembly are detector-adapter policy rather than parser
policy.

## Execution and ownership

`GpuEventManager` is run-scoped and owns:

- Configure-derived routing and parser tables.
- KvikIO readers and GPU detector adapters.
- The per-BD device-memory budget and asynchronous D2H pipeline.
- `EventPool`, whose reusable slots each own a non-blocking CUDA stream.

Each occupied slot owns its input bytes, parser rows, detector buffers, result
views, and completion state. Per-event `GpuEventState` objects expose only that
event's results and input bindings; they do not own the manager.

The intended lifetime rule is:

> A reusable slot cannot be overwritten until every terminal GPU or D2H
> consumer of its contents has completed.

Advancing the Python generator is not proof of CUDA completion. See
[Memory backpressure and results](memory_backpressure_and_results.md). Parsed
input leases already retain multiple consumer events. A normal detector-result
`SlotLease` currently retains only one event, so multiple zero-copy consumers
of the same result are not yet safe; see
[Known problems and limitations](known_issues.md#result-lease-fan-out).

## Transitions

- `BeginStep` drains work that depends on old calibration constants before
  updating them.
- `EndRun` drains pending GPU results exactly once.
- KvikIO handles and run-scoped GPU resources are closed when iteration exits.
- Intermediate transitions do not introduce unnecessary full-pipeline drains.

## MPI placement

GPU selection happens on BD ranks before CuPy import. Non-BD ranks hide CUDA
devices so Smd0 and EventBuilder do not allocate GPU state. Multiple BD
processes may share one physical GPU; calibration constants may be shared
through CUDA IPC, while input, parser, gather, and output slots remain
process-owned.

Multi-EventBuilder GPU placement is an open design issue. The architecture
needs a node-wide GPU ownership and budgeting model across EventBuilder groups;
this overview does not define `PS_EB_NODES=1` as the intended solution.

## Current limitations

- Calibrated detector processing is currently Jungfrau-specific.
- Common-mode correction is not implemented in the GPU calibration kernel.
- Geometry can be prepared and an image-assembly helper exists, but
  `GPUDetector.process_batch()` does not currently publish `<det>.image`.
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
| `gpu_events.py` | Run-scoped orchestration, joins, D2H, and transitions |
| `gpu_kvikio_read.py` | Per-slot asynchronous bigdata reads |
| `gpudgram/` | Configure tables, device XTC walk, and field locators |
| `gpu_input.py` | Event input views, detector bindings, and input leases |
| `gpu_detector.py` | Dense gather, calibration, passthrough, and geometry |
| `gpu_stream.py` | Reusable execution slots and retirement |
| `context.py` | `GpuEventState`, `GPUResult`, and result access modes |
| `gpu_budget.py` | Per-BD accounting for explicitly tracked device allocations |
| `gpu_mpi.py` | Device assignment and CUDA IPC calibration sharing |
