## Bulk Reader Draft Plan

This note defines the first draft for an experimental payload-descriptor /
bulk-reader pipeline on top of the `psana2-gpu` baseline.

The purpose of this draft is to test whether we can bypass the current hot-path
costs from:

- host bigdata object handling
- `Event._assign_det_segments()`
- `det_raw._segments(evt)`
- extra host-to-host staging into the slot buffer

while still reusing the existing GPU calibration path where practical.

### Why This Is A Separate Experiment

The current `cupy + async 3stage` branch is useful as:

- correctness baseline
- profiling baseline
- bottleneck baseline

The bulk-reader direction changes the ingest model enough that it should be
treated as a parallel experimental path rather than an in-place tweak to the
existing pipeline.

### First-Draft Goal

Prove that a descriptor-driven Jungfrau GPU path can:

- bypass `_det_segments` / `_segments(evt)` on the hot path
- read detector payload by offset/size into pinned host buffers
- feed the existing CuPy Jungfrau calibration path
- produce correct `det.raw.calib(evt)` results
- provide a clean abstraction that can later swap pinned-host I/O for GDS

### First-Draft Scope

Included:

- Jungfrau only
- `det.raw.calib(evt)` only
- CPU control plane
- descriptor batching
- pinned-host bulk read path
- reuse of current CuPy calibration path when possible

Not required in the first draft:

- GDS
- arbitrary user GPU stages
- multi-detector generalization
- final public API design
- multi-GPU scheduling policy
- direct replacement of the existing `cupy + 3stage` runtime

### Target Model

```text
CPU metadata path
    ->
descriptor batch
    ->
pinned-host bulk reader
    ->
GPU raw/calib path
    ->
result event or small result
```

The CPU remains the control plane:

- timestamps
- transitions
- config/cache identity
- descriptor construction
- batch routing

The GPU-side path becomes the detector payload plane:

- fetch payload
- build GPU-native Jungfrau raw representation
- run calibration
- later support additional device-resident stages

### Ownership Hierarchy

This section describes the intended top-level ownership model, starting from the
normal psana entry point.

```text
DataSource(...)
  ->
RunGpu
  ->
BulkReaderRuntime
      owns:
        - DescriptorBuilder
        - BulkReader
        - IngressBuffer pool / ring
        - GPU execution backend
        - result readiness / batching state
  ->
detector-facing GPU wrapper
  ->
det.raw.calib(evt)
```

More concretely:

#### 1. `DataSource(...)`

Normal psana entry point.

- still responsible for creating the run object
- still carries `gpu_detector`, `gpu_runtime`, and related GPU knobs

#### 2. `RunGpu`

Top-level GPU run façade.

- created through the normal psana run selection path
- owns the selected runtime
- still provides the public psana-facing interface:
  - `run.Detector(...)`
  - `run.events()`

#### 3. `BulkReaderRuntime`

Main owner of the experimental pipeline.

- created by the runtime factory when:
  - `gpu_runtime="bulk-reader"`
  - `gpu_pipeline="descriptor"`
- owns the ingest/data-plane state for the GPU detector path
- should be the place where batching, readiness, and transition handling live

Planned owned subcomponents:

- `DescriptorBuilder`
- `BulkReader`
- `IngressBuffer` pool or ring
- GPU execution backend

#### 4. `DescriptorBuilder`

Owned by `BulkReaderRuntime`.

- converts CPU-side event metadata into lightweight detector payload
  descriptors
- should use offset/size and other control metadata
- should not depend on `_det_segments` for the hot path

#### 5. `BulkReader`

Owned by `BulkReaderRuntime`.

- consumes descriptor batches
- fills ingress buffers
- first implementation:
  - `PinnedHostBulkReader`
- later implementation:
  - `GdsBulkReader`

#### 6. `IngressBuffer`

Owned and recycled by `BulkReaderRuntime`, but filled by `BulkReader`.

- holds detector payload bytes for one or more descriptors
- first draft uses pinned host buffers
- later versions may use device-resident ingress buffers

#### 7. GPU Execution Backend

Owned by `BulkReaderRuntime`.

- consumes ingress-buffer data
- performs H2D transfer when needed
- launches Jungfrau calibration and later device-resident stages

#### 8. Detector-Facing Wrapper

Exposed back through `RunGpu.Detector(...)`.

- should preserve psana-facing usage as much as practical
- returns calibrated data for:
  - `det.raw.calib(evt)`

### Top-Level Object Flow

Planned first-draft object flow:

```text
DataSource(...)
  ->
RunGpu
  ->
BulkReaderRuntime
  ->
DescriptorBuilder builds JungfrauPayloadDescriptor batch
  ->
BulkReader fills IngressBuffer
  ->
GPU execution backend consumes ingress data
  ->
result attached to event-facing wrapper
  ->
user calls det.raw.calib(evt)
```

This is intentionally different from the current hot path:

```text
Event
  ->
_det_segments
  ->
det_raw._segments(evt)
  ->
host slot staging
  ->
GPU
```

The bulk-reader experiment is meant to replace that hot path for the GPU
detector case, while keeping the top-level psana ownership model recognizable.

### Explicit Non-Goal

The first draft should **not** attempt to reproduce the current full:

```text
Dgram -> Event._det_segments -> DetectorImpl._segments()
```

object model on the GPU.

That would preserve the current hot-path costs we want to escape and would
entangle the draft with CPU/Python object semantics.

### Proposed Components

#### 1. Descriptor Builder

Build a lightweight per-event descriptor for Jungfrau payloads.

Minimum descriptor fields:

- timestamp
- service
- file/chunk identifier
- payload offsets
- payload sizes
- detector name
- segment count / layout metadata
- cache/config identity

#### 2. Bulk Reader Interface

Common interface for reading detector payload from descriptors.

First implementation:

- `PinnedHostBulkReader`

Future implementation:

- `GdsBulkReader`

The rest of the pipeline should depend on the reader interface rather than on
how the bytes arrived.

#### 3. Ingress Buffer

Reusable long-lived pinned host buffers used by the reader.

First draft should try to make these buffers already match the expected detector
layout so that we do not reintroduce an extra host-to-host repack step.

#### 4. Experimental Runtime

A separate runtime should own:

- descriptor batching
- bulk reading
- GPU submission
- result readiness

This should be treated as a parallel runtime path rather than folded into the
current `3stage` pipeline immediately.

### Implementation Order

#### Phase 1: Design + Skeleton

- descriptor dataclass for Jungfrau
- bulk reader base interface
- pinned-host bulk reader skeleton
- runtime skeleton for the bulk-reader experiment

#### Phase 2: Minimal Correctness Path

- produce descriptors for Jungfrau events
- fill pinned host buffers from file offsets/sizes
- hand off data to the existing CuPy Jungfrau execution path
- validate against current CPU reference data

#### Phase 3: Profiling

Measure:

- descriptor build time
- bulk read time
- H2D time
- kernel time
- end-to-end loop wall time

Compare against:

- current `cupy + async 3stage`

### Success Criteria

The first draft is successful if it demonstrates at least these:

- correctness for `det.raw.calib(evt)`
- removal of `_det_segments` / `_segments(evt)` from the hot GPU path
- removal of the extra host-to-host staging copy
- a clean reader abstraction that can later swap in GDS

Performance improvement is highly desirable, but the first draft is primarily
about validating the architecture and isolating the ingest bottlenecks.

### Open Questions

- Where should descriptor construction live relative to current psana event
  iteration?
- Can the payload be read directly into final detector layout in pinned memory?
- What is the best boundary between CPU event tokens and GPU detector payload?
- How should transition/config changes invalidate descriptor/cache state?
- What is the minimum event object needed to preserve a usable psana-facing
  interface while bypassing the old segment path?
