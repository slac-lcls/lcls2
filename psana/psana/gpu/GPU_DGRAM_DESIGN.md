## GPU Dgram / GPU-Side XTC Parser Design

This note defines a new project direction for `psana2-gpu`: a GPU-side
`Dgram`/XTC parsing layer that can consume bulk event bytes on device and
extract detector payload there.

It is intentionally separate from the current
[`BULK_READER_PLAN.md`](./BULK_READER_PLAN.md)
prototype. That branch remains useful as the current ingest/control-plane
baseline, but its transitional host `Dgram` rebuild is not the target
architecture.

### Problem Statement

The current GPU path and the experimental bulk-reader path both still depend,
directly or indirectly, on host-side `XtcData` parsing semantics:

- host `Dgram(view=...)` creation
- host `Event._assign_det_segments()`
- host `DetectorImpl._segments(evt)`
- host-side detector object trees

That stack is correct, but it forces the hot path through CPU object assembly
even when the useful detector bytes are already known by offset/size or could be
made GPU-resident.

The long-term goal is therefore:

```text
bulk event bytes on GPU
    ->
GPU-side structural XTC walk
    ->
GPU-side detector extraction
    ->
GPU-native raw/calib/user stages
```

### Why This Is A Separate Project

The current bulk-reader branch already proves useful things:

- BD-side SMD interception works
- descriptor batching exists
- pinned-host bulk read exists
- naming is now explicit about `smd` vs future `bd`
- tests exist for the prototype seam

But the current parser stage:

- rebuilds host `dgram.Dgram`
- rebuilds host `Event`
- re-enters the normal host detector stack

That is a good bridge for validation, but it does not remove the structural
cost we eventually want to eliminate.

### Architectural Decision

Two long-term directions were considered.

#### Option A: CPU XTC Parse + GPU Payload Transfer

Model:

```text
CPU parses full XTC/Dgram
    ->
CPU extracts detector payload
    ->
GPU receives already-extracted raw array
```

Strengths:

- lower implementation risk
- easiest short path to more detector support
- keeps using existing detector semantics and calibration contracts

Weaknesses:

- still pays host structural parse costs
- still ties GPU work to detector-specific host extraction
- less attractive if bytes already land on GPU through GDS or another
  GPU-visible ingress path
- does not create a reusable "GPU can interpret XTC bytes" capability

#### Option B: GPU-Side XTC Parse / GPUDgram Runner

Model:

```text
event bytes arrive in GPU-visible memory
    ->
GPU walks Dgram/Xtc structure
    ->
GPU extracts detector arrays
    ->
GPU calib and later stages stay device-resident
```

Strengths:

- aligns with GDS and future GPU-visible ingest
- creates a reusable capability for any GPU-side XTC bytes
- avoids routing every detector through host object reconstruction
- cleaner foundation for later GPU-resident common mode, assembly,
  integration, and user kernels

Weaknesses:

- materially larger project
- requires a GPU-native subset of `XtcData` semantics
- requires careful validation against real XTC layout

Decision:

- treat Option B as the target architecture
- keep Option A and the current bulk-reader branch as validation and fallback
  infrastructure

### Key Constraint

This project should reuse `XtcData` byte layout and semantics, but it should
not assume the existing host implementation can simply be compiled into CUDA.

What is realistic:

- reuse on-disk/in-memory XTC layout
- reuse `TypeId`, `NamesId`, `NamesLookup`, `ShapesData`, and `DescData`
  semantics
- write GPU-compatible views and traversal code for the supported subset

What is not realistic as a primary plan:

- compiling the current host `XtcData` codebase unchanged into device code
- reproducing Python `Event` or detector object semantics on GPU

### Scope

#### In Scope For The First GPUDgram Project

- offline xtc datagrams read by psana
- GPU-side structural parsing of a minimal `XtcData` subset
- Jungfrau as the first detector target
- output sufficient for `det.raw.raw(evt)` and `det.raw.calib(evt)` equivalent
  GPU paths
- compatibility with current CPU control-plane ownership
- eventual compatibility with pinned-host ingress first, GDS later

#### Out Of Scope Initially

- full host/Python `Event` semantics on GPU
- arbitrary detector support in phase 1
- complete device implementation of every `XtcData` feature
- direct drop-in replacement for all psana detector APIs
- full multi-GPU scheduling policy

### Design Principle

The CPU remains the control plane. The GPU becomes the event payload plane.

CPU responsibilities:

- transitions
- run/config lifetime
- stream routing
- batch planning
- detector selection
- cache ownership and invalidation policy
- fallback decisions

GPU responsibilities:

- interpret event bytes as `Dgram`/`Xtc`
- locate detector payload using `XtcData` semantics
- extract detector arrays
- keep raw/calib/intermediate results device-resident

### Minimum GPU-Side XtcData Subset

The first project should support the smallest subset that can correctly parse
Jungfrau event bytes without host `Dgram` reconstruction.

#### 1. `GpuDgramView`

GPU view over the event root:

- pointer to start of event bytes
- pointer or size limit for bounds checking
- transition timestamp/env access
- root `Xtc` access

This mirrors the structural role of `XtcData::Dgram`, not the host Python
wrapper.

#### 2. `GpuXtcView`

GPU view over an `Xtc` node:

- `src`
- `damage`
- `contains`
- `extent`
- `payload()`
- `next()`

This is the basic building block for traversal.

#### 3. `GpuTypeId`

Support only the types needed for the initial parser:

- `Parent`
- `Names`
- `Shapes`
- `Data`
- `ShapesData`

#### 4. `GpuNamesId`

Support the same logical role as `XtcData::NamesId`:

- map event `ShapesData` payloads to configure-time metadata
- preserve `nodeId` + `namesId` identity semantics

The device does not need to use the same host container type, but it must
consume the same encoded identifier.

#### 5. `GpuNamesLookup`

Device-readable lookup keyed by `NamesId`, containing only what event parsing
needs:

- detector name / detector type / segment
- algorithm identity and version
- field list
- per-field scalar/array type
- rank
- shape-offset mapping or equivalent precomputed metadata
- offsets or enough metadata to compute offsets in `ShapesData`

Important design choice:

- configure transitions should be parsed once on the CPU first
- the result should be uploaded as a compact device lookup

This preserves the CPU control plane while still making event parsing GPU-side.
It also avoids forcing the first phase to implement full device parsing of
configure `Names` trees.

#### 6. `GpuShapesDataView`

GPU view over one `ShapesData` payload:

- associated `NamesId`
- access to shapes block
- access to data block

#### 7. `GpuDescDataView`

GPU equivalent of the minimum useful part of `DescData`:

- resolve a field index from `GpuNamesLookup`
- compute byte offsets inside the `Data` payload
- return scalar or array pointer/shape metadata

The first implementation should avoid string lookup in hot kernels. Instead,
field indices should be resolved ahead of time by detector-specific setup.

#### 8. `GpuXtcWalker`

GPU-side iterator or walker that traverses nested `Parent` xtcs:

- non-recursive implementation preferred
- explicit stack with fixed maximum depth
- bounds checking on every `extent`
- collect only matching nodes instead of materializing a full tree

The goal is traversal, not object graph construction.

#### 9. Detector Extractor Interface

First detector-specific consumer of the walker.

Suggested interface:

```text
GpuDetectorExtractor
    parse_event(GpuDgramView, GpuNamesLookup, output)
```

Responsibilities:

- identify the xtc nodes relevant to one detector
- validate expected field names/types/ranks
- extract payload pointers and metadata
- convert to a stable GPU raw representation

### Jungfrau First Target

The Jungfrau path is the first correctness target because the current
`psana2-gpu` work already centers on it.

The practical first output should be a GPU-native raw representation equivalent
to what existing calibration expects:

- shape: `(<nsegments>, 512, 1024)`
- dtype: `uint16`

That means the first detector-specific milestone is not "generic GPU event
objects". It is:

- find Jungfrau segment payloads in event bytes
- decode enough metadata to identify segment index and payload location
- produce the contiguous raw stack expected by the calibration path

### Proposed Object Model

The project should keep the object model shallow and explicit.

#### Host-Side Control Objects

- `GpuDgramPlan`
  - batch-level plan produced by the runtime
  - owns timestamps, service, ingress source, and the device lookup version

- `GpuNamesLookupHost`
  - compact host representation built from configure transitions
  - uploadable to device

- `GpuParserCache`
  - keyed by run/config identity
  - invalidated on relevant transitions

#### Device-Side Structural Views

- `GpuDgramView`
- `GpuXtcView`
- `GpuShapesDataView`
- `GpuDescDataView`
- `GpuNamesLookupDevice`

These should be plain C++/CUDA views over bytes, not heap-owning device objects.

#### Device-Side Detector Outputs

- `GpuJungfrauSegments`
  - per-segment payload pointer and segment id

- `GpuJungfrauRawBatch`
  - contiguous `(nsegments, 512, 1024)` output buffer
  - per-event validity and damage state

The runtime should traffic in these detector outputs, not a device clone of
host `Event`.

### Runtime Architecture

Target flow:

```text
SMD/BD metadata on CPU
    ->
descriptor / batch plan
    ->
bulk bytes into GPU-visible buffer
    ->
GpuDgram parser kernel
    ->
detector extractor kernel
    ->
raw/calib/user kernels
```

The current branch already provides a useful starting point for the left half of
that pipeline:

- `RunGpu.start()` SMD-only interception
- `BigDataNode.iter_smdonly()`
- descriptor batching
- pinned-host bulk read

What should change is the right half:

- replace `TransitionalJungfrauIngressParser`
- remove host `dgram.Dgram(config=..., view=...)` from the hot path
- stop rebuilding host `Event` merely to feed `_segments(evt)`

### Configure-Time Strategy

The biggest semantic dependency is `NamesLookup`.

Recommended first implementation:

1. Parse configure transitions on CPU using existing trusted host code.
2. Build a compact parser lookup for only the needed detector fields.
3. Upload that lookup to device.
4. Let event kernels use the uploaded lookup to decode event `ShapesData`.

This yields a real GPU-side event parser while avoiding a much larger phase-1
task of fully parsing configure `Names` xtcs on device.

Later, if needed, configure parsing can also move device-side.

### Why drpGpu Is Relevant And Why It Is Not Enough

`psdaq/drpGpu` contains useful implementation patterns:

- buffer pool ownership
- queue and ring management
- host/device synchronization style
- CUDA graph orchestration
- GPU-visible DMA and staging patterns

Those patterns are relevant to runtime design and buffer lifecycle.

`drpGpu` does not appear to provide the missing abstraction here:

- generic offline `Dgram`/XTC parsing on GPU
- GPU `NamesLookup`
- GPU `DescData`
- general-purpose event-byte interpretation for psana offline xtc

So the reuse should be selective:

- reuse queue/buffer/ring ideas
- do not treat `drpGpu` as the parser itself

### Phased Implementation Plan

#### Phase 0: Preserve The Current Prototype As Groundwork

Keep and use the current branch for:

- SMD interception
- descriptor batching
- pinned-host ingress
- transitional validation path

Do not expand the host `Dgram` rebuild further than necessary.

#### Phase 1: Define The Minimal Metadata Contract

Deliverables:

- exact Jungfrau field map from real event bytes to raw output
- compact host/device `NamesLookup` schema
- explicit list of `TypeId`, field types, ranks, and algorithms required

Success criterion:

- enough information is known to parse one Jungfrau event without host detector
  objects

#### Phase 2: Build A CPU Reference Walker For Validation

Implement a small parser that:

- walks bytes structurally using the intended GPUDgram semantics
- avoids Python `Event` and `DetectorImpl`
- runs on CPU for debugging and validation

This is not the final runtime. It is the correctness oracle for device code.

Success criterion:

- CPU reference walker can extract the same Jungfrau raw data as the current
  host stack from the same event bytes

#### Phase 3: Implement Device Structural Views And Walker

Implement:

- `GpuDgramView`
- `GpuXtcView`
- explicit-stack walker
- device lookup consumption
- bounds and damage checks

Success criterion:

- a CUDA test can locate the same relevant event nodes as the CPU reference

#### Phase 4: Implement Jungfrau Device Extractor

Implement:

- mapping from relevant event nodes to Jungfrau segment payloads
- packing into contiguous raw batch buffers
- per-event status for missing segments, damage, or schema mismatch

Success criterion:

- extracted raw bytes match the CPU reference on staged data

#### Phase 5: Integrate With The Existing GPU Calibration Runtime

Replace the transitional parser/output seam with:

- device raw extraction
- direct calibration input
- optional host fallback for unsupported events

Success criterion:

- `det.raw.calib(evt)` equivalent works without host `Dgram` rebuild on the
  fast path

#### Phase 6: Upgrade Ingress

Ingress evolution:

1. pinned host bulk read + async H2D
2. better batch packing
3. GDS or other direct GPU-visible ingress where available

Success criterion:

- the parser contract remains unchanged while ingress improves underneath it

### Validation Strategy

Validation needs to happen at three levels.

#### Structural Validation

- compare parser-discovered nodes against host `XtcData` interpretation
- validate `NamesId`, field types, ranks, and extents

#### Detector Validation

- compare extracted Jungfrau raw arrays against current host extraction
- compare calibration output against the existing GPU and CPU reference paths

#### Runtime Validation

- mixed batches with missing streams
- damaged events
- transitions and config changes
- cache invalidation on detector schema changes

### Failure Model

The parser should fail explicitly, not silently.

Per-event status should distinguish:

- malformed xtc structure
- unsupported schema
- missing required detector fields
- damaged detector payload
- partial segment coverage

The runtime can then choose:

- drop event
- fall back to host path
- mark result invalid

### Initial Code Organization

Suggested new module family under `psana/psana/gpu/`:

- `gpudgram/`
- `gpudgram/__init__.py`
- `gpudgram/schema.py`
- `gpudgram/lookup.py`
- `gpudgram/reference.py`
- `gpudgram/jungfrau.py`

And later, if CUDA/C++ pieces grow enough:

- `psana/src/gpudgram/` or a dedicated compiled extension area for the
  host/device implementation

The important part is keeping GPUDgram conceptually separate from the current
bulk-reader bridge parser.

### Immediate Next Deliverables

The next concrete deliverables should be:

1. a schema note enumerating the exact Jungfrau-related `NamesLookup` entries
   and fields needed from real data
2. a CPU reference parser that walks bytes without host `Event` reconstruction
3. tests that compare:
   - current host extraction
   - transitional bulk-reader extraction
   - new reference parser extraction

### Summary

The current bulk-reader branch should be kept as exploratory groundwork for the
ingest and control plane.

The new project target should be a GPU-side XTC parser, or "GPUDgram runner",
with these core properties:

- uses real `XtcData` layout and semantics
- does not try to clone host Python event semantics on GPU
- treats configure-time `NamesLookup` as cacheable metadata
- extracts detector payload on device
- feeds GPU-native raw/calib/user pipelines

That is the cleanest long-term design for `psana2-gpu`, especially when future
ingress paths can place bytes directly in GPU-visible memory.
