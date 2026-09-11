# GPU XTC parser

## Parser contract and event-loop integration

Stage 1 defines a detector-independent, device-resident parser contract. It
can still be exercised independently of `DataSource`. Stage 2 connected that
contract to the existing KvikIO and `EventPool` path in shadow mode. Stage 3
replaced the detector kernel's legacy inferred raw offset and fixed-stride ABI
with parser-produced field locators.

The ownership model is:

```text
Run / CPU
  Run.configs[stream_id]
    -> GpuStreamConfigTable       strings + stream ownership
    -> DeviceConfigTables         numeric tables uploaded once per run
    -> GpuFieldHandle             numeric det/segment/alg/field selection

Read slot / GPU
  data_gpu                        raw bytes for all selected stream dgrams
  dgram_records_gpu               event, stream, byte offset, byte size
    -> GpuEventBatch.walk_xtc
       enrich dgram records       timestamp, service, damage, parse status
       produce shape_refs_gpu     ShapesData -> Configure Names row
       produce shape_counts_gpu
    -> GpuEventBatch.locate(handle)
       produce locator rows       type, shape, device offset, byte count
    -> detector CUDA kernel       dereference locator and consume field bytes
```

There is no parser-produced metadata D2H copy. CPU descriptor metadata still
maps an event and stream to its dense dgram index, but it does not inspect XTC
or calculate payload addresses. Tests copy result rows to the CPU only after
parsing to assert correctness. The former standalone `gpudgram_driver.py`,
`GPUDgramBatch`, and Python `GpuDgramRef`/`GpuFieldView` compatibility API were
removed because they encouraged a CPU round trip that the integrated design
will not use.

## Configure tables

Configure dgrams contain `Names` records. Event dgrams contain `ShapesData`.
A ShapesData `NamesId` joins the event bytes to a Names definition in the
Configure from the same physical stream:

```text
(stream_id, NamesId)
  -> detector name, segment, algorithm
  -> ordered fields: name, type, element size, rank, shape index
```

`dgram.Dgram.config_names()` exports the C++ Configure `NamesLookup` metadata.
`GpuStreamConfigTable.from_configs(run.configs)` flattens all stream Configures
on the CPU. Human-readable strings and the reverse mappings stay there. The
three device arrays are:

```text
stream_names_index  uint64[n_streams + 1]
names               uint64[n_names, 7]
fields              uint64[n_fields, 5]
```

Names rows are sorted by NamesId within each stream and have this layout:

```text
[stream_id, NamesId, segment, det_key, alg_key, first_field, n_fields]
```

Field rows have this layout:

```text
[field_key, type, element_size, rank, shape_index]
```

`first_field` is an index into the flattened field table, not an XTC byte
offset. String keys are deterministic run-local integers created by sorting
the unique strings. Scalars have rank zero and use `UINT64_MAX` for
`shape_index`; arrays use the Shape row recorded by Configure.

The CPU resolves a user selector once:

```python
configs = GpuStreamConfigTable.from_configs(run.configs)
device_configs = configs.to_device(cp)  # once per run
handle = configs.resolve("xppcspad", 1, "raw", "arrayRaw")
```

The resulting `GpuFieldHandle` contains only:

```text
stream_id, NamesId, config_names_index, config_field_index, field_index,
type, element_size, rank, shape_index
```

`resolve_all()` preserves multiple stream owners. This is needed when one
logical detector has segments supplied by more than one stream.

## Device dgram records and XTC walk

The read/EventBuilder side creates one device row for every physical dgram in
a GPU batch:

```text
[event_index, stream_id, offset, size,
 timestamp, env, service, damage, type, status]
```

Only the first four values are required on input. `walk_xtc` validates the
dgram range and root header, then fills the remaining values in place. One
CUDA thread walks one dgram. Nested Parent XTCs are traversed with fixed-size
`cursor[]` and `end[]` stacks.

`depth` is the current Parent nesting level, not a ShapesData number. Depth
zero starts at the root dgram payload. Entering a nested Parent increments
depth; finishing its children decrements it. A ShapesData found at any depth
produces the same compact reference:

```text
[dgram_index, config_names_index, offset, extent, damage]
```

The walker uses the dgram's `stream_id` and the ShapesData `NamesId` to binary
search the correct range in `stream_names_index`. Consequently NamesId values
may repeat in different streams without ambiguity.

`max_shapes_per_dgram` bounds the per-dgram reference storage. Overflow,
unknown NamesIds, corrupt XTC, invalid streams, and invalid dgram ranges are
reported in `dgram_records_gpu[:, DGRAM_STATUS]`.

## Field location

`GpuEventBatch.locate(handle)` launches one work item per allocated ShapesData
reference slot. References for other Configure Names rows are ignored. For a
match, the kernel finds the Shapes and Data children and walks fields in
Configure order until the requested field:

```text
scalar bytes = element_size
array bytes  = element_size * product(runtime Shape dimensions)
field offset = Data payload offset + bytes of preceding fields
```

The result is `uint64[n_dgrams, 11]`:

```text
[config_field_index, type, rank, dim0, dim1, dim2, dim3, dim4,
 device_offset, nbytes, status]
```

Absent fields stay `STATUS_NOT_PRESENT`; valid matches become
`STATUS_FOUND`. The locator event records which CUDA stream produced the
table. A kernel on another stream calls `locators.wait_on(stream)` before it
uses the rows. No host synchronization is required.

## xpptut15 example

The focused integration test uses:

```text
psana/psana/tests/.tmp/xpptut15-r0014-s000-c000.xtc2
```

Its Configure is 40,608 bytes. The xppcspad segment-1 raw Names entry is:

```text
NamesId            0x10c
config_names_index 8
Names row          [0, 268, 1, 4, 3, 36, 1]
field              arrayRaw
config field index 36 (field index 0 within this Names entry)
field metadata     [arrayRaw_key, UINT16, 2 bytes, rank 2, shape index 0]
```

For the first L1Accept (post-Configure dgram index 3), walking finds six
ShapesData records. The segment-1 `arrayRaw` reference resolves to shape
`(3, 6)`, 36 bytes. The test makes a CuPy view directly over the located byte
range and compares its values with the normal CPU Dgram result. Neither the
field offset nor its byte count is supplied by the CPU parser.

Run the CPU contract tests with:

```bash
pytest -q psana/psana/tests/gpu/unit/test_gpudgram.py
```

Run the real device test on a GPU node with the xpptut file present:

```bash
pytest -q psana/psana/tests/gpu/integration/test_gpudgram_device.py
```

## Integrated ownership and execution order

`GpuEventManager` compiles `GpuStreamConfigTable` from `Run.configs` and
constructs one `GpuXtcBatchPool` for the run. The pool uploads the three
numeric Configure tables once and records a CUDA completion event for that
upload. Each EventPool stream waits on this event before its first parse.

After KvikIO completes a read, its CPU descriptor table has one dense row per
valid dgram:

```text
[event_index, stream_id, timestamp, file_offset, read_size, device_offset]
```

`build_dgram_records()` copies only `event_index`, `stream_id`, `read_size`,
and `device_offset` into a small CPU staging table. That table is uploaded on
the selected slot stream. It does not inspect XTC bytes and it does not
contain field offsets. The GPU walker produces those results.

Each `GpuXtcBatchPool` slot owns reusable high-water buffers for:

```text
dgram records
ShapesData counts and references
one locator table per registered field handle
```

Their allocations are charged to the same `_GpuBudget` as KvikIO input,
calibration, raw, and geometry buffers. Parser bytes are also included in
subbatch sizing. `EventPool.submit()` queues the parser before detector work
on the same non-blocking slot stream and retains the `GpuEventBatch` in
`_EventSlot.xtc_batch`. Two-phase retirement synchronizes the producer and
waits for consumer leases before the object is released and the slot buffers
may be overwritten.

`GpuEventManager` resolves one unambiguous Configure array handle for every
routed detector segment. `EventPool` uses CPU descriptor metadata to construct
one immutable `GpuEventDgrams` mapping per event. The same stream-indexed
mapping is passed to every detector adapter, so event/stream ownership is not
rebuilt per detector. `GPUDetector.process_batch()` reads the corresponding
device locator row, validates type, rank, payload size, and bounds, and copies
the field into canonical segment order. No XTC bytes or locator results make a
GPU-to-CPU round trip.

Stage 4A introduces the input-to-detector ownership contracts in
`gpu_input.py`:

```text
GpuEventDgrams
  dgrams[stream_id] -> GpuStreamDgramView(batch, dgram_index)

GpuDetectorBinding
  canonical segment -> Configure field handle
  canonical segment -> output row
  Configure field handle -> owning stream
```

`GpuDetectorBinding` is created once during run setup. It owns detector
membership, canonical segment routing, and all `(algorithm, field)` handles,
but deliberately has no dense-shape or calibration policy. `GpuEventDgrams`
and its parsed batch are retained by the EventPool slot and cleared together
at retirement.

Stage 4B exposes those contracts through the public event state:

```python
field = evt.gpu.detector("jungfrau").field("raw", "raw")

# Segment-preserving independent host values.
host_segments = field.on_cpu
panel_3 = host_segments[3]

# Independent device copies. The EventPool input slot may be reused.
gpu_segments = field.on_gpu

# Zero-copy views into the KvikIO input buffer.
with field.on_gpu_view(user_stream) as segment_views:
    my_kernel(segment_views[3], stream=user_stream)

# Select one segment while retaining an explicit segment mapping.
frame_count = evt.gpu.detector("jungfrau").field(
    "raw", "frame_cnt", segment=3
).on_cpu.only()
```

`GpuFieldData` is always keyed by physical segment id. It does not implicitly
stack, squeeze, pad, or reorder arbitrary field shapes. `.only()` is a
convenience for an explicitly selected single segment. Array rank, dimensions,
type, byte offset, and byte count come from the GPU locator row; copying that
small row to the CPU when the user first accesses a field does not parse or
copy the XTC payload on the CPU.

Every configured event field for the selected GPU detectors is located
eagerly so parser memory remains part of subbatch admission. A per-event
`InputSlotLease` accepts multiple CUDA completion events, allowing more than
one field or consumer stream to use the same input safely. Parsed input has no
automatic host handoff, so the EventPool preserves the yield-before-release
window even when automatic calibrated-result D2H is enabled.

## Current integration scope

The parser, field locators, and event field interface are detector-independent.
They can be used without a CPU detector implementation, calibration constants,
or a pedestal-derived shape. `GPUDetector` still
materializes one array field per segment into a dense, fixed-shape detector
tensor and either calibrates `uint16` Jungfrau raw data or passes through
pre-calibrated `float32` data. Its row and column dimensions are still derived
from pedestal calibration constants. Detectors that do not meet those adapter
requirements expose named parser fields but do not produce legacy
`det.raw`/`det.calib` GPU result keys.

The event interface intentionally leaves dense materialization as an adapter
policy. Variable/ragged fields remain separate segment arrays. Calibration is
one consumer of the shared input interface rather than a requirement of GPU
field access.

### Exclusive and mirrored stream routing

The detector selection has two intentionally simple modes:

- `gpu_det="detname"` gives the GPU path exclusive ownership of every stream
  containing that detector. EventBuilder emits GPUBAT1 offset/size descriptors
  and removes those stream proxies from the CPU batch. This retains the
  existing no-duplicate-read behavior and still requires the detector to be
  the stream's only normal detector.
- `hybrid_det="detname"` emits the same GPUBAT1 descriptors but also retains
  those stream proxies in the CPU batch. The normal EventManager and the GPU
  KvikIO reader therefore read the complete bigdata dgram independently. This
  permits a GPU-selected detector to share a stream with other detectors, at
  the explicit cost of duplicate bigdata I/O for that stream.

Both arguments accept a detector-name list. Several hybrid detectors may map
to the same stream; the stream is represented once in each batch. A detector
cannot appear in both arguments, and an exclusive stream cannot overlap a
hybrid stream, because either case would silently change `gpu_det` ownership.
No detector-size policy is inferred: callers choose whether the duplicated I/O
of `hybrid_det` is appropriate.

`smd_callback` is not supported with either GPU routing mode. Callback batching
currently produces only the CPU and step batches, so psana rejects this
combination at DataSource parameter construction rather than silently dropping
the GPUBAT1 descriptors.

## Later integration stages

Stage 3 switched `GPUDetector` from `_raw_data_offset` and fixed segment
stride addressing to field locators. Stage 4A moved stream/dgram ownership and
canonical segment binding out of the calibration adapter. Stage 4B added
general field selection, segment-preserving shape materialization, and the
input-buffer lease used by `on_gpu`, `on_gpu_view`, and `on_cpu`. Remaining
cleanup can remove the unused legacy layout helper and its tests. Exclusive
`gpu_det` routing continues to reject shared streams; `hybrid_det` is the
explicit mirrored-I/O path for those streams.

The run-scoped Configure allocation must outlive every batch. Batch bytes,
dgram records, ShapesData references, locators, and downstream detector work
must share the existing slot lease. A slot cannot be reused until all CUDA
consumers have completed.
