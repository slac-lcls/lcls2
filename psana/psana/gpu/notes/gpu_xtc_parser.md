# GPU XTC parser

## Stage 1 boundary

Stage 1 defines a detector-independent, device-resident parser contract. It
is intentionally isolated from `DataSource`, `EventPool`, and `GPUDetector`.
Those integrations are later review stages.

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

There is no parser metadata D2H copy and no CPU event/stream regrouping in
this stage. Tests copy result rows to the CPU only after parsing to assert
correctness. The former standalone `gpudgram_driver.py`, `GPUDgramBatch`, and
Python `GpuDgramRef`/`GpuFieldView` compatibility API were removed because
they encouraged a CPU round trip that the integrated design will not use.

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

The read/EventBuilder side will eventually create one device row for every
physical dgram in a GPU batch:

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

## Later integration stages

Stage 2 will translate the existing GPUBAT1/KvikIO descriptors directly into
`dgram_records_gpu` and make parser scratch buffers EventPool-slot-owned and
budgeted. Stage 3 will switch `GPUDetector` from `_raw_data_offset` and fixed
segment stride addressing to field locators, then remove that old GPU ABI.
Stage 4 will expose general `on_gpu`, `on_gpu_view`, and `on_cpu` field access.

The run-scoped Configure allocation must outlive every batch. Batch bytes,
dgram records, ShapesData references, locators, and downstream detector work
must share the existing slot lease. A slot cannot be reused until all CUDA
consumers have completed.
