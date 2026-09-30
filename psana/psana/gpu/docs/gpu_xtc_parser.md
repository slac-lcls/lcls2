# GPU XTC parser and field access

The current parser resolves event fields using Configure metadata and device
XTC bytes. It is detector-independent; the separate Jungfrau dense adapter is
one consumer. Calibration constants are not required to parse a field.

## Configure and descriptor identity

`GpuStreamConfigTable.from_configs()` consumes `Dgram.config_names()` exports.
The host resolves detector/segment/algorithm/field selectors once into
`GpuFieldHandle` records. Stream ID and NamesId both participate in identity;
NamesId alone is not globally unique. Configure payload fields are excluded
from the public event-field interface.

Three run-scoped device tables hold stream ranges, Names and fields:

```text
stream_names_index: uint64[n_streams + 1]
names:             uint64[n_names, 7]
  [stream_id, NamesId, segment, det_key, alg_key, first_field, n_fields]
fields:            uint64[n_fields, 5]
  [field_key, type, element_size, rank, shape_index]
```

`first_field` is a table index, not a byte offset. Scalars have rank zero;
runtime Shape records provide array dimensions. `GPUBAT1` carries event identity
and per-stream bigdata descriptors; it is a transport format, separate from
these parser tables. The BD resolves file/chunk identities before I/O.

## Device parsing and locators

`GpuXtcBatchPool` owns reusable metadata allocations. Device dgram rows describe
input-buffer offset/size and event/stream identity. The walker validates bounds,
traverses nested Parent XTCs and emits bounded ShapesData references. Invalid
headers, unknown NamesIds, corrupt payloads or capacity overflow become status
codes rather than trusted payload pointers.

Configured field requests are compiled by stream. After walking XTC,
`init_locators` initializes active rows and `locate_fields` decodes fields in
parallel. The backing is `uint64[n_handles, capacity, 11]`; the capacity stride
is preserved across smaller tail batches. The locator layout is:

```text
[config_field_index, type, rank, dim0, dim1, dim2, dim3, dim4,
 device_offset, nbytes, status]
```

The decoder walks preceding fields in Configure order: scalars consume one
element; arrays consume the product of runtime dimensions times element size.
It validates bounds and detects duplicate matches. Valid rows are `STATUS_FOUND`;
missing, malformed, duplicate and Corrupted rows have distinct statuses. A
non-Corrupted damage flag alone does not reject an otherwise valid field.

Configured handles share a completion event. `configured_locations()` exposes
the combined backing without per-handle wrappers. `locate(handle)` lazily creates
a view for a configured handle without another kernel or allocation; an
unregistered handle uses the single-handle lazy decoder. A consumer stream must
wait on readiness before use. Normal task input preparation requires no locator
D2H or CPU parsing of event payloads.

## Inputs, bindings and dense preparation

`InputWindow` retains input bytes and parsed metadata independently of execution
slots. `GpuEventDgrams` maps an event's streams to owner-local dgram rows, including
executions drawing from multiple input windows. `GpuDetectorBinding` records
physical segment membership and configured field handles. The current manager
uses sorted Configure segment IDs for the canonical axis.

`DenseInputPreparer.jungfrau_raw()` validates the Jungfrau raw binding and gathers
uint16 `(N, S, 512, 1024)` input, accepting per-panel rank-two or singleton-leading
rank-three shapes. One gather per declared detector/execution writes all pixels
and presence flags. Invalid/missing segments are zero. It uses the combined
locator backing and owner routing tables, so event buffers need not share one
base pointer. Dense shape does not come from pedestal constants. This adapter
performs no calibration; user kernels consume the prepared array.

## Public event fields

```python
field = evt.gpu.detector("jungfrau").field("raw", "raw")
segments = field.on_cpu              # independent arrays by physical segment ID
panel = segments[segments.segment_ids[0]]

with field.on_gpu_view(user_stream) as views:
    # Submit work using views[physical_segment_id] on user_stream.
    pass
```

`GpuFieldData` preserves physical segment IDs and each field's shape. It does not
implicitly stack ragged arrays. Explicit single-segment selection supports
`.only()`. `.on_gpu` makes independent device copies; `.on_gpu_view(stream)`
borrows input storage and registers consumer completion on context exit. Do not
use escaped borrowed arrays after their access context ends. Input access must
occur while the event's input lease is active; cached independent CPU values can
survive retirement.

Unlike task descriptors, the first public materialization may copy the small
locator row to the CPU to determine dtype/shape. Copying a locator is not CPU
parsing of XTC bytes. Event input access and published task output access are
distinct APIs with distinct lifetimes.

## Task field descriptors

Declare `(detector, algorithm, field)` in `GpuTask.inputs`, then call
`batch.field(detector, algorithm, field)`. Its `rows` is a uint64 device table
of shape `(N, S, 8)` and `segment_ids` identifies the second axis:

```text
[raw_ptr, raw_nbytes, locator_ptr, locator_row,
 type, rank, element_size, source_present]
```

`source_present` means a host-known source dgram exists, not that the device
accepted the field. Check it before dereferencing; then check the locator's
status, type/rank and bounds. `locator_ptr` addresses the configured field's row
table, not the selected row. Different entries may refer to different owners.
These tables and device event identities upload together on first metadata
access during the callback. The producer's leases retain all referenced storage.

## Source and validation

Table definitions and validation are in [config.py](../gpudgram/config.py),
[batch.py](../gpudgram/batch.py) and [parser.py](../gpudgram/parser.py).
The CUDA decoder is in the [gpudgram directory](../gpudgram).
Bindings/views are in [gpu_input.py](../gpu_input.py), dense preparation in
[gpu_detector.py](../gpu_detector.py), and task descriptors in
[gpu_task_batch.py](../gpu_task_batch.py). [Design](design.md) explains retirement.

The GPU integration suite checks batched/lazy equivalence, damage and malformed
payloads, physical segment routing, capacity growth/tails, cross-stream readiness,
independent owners, public field access and task input preservation. See
[accepted validation](design.md#validation).
