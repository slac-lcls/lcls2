# psana2 GPU CPU-Push Prototype

This note summarizes the `features/psana2-gpu` prototype against `master`.
The current branch keeps Smd0 unchanged, splits GPU detector work in
EventBuilder, reads full bigdata dgrams to GPU with KvikIO, and uses CPU-built
metadata tables to locate Jungfrau raw payloads. The GPU no longer parses
Dgram/XTC headers.

## EventBuilder Side

Changed files:

- `psana/psana/eventbuilder.pyx`
- `psana/psana/psexp/eventbuilder_manager.py`
- `psana/psana/dgramlite.pyx`
- `psana/psana/gpu/gpu_batch.py`

What changed:

- `eventbuilder.pyx` adds the GPU split path. When
  `PS_TEST_GPU_STREAM_IDS` is set, EventBuilder builds:
  - `cpu_batch`: normal SMD batch with GPU stream dgrams omitted and footer
    sizes set to `0`.
  - `gpu_batch`: fixed-width GPU batch ABI for GPU detector streams.
  - `step_batch`: transition/step handling as in the non-split path.
- `eventbuilder_manager.py` adds `batches_with_gpu()` and normalizes
  EventBuilder output to:

```text
cpu_batch_dict, gpu_batch_dict, step_dict
```

- `dgramlite.pyx` adds `SmdInfoLite` / `SmdInfoLiteReader`. This is a CPU-side
  lightweight reader for:

```text
smdinfo.offsetAlg.intOffset
smdinfo.offsetAlg.intDgramSize
```

- `gpu_batch.py` defines the GPU batch ABI:

```text
[GpuBatchHeader][GpuEventTable][GpuDescTable]
```

The `GpuDescTable` gives GPU BD enough information to read bigdata dgrams:

```text
batch_event_index, stream_id, bd_offset, bd_size, smd_size, flags, reserved
```

## BigData / Prototype Driver

Changed files:

- `psana/psana/gpu/gpu_bd_read.py`
- `psana/psana/gpu/gpu_kvikio_read.py`

What changed:

- `gpu_bd_read.py` is the prototype driver. It:
  - reads SMD chunks,
  - asks EventBuilder for `cpu_batch` and `gpu_batch`,
  - reads CPU dgrams through the existing `EventManager`,
  - reads GPU-stream bigdata dgrams through `KvikioGpuReader`,
  - bootstraps `GpuRawOffsetCache`,
  - builds a CPU raw locator table,
  - sends that locator table to GPU,
  - launches the Jungfrau raw assembly kernel,
  - optionally compares with no-split CPU reference output.
- `gpu_kvikio_read.py` maps `GpuBatchView` read descriptors into a KvikIO read
  plan. It reads full bigdata dgrams into a contiguous `data_gpu` byte buffer.

The production BD node is not refactored yet. This is still a standalone
prototype path.

## Metadata Handling

Changed files:

- `psana/psana/gpu/gpu_jungfrau.py`
- `psana/psana/gpu/gpu_raw_offset_cache.py`
- `psana/src/dgram.cc`

What changed:

- `PS_TEST_GPU_STREAM_IDS` currently selects which stream-list indexes are
  routed to the GPU prototype. This is debug routing metadata; later it should
  become a DataSource/EventBuilder attribute derived from GPU detector
  selection.
- `gpu_raw_offset_cache.py` reads one representative bigdata dgram per GPU
  stream, calls `dgram.raw_descriptors()`, and caches dgram-relative raw
  payload offsets, names ids, segment ids, shapes, and dtype sizes.
- `gpu_jungfrau.py`:
  - prepares the Jungfrau raw output layout,
  - builds a CPU raw locator table from the KvikIO descriptor table and raw
    offset cache,
  - launches a RawKernel that only copies raw payload bytes into the assembled
    output array.
- `dgram.cc` exposes `raw_descriptors()` for lazy targeted L1Accept raw
  payload descriptors. This method reuses the Configure dgram's internal
  `NamesLookup` and `DescData`; it does not require a separate public
  Configure table API.

The metadata split is:

```text
Selected GPU stream ids:
  stream routing indexes from PS_TEST_GPU_STREAM_IDS

Raw offset cache:
  (stream_id, names_id_value) -> dgram-relative raw payload offset,
                                 segment identity, actual raw shape, dtype size

GPU locator table:
  read descriptor + raw offset cache -> device raw payload offset
```

## Tool And Verification

Changed files:

- `psana/psana/gpu/gpu_compare.py`
- `psana/meson.build`

What changed:

- `gpu_compare.py` provides prototype validation helpers:
  - compare split vs no-split bigdata dgram digests,
  - compare GPU-assembled Jungfrau raw against CPU `det.raw.raw`.
- `meson.build` is updated for the changed Cython extension inputs.

## Example Tables

The examples below came from run 77 with:

```bash
PS_TEST_GPU_STREAM_IDS=0,6,7,8,9
```

They are representative examples, not an ABI promise for all experiments.

### GPU Batch Desc Rows

These rows come from `GpuBatchView.iter_read_descs()`. They are produced from
`GpuDescTable` and include the bigdata file offset and size for each GPU stream.

Columns:

```text
batch_event_index, timestamp, stream_id, bd_offset, bd_size, smd_size, flags
```

Example:

```text
(0, 4801813440799607242, 0, 152046, 7340630, 76, 1)
(0, 4801813440799607242, 6, 109858, 5243314, 76, 1)
(0, 4801813440799607242, 7, 140624, 6291972, 76, 1)
(0, 4801813440799607242, 8, 152046, 7340630, 76, 1)
(0, 4801813440799607242, 9, 152046, 7340630, 76, 1)
```

### KvikIO Read Descriptor Table

`gpu_kvikio_read.py` converts the GPU batch descriptors into a CPU table used
to schedule KvikIO reads. `device_offset` is the offset into the contiguous
`data_gpu` byte buffer where that bigdata dgram is read.

Columns:

```text
event_index, stream_id, timestamp, file_offset, read_size, device_offset
```

Example:

```text
(0, 0, 4801813440799607242, 152046, 7340630, 0)
(0, 6, 4801813440799607242, 109858, 5243314, 7340630)
(0, 7, 4801813440799607242, 140624, 6291972, 12583944)
(0, 8, 4801813440799607242, 152046, 7340630, 18875916)
(0, 9, 4801813440799607242, 152046, 7340630, 26216546)
```

### Raw Offset Cache

This table comes from one representative bigdata dgram per GPU stream through
`dgram.raw_descriptors()`. `raw_rel_offset` is relative to the start of that
bigdata dgram. `dim0`, `dim1`, `dim2`, and `dtype_size` are derived from the
actual event-time raw descriptor shape and field byte size. `segment` comes
from the Configure `Names` joined through the event `ShapesData` names id.

Columns:

```text
stream_id, names_id_value, segment, raw_rel_offset, raw_nbytes,
dim0, dim1, dim2, dtype_size, expected_bd_size
```

Example:

```text
(0, 10, 18, 80, 1048576, 1, 512, 1024, 2, 7340630)
(0, 11, 14, 1048738, 1048576, 1, 512, 1024, 2, 7340630)
(0, 12, 10, 2097396, 1048576, 1, 512, 1024, 2, 7340630)
(0, 13, 6, 3146054, 1048576, 1, 512, 1024, 2, 7340630)
(0, 14, 30, 4194712, 1048576, 1, 512, 1024, 2, 7340630)
(0, 15, 26, 5243370, 1048576, 1, 512, 1024, 2, 7340630)
```

### What Is Sent To GPU

Two GPU buffers matter in the current prototype:

```text
data_gpu:
  contiguous byte buffer containing full bigdata dgrams read by KvikIO

loc_gpu:
  CPU-built raw locator table copied to GPU
```

The locator table is built from:

```text
KvikIO desc table + raw offset cache
```

The key computation is:

```text
raw_device_offset = device_offset + raw_rel_offset
```

Columns:

```text
desc_index, raw_row_index, event_index, stream_id, timestamp,
names_id_value, segment, raw_device_offset, raw_nbytes,
dim0, dim1, dim2, dtype_size, status
```

Example:

```text
(0, 0, 0, 0, 4801813440799607242, 10, 18, 80, 1048576, 1, 512, 1024, 2, 1)
(0, 1, 0, 0, 4801813440799607242, 11, 14, 1048738, 1048576, 1, 512, 1024, 2, 1)
(0, 2, 0, 0, 4801813440799607242, 12, 10, 2097396, 1048576, 1, 512, 1024, 2, 1)
(0, 3, 0, 0, 4801813440799607242, 13, 6, 3146054, 1048576, 1, 512, 1024, 2, 1)
(0, 4, 0, 0, 4801813440799607242, 14, 30, 4194712, 1048576, 1, 512, 1024, 2, 1)
(0, 5, 0, 0, 4801813440799607242, 15, 26, 5243370, 1048576, 1, 512, 1024, 2, 1)
```

The RawKernel in `gpu_jungfrau.py` consumes `data_gpu` and `loc_gpu`. It does
not parse Dgram/XTC headers. It only copies:

```text
data_gpu[raw_device_offset : raw_device_offset + raw_nbytes]
```

into the assembled Jungfrau raw output array.
