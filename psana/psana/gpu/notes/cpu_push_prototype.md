# psana2 GPU CPU-Push Prototype

This note summarizes the current `features/psana2-gpu` CPU-push prototype.
It was updated after the GPU path was wired into normal psana event iteration.

The current branch keeps Smd0 unchanged, splits CPU and GPU detector work in
EventBuilder, sends a CPU-readable SMD batch plus a GPU-friendly descriptor
batch to BD ranks, reads full bigdata dgrams into GPU memory with KvikIO, and
uses CPU-built metadata tables to locate Jungfrau raw payloads. The GPU does
not parse Dgram/XTC headers to find raw payload offsets in the current path.

## Current CPU-Push Direction

Agreed model:

- Smd0 stays unchanged.
- EventBuilder creates coherent `cpu_smd_batch` and `gpu_smd_batch` for the
  same event range.
- A CPU BD process schedules work onto a GPU. Multiple BD ranks can share one
  GPU, so GPU assignment, queueing, memory ownership, and backpressure must be
  explicit.
- After a BD has finished CPU work for a batch and scheduled GPU work, it can
  request the next CPU/GPU SMD batch from EB without waiting for GPU result
  D2H.
- GPU results stay on GPU by default. CPU join / D2H should happen only at a
  configured interval or by explicit user request such as `.on_cpu`.
- The current MPI path exists. The active work is performance, scheduling, and
  join/D2H behavior, especially when multiple BD ranks share one GPU.

Design implications:

- `cpu_smd_batch` and `gpu_smd_batch` need shared event identity. The current
  implementation uses timestamp and batch event index.
- BD-side GPU buffers must remain valid until the associated KvikIO reads,
  kernels, and any requested D2H/join have completed.
- For multiple BDs sharing one GPU, the scheduling model must define GPU
  ownership, CUDA stream allocation, queue ownership, memory limits, and
  fairness/backpressure between BD ranks.
- Mixed CPU/GPU routing for the same large detector should be avoided in the
  performance path. The current `DetectorRouter` can bridge partial routing for
  correctness/debug, but it may copy CPU-calibrated segments back to GPU.

## Current Integrated Flow

User-facing entry point:

```python
from psana import DataSource

ds = DataSource(
    exp="mfx101210926",
    run=88,
    dir="/path/to/xtc",
    gpu_det="jungfrau",
    batch_size=1,
)
run = next(ds.runs())

for ctx in run.events():
    calib_gpu = ctx.get("calib").on_gpu
    # calib_cpu = ctx.get("calib").on_cpu  # explicit D2H
```

Current high-level path:

```text
Smd0
  unchanged SMD chunk production

EventBuilder
  builds cpu_smd_batch
  builds gpu_smd_batch in GPUBAT1 ABI
  sends both batches to BD

BD / GpuEvents
  issues KvikIO reads for gpu_smd_batch descriptors
  builds CPU Event objects from cpu_smd_batch through EventManager
  waits for GPU reads
  launches GPU detector work
  joins GPU results to CPU event context by timestamp
```

`gpu_bd_read.py` still exists as a prototype/debug driver, but it is no longer
the only path. The integrated path is:

- Serial: `RunSerial` uses `GpuEvents` when `DataSource(..., gpu_det=...)` is
  set.
- MPI: BD ranks in `RunParallel.events()` delegate to the MPI-aware GPU event
  path. This wraps `BigDataNode.start_gpu()` batches into the same `GpuEvents`
  class used by the serial path.

## EventBuilder Side

Changed files:

- `psana/psana/eventbuilder.pyx`
- `psana/psana/psexp/eventbuilder_manager.py`
- `psana/psana/dgramlite.pyx`
- `psana/psana/gpu/gpu_batch.py`

What changed:

- `eventbuilder.pyx` adds `_build_fast_batch_gpu_split()`.
- GPU splitting is enabled when EventBuilder receives `gpu_stream_ids` from
  `EventBuilderManager` / `DsParms`.
- `PS_TEST_GPU_STREAM_IDS` still exists as a legacy/direct-test fallback, but
  it is not the preferred integrated route.
- The split path builds:

```text
cpu_batch:
  normal SMD batch with GPU stream dgrams omitted
  PacketFooter entries for GPU streams set to 0

gpu_batch:
  fixed-width GPU batch ABI for GPU detector streams

step_batch:
  transition/step handling as in the non-GPU path
```

- `eventbuilder_manager.py` passes `gpu_stream_ids` into EventBuilder and adds
  `batches_with_gpu()`, which normalizes the output to:

```text
cpu_batch_dict, gpu_batch_dict, step_dict
```

- `dgramlite.pyx` adds `SmdInfoLite` / `SmdInfoLiteReader`. This is a CPU-side
  lightweight reader used by the fast EventBuilder path for:

```text
smdinfo.offsetAlg.intOffset
smdinfo.offsetAlg.intDgramSize
```

- `gpu_batch.py` defines the GPU batch ABI:

```text
[GpuBatchHeader][GpuEventTable][GpuDescTable]
```

The `GpuDescTable` gives BD/GPU code enough information to read bigdata dgrams:

```text
batch_event_index, stream_id, bd_offset, bd_size, smd_size, flags, reserved
```

## Stream Discovery

Changed files:

- `psana/psana/dgrammanager.py`
- `psana/psana/psexp/mpi_ds.py`
- `psana/psana/gpu/gpu_events.py`

Current behavior:

- `DgramManager` inspects Configure dgrams and builds detector-to-stream
  metadata:

```text
det_stream_ids_table:
  det_name -> [stream_id, ...]

det_stream_segments_table:
  det_name -> {stream_id: [segment_id, ...]}

stream_id_to_detnames:
  stream_id -> [det_name, ...]
```

- These tables are copied to config consumers, including `dsparms`.
- In MPI setup, when `gpu_det` is set, `MPIDataSource` derives
  `dsparms.gpu_stream_ids` from those Configure-derived tables on all ranks,
  including EB ranks. This is what enables EventBuilder GPU splitting in the
  integrated path.
- `GpuEvents._setup_detectors()` also uses these tables to decide which streams
  and segments belong to the GPU detector and which, if any, remain CPU-routed.

Important distinction:

```text
stream_id is an index into psana's stream/file list.
segment_id is detector segment identity from Configure metadata.
names_id links event-time ShapesData/Data to Configure Names.
```

For a normal `DataSource(..., gpu_det="jungfrau")` run, stream selection should
come from Configure metadata through `DgramManager`, not from
`PS_TEST_GPU_STREAM_IDS`.

## BigData / GPU Read Path

Changed files:

- `psana/psana/gpu/gpu_events.py`
- `psana/psana/gpu/gpu_kvikio_read.py`
- `psana/psana/psexp/node.py`
- `psana/psana/psexp/mpi_ds.py`

What changed:

- `BigDataNode.start_gpu()` yields one `(smd_batch, gpubat1_bytes)` pair per
  EB batch for GPU BD ranks.
- `RunParallel._gpu_events_mpi()` adapts those MPI batches into the interface
  expected by `GpuEvents`.
- `GpuEvents` handles both serial and MPI GPU event iteration.
- For each batch, `GpuEvents`:
  - receives `cpu_batch_dict`, `gpu_batch_dict`, and `step_dict`;
  - issues KvikIO reads from `GpuBatchView` descriptors;
  - builds CPU `Event` objects from `cpu_batch_dict` through normal
    `EventManager`;
  - waits for KvikIO futures;
  - submits GPU data and CPU events into `EventPool`;
  - yields `GpuEventContext` objects when GPU results and CPU events are ready.

`KvikioGpuReader` details:

- `issue_batch()` converts `GpuBatchView.iter_read_descs()` into a CPU
  descriptor table and copies it to `desc_table_gpu`.
- It allocates or reuses a per-slot `data_gpu` byte buffer.
- It issues one KvikIO `pread()` future per descriptor.
- `wait_batch()` calls `future.get()` for all reads and returns a
  `KvikioBatchRead`.
- On S3DF filesystems where true GDS is unavailable, KvikIO normally uses the
  CPU-fallback path:

```text
filesystem -> CPU DRAM -> GPU VRAM
```

Current assumption: the whole GPU descriptor batch for one EB batch is copied
to one GPU as `desc_table_gpu`, and KvikIO reads all described bigdata dgrams
into one `data_gpu` buffer on that GPU. If one CPU BD rank later drives more
than one GPU, this batch will need an additional partitioning step. That split
could be by event range, stream id, total read bytes, or another scheduling
policy, but each GPU should receive only the descriptor subset and `data_gpu`
allocation it owns.

## GPU Assignment / MPI

Changed files:

- `psana/psana/gpu/gpu_mpi.py`
- `psana/psana/psexp/mpi_ds.py`

Current behavior:

- GPU pinning is performed for BD ranks before CuPy import.
- Non-BD ranks in a GPU job clear `CUDA_VISIBLE_DEVICES` so Smd0/EB/SRV ranks
  do not accidentally create CUDA contexts.
- BD GPU assignment uses BD-local rank modulo `SLURM_GPUS_ON_NODE`.
- Multiple BD ranks can therefore share one physical GPU.
- `share_calib_between_gpu_peers()` supports CUDA IPC sharing for BD ranks that
  are peers on the same GPU.

Useful debug signal:

```text
[GPU-PIN] rank=... bd_rank=... bd_local_rank=... n_gpus=... gpu_id=...
```

## GPU Event Join / D2H

Changed files:

- `psana/psana/gpu/gpu_events.py`
- `psana/psana/gpu/gpu_stream.py`
- `psana/psana/gpu/context.py`
- `psana/psana/gpu/detector_router.py`

Current behavior:

- `EventPool` keeps multiple GPU batches in flight. The depth is controlled by
  `dsparms.n_gpu_streams` / `gpu_pool_depth` in debug tooling.
- `GpuEvents` joins GPU results to CPU events by timestamp when an EventPool
  slot is recycled or flushed:

```text
CPU Event timestamp -> gpu_results_by_ts[timestamp]
```

- The joined user object is `GpuEventContext`.
- `ctx.get("calib").on_gpu` returns the CuPy result already resident on the
  GPU.
- `ctx.get("calib").on_cpu` is the point where D2H happens for that result.
- Current benchmark/debug code also supports interval-based D2H testing. The
  longer-term design target is an asynchronous D2H join-size path that can copy
  groups of GPU results without forcing every event to synchronize with the
  CPU.

Known inefficient fallback:

- `DetectorRouter` supports partial detector routing, where some segments for
  a large detector are GPU-routed and remaining segments are CPU-routed.
- For partial routing, `DetectorRouter.compute_cpu_calib()` computes the CPU
  segments on the host, then `DetectorRouter.assemble_full_calib()` copies
  those CPU-calibrated segments back to the GPU with:

```python
cpu_gpu = cp.asarray(cpu_calib.astype(np.float32))
```

- This is intentionally a correctness bridge, not the desired performance path.
  For Jungfrau, one calibrated segment is:

```text
512 * 1024 * 4 bytes = 2 MiB
```

  so leaving 13 segments on CPU would require roughly `26 MiB/event` of H2D
  traffic just to assemble a full GPU result.
- Preferred production behavior is to route all streams/segments for a GPU
  detector to the GPU. CPU-only detectors should remain on CPU, but mixed
  CPU/GPU routing for the same large detector should be avoided unless it is
  explicitly requested for fallback/debug.

## Raw Metadata Handling

Changed files:

- `psana/psana/gpu/gpu_raw_offset_cache.py`
- `psana/psana/gpu/gpu_jungfrau.py`
- `psana/src/dgram.cc`

What changed:

- `gpu_raw_offset_cache.py` reads one representative bigdata dgram per GPU
  stream, creates a normal `dgram.Dgram`, calls `dgram.raw_descriptors()`, and
  caches dgram-relative raw payload offsets.
- `dgram.cc` exposes `raw_descriptors()` for lazy targeted L1Accept raw
  payload descriptors. This method reuses the Configure dgram's internal
  `NamesLookup` and `DescData`; it does not require a separate public
  Configure names table API.
- `gpu_jungfrau.py`:
  - prepares the Jungfrau raw output layout;
  - builds a CPU raw locator table from the KvikIO descriptor table and raw
    offset cache;
  - copies the locator table to GPU;
  - launches a RawKernel that copies raw payload bytes into the assembled
    output array.

The metadata split is:

```text
Selected GPU stream ids:
  stream routing indexes from Configure-derived dsparms.gpu_stream_ids

Raw offset cache:
  (stream_id, names_id_value) -> dgram-relative raw payload offset,
                                 segment identity, actual raw shape, dtype size

GPU locator table:
  read descriptor + raw offset cache -> device raw payload offset
```

Key point: the GPU kernel does not parse Dgram/XTC headers in the current path.
It consumes `data_gpu` plus a CPU-built locator table.

## Prototype / Debug Tools

Useful files:

- `psana/psana/gpu/gpu_bd_read.py`
  - standalone prototype/debug driver for split batches, KvikIO reads, raw
    offset cache bootstrapping, and split/no-split comparison.
- `psana/psana/debugtools/ds_count_events.py`
  - current lightweight CPU/GPU event-counting and timing driver; supports
    `gpu_det`, `gpu_pool_depth`, and D2H interval testing.
- `psana/psana/debugtools/net_bandwidth.py`
  - node-level NIC/read-bandwidth helper used during CPU/GPU read benchmarks.
- `psana/psana/gpu/gpu_mpi_perf_compare.py`
  - MPI GPU performance comparison driver.
- `psana/psana/gpu/gpu_performance_benchmark.py`
  - GPU performance benchmark helper.
- `psana/psana/gpu/gpu_compare.py`
  - prototype validation helpers for split/no-split and GPU/CPU raw
    comparison.
- `psana/psana/tests/test_gpu_multi_rank.py`
  - MPI GPU test/driver.
- `psana/psana/tests/test_gpu_mpi_transport.py`
  - GPU MPI transport test coverage.

Build-related files:

- `psana/meson.build`
  - updated as needed for Cython extension inputs.

## Example Tables

The examples below came from early prototype runs with explicit debug stream
selection:

```bash
PS_TEST_GPU_STREAM_IDS=0,6,7,8,9
```

They are representative examples for understanding the tables. For normal
integrated usage, prefer:

```python
DataSource(..., gpu_det="jungfrau")
```

so stream ids are discovered from Configure metadata.

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

Two GPU buffers matter in the current raw path:

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
