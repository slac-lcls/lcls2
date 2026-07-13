# psana2 GPU CPU-Push Model

This note describes the integrated CPU-push implementation on
`features/psana2-gpu`. It is a current-state summary, not a record of the
removed standalone prototypes.

The GPU path currently supports uncompressed Jungfrau data. Smd0 remains
unchanged, EventBuilder splits CPU and GPU work for the same event range, and a
CPU BigData (BD) process schedules KvikIO reads and calibration on its assigned
GPU. Results remain on the GPU unless the user explicitly requests a CPU copy.

## User Interface

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

for ctx in run.events():
    calib_gpu = ctx.get("calib").on_gpu
    # calib_cpu = ctx.get("calib").on_cpu  # explicit synchronous D2H
    # energy = ctx.raw("gmd").energy       # normal CPU detector access
```

`ctx.get("calib").on_gpu` is a CuPy view into a reusable EventPool slot. It
must be consumed before that slot is recycled. `.on_cpu` returns an independent
NumPy copy.

## Integrated Flow

```text
Smd0
  produces normal SMD chunks

EventBuilder
  aligns events and transitions
  builds a CPU-readable SMD batch
  builds a GPUBAT1 descriptor batch for GPU-routed streams
  sends CPU, GPU, and step batches to BD

BD / GpuEvents
  issues KvikIO reads for the GPUBAT1 descriptors
  builds normal CPU Event objects through EventManager
  waits for the KvikIO futures
  launches Jungfrau raw gathering and calibration
  joins CPU events and GPU results by timestamp
  yields GpuEventContext objects
```

The same `GpuEvents` implementation is used by both execution modes:

- `RunSerial` uses it directly when `DataSource(..., gpu_det=...)` is set.
- MPI BD ranks use `RunParallel._gpu_events_mpi()` and
  `BigDataNode.start_gpu()` as the batch source.

Smd0, EventBuilder, and service ranks do not need CUDA contexts.

## EventBuilder Split and Batch ABI

`DgramManager` derives detector routing metadata from Configure:

```text
det_stream_ids_table:
  det_name -> [stream_id, ...]

det_stream_segments_table:
  det_name -> {stream_id: [segment_id, ...]}
```

Here `stream_id` indexes psana's stream/file list, while `segment_id` is the
physical detector segment from Configure. An event ShapesData `names_id` links
the event payload back to its Configure Names record.

For normal integrated use, `gpu_det="jungfrau"` selects streams through these
Configure-derived tables. `PS_TEST_GPU_STREAM_IDS` remains only as a legacy
direct-test override.

EventBuilder's GPU split produces:

```text
cpu_batch:
  normal SMD event data with GPU stream dgrams omitted

gpu_batch:
  [GpuBatchHeader][GpuEventTable][GpuDescTable]

step_batch:
  transitions required by the BD event loop
```

Each GPU descriptor contains the event identity, stream ID, bigdata file
offset, and dgram size. Timestamp plus batch event index keep the CPU and GPU
batches coherent.

## Segment Identity and Ordering

Configure identifies which physical segments belong to a stream, but its
dictionary order is not assumed to match the child-XTC order in L1Accept. The
fixed-stride GPU gather preserves L1Accept order, so calibration constants must
use that same order.

During GPU detector setup, on CPU:

1. `build_stream_seg_map()` opens each detector-bearing bigdata stream.
2. It scans to the first L1Accept containing Jungfrau data.
3. Psana joins each ShapesData `names_id` to Configure and exposes physical
   segment IDs in XTC traversal order.
4. The resulting ordered segment list is checked against Configure membership.
5. That order is treated as fixed for the run.

For example:

```text
L1Accept raw order:          [17, 13, 9, 5]
per-stream pedestal order:   [17, 13, 9, 5]
per-stream gain/mask order:  [17, 13, 9, 5]
calibrated output order:     [17, 13, 9, 5]
geometry index order:        [17, 13, 9, 5]
```

The output is correctly associated with physical segment IDs but is not
numerically sorted by segment ID.

## BigData Read, Raw Gather, and Calibration

`KvikioGpuReader` creates one reusable `data_gpu` byte buffer per EventPool
slot. All selected bigdata dgrams for the GPU batch are packed into that buffer
at descriptor-specific `device_offset` values. It issues one KvikIO `pread()`
future per descriptor.

The GPU does not parse the XTC tree. On the first batch, `GPUDetector` copies a
small sample to CPU and detects the fixed raw payload offset and segment stride.
For each stream dgram:

- A single-segment raw array is a view into `data_gpu`.
- For multiple segments, `as_strided()` creates a no-copy view that skips XTC
  header gaps, and `buf[:] = src_view` performs one device-to-device compaction
  into a reusable contiguous raw buffer.
- This compaction preserves L1Accept panel order; it does not permute panels by
  segment ID.
- The stream's pedestal and gain-mask arrays are copied into that order once
  and cached.
- `fused_calib_gpu()` calibrates one stream dgram and writes directly into its
  slice of the preallocated batch output buffer.

The Jungfrau kernel computes, per pixel:

```text
(raw ADC - pedestal - pixel_offset) * (1 / pixel_gain) * mask
```

with the pedestal and gain factor selected from the raw pixel's Jungfrau gain
bits. Common-mode correction is not implemented on the GPU path.

Geometry indices are selected in the same stream/segment order. `GPUDetector`
can assemble an image when requested, although the normal EventPool path
currently produces calibrated panel stacks.

## Buffer Ownership and EventPool Lifetime

The active GPU buffers are:

| Buffer | Owner and lifetime |
|---|---|
| KvikIO `data_gpu` | One per EventPool slot; grows to the largest batch read |
| Contiguous raw gather | Cached per slot and stream segment layout |
| Calibration output | One whole-batch buffer per slot; events receive slices |
| Full pedestal/gain-mask | One detector-wide set, normally run lifetime |
| Reordered stream constants | Cached per unique L1Accept segment order |
| Geometry arrays | Detector/run lifetime when geometry is enabled |

`EventPool` owns one non-blocking CUDA stream per slot. Before a slot is reused,
it synchronizes the old stream and yields every result from the retired batch.
Only after those contexts have been consumed does it submit new calibration
work into that slot.

This bounds the number of in-flight batches, but not GPU memory by bytes. The
buffers generally grow to their largest observed size and do not shrink during
normal iteration.

## Transitions

Transition handling preserves calibration and buffer lifetime:

- `BeginStep` drains pending GPU work before calibration constants change.
- Updated constants are written in place; reordered stream caches are cleared
  and rebuilt on demand.
- `EndRun` drains remaining results exactly once and terminates the GPU loop.
- `Enable`, `Disable`, and `EndStep` do not force an unnecessary drain.
- KvikIO file handles are closed when iteration exits.

## MPI GPU Assignment

GPU pinning is performed for MPI BD ranks before CuPy import. A BD rank selects
a physical GPU from its BD-local placement and `SLURM_GPUS_ON_NODE`; non-BD
ranks clear GPU visibility.

Multiple BD ranks may share one physical GPU. BD peers on the same GPU can
share read-only detector calibration buffers through CUDA IPC, avoiding one
full constant allocation per follower rank. Event read, raw gather, and output
slot buffers remain owned by each BD process.

Mixed CPU/GPU routing of one large detector remains available through
`DetectorRouter` as a correctness/debug bridge. It can copy CPU-calibrated
segments back to GPU and is not the preferred performance path.

## D2H and Backpressure Status

Current behavior:

- GPU results remain on device by default.
- `.on_cpu` performs a synchronous copy for one result.
- Debug tooling can sample D2H at a configured event interval.
- EventPool depth bounds batches within one BD process.

Not yet implemented:

- Asynchronous grouped D2H/join with pinned host slots.
- A completion token that prevents GPU slot reuse while asynchronous D2H is
  still reading the calibrated output.
- A byte-based GPU memory budget or high/low watermarks.
- Fair cross-BD backpressure when multiple BD processes share one GPU.

Any asynchronous D2H implementation must extend slot ownership through D2H
completion. Synchronizing only the calibration stream is not sufficient if a
separate D2H stream still reads the slot.

## Current Limitations and Next Optimizations

- Integrated calibration supports Jungfrau only.
- Raw extraction assumes uncompressed, fixed-stride segment payloads.
- L1Accept segment order is discovered once per stream and assumed stable for
  the configured run.
- Common-mode correction is not implemented.
- True GDS depends on filesystem and driver support; otherwise KvikIO uses
  `filesystem -> CPU DRAM -> GPU VRAM` compatibility mode.
- One BD process currently drives one assigned GPU.

A useful future kernel optimization is to read the strided raw payload and
apply segment mapping and calibration in one launch. That would eliminate the
contiguous raw gather buffer and its device-to-device copy. It may also allow
the kernel to use canonical detector constants plus a small segment map instead
of retaining reordered per-stream constant copies.

## Validation and Tools

The automated suite intentionally has two layers:

```bash
python -m pytest -q psana/psana/tests/gpu/unit
python -m pytest -q -m slow \
  psana/psana/tests/gpu/integration/test_pixel_exact.py
```

The pixel-exact acceptance test uses public Lysozyme Jungfrau data,
`mfx100848724` run 51. It compares CPU and GPU float32 calibration by timestamp
for single-event and batched slot-reuse modes.

Useful performance/debug entry points are:

- `psana/psana/debugtools/ds_count_events.py`
- `psana/psana/gpu/gpu_mpi_perf_compare.py`
- `psana/psana/gpu/gpu_performance_benchmark.py`
- `psana/psana/gpu/scripts/gpu_multi_rank_smoke.py`

## Active Source Map

- `eventbuilder.pyx`, `eventbuilder_manager.py`: CPU/GPU batch split.
- `gpu_batch.py`: GPUBAT1 ABI and descriptor views.
- `gpu_events.py`: serial/MPI batch orchestration and timestamp join.
- `gpu_kvikio_read.py`: per-slot bigdata reads and descriptor table.
- `gpu_calib.py`, `cuda/fused_calib.cuh`: ordering, raw gathering, constants,
  geometry, and Jungfrau calibration.
- `gpu_stream.py`: reusable CUDA streams and slot lifetime.
- `context.py`: user-facing GPU result/context wrappers.
- `gpu_mpi.py`: GPU pinning, CUDA IPC calibration sharing, and fatal-error
  handling.

The removed standalone raw-locator path (`gpu_raw_offset_cache.py`,
`gpu_jungfrau.py`, and its `loc_gpu` table) is not part of the integrated
implementation.
