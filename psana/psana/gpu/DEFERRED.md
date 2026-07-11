# Deferred Features

Features removed from the MVP branch. Each entry records what was removed,
why it was premature, and what benchmark signal would justify bringing it back.

Removal happened in two passes: the slim commit (958878d6e) removed the
speculative infrastructure from `psana/psana/gpu/`, and a follow-up cleanup
(8ea55e898) removed the hooks that infrastructure had grown into the core
event loop, C extensions, and test suite. Entries below marked *(second
pass)* were removed in the cleanup.

---

## AsyncD2HJoiner (`notes/async_d2h_join_size_prototype.md`)

**What it was:** A proposed `gpu_d2h_joiner.py` implementing batched async
D2H via pinned host memory. Per-event CuPy arrays would be D2D-copied into a
`(join_size, *shape)` staging buffer and transferred to CPU in one async
`cudaMemcpy` with a CUDA event for completion signaling.

**Why removed:** Not yet implemented — only a design note existed. The MVP
must validate correctness before optimizing D2H.

**Benchmark signal to justify:** This is the *highest-priority* deferred item.
The July 6 measurement already confirmed the need:

| Run | Rate | D2H time |
|---|---:|---:|
| GPU, no D2H | 100.3 Hz | 0 |
| GPU, D2H every event | 39.6 Hz | 246 s / 16k events |

Synchronous `.get()` is fully additive (~15 ms/event, not overlapped). Once
MVP correctness is confirmed, implement AsyncD2HJoiner. Target: D2H rate
approaches no-D2H rate (≥80 Hz) for `join_size=32`.

**2026-07-08: trigger condition MET at scale** (job 31049831, 32 BD ranks /
2 nodes / FFB): sync `.get()` under 32-rank PCIe contention costs 72 ms/event
(vs 12.5 ms serial) and is ~73% additive — aggregate drops 236 -> 170 Hz
(-28%). Not hidden by event-wait: ranks' transfers already overlap each other
on the shared link. AsyncD2HJoiner (or keeping results GPU-resident) is
justified for any workflow that returns calibrated frames to host.

---

## EventPool / StreamPool (`gpu_stream.py`)

**What it was:** `StreamPool` pre-allocated N non-blocking CUDA streams.
`EventPool` kept N GPU batches in flight, submitting calibration on stream[N%n]
without synchronizing, and returning results from n batches ago.

**Why removed:** Adds N-in-flight pipeline complexity before knowing whether
single-event GPU occupancy is the bottleneck. For Jungfrau 4M (~9.96M pixels),
one event launches ~38,900 blocks vs the A100's ~864-block saturation — the
GPU is already full on a single event.

**Benchmark signal to justify:** Measure single-event GPU occupancy via
`cupyx.profiler.benchmark`. If SM utilization < 50% for a single Jungfrau
event, EventPool is worth adding. The July 1 benchmark showed GPU kernel time
of 0.002–0.009 ms/event vs 30–40 ms wall time — the GPU is not the bottleneck.

---

## GpuBatchView / GPUBAT1 wire format (`gpu_batch.py`)

**What it was:** An 11-field binary ABI (magic `GPUBAT1\0`, version, event
table, desc table) for passing GPU work metadata from EB to BD ranks. Carried
stream IDs, BD file offsets, and sizes for each event's bigdata read.

**Why removed:** The format was designed for stability before it was validated.
The bottleneck is NVMe I/O and EB batch latency, not metadata serialization.
MVP uses the CPU event loop (`det.raw.raw(evt)`) instead.

**Benchmark signal to justify:** GPU calib throughput is demonstrated to be
NVMe/EB-bound in the CPU-path integration, AND a direct BD XTC read path
(bypassing EventManager) is shown to be faster. Both conditions must hold.

---

## KvikioGpuReader (`gpu_kvikio_read.py`)

**What it was:** A GDS read layer wrapping `kvikio.CuFile`. Detected
`compat_mode` (CPU-fallback) vs true GDS, pre-allocated per-slot `data_gpu`
buffers, and issued async batch reads keyed by the GPUBAT1 desc table.

**Why removed:** All benchmarks used the CPU-fallback path (Lustre on S3DF
has no GDS). Starting with `cp.asarray(det.raw.raw(evt))` is equivalent on
this filesystem and requires no abstraction layer.

**Benchmark signal to justify:** True GDS is available (non-Lustre filesystem
or cuFile driver loaded), AND the CPU-fallback H→D transfer is measured to be
the throughput bottleneck. Log line to watch: `GpuEvents: kvikio I/O path =
... (NVMe → GPU VRAM direct)` vs `(NVMe → CPU DRAM → GPU VRAM via cudaMemcpy)`.

---

## GPUKernelRegistry / GPUFileKernel (`gpu_kernel_registry.py`)

**What it was:** `GPUKernelRegistry` mapping `(det_type, result_name) →
GPUKernel`. Built-in kernels: `JungfrauCalibKernel` (3-mode gain bits),
`SimpleAreaCalibKernel` (single-mode). `GPUFileKernel` base class for CUDA
files. `gpu_kernel_from_file()` factory.

**Why removed:** Only one kernel (`JungfrauCalibKernel`) does real work;
`SimpleAreaCalibKernel` is three lines of arithmetic. The registry was built
for a second detector that doesn't exist yet.

**Benchmark signal to justify:** A second detector type (ePix10k, ePix100,
CSPAD) requires GPU calibration for a production use case.

---

## DetectorRouter (`detector_router.py`)

**What it was:** Tracked GPU-routed vs CPU-only detectors. Resolved
unqualified keys (`ctx.get('calib')` → `ctx.get('jungfrau.calib')`).
Implemented GPU+CPU segment combining: pre-extracted CPU calibconst rows at
BeginRun, per-event scattering of GPU and CPU panels into a complete
`(n_calibconst_segs, 512, 1024)` array.

**Why removed:** All benchmark runs used a single all-GPU Jungfrau
configuration. Split readout (some panels GPU, some CPU) was never exercised.
The key resolution is trivial without multiple detectors.

**Benchmark signal to justify:** A real run requires mixing GPU-path and
CPU-path Jungfrau panels (e.g. only streams 6, 8, 9 are on the GPU BD rank;
streams 0, 7 go CPU). Correctness requires segment scatter.

---

## GpuEvents / gpu_events_prototype.py

**What they were:** `GpuEvents` was a full replacement for psana2's `RunSerial`
event iterator, hooking into `SmdReaderManager` / `BatchIterator` / 
`EventBuilderManager`. `gpu_events_prototype.py` was a standalone version
taking SMD glob paths. Both consumed GPUBAT1 batches and maintained an
EventPool.

**Why removed:** The MVP uses psana2's standard `run.events()` loop unchanged.
`det.raw.raw(evt)` gives the raw array; `cp.asarray()` moves it to GPU. No
custom iterator is needed to validate calibration correctness.

**Benchmark signal to justify:** The CPU-path H→D transfer (`cp.asarray(raw)`)
is measured to be slower than a direct BD XTC read, AND the performance gap
matters for a real production use case.

---

## GpuEventContext / GPUResult (`context.py`)

**What it was:** `GPUResult` wrapped a CuPy array with `.on_gpu` (no cost) and
`.on_cpu` (stream sync + `.get()`). `GpuEventContext` combined GPU results with
CPU detector access and forwarded psana2 metadata. Provided `ctx.get('calib')`,
`ctx.raw('gmd').energy`, `ctx.timestamp`.

**Why removed:** The MVP API is two functions. CuPy arrays have `.get()`.
The lazy-sync behavior of `.on_cpu` is only needed when streams are in use,
which requires EventPool (also deferred).

**Benchmark signal to justify:** Users find the `fused_calib_gpu()` call +
`.get()` API insufficient for production workflows that mix GPU and CPU
detectors in the same loop.

---

## gpu_mpi.py (beyond init_gpu_rank)

**What was removed:** `create_gpu_communicators()` (bd_comm + node_comm),
`share_calib_between_gpu_peers()` (CUDA IPC handle exchange for calibration
constant sharing between BD ranks on the same GPU), `gpu_error_handler`
context manager, `log_gpu_mem`, `verify_gpu_pinning`.

**`init_gpu_rank()` was kept** (moved inline to `__init__.py`).

**Why removed:**
- `create_gpu_communicators`: no NCCL AllReduce implemented yet
- `share_calib_between_gpu_peers`: complex deterministic peer discovery;
  built before confirming multi-BD-per-GPU is the target topology or that
  400 MB × N duplicate calibration buffers are a real memory problem
- `gpu_error_handler`: needed when bd_comm + GpuEvents are in use; MVP has
  neither

**Benchmark signal to justify IPC sharing:** Multiple BD ranks on the same
GPU AND OOM errors or measurable memory pressure from duplicate calibration
buffers. Memory per BD rank for Jungfrau 4M: `peds_gpu + gmask_gpu ≈ 400 MB`.
On a 40 GB A100 with 2 BD ranks: 800 MB — not a problem. Revisit if N_BD > 4.

**2026-07-10 (Ralph iter 5): trigger MET.** The BD-per-node concurrency sweep
(TASK.md) OOMs the 40 GB A100 at **48 BD ranks** (`OutOfMemoryError: allocating
201,326,592 bytes`; 48 and 64 both abort), whereas 32 BD fits. Each rank
replicates `peds_gpu + gmask_gpu ≈ 384 MB` on the 32-seg detector plus its own
CUDA context — the duplicate-buffer memory pressure is now real and caps
single-A100 concurrency at ~32 BD. **Caveat before building:** the same sweep
showed throughput is FLAT 16→32 BD (per-rank serialization, not concurrency,
is the wall), so IPC sharing would buy memory headroom / denser packing, NOT
throughput. Justified for topologies needing >32 BD/GPU; not a throughput lever.

---

## gpudgramlite.py / cuda/gpudgramlite.cuh

**What it was:** A CuPy RawKernel that parsed Dgram/XTC2 header fields
(timestamp, service, extent, payload_size) directly on device, returning a
`(n_desc, INFO_NCOLS)` uint64 table. Used the `GpuDgramLite` / `GpuXtcLite`
structs in `gpudgramlite.cuh`.

**Why removed:** Not used in the current hot path. The desc table (GPUBAT1)
already carries offsets/sizes from CPU-side SMD parsing. No GPU-side routing
or filtering is needed in the MVP.

**Benchmark signal to justify:** CPU-side event routing (EventManager) is
measured to be the bottleneck in the BD pipeline, AND GPU-side filtering
(e.g. skipping non-L1Accept dgrams) would recover measurable throughput.

---

## GPU BD rank pipeline (`gpu_bd_read.py`)

**What it was:** A 516-line standalone pipeline that read SMD files directly,
built GPUBAT1 batches, issued KvikIO reads, ran GPUDetector calibration, and
included correctness validation tools (`compare_jungfrau_raw`,
`compare_split_event`, `digest_bytes`).

**Why removed:** Parallel implementation to `GpuEvents` (also removed).
Correctness validation belongs in a simple test script using the standard
psana2 event loop, not a bespoke standalone pipeline.

**Benchmark signal to justify:** The standard-loop integration path
(`det.raw.raw(evt)` → `cp.asarray`) is shown to be slower than direct SMD
read for a concrete reason (e.g. CPU calibration overhead in EventManager).

---

## DataSource(gpu_det=...) event-loop integration *(second pass)*

**What it was:** The psexp-side surface that wired the removed GPU iterator
into psana2: `gpu_det` / `n_gpu_streams` / `gpu_stream_ids` on `DsParms` and
`DataSource(...)`, `GpuEvents` dispatch in `run.py` (RunSerial) and
`mpi_ds.py`, GPUBAT1 byte packing alongside SMD batches in `node.py`,
`batches_with_gpu()` / 3-tuple `eb.build()` in `eventbuilder_manager.py` and
`smdreader_manager.py`, GPUBAT1 stream splitting in `eventbuilder.pyx`,
multi-stream routing tables (`det_stream_ids_table`,
`det_stream_segments_table`) in `dgrammanager.py`, the GPU SMD-file path in
`singlefile_ds.py`, `_setup_gpu_geometry()` / shared-calib logging in
`mpi_ds.py`, and `--gpu_det` driver options in
`debugtools/ds_count_events.py`.

**Why removed:** It existed only to serve the removed `GpuEvents`/GPUBAT1
machinery, and after the first pass it referenced deleted modules —
`mpi_ds.py` broke every MPI DataSource run and `eventbuilder.pyx` broke the
eventbuilder extension at import time. The MVP deliberately has no DataSource
integration: the caller uses the standard event loop and two functions.

**Benchmark signal to justify:** Same as GpuEvents — a validated MVP shows
GPU calib is production-worthy AND the explicit two-function API is shown to
be a real adoption barrier for users. Any future integration should be
type-stable (never silently returning CuPy where NumPy is expected).

---

## Dgram.raw_descriptors() raw payload locator *(second pass)*

**What it was:** ~270 lines in `psana/src/dgram.cc` adding a
`raw_descriptors()` method to Dgram: per-segment file offset/size descriptors
for L1Accept raw payloads, used to build GPUBAT1 desc tables for direct BD
reads (KvikIO/GDS path).

**Why removed:** Only consumer was the removed BD pipeline; nothing imported
it after the first pass. The MVP gets raw data via `det.raw.raw(evt)`.

**Benchmark signal to justify:** Same as KvikioGpuReader — a direct
BD-offset read path is demonstrated to beat the standard event loop.

---

## SmdInfoExtractIter (`dgramlite.pyx`) *(second pass)*

**What it was:** An embedded C++ `XtcIterator` in `dgramlite.pyx` (plus the
`xtc_shlib` meson link it required) extracting bigdata offset/size fields
from SMD dgrams — the CPU-side producer for GPUBAT1 desc tables.

**Why removed:** Unused after the first pass; only the removed BD pipeline
consumed its output.

**Benchmark signal to justify:** Same as GpuBatchView/GPUBAT1.

---

## Benchmark and test scripts

**Removed (first pass):** `gpu_mpi_benchmark.py`, `gpu_performance_benchmark.py`,
`gpu_mpi_perf_compare.py`, `bench_wall.py`, `gpu_event_loop_test.py`,
`gpu_batch_size_test.py`, `test_ipc_sharing.py`, `gpu_jungfrau_calib_example.py`,
`scripts/` directory.

**Removed (second pass):** the seven `psana/psana/tests/test_gpu_*.py` files
(~2,700 lines: calib, event_loop, kernel_registry, mpi_transport, multi_rank,
performance, transition_handling) — they exercised the removed modules, and
two failed pytest collection by importing them at module level.

**Replaced by:** `test_jungfrau_calib.py` (correctness) and `bench_calib.py`
(performance). The correctness test passed bit-exact (max_diff 0.0 vs
`det.raw.calib()`, 20 events, mfx101210926 r387, A100) on 2026-07-07 after
the second-pass cleanup and a clean meson rebuild.

---

## Design notes (`notes/` directory)

The following notes were removed from the branch but their content is preserved
here for reference:

- `async_d2h_join_size_prototype.md` — design for AsyncD2HJoiner (see entry above)
- `cpu_push_prototype.md` — CPU-push model for GPU pipeline
- `d2h_interval_bandwidth_results.md` — July 6 D2H bandwidth measurements (key result)
- `event_joiner_implementation.md` — general CPU/GPU result combining design
- `join_operation.md` — join semantics for GPU/CPU mode
- `performance_report_gpu.md` — July 1 batch_size × pool_depth sweep results

The July 6 and July 1 benchmark results are the most important prior work.
Key numbers to carry forward:

| Configuration | Rate |
|---|---:|
| CPU baseline | 38 Hz |
| GPU, no D2H | 100 Hz |
| GPU, D2H every event | 40 Hz |
| GPU, bs=50 pd=4 (old EventPool path) | 60 Hz |

GPU kernel time: 0.002–0.009 ms/event. Wall time dominated by NVMe I/O + EB
batch latency (~30–40 ms/event). D2H per Jungfrau 4M event: ~15 ms (synchronous).
