# GPU Calibration MVP — Planning Document

## Project Overview

Minimum viable GPU calibration for the Jungfrau detector in psana2.

The goal is narrow and deliberate: demonstrate that GPU-accelerated Jungfrau
calibration (a) produces pixel-level correct results relative to the CPU path
and (b) is measurably faster. Everything else is deferred until a benchmark
shows it is necessary.

Integration point: the existing psana2 event loop is used unchanged. Per event,
the caller transfers raw uint16 pixels to device, runs the fused calibration
kernel, and gets a CuPy float32 array back. No new event iterator, no batch
wire format, no stream pooling, no MPI utilities beyond optional GPU pinning.

**Out of scope for MVP:**
- Direct XTC read from BD rank (KvikIO / GDS)
- Batched kernel launches (EventPool, N-in-flight)
- Multi-detector routing (DetectorRouter)
- CUDA IPC calibration sharing between ranks
- GPU-side dgram header parsing
- Async batched D2H (AsyncD2HJoiner)
- Any detector other than Jungfrau

See `DEFERRED.md` in this directory for the full removed-feature catalogue with
benchmarking criteria for when each becomes justified.

---

## Architecture

### Core components (MVP)

```
psana/psana/gpu/
├── __init__.py          public surface: prep_calib_constants, fused_calib_gpu
├── gpu_calib.py         two functions + the kernel dispatch
└── cuda/
    └── fused_calib.cuh  jungfrau_calib_pixel() device inline
```

**`prep_calib_constants(det) -> (peds_gpu, gmask_gpu)`**
Called once per run (or per BeginStep if calibration constants change).
Reads `det.calibconst['pedestals']` and `det.calibconst['pixel_gain']` and
`det.calibconst['pixel_mask']`, computes `gmask = (1/gain) * mask` on CPU,
then H→D transfers both flat float32 arrays. Layout is mode-major:
`[mode0_pixels..., mode1_pixels..., mode2_pixels...]`, length `3 * n_pixels`.

**`fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu) -> calib_gpu`**
Per-event. `raw_gpu` is a CuPy uint16 array already on device. Dispatches the
`jungfrau_calib_kernel` RawKernel (compiled JIT via NVRTC on first call,
cached for the process lifetime). Returns a float32 CuPy array of the same
shape. The caller decides when to call `.get()` for D→H.

**`cuda/fused_calib.cuh`**
Contains the `jungfrau_calib_pixel()` device inline implementing 3-gain-mode
calibration:
```
gain_bits = raw >> 14        # 00→g0, 01→g1, 11→g2, 10→bad(→0.0)
calib[i]  = (raw & 0x3fff − peds[mode*npixels + i]) * gmask[mode*npixels + i]
```

### What is intentionally absent

- `GPUDetector` class (slot buffers, beginstep, stream-segment map, geometry)
- `StreamPool` / `EventPool` (CUDA stream management)
- `GpuBatchView` / GPUBAT1 wire format
- `KvikioGpuReader` (direct BD XTC read)
- `GPUKernelRegistry` / `GPUFileKernel` (multi-detector plugin system)
- `DetectorRouter` (GPU+CPU segment scatter)
- `gpudgramlite.py` / `cuda/gpudgramlite.cuh` (GPU-side XTC header parsing)
- `GpuEvents` / `gpu_events_prototype.py` (custom event iterators)
- `gpu_mpi.py` beyond a 20-line `init_gpu_rank()` helper
- All benchmark scripts (replaced by a single minimal bench script)
- All design notes (preserved in `DEFERRED.md`)

### Data model

No database. Calibration constants live in psana2's `det.calibconst` dict,
loaded from the calibration store at run open.

---

## API

No HTTP endpoints. The public Python API surface for the MVP is two functions:

```python
from psana.gpu import prep_calib_constants, fused_calib_gpu

# Once per run:
peds_gpu, gmask_gpu = prep_calib_constants(det)

# Per event (inside the psana2 event loop):
import cupy as cp
raw = det.raw.raw(evt)        # numpy uint16, shape (n_segs, 512, 1024)
if raw is not None:
    raw_gpu  = cp.asarray(raw)
    calib_gpu = fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu)
    # calib_gpu: CuPy float32, same shape
    # stay on GPU for further computation, or:
    calib_cpu = calib_gpu.get()   # D→H when needed
```

Optional GPU pinning for MPI runs (must be called before `import cupy`):

```python
from psana.gpu import init_gpu_rank
gpu_id = init_gpu_rank()   # reads SLURM_LOCALID, sets CUDA_VISIBLE_DEVICES
import cupy as cp
```

---

## Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| GPU compute | CuPy (CUDA via NVRTC JIT) |
| CUDA kernel | Inline C++ via `cp.RawKernel`, kernel source in `.cuh` |
| Data framework | psana2 (lcls2) — DataSource, Detector, events() |
| MPI (optional) | mpi4py |
| Hardware target | NVIDIA A100 40 GB (S3DF sdfampere nodes) |
| Filesystem | Lustre (S3DF) — no GDS available |

---

## Project Structure

```
psana/psana/gpu/
├── PLANNING.md              this file
├── TASK.md                  task tracking
├── DEFERRED.md              removed features + benchmarking criteria
├── __init__.py              exports: prep_calib_constants, fused_calib_gpu, init_gpu_rank
├── gpu_calib.py             ~150 lines: prep_calib_constants + fused_calib_gpu + kernel dispatch
└── cuda/
    └── fused_calib.cuh      jungfrau_calib_pixel() device inline (~60 lines, unchanged)
```

Files to be deleted from `features/psana2-gpu` when creating the MVP branch:

```
gpu_batch.py                 GPUBAT1 wire format
gpu_stream.py                StreamPool / EventPool
detector_router.py           DetectorRouter
gpu_events.py                GpuEvents integrated iterator
gpu_events_prototype.py      standalone prototype iterator
gpu_kvikio_read.py           KvikIO / GDS read layer
gpu_kernel_registry.py       GPUKernelRegistry + built-in kernels
gpudgramlite.py              GPU-side dgram parsing
gpu_bd_read.py               standalone BD rank pipeline
gpu_compare.py               correctness validation tools
gpu_jungfrau.py              Jungfrau raw locator / layout tables
gpu_raw_offset_cache.py      per-stream BD offset cache
gpu_mpi_benchmark.py         MPI benchmark harness
gpu_performance_benchmark.py performance benchmark
gpu_mpi_perf_compare.py      MPI perf comparison
bench_wall.py                wall-clock timing utilities
test_ipc_sharing.py          IPC sharing test
gpu_batch_size_test.py       batch size test
gpu_event_loop_test.py       event loop test
gpu_jungfrau_calib_example.py  example script
context.py                   GPUResult / GpuEventContext
cuda/gpudgramlite.cuh        GPU XTC header structs
notes/                       all design notes (content preserved in DEFERRED.md)
scripts/                     all run/submit scripts
```

Functions to be deleted from `gpu_calib.py`:

```
optimal_kernel_batch_size()    no batching in MVP
_detect_dgram_layout()         BD-path XTC parsing, not needed
EventContext dataclass         BD-path event context
GPUDetector class (~650 lines) slot buffers, beginstep, geometry, stream routing
build_stream_seg_map()         BD-path stream mapping
_compute_calib_constants_cpu() used only by GPUDetector.beginstep()
```

---

## Testing Strategy

### Correctness test (required before any performance work)

Compare GPU calib output against psana2's reference CPU path pixel-by-pixel:

```python
# psana/psana/gpu/test_jungfrau_calib.py
import numpy as np, cupy as cp
from psana import DataSource
from psana.gpu import prep_calib_constants, fused_calib_gpu

ds  = DataSource(exp=EXP, run=RUN)
run = next(ds.runs())
det = run.Detector('jungfrau')
peds_gpu, gmask_gpu = prep_calib_constants(det)

mismatches = 0
for i, evt in enumerate(run.events()):
    if i >= 50: break
    raw = det.raw.raw(evt)
    if raw is None: continue
    cpu_ref  = det.raw.calib(evt).astype(np.float32)
    gpu_out  = fused_calib_gpu(cp.asarray(raw), peds_gpu, gmask_gpu).get()
    if not np.allclose(cpu_ref, gpu_out, atol=1e-3, rtol=0):
        mismatches += 1
        print(f"evt {i}: max_diff={np.abs(cpu_ref - gpu_out).max():.6f}")

print(f"Checked {i+1} events, {mismatches} mismatches")
```

Pass criterion: zero mismatches at `atol=1e-3` for 50 events on the reference
run (mfx100852324-r0077 or mfx101210926-r0387).

### Performance benchmark (after correctness is confirmed)

Measure events/sec for GPU path vs CPU path, H→D time, kernel time, D→H time:

```
python psana/psana/gpu/bench_calib.py \
    -e mfx101210926 -r 387 -n 500 --compare-cpu
```

Record: events/sec GPU (no D2H), events/sec GPU (with D2H), events/sec CPU,
H→D ms/event, kernel ms/event, D→H ms/event.

### No unit tests for MVP

The correctness test IS the test. Do not add pytest fixtures or mocking before
the kernel is validated against real data.

---

## Development Commands

```bash
# Environment (S3DF sdfampere node):
source /sdf/scratch/users/a/ajshack/lcls2/setup_env.sh

# Install psana in editable mode:
cd /sdf/scratch/users/a/ajshack/lcls2/psana
pip install -e .

# Correctness test (1 BD rank — no MPI needed):
python psana/psana/gpu/test_jungfrau_calib.py

# Performance benchmark (MPI: 1 smd0 + 1 EB + 1 BD):
mpirun -n 3 python psana/psana/gpu/bench_calib.py \
    -e mfx101210926 -r 387 -n 500 --compare-cpu
```

---

## Multi-node launch: `PS_EB_NODE_LOCAL=1` (recommended for ≥2 nodes)

For any GPU calibration run spanning **more than one compute node**, set
`PS_EB_NODE_LOCAL=1`. It places **one EventBuilder rank per physical node**
(each serving only that node's BD ranks) instead of the default single global
EB (on N0) that serves every BD rank across all nodes.

This is an **opt-in flag, default off.** Unset (or `0`) leaves the standard
psana MPI event loop completely unchanged — the placement change is gated
entirely behind this env var (`_ensure_local_eb_nodes` in `datasource.py`;
`colocate_non_marching` in `psexp/node.py`). It is safe for non-GPU psana runs
too but is only *measured* for the GPU calib path; leave it off for the default
path unless you have a reason.

**Why — measured (loop iterations 18 & 20, r47 on FFB, A100):** the multi-node
throughput plateau is caused by a single EB rank serializing as it serves more
BD ranks (`eb_wait` climbs 3.5 → 70 ms as BDs go 32 → 96). One EB per node holds
`eb_wait` down and the win **compounds with node count**:

| nodes | default agg Hz | node-local agg Hz | gain | node-local per-node eff |
|------:|---------------:|------------------:|-----:|------------------------:|
| 2     | 301.0          | 328.4             | +9.1%  | 90% |
| 3     | 338.3          | 444.0             | +31.3% | 77% |

`bd_read` is flat across placements (storage scales per-node); the entire
differential is EB serving latency. See `TASK.md` (2026-07-13 entry) and
`ralph/PROGRESS.md` iters 18/20 for full attribution.

```bash
# Multi-node GPU calib launch (sbatch): one EB per node, colocated with BDs.
# Template: bench_mpi_sweep/eb_node_local_3n_v2.sbatch
export PS_EB_NODE_LOCAL=1        # overrides PS_EB_NODES -> node count automatically
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
mpirun --bind-to none -x PS_EB_NODE_LOCAL \
    python psana/psana/gpu/bench_calib.py \
    -e mfx101572426 -r 47 -n 500 --warmup 10 --dir $FFB
```

Do **not** set `PS_EB_NODES` by hand alongside `PS_EB_NODE_LOCAL` — the flag
derives the EB count from the node count and overrides `PS_EB_NODES`.

Residual per-node loss at 3 nodes (77% eff, `eb_wait` still 26 ms) points at
*intra-node* EB serialization as the next lever — now measured and controlled by
`PS_EB_PER_NODE` (below).

### `PS_EB_PER_NODE`: multiple EBs per node — the knee grows with node count

`PS_EB_NODE_LOCAL=1` places one EB per node; that one EB still serializes as it
serves ~31 BDs on its node. `PS_EB_PER_NODE=k` (requires `PS_EB_NODE_LOCAL=1`)
sub-splits each node into **k EventBuilders**, each serving ~`32/k` BD ranks, so
`eb_wait` drops further — at the cost of `k` BD-reader slots per node. The
optimum `k` is a **crossover, and it grows with node count**:

| nodes | knee (best k) | best-k agg Hz | k=1 agg Hz | best-k gain vs k=1 |
|------:|:-------------:|--------------:|-----------:|-------------------:|
| 2     | **2**         | 502.9         | 410.6      | +23%  (k=3/4 regress −18%) |
| 3     | **3**         | 802.5 / 791.5 | 622.9/419.5| +24–32% (k=2 between; k=4/5 lose) |

*(2-node: iters 22/23, jobs 31586774/31587292. 3-node knee pinned across the full
{1,2,3,4,5} sweep in three windows — iter 24 forward job 31588126 + reversed
confirm 31589010 gave `3 > 2 > 1` (`epn3n*`/`epn3nrev*`); iter 25 job 31590259
swept the upper end and found k=3 (791.5 Hz) beats k=4 (658.9) and k=5 (757.1)
even as the cold first phase (`epn3nk_*` logs). The k=3 anchor matches across
windows (802.5 vs 791.5, ~1.4%), so the peak is real, not window drift. Past the
knee, over-provisioned EBs starve BD-reader ranks into idleness — at 3n k=5 only
67 of ~83 BD ranks reported work.)*

**Mechanism:** `eb_wait` falls monotonically with `k` at every node count (more
EBs, fewer BDs each). But the single `smd0` feeds *all* node-local EBs, so with
more nodes there are more EBs contending for smd batches and `eb_wait` at low `k`
is larger — pushing the crossover (where the marginal EB's lost BD slot + `bd_read`
inflation outweighs the `eb_wait` it saves) to higher `k`. Empirically the knee
= node count (2 nodes → k=2, 3 nodes → k=3, both pinned by full sweeps).
**Guidance: scale `PS_EB_PER_NODE` up with node count; set `k = node_count`.** Do
not set `k` above the knee — past it, aggregate regresses (measured −18% at 2
nodes with k=3/4; at 3 nodes k=4/5 both lose to k=3, and k=5 starves ~16 of ~83
BD ranks into idleness).

```bash
# Multi-node GPU calib, tuned EB fan-out. Template: bench_mpi_sweep/eb_per_node_3n.sbatch
export PS_EB_NODE_LOCAL=1
export PS_EB_PER_NODE=3           # ~= node count; overrides PS_EB_NODES -> nodes*k
```

The 3-node knee is now pinned at exactly 3 (iter 25). The remaining
characterization is the 4-node point — does `k = node_count` predict k=4 wins at
4 nodes, turning the two-point law into three? See `ralph/PROGRESS.md` iters 24–25.

---

## Environment Setup

- Compute: `sdfampere` nodes (NVIDIA A100 40 GB, 108 SMs)
- Login: `sdfiana026` → `ssh sdfampere001`
- Filesystem: Lustre at `/sdf/data/lcls/...` (no GDS — all I/O via CPU DRAM)
- Python env: managed by `setup_env.sh` in lcls2 root
- CuPy: available in the psana conda env
- `CUDA_VISIBLE_DEVICES` must be set before `import cupy` in MPI runs;
  use `init_gpu_rank()` or set manually: `export CUDA_VISIBLE_DEVICES=0`

---

## Development Guidelines

1. **Measure before building.** Each new component requires a benchmark result
   showing the current bottleneck. Check `DEFERRED.md` for the relevant signal.

2. **One kernel, one detector.** Do not generalize `fused_calib_gpu` or add
   a kernel registry until a second detector needs GPU calib.

3. **No batching until profiled.** Don't add batch dimensions or EventPool
   until per-event occupancy is measured and shown to be too low. For Jungfrau
   4M (9.96M pixels) a single event likely saturates the GPU already.

4. **No new abstractions for single use cases.** `GpuBatchView` was built for
   one format; `DetectorRouter` was built for one detector. Don't repeat this.

5. **The deferred doc is a contract.** Before adding anything from `DEFERRED.md`
   back, the corresponding benchmark signal must be present.

6. **Keep the API flat.** Two functions (`prep_calib_constants`, `fused_calib_gpu`)
   is the right size for MVP. Resist wrapping them in a class.

---

## Security Considerations

This is a scientific computing project running in a controlled HPC environment
(SLAC S3DF). No external network exposure, no user-supplied inputs beyond file
paths, no authentication tokens in code. The main considerations are:

- Do not hardcode experiment names, run numbers, or file paths in committed code.
- Calibration constants come from the LCLS calibration store — treat them as
  trusted but validate shape/dtype before GPU transfer.
- `CUDA_VISIBLE_DEVICES` manipulation in `init_gpu_rank()` is intentional and
  safe within a single-node Slurm job.

---

## Future Considerations

All items below are tracked in `DEFERRED.md` with specific benchmarking criteria.
Do not implement until the criterion is met.

| Feature | Criterion |
|---|---|
| Direct XTC read (KvikIO) | GPU calib throughput is I/O bound on CPU-path H→D |
| EventPool N-in-flight | Single-event GPU occupancy < 50% on A100 |
| GPUBAT1 batch format | Batched read is faster than per-event; format needs to be stable |
| StreamPool | Serialized stream launches are measurable bottleneck |
| AsyncD2HJoiner | D2H measured as bottleneck (already shown: 100→40 Hz) — implement next after MVP validated |
| DetectorRouter combining | Split GPU/CPU Jungfrau segments needed for a real run |
| CUDA IPC calib sharing | Multiple BD ranks on same GPU, memory pressure confirmed |
| GPU dgram parsing | CPU-side event routing measured as bottleneck |
| MPI communicators | NCCL AllReduce needed (XPCS use case) |
| Second detector (ePix10k) | Jungfrau GPU calib is in production use |
| GpuEventContext / ctx API | Functional API (`fused_calib_gpu` call) is limiting user workflows |
