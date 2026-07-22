# GPU Kernel Campaign — Findings

Branch `features/psana2-gpu-kernels` (off `features/psana2-gpu`).
Data: mfx101572426 run 47, Jungfrau 16.8 Mpix (~33.5 MB/event raw), read from
FFB. All pipeline measurements on a single A100.

## What was built

**Analysis kernels** (`cuda/analysis_kernels.cu`, exposed via
`gpu_azint_kernel.py`): azimuthal integration of the calibrated image into a
q-histogram, ported from the standalone jungfrau_gpu_azint project. Two
implementations of the same computation, verified to produce identical
results against a float64 CPU reference:

- **sorted** — pixels pre-sorted by bin at setup; the kernel gathers and
  tree-reduces with no atomics. ~0.3 ms/event in-pipeline.
- **atomic** — one thread per pixel, `atomicAdd` into a global histogram.
  ~9–11 ms/event solo. Kept deliberately as a heavyweight workload: it makes
  the GPU the bottleneck so framework behavior under compute load is visible.

Both consume the calibrated buffer the framework already materializes; a
fused raw→histogram variant was measured and rejected (random-access
amplification on the calibration constants makes it slower than the
two-kernel chain).

**Registry extension** (~100 lines across `gpu_kernel_registry.py`,
`gpu_calib.py`, `gpu_events.py`, `gpu_stream.py`): kernels registered under a
name other than `'calib'` run after calibration, on-device, inside the same
stream, and surface to the user as `ctx.get('jungfrau.azint')`. The `'calib'`
contract is untouched; with no extra kernels registered, nothing changes.

**Per-stage profiler** (`bench_pipeline_stages.py`): times every stage of the
BD event loop with CUDA events — `feed` (wait for the next batch), `k_calib`,
`k_azint`, `d2h` — under the knobs a user would turn: `--azint
off|sorted|atomic`, `--n_gpu_streams`, `--batch_size`, `--no-share-calib`,
`--d2h none|azint|calib`. Per-rank stats persist to files as ranks finish
(`--stats-dir`), so partial results survive stragglers; `--report-dir`
aggregates.

**Wire-format fix** (`psexp/node.py`): the event builder's end-of-run
"missing step" sends bypassed the GPU-path batch wrapper, so BD ranks
misparsed step-only batches as GPU work — a crash at 8 BD ranks, a silent
hang at 4. Both send sites now wrap, and the receiver validates the GPUBAT1
magic. (A second termination issue remains open: `max_events` truncation can
strand GPU BD ranks in the event builder's kill accounting; the bench works
around it by breaking at an event target.)

**Tests**: `test_analysis_kernels.py` (standalone, synthetic data, GPU vs
CPU-float64 reference), `test_azint_kernel.py` (registry plumbing against a
stub detector).

## Measurement discipline

The FFB serves ~230 Hz/node of this run's raw data cold (~7.9 GB/s ÷
33.5 MB/event). Re-reading the same files warms the page cache and inflates
later measurements: an identical config read 168 Hz at the start of a job and
270 Hz ten minutes in. Every figure below therefore carries a window class:

- **cold** — first config of a fresh allocation on a node that had not
  recently touched the files.
- **warm** — any subsequent config. Warm A/Bs are valid *relative*
  comparisons (verified with order-swapped pairs and repeated sentinel
  configs); warm absolute numbers are cache measurements, not storage ones.

## What was measured

### Topology: ranks vs streams (warm, order-swapped pairs, sorted kernel)

At matched total concurrency (ranks × streams = 32, batch 1):

| ranks × streams | Hz | feed ms/event/rank |
|---|---:|---:|
| 16 × 2 | 255 | 45 |
| 8 × 4 | 302 | 22 |
| **4 × 8** | **371** | 10 |
| 2 × 16 | 229 | 8.5 |

Per-rank feed cost grows ~linearly with rank count — the delivery path
serializes across processes — so fewer, better-fed processes win. Below ~4
processes the CPU side can no longer keep the GPU supplied (the 8.5 ms/rank
feed floor caps 2 ranks at ~230 Hz). Optimum: the fewest ranks that can still
feed, with streams providing concurrency inside each.

### Batching (warm, 4 BD × 4 streams, sorted)

| batch size | Hz |
|---|---:|
| 1 (before) | 280 |
| 4 | 353 |
| 16 | 413 |
| 1 (drift control, after) | 347 |

Batch 16 beat even the bs=1 control that ran after it, so the gain is real,
not cache drift. 413 Hz is the highest rate measured on this branch.

### The tuned configuration, cold (fresh allocation, different node)

| config | window | Hz |
|---|---|---:|
| tuned: 4 BD × 4 streams × batch 16 | **cold** | **218** |
| tuned, repeated | warm | 354 |
| default: 16 BD × 2 streams × batch 1 | warm | 192 |

The tuned pipeline reads within ~5% of the cold storage floor: in that
configuration psana's GPU path is no longer the limiting factor — the
filesystem is. The default topology delivers 192 Hz *with warm data*, i.e.
below the storage floor: it is bottlenecked by the pipeline itself, not by
storage. Warm inflation on the tuned config is +62% (218 → 354), which is
why absolute figures must carry a window class.

### Kernel weight and GPU contention (atomic kernel, per-event kernel time)

| BD ranks sharing the GPU | 1 | 2 | 4 | 8 | 16 |
|---|--:|--:|--:|--:|--:|
| k_azint ms/event | 9.4 | 16.1 | 35.5 | 70.5 | 133.7 |
| aggregate Hz | 45 | 71 | 75 | 72 | 69 |

Dilation is linear from the first added rank — the GPU serializes the
launches — and aggregate throughput pins at the kernel's serialized ceiling
(~72–75 Hz) from 2 ranks onward. There is no rank count at which an
atomic-heavy kernel shares a GPU without penalty. The sorted kernel shows
zero dilation at any rank count (0.30–0.34 ms throughout): contention
behavior is a property of kernel style, not just kernel cost. Expensive
per-process work is another argument for the few-ranks topology.

### Returning results to the host (warm, 16 BD, sorted)

| what crosses PCIe per event | Hz |
|---|---:|
| nothing | 271 |
| 3 KB histogram | ~271 (d2h 5.9 ms exposed, no throughput cost) |
| 67 MB calibrated frame | 142 (d2h 74.7 ms) |

Reducing on-device and returning the small result is free; returning frames
halves throughput. This is the concrete case for running analysis kernels
inside the pipeline rather than pulling calibrated images back.

### Memory and the rank ceiling

Each BD rank is a process carrying its own CUDA context (~0.3–0.5 GB), a
copy of the calibration constants (~384 MB), geometry arrays, **and one
additional ~384 MB constants copy per GPU stream** (the per-stream cache in
`gpu_calib.py`), plus batch-sized frame buffers (4 slots × batch × 67 MB).
On a 40 GB A100 this puts the ceiling between 16 and 24 ranks (24 OOMs).

Two mitigations were tested:

- **CUDA-IPC constant sharing** shares only the base constants copy, and
  only *after* every rank has already allocated its own — the peak is
  unchanged, and the per-stream caches are not covered. It saves steady-state
  memory at small rank counts but does not raise the ceiling.
- **CUDA MPS** improves throughput (+20% on sorted at 16 ranks; atomic
  dilation 134 → 91 ms) but does not move the ceiling either: 24 and 32
  ranks OOM under MPS identically. The ceiling is data (constants, caches,
  buffers), not contexts.

The few-ranks topology sidesteps rather than solves this: 4 ranks × 4
streams uses roughly a quarter of the per-rank overhead and never approaches
the ceiling.

## Why the design choices matter under a storage cap

The storage rate is a ceiling, not a guarantee. A configuration only runs at
~230 Hz if every other part of the pipeline can sustain more than that; the
measurements show the default configuration cannot — it delivers 192 Hz even
when storage is effectively removed from the problem by the page cache. Its
binding constraint is the delivery path's per-rank serialization, which no
storage upgrade would fix.

The design features — few ranks, streams, batching, on-device reduction —
raise the *pipeline's own capability* from below the storage floor (~192 Hz
warm) to well above it (413 Hz warm). Cold, that capability is clipped at
the floor (218 Hz measured), which is precisely the desired end state: the
pipeline saturates the storage it is given. The same tuning is what makes
the remaining headroom usable — faster storage windows, multi-node scaling,
or cached re-analysis immediately translate to throughput instead of being
absorbed by pipeline overhead, and the topology that achieves it is also the
one that tolerates expensive user kernels and stays far from the memory
ceiling.

## Reproducing

```
sbatch psana/psana/gpu/scripts/submit_design_space.sh   # matrix (jobs like 32765825)
sbatch psana/psana/gpu/scripts/submit_cold_control.sh   # cold control (32779707)
```

The first script runs the sentinel/knee/topology/batch/D2H/MPS matrix in one
allocation (~35 min); the second measures the tuned config cold-first on a
fresh node. Standalone kernel checks: `python
psana/psana/gpu/test_analysis_kernels.py` (no psana required).
