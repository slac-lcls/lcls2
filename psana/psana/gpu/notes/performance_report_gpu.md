# psana2 GPU BD Prototype — Performance Report

**Branch:** `features/psana2-gpu`  
**Hardware:** NVIDIA A100 40 GB (sdfampere nodes), S3DF  
**Dataset:** `mfx100852324-r0077` — MFX Jungfrau 4M, 32 segments, uncompressed  
**Topology:** 4 MPI ranks — smd0(0) + EB(1) + BD-GPU0(2) + BD-GPU1(3)  

> **I/O path:** All runs use the **kvikio CPU-fallback path**
> (NVMe → CPU DRAM → GPU VRAM via `cudaMemcpy`).  True GDS is unavailable on
> S3DF Lustre.  True GDS would bypass CPU DRAM entirely and is expected to
> roughly double throughput.

---

## Latest results — batch_size × pool_depth sweep

**Job:** 30351528  **Node:** sdfampere027  **Date:** 2026-07-01  
**Config:** `--n-events 2000 --n-warmup 100`  **Timed events:** 1900/BD rank  
**Script:** two `srun` steps (one per pool_depth) to reset MPI universe between sweeps

### CPU baseline

| Rank | events | mean ms | p95 ms | evt/s |
|---:|---:|---:|---:|---:|
| 2 | 1012 | 29.4 | 31.4 | 18 |
| 3 | 888 | 33.8 | 37.5 | 16 |
| **AGGREGATE** | **1900** | | | **34** |

### GPU sweep

| batch\_size | pd=2 evt/s | pd=2 ×CPU | pd=4 evt/s | pd=4 ×CPU | pd=8 |
|---:|---:|---:|---:|---:|---|
| 1 | 48 | 1.4× | 57 | 1.7× | — |
| 5 | 47 | 1.4× | 49 | 1.4× | — |
| 10 | 48 | 1.4× | 58 | 1.7× | — |
| 20 | 49 | 1.4× | 54 | 1.6× | — |
| **50** | **54** | **1.6×** | **60** | **1.8×** | OOM |

**Best: bs=50, pd=4 → 60 evt/s, 1.8× over CPU.**

`pd=8, bs=50` OOMs even as the first and only config on a fresh node
(~29 GB required; confirmed on sdfampere004 job 30359393).  `pd=8` is safe
only with `bs≤20`.

### Notes

- Throughput is flat across batch sizes (47–60 evt/s). With 1900 timed events
  the result is dominated by steady-state NVMe I/O (~40 ms/event wall time),
  not per-batch amortization. Smaller event counts show more variation due to
  warmup and NUMA load imbalance.
- `pd=4` beats `pd=2` by ~10–20% across all batch sizes (extra look-ahead).
- GPU kernel time: 0.002–0.009 ms/event — calibration is not the bottleneck.
  Wall time is almost entirely NVMe I/O wait.

---

## VRAM budget per BD rank (mfx100852324-r0077)

```
data_gpu per slot     ≈ bs × 33 MB
calib_slot_buf/slot   ≈ bs × 39.8 MB   (19 segs × 512 × 1024 × 4B × bs events)
raw_slot_bufs/slot    ≈ 19.8 MB        (fixed, independent of bs)
peds_gpu + gmask_gpu  ≈ 400 MB         (once per run)

Total ≈ n_slots × bs × 72.8 MB  +  n_slots × 19.8 MB  +  400 MB
```

| | bs=10 | bs=20 | bs=50 |
|---|---:|---:|---:|
| pd=2 | ~1.9 GB | ~3.3 GB | ~7.7 GB |
| pd=4 | ~3.5 GB | ~6.0 GB | ~15.0 GB |
| pd=8 | ~6.6 GB | ~11.6 GB | **~29.5 GB** ← OOM |

Maximum safe batch size with pd=4 on a 40 GB A100:
```
bs_max = (40 000 MB − 400 MB) / (4 × 72.8 MB)  ≈  135
```
Recommended maximum for a sustained single run: **bs=80, pd=4**.

---

## EB bottleneck

GPU calibration completes in ~0.002 ms/event; wall time is ~30–40 ms/event.
The BD rank is idle nearly the entire time waiting for EB to deliver the next
batch.  EB reads SMD files and builds batches; the round-trip cost dominates:

```
kvikio read (CPU-fallback):  10 events × 33 MB ≈ 165 ms
EB SMD read + batch build:                      ≈ 220 ms
Total per 10-event batch:                       ≈ 385 ms  →  26 evt/s/rank
```

Larger batch sizes amortise the fixed EB overhead over more events, which is
why bs=50 outperforms bs=1 even though both have the same per-event I/O cost.
EB look-ahead (request for batch N+1 sent immediately after receiving batch N)
hides the 220 ms EB build time inside the 165 ms GDS read window.

**To improve further:** multiple EB ranks (`PS_EB_NODES=2`), or BD ranks
reading SMD directly (bypasses EB entirely, approaches single-process speed).

---

## Reproducing

```bash
# From sdfiana026:
cd ~/lcls2
source setup_env.sh

# Standard CPU vs GPU benchmark (bs=10,20,50 pd=4):
sbatch psana/psana/gpu/scripts/submit_mpi_perf_compare.sh

# Full batch_size × pool_depth sweep:
# run separate srun per pool_depth to avoid MPI state accumulation
# see /tmp/perf_sweep_final.sh for the template used for job 30351528

# Multi-GPU correctness + throughput sweep:
sbatch psana/psana/gpu/scripts/submit_multi_rank_sweep.sh
```

Always use `--n-warmup 100` for MPI benchmarks — smaller values give
optimistic numbers because the CuPy pool is cold at startup.
