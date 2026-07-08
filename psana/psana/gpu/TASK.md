# GPU Calibration MVP — Task Tracking

## Setup

- [x] Create branch `features/psana2-gpu-mvp` from `origin/features/psana2-gpu`
- [x] Delete all GPU files except `gpu_calib.py`, `cuda/fused_calib.cuh`, and this planning directory
- [x] Gut `gpu_calib.py` to ~150 lines: keep only `prep_calib_constants()`, `fused_calib_gpu()`, `_cupy()`, `_jungfrau_calib_kernel()`, `_kernel_source()`
- [x] Remove `GPUDetector`, `EventContext`, `optimal_kernel_batch_size`, `_detect_dgram_layout`, `build_stream_seg_map`, `_compute_calib_constants_cpu` from `gpu_calib.py`
- [x] Remove import of `gpu_kvikio_read` from `gpu_calib.py`
- [x] Rewrite `__init__.py` to export only: `prep_calib_constants`, `fused_calib_gpu`, `init_gpu_rank`
- [x] Write `init_gpu_rank()` as a standalone ~20-line function directly in `__init__.py` (no `gpu_mpi.py`)
- [x] Write `DEFERRED.md` cataloguing every removed feature with its benchmarking criterion
- [x] Verify `cuda/fused_calib.cuh` has no imports or dependencies on removed files
- [x] Second-pass cleanup (8ea55e898): remove GPU-era residue outside `psana/gpu/`
  (psexp plumbing, eventbuilder.pyx GPUBAT1, dgram.cc raw_descriptors,
  dgramlite.pyx SmdInfoExtractIter, seven stale `tests/test_gpu_*.py`) —
  see DEFERRED.md *(second pass)* entries

## Core Functionality

- [x] Confirm `prep_calib_constants(det)` produces correct pedestal/gmask layout on a reference run
  - Expected: flat float32, length `3 * n_segs * 512 * 1024`, mode-major order
  - Check: `peds_gpu.shape == (3 * n_pixels,)` and `gmask_gpu.shape == (3 * n_pixels,)`
  - 2026-07-07: shape `(50331648,)` float32 on mfx101210926 r387 (32 segs) ✓
- [x] Confirm `fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu)` returns float32 array of same shape as input
- [x] Confirm the kernel compiles without error on first call (NVRTC JIT)
  - 2026-07-07: compiled on A100 (sdfampere004), cupy-cuda12x 13.6.0 + pip nvrtc-cu12 libs
- [x] Confirm `fused_calib.cuh` is found via `Path(__file__).parent / 'cuda' / 'fused_calib.cuh'`

## Correctness Validation

- [x] Write `test_jungfrau_calib.py` — pixel-level comparison against `det.raw.calib(evt)`
  - Run on mfx101210926-r0387 (50 events)
  - Pass criterion: zero mismatches at `atol=1e-3`
- [x] Run correctness test and record result
  - 2026-07-07: **PASS** — 50 events, 0 skipped, 0 mismatches, max_diff_seen=0.000000
    (bit-exact, not just within atol) on A100, after second-pass cleanup + clean meson rebuild
- [x] Identify and document any known divergences (e.g. common-mode correction applied by CPU but not GPU)
  - Documented in `test_jungfrau_calib.py` docstring; none observed on the reference run

## Performance Benchmarking

- [x] Write `bench_calib.py` — minimal benchmark script
  - Flags: `-e EXP`, `-r RUN`, `-n N_EVENTS`, `--compare-cpu`, `--d2h`
  - Records: events/sec GPU (no D2H), events/sec GPU (with D2H), events/sec CPU
  - Records: H→D ms/event, kernel ms/event, D→H ms/event
- [x] Run benchmark: GPU no-D2H vs GPU with-D2H vs CPU
  - 2026-07-07, A100 (sdfampere004), mfx101210926 r387 (32-seg Jungfrau, 16.78M pixels):
  - Serial single-process, 500 events: GPU 0.9 Hz = CPU 0.9 Hz (1.01x) — the
    serial event loop costs ~1.1 s/event on /sdf/data Lustre (~30 MB/s effective)
    and dominates both paths; calibration is 0.4% of wall time. Per-stage:
    H→D 4.23 ms, kernel 0.32 ms.
  - Compute-only (10 in-memory events, event-loop I/O excluded):
    CPU `det.raw.calib()` 30.08 ms/event; GPU H→D+kernel 3.69 ms/event =
    **8.1x**; with synchronous `.get()` 16.19 ms/event = 1.9x. The ~12.5 ms
    sync D2H cost reconfirms AsyncD2HJoiner as the top deferred item.
  - Conclusion: MVP claim holds — calibration itself is 8x faster on GPU;
    end-to-end rate is bounded by event iteration, not calibration
    (see DEFERRED.md for the escalation criteria).
- [x] Measure single-event GPU occupancy on A100 (blocks launched vs saturation point)
  - Expected for Jungfrau 4M: ~9.96M pixels / 256 threads = ~38,900 blocks >> 864 saturation → occupancy > 100%, single event saturates GPU
  - Record this — it is the justification for NOT adding EventPool
  - 2026-07-07 measured (16.78M-pixel run): 65,536 blocks/event vs 864
    saturation = **7,585%** — one event over-saturates the A100 ~76x.
    EventPool remains unjustified.
- [x] MPI scaling benchmark: N BD ranks feeding one A100 via `init_gpu_rank()`
  - 2026-07-08, jobs 31043622 (Lustre) / 31047910 (FFB), single ampere node,
    `mpirun --bind-to none`, PS_EB_NODES=1, mfx101210926 r387, logs in
    `bench_mpi_sweep/` at repo root:

    | BD ranks | /sdf/data Lustre | FFB (drpsrcf) | FFB per-rank | FFB H->D ms |
    |---:|---:|---:|---:|---:|
    | 1  | 1.2 Hz  | 36.4 Hz  | 36.4 | 4.07 |
    | 2  | 6.2 Hz  | 68.1 Hz  | 34.0 | 4.03 |
    | 4  | 3.3 Hz  | 134.0 Hz | 33.5 | 4.10 |
    | 8  | 6.3 Hz  | 195.0 Hz | 24.4 | 5.53 |
    | 16 | 9.4 Hz  | 205.1 Hz | 12.8 | 11.01 |
    | 32 | 15.0 Hz | 210.4 Hz | 6.6  | 15.95 |

  - Multi-node sweep (2026-07-08, jobs 31048350/31048352, FFB, 1 GPU/node,
    ~16 BD ranks/node): 32 ranks / 2 nodes / 2 GPUs -> **258.1 Hz**;
    64 ranks / 4 nodes / 4 GPUs -> **306.4 Hz**.
  - Storage is the dominant ceiling: identical code/events, 14x aggregate
    difference. On FFB the standard loop reaches **210 Hz ≈ 7 GB/s = 78%**
    of the ~270 Hz one-A100 absorption limit with no deferred machinery.
  - The plateau is **central, not per-node**: doubling nodes gave +23%
    (not 2x), quadrupling +46%. The single smd0 + single EB serving chain
    saturates around ~300 Hz. Per-node PCIe H->D contention is real but
    secondary (H->D 16.0 -> 10.6 ms when 32 ranks split across 2 nodes;
    kernel always <1 ms). At mn4 each GPU gets only ~77 Hz of its ~270 Hz
    capacity — GPUs are nowhere near the bottleneck.
  - Next lever, in order: PS_EB_NODES > 1 (configuration, discriminates
    EB vs smd0 as the serializer), then pinned-host H->D staging if
    per-node rates climb enough for PCIe contention to bind again.
  - Gotcha for reproduction: OpenMPI default core binding silently
    distorts multi-rank rates and refuses >17 procs on a 17-core
    allocation — always `--bind-to none` (attempt-1 logs show the artifact).
- [x] Record NIC recv bandwidth during GPU run vs CPU run
  - 2026-07-08: sampled /proc/net/dev at 2 s during every sweep config.
    Finding: bulk storage I/O (~7 GB/s at 210 Hz) is INVISIBLE to netdev
    byte counters — both Lustre and FFB mounts move data over RDMA/verbs.
    Only TCP-side chatter (20-60 MB/s on enp225s0) appears. NIC saturation
    must be assessed via IB counters (/sys/class/infiniband) in future runs.

## Documentation

- [x] Write `DEFERRED.md` with all removed features and benchmark criteria (see PLANNING.md §Future Considerations for the table)
- [x] Add brief usage example to `__init__.py` docstring showing the two-function API

## Completed Work

- 2026-07-07 (958878d6e): slimmed `psana/psana/gpu/` to the two-function MVP;
  wrote DEFERRED.md / PLANNING.md / TASK.md, `test_jungfrau_calib.py`, `bench_calib.py`.
- 2026-07-07 (8ea55e898): second-pass cleanup — reverted all GPU-era residue
  outside `psana/gpu/` to the master merge-base (4d7254c99); psana imports and
  MPI DataSource work again. 21 files, −4,374 lines.
- 2026-07-07: clean meson rebuild via `./build_all.sh` (replaced a broken stale
  editable install); installed `cupy-cuda12x` (--user) for the ps_20241122 py3.9 env.
- 2026-07-07: correctness validation **PASS** — bit-exact vs `det.raw.calib()`,
  50 events, mfx101210926 r387, A100.
- 2026-07-07: fixed two bugs in `bench_calib.py` from the slim commit —
  `_is_bd_rank()` skipped the whole benchmark in single-process runs (mpi4py
  imports fine at world size 1, so rank 0 < 2 short-circuited everything,
  silent exit 0), and `run_gpu_bench()` referenced `fused_calib_gpu` that was
  only imported inside `main()` (latent NameError).
- 2026-07-07: performance benchmark run — compute speedup **8.1x**
  (30.08 → 3.69 ms/event), occupancy 7,585%, serial event loop identified as
  the end-to-end bottleneck (~1.1 s/event vs 4.5 ms GPU work). Numbers above.

- 2026-07-08 (f8ae68984): restructured `bench_calib.py` for MPI — single
  collective DataSource, max_events-based termination, rank-0 aggregate
  report. The previous structure deadlocked under mpirun.
- 2026-07-08: MPI scaling sweep run (results above). Verdict: the standard
  event loop + FFB storage feeds one A100 at 210 Hz (78% of absorption);
  /sdf/data Lustre caps the same code at 15 Hz. No deferred I/O feature is
  justified by these data; the first indicated optimization is pinned-host
  H->D staging once per-node rates matter.

- 2026-07-08: multi-node sweep done (results above). Verdict: ceiling is the
  central smd0/EB chain at ~300 Hz, not per-node resources. Full campaign
  logs in `bench_mpi_sweep/` (three single-node suffix families + mn2/mn4).

Remaining, in priority order:
1. PS_EB_NODES sweep (1/2/4 EB ranks at fixed 32 BD / 2 nodes, FFB) — if
   rates rise, EB was the serializer; if flat, it is smd0 itself.
2. Correctness + sweep on mfx100852324 r0078 (r0077 bigdata is tape-only)
   to compare against the original branch's July dataset.
3. `--d2h` sweep at the best-known config to quantify the AsyncD2HJoiner
   trigger from DEFERRED.md.
