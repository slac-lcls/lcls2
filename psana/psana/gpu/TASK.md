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
- [ ] Run benchmark: GPU no-D2H vs GPU with-D2H vs CPU
- [ ] Measure single-event GPU occupancy on A100 (blocks launched vs saturation point)
  - Expected for Jungfrau 4M: ~9.96M pixels / 256 threads = ~38,900 blocks >> 864 saturation → occupancy > 100%, single event saturates GPU
  - Record this — it is the justification for NOT adding EventPool
- [ ] Record NIC recv bandwidth during GPU run vs CPU run (using `net_bandwidth.py` from the original branch if needed)

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

Remaining: run `bench_calib.py` (GPU vs CPU throughput + per-stage timing) and
record occupancy + NIC bandwidth numbers.
