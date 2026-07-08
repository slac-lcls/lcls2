# GPU Calibration MVP — Task Tracking

## Setup

- [ ] Create branch `features/psana2-gpu-mvp` from `origin/features/psana2-gpu`
- [ ] Delete all GPU files except `gpu_calib.py`, `cuda/fused_calib.cuh`, and this planning directory
- [ ] Gut `gpu_calib.py` to ~150 lines: keep only `prep_calib_constants()`, `fused_calib_gpu()`, `_cupy()`, `_jungfrau_calib_kernel()`, `_kernel_source()`
- [ ] Remove `GPUDetector`, `EventContext`, `optimal_kernel_batch_size`, `_detect_dgram_layout`, `build_stream_seg_map`, `_compute_calib_constants_cpu` from `gpu_calib.py`
- [ ] Remove import of `gpu_kvikio_read` from `gpu_calib.py`
- [ ] Rewrite `__init__.py` to export only: `prep_calib_constants`, `fused_calib_gpu`, `init_gpu_rank`
- [ ] Write `init_gpu_rank()` as a standalone ~20-line function directly in `__init__.py` (no `gpu_mpi.py`)
- [ ] Write `DEFERRED.md` cataloguing every removed feature with its benchmarking criterion
- [ ] Verify `cuda/fused_calib.cuh` has no imports or dependencies on removed files

## Core Functionality

- [ ] Confirm `prep_calib_constants(det)` produces correct pedestal/gmask layout on a reference run
  - Expected: flat float32, length `3 * n_segs * 512 * 1024`, mode-major order
  - Check: `peds_gpu.shape == (3 * n_pixels,)` and `gmask_gpu.shape == (3 * n_pixels,)`
- [ ] Confirm `fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu)` returns float32 array of same shape as input
- [ ] Confirm the kernel compiles without error on first call (NVRTC JIT)
- [ ] Confirm `fused_calib.cuh` is found via `Path(__file__).parent / 'cuda' / 'fused_calib.cuh'`

## Correctness Validation

- [ ] Write `test_jungfrau_calib.py` — pixel-level comparison against `det.raw.calib(evt)`
  - Run on mfx101210926-r0387 (50 events)
  - Pass criterion: zero mismatches at `atol=1e-3`
- [ ] Run correctness test and record result
- [ ] Identify and document any known divergences (e.g. common-mode correction applied by CPU but not GPU)

## Performance Benchmarking

- [ ] Write `bench_calib.py` — minimal benchmark script
  - Flags: `-e EXP`, `-r RUN`, `-n N_EVENTS`, `--compare-cpu`, `--d2h`
  - Records: events/sec GPU (no D2H), events/sec GPU (with D2H), events/sec CPU
  - Records: H→D ms/event, kernel ms/event, D→H ms/event
- [ ] Run benchmark: GPU no-D2H vs GPU with-D2H vs CPU
- [ ] Measure single-event GPU occupancy on A100 (blocks launched vs saturation point)
  - Expected for Jungfrau 4M: ~9.96M pixels / 256 threads = ~38,900 blocks >> 864 saturation → occupancy > 100%, single event saturates GPU
  - Record this — it is the justification for NOT adding EventPool
- [ ] Record NIC recv bandwidth during GPU run vs CPU run (using `net_bandwidth.py` from the original branch if needed)

## Documentation

- [ ] Write `DEFERRED.md` with all removed features and benchmark criteria (see PLANNING.md §Future Considerations for the table)
- [ ] Add brief usage example to `__init__.py` docstring showing the two-function API

## Completed Work

*(empty — populated as tasks are finished)*
