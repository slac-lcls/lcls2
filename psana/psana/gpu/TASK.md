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
- [x] Correctness on a second dataset: mfx100852324 r0078 (the original
  branch's experiment family; r0077 bigdata is tape-only)
  - 2026-07-08 (job 31049526): **PASS** — 20 events, 0 mismatches,
    max_diff_seen=0.000000 (bit-exact), detector name 'jungfrau', same
    16.78M-pixel geometry. Log: `bench_mpi_sweep/r78_correctness.log`.

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
  - PS_EB_NODES sweep (2026-07-08, job 31049525, 32 BD / 2 nodes / FFB):
    EB=1 -> 226.4 Hz, EB=2 -> 248.7 Hz, EB=4 -> 212.4 Hz. Flat within
    allocation-to-allocation noise (~10-15%): the EventBuilder is not
    the serializer. (EB=4 also stranded ~900 events in partial batches
    at stream end — rates still valid, per-rank timed. The flat result
    was initially over-read as implicating smd0; corrected below.)
  - PS_SMD_N_EVENTS sweep (2026-07-08, job 31049831, 32 BD / 2 nodes /
    FFB): 1000 -> 253.7 Hz, 4000 -> 254.0 Hz, 16000 -> 242.3 Hz. Flat.
    (Initially read, with the EB sweep, as implicating smd0. WRONG — see
    the smd0 diary and raw I/O matrix below.)
  - smd0 diary (2026-07-08, job 31055638, `--smd0-debug`): smd0 read the
    run's ENTIRE smalldata (297 MB) at 2.2 GB/s in one chunk, then sat
    19.2 s idle in eb_wait out of a ~32 s event phase. **smd0 exonerated;
    the parallel-smd0 design is dead.** The pull-based chain means an
    idle smd0 and a saturated one look identical from outside — only the
    diary discriminates.
  - Raw I/O matrix, no psana (2026-07-08, jobs 31058104/05, 31058810,
    dumb parallel os.pread readers on the FFB files). Reads are
    **per-node bandwidth limited at ~7.9 GB/s, and that scales with
    nodes**:

    | config | nodes | readers/node | GB/s |
    |---|---:|---:|---:|
    | B1_1   | 1 | 1  | 0.71 |
    | B1_8   | 1 | 8  | 2.22 |
    | B1_32  | 1 | 32 | 7.87 |
    | B3 (random offsets) | 1 | 32 | 7.28 (-8%) |
    | B2_2n  | 2 | 16 | 12.0 (6.0/node) |
    | B5_2x32| 2 | 32 | 15.2 (7.6/node) |

    A node needs ~32 concurrent readers to saturate its ~7.9 GB/s; at
    16/node it only reaches ~6. Random access costs 8%. Two nodes on the
    SAME run scale cleanly (15.2 = 2x); B4 (two nodes each on a DIFFERENT
    run) got only 9.6 total, i.e. worse, which points to shared-backend
    contention plus minute-to-minute variance on a live production FS.
  - **Correction to the earlier "aggregate ~10-12 GB/s cap" note (itself
    an over-attribution): there is no hard aggregate cap in this range.**
    The limit is per-node (~7.9 GB/s at full concurrency) and scales with
    node count. Raw 2-node reaches 15.2 GB/s (= ~450 Hz of Jungfrau).
  - psana vs raw, apples-to-apples by layout: single node psana 7.0 vs
    raw 7.9 GB/s (89%); 2 nodes at 16 BD/node psana 8.6 vs raw 12.0
    (72%). So psana leaves the most on the table at multi-node, and it
    was also running too few BD ranks per node (16, where raw needs 32 to
    saturate). RESOLVED below.
  - psana 32 BD/node x 2 nodes (2026-07-08, job 31062160, 64 BD): **294.9
    Hz** = 9.9 GB/s. Up from 258 Hz at 16 BD/node (+14%), so more workers
    per node helps modestly but does NOT reach raw's 15.2 GB/s (~450 Hz)
    at the same layout. psana attains 65% of raw storage at 2 nodes vs
    89% at 1 node. **Verdict: the multi-node shortfall is a real psana
    inefficiency, not just under-concurrency.**
  - Synthesis (two ceilings, dominating at different scales):
      * Storage: per-node ~7.9 GB/s (~235 Hz/node), scales with nodes.
      * psana serving/pipeline: a ~260-306 Hz plateau independent of node
        count (2 vs 4), BD ranks/node (16 vs 32), EB count, and batch.
    At 1 node the two nearly coincide (storage 235, psana 210 = storage-
    bound). Beyond ~1 node the psana plateau (~300 Hz) binds first while
    storage still has headroom (450+ at 2 nodes). So ~300 Hz IS a psana
    ceiling at multi-node after all — but NOT smd0 (proven idle), NOT EB
    count, NOT batch size.
  - Most likely remaining cause: the per-BD-rank synchronous pipeline
    (read bigdata -> H2D -> kernel with no overlap; 32 ranks/node contend
    on one PCIe link, H2D already measured inflating to 16 ms). Ranks
    cannot consume the extra storage bandwidth a second node exposes.
    This RAISES the priority of pinned-host + overlapped H2D (previously
    "later") — it is now the one code lever that could unlock multi-node
    headroom. Confirm with a BD-rank profile (read vs H2D vs kernel vs
    MPI-wait) before building.
  - Levers, updated: (a) pinned + overlapped H2D per BD rank — promoted,
    the likely multi-node unlock; (b) ~32 BD ranks/node — modest, free;
    (c) detector compression at write time (~2x); (d) NVMe staging for
    reprocessing; (e) more nodes; (f) facility FFB QoS. Dead: parallel
    smd0 (idle), more EBs (flat), locality-aware EB scheduling (no
    pattern penalty), spreading across runs (B4 worse).
  - D2H at scale (same job, 32 BD ranks): --d2h off 236.3 Hz vs on
    170.0 Hz (-28%). Sync `.get()` costs 72 ms/event under 32-rank PCIe
    contention (vs 12.5 ms serial) and is ~73% additive — NOT hidden by
    event-wait, because ranks' transfers already overlap each other on
    the shared link. **AsyncD2HJoiner trigger condition: MET** (see
    DEFERRED.md) for any workflow returning calibrated frames to host.
  - Gotcha for reproduction: OpenMPI default core binding silently
    distorts multi-rank rates and refuses >17 procs on a 17-core
    allocation — always `--bind-to none` (attempt-1 logs show the artifact).
- [~] Establish B-CPU (CPU-only psana, det.raw.calib(), MPI at scale on FFB)
  - 2026-07-10 (Ralph iter 1): added `--cpu` mode to bench_calib.py — BD ranks
    run det.raw.calib() on the shared collective DataSource, MPI-capable at the
    SAME rank layout as the GPU path. Code verified working at 1 and 32 BD ranks.
  - **Reference dataset is now mfx101572426 r47 on FFB** (2026-07-10): 32-seg
    Jungfrau (32, 512, 1024) = 16.78M pixels, det `jungfrau`, ~37k events,
    calib constants deployed and verified. FFB dir
    `/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc`. Measure B-CPU **and**
    re-measure B-MVP on r47 at 1 and 32 BD ranks — the identical run anchors
    the comparison. The Lustre numbers below were verification-only.
  - Verification numbers on Lustre r387 (job 31260656, sdfampere042,
    `bench_mpi_sweep/cpu_lustre_bd{1,32}.log`, `cpu_baseline_README.md`):

    | BD ranks | aggregate Hz | per-rank Hz | CPU calib ms/event |
    |---:|---:|---:|---:|
    | 1  | 10.1  | 10.06 | 29.75 |
    | 32 | 106.5 | 3.33  | 54.48 |

    These are Lustre + cache-contaminated (r387 is warm now) — NOT comparable
    to the FFB B-MVP rows nor to the historical cold-Lustre GPU column.
  - Robust finding: CPU calib inflates 29.7 -> 54.5 ms/event from 1 -> 32 ranks
    (32-way core/memory-bandwidth contention). CPU compute ceiling at 32 ranks
    ~= 32/0.0545 = ~587 Hz, vs the GPU kernel ceiling ~3100 Hz. CPU calib does
    not scale linearly across cores.
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

- 2026-07-08: PS_EB_NODES sweep — flat; initially attributed the ceiling
  to smd0 (WRONG — corrected same day by the smd0 diary + raw I/O matrix:
  the ceiling is aggregate FFB bandwidth, ~10-12 GB/s; smd0 sits idle).
  r0078 correctness PASS bit-exact (second dataset). Old-branch benchmark
  comparison on mfx100852324 dropped: no FFB copy exists, so it would be
  Lustre-bound and uninformative.

- 2026-07-08: PS_SMD_N_EVENTS sweep flat (253.7/254.0/242.3 Hz) and D2H
  sweep (236.3 -> 170.0 Hz with sync .get()) — **measurement campaign
  complete.** Every DEFERRED.md item now has a data-backed verdict.

- 2026-07-08 (later): smd0 diary + raw I/O matrix (results above) —
  **ceiling re-attributed from smd0 to aggregate FFB bandwidth**. smd0
  profiling done (it is idle); parallel-smd0 removed from the roadmap.

Remaining (all beyond pure measurement):
1. AsyncD2HJoiner — trigger MET (DEFERRED.md updated); build when a
   production workflow needs calibrated frames back on host.
2. Storage-side levers for the ~10-12 GB/s FFB ceiling — none are psana
   code: detector-data compression at write time (~2x), node-local NVMe
   staging for reprocessing, facility conversation on FFB bandwidth/QoS.
3. Pinned-host H->D staging — secondary, only relevant once per-GPU feed
   can exceed ~270 Hz (i.e., after the storage ceiling moves).
