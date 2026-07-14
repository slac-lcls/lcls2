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
  - **RE-MEASURED post-wins on r47 (2026-07-10, iter 16, job 31291349, 64
    BD, log `bench_mpi_sweep/sweep_mn2x32_r47.log`): 284.4 Hz** — the
    plateau HELD. Despite copy=False + seg-h2d lifting single-node 32-BD
    +70% (210 -> 337 Hz), the 2-node aggregate is unchanged (~295 -> 284,
    window variance). `--wait-split` attribution at 64 BD: bd_read 133.1
    ms/event (59% of the 225 ms/rank wall), eb_wait only 9.15 ms (4%),
    H->D 8.1, kernel 0.65. So the multi-node ceiling is **bd_read/storage-
    read-bound, NOT the EB serving chain** — refutes the "single EB serving
    64 BD becomes the ceiling" hypothesis; PS_EB_NODES>1 stays a dead end.
    Aggregate read demand 9.5 GB/s / 2 nodes = 4.76 GB/s/node = ~60% of the
    7.9 GB/s/node FFB ceiling.
  - **Same-window bracket (2026-07-10, iter 16 addendum, job 31291618,
    --oversubscribe fix; single/multi/single in ONE allocation):** single
    32-BD 189.0 / 174.4 Hz (mean 181.7), two-node 64-BD 291.6 Hz. **Clean
    scaling 291.6/181.7 = 1.60x for 2x nodes = 80% per-node efficiency** —
    sub-linear plateau CONFIRMED in a single window (not an artifact).
    Attribution single→multi: **bd_read FLAT (132 → 127 ms) — storage read
    scales cleanly per-node** (the earlier 43.6→133 "3x" was purely cold/warm
    window state, iters 13-14). The scaling loss is in **eb_wait (5.4 → 11.2,
    2.1x) + wait-residual (39.6 → 74.4, 1.9x)** — coordination/serving
    overhead in the wait path (dgram/generator/smd0-EB cross-node roundtrip),
    NOT storage and NOT the eb_wait batch-block specifically. Residual carries
    the iter-15 denominator caveat but its doubling is a real relative signal.
    Next: plumb dgram/smdparse counters into the multi-node aggregate report
    (currently only eb_wait/bd_read surface there) to split residual =
    dgram-plumbing vs pure smd0/EB coordination latency.
  - Synthesis (two ceilings, dominating at different scales):
      * Storage: per-node ~7.9 GB/s (~235 Hz/node), scales with nodes.
      * psana serving/pipeline: a ~260-306 Hz plateau independent of node
        count (2 vs 4), BD ranks/node (16 vs 32), EB count, and batch.
    At 1 node the two nearly coincide (storage 235, psana 210 = storage-
    bound). Beyond ~1 node the psana plateau (~300 Hz) binds first while
    storage still has headroom (450+ at 2 nodes). So ~300 Hz IS a psana
    ceiling at multi-node after all — but NOT smd0 (proven idle), NOT EB
    count, NOT batch size.
  - ~~Most likely remaining cause: the per-BD-rank synchronous pipeline
    (read bigdata -> H2D -> kernel with no overlap; ... H2D ... inflating to
    16 ms) ... RAISES the priority of pinned-host + overlapped H2D ... the one
    code lever that could unlock multi-node headroom.~~ **REFUTED 2026-07-10 by
    the BD-rank profile (iter 3, below):** H2D+kernel are only ~12% of per-event
    wall at 32 BD; pinned+async H2D gains ≤13%. The suspect was wrong — the
    bottleneck is the read/serving path (`det.raw.raw` 119 ms + gen-advance
    232 ms of a 389 ms event). See the `--profile` row further down.
  - Levers, updated (H2D priority CORRECTED by iter-3 profile — see below):
    (a) ~~pinned + overlapped H2D per BD rank — promoted, the likely multi-node
    unlock~~ DE-PRIORITIZED (≤13% ceiling, proven); (b) ~32 BD ranks/node — modest, free;
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
- [x] Establish B-CPU (CPU-only psana, det.raw.calib(), MPI at scale on FFB)
  - 2026-07-10 (Ralph iter 1): added `--cpu` mode to bench_calib.py — BD ranks
    run det.raw.calib() on the shared collective DataSource, MPI-capable at the
    SAME rank layout as the GPU path. Code verified working at 1 and 32 BD ranks.
  - **Reference dataset is now mfx101572426 r47 on FFB** (2026-07-10): 32-seg
    Jungfrau (32, 512, 1024) = 16.78M pixels, det `jungfrau`, ~37k events,
    calib constants deployed and verified. FFB dir
    `/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc`. B-MVP measured on r47
    2026-07-10: **175.3 Hz @ 32 BD**, 36.8 Hz @ 1 BD (H->D 11.85 ms, kernel
    0.614 ms/event; historical r387 210 Hz).
  - **B-CPU on r47/FFB measured 2026-07-10 (Ralph iter 2)** — anchor pair now
    complete. Job 31267701 (sdfampere029), `--cpu -n 200 --warmup 10`,
    `mpirun --bind-to none`; logs `bench_mpi_sweep/cpu_ffb_r47_bd{1,32}.log`:

    | BD ranks | CPU agg Hz | per-rank Hz | CPU calib ms/event | B-MVP GPU agg Hz | GPU speedup |
    |---:|---:|---:|---:|---:|---:|
    | 1  | 22.5 | 22.53 | 31.6  | 36.8  | 1.6× |
    | 32 | 44.9 | 1.40  | 433.7 | 175.3 | 3.9× |

    CPU calib inflates **31.6 -> 433.7 ms/event (13.7×)** from 1 -> 32 ranks on the
    16.78M-px detector: `det.raw.calib()` on a 33.5 MB array is memory-bandwidth
    bound and collapses under 32-way contention, so 32 cores buy only ~2× the
    aggregate. CPU compute ceiling at 32 ranks ≈ 32/0.4337 = ~74 Hz — the CPU path
    is compute-bound, not I/O-bound, at scale on this detector. GPU advantage
    widens 1.6× -> 3.9× with scale because the kernel (0.32–0.6 ms) has 76×
    headroom while the CPU stops scaling.
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
- [x] BD-rank profile of the GPU path — attribute the 32-BD per-event wall time
  - 2026-07-10 (Ralph iter 3): added `--profile` mode to `bench_calib.py`
    (`run_gpu_bench_profile`) bucketing each event's WALL time into wait (gen
    advance = bigdata read + EB/MPI serving), read (`det.raw.raw`), H2D, kernel.
    Job 31267701 (sdfampere029), r47/FFB, `mpirun --bind-to none`; logs
    `bench_mpi_sweep/prof_ffb_r47_bd{1,32}.log` + `..._bd32_dmon.log`.

    | bucket (ms/event) | 1 BD | 32 BD | share @32BD |
    |---|---:|---:|---:|
    | wait (gen advance = read+EB/MPI) | 12.0 | 231.8 | 60% |
    | read (`det.raw.raw`) | 7.9 | 119.3 | 31% |
    | H->D | 3.9 | 43.6 | 11% |
    | kernel | 0.32 | 2.3 | 0.6% |
    | wall/event | 24.2 | 388.7 | |
    | aggregate Hz | 41.4 | 82.3 | |

    (sum of buckets ≈ wall in every rank block → attribution trustworthy.)
  - **Finding — overturns the standing H2D hypothesis:** at 32 BD, GPU-side work
    (H2D + kernel) is only **~12%** of per-event wall; psana bigdata delivery
    (wait + read) is **~91%**. Driving H2D to zero would gain **at most +13%**.
    `nvidia-smi dmon` during the run: rxpci sustained **~3.7 GB/s (~15% of the
    25 GB/s gen4 wire)**, SM ~16% — GPU and PCIe link both largely idle. So the
    per-rank H2D-under-contention (11.85→16 ms) named above as "the leading
    suspect" / the reason pinned+async H2D was "promoted to the multi-node
    unlock" is **NOT the ceiling**. Lever priority inverts: de-prioritize
    pinned+async H2D; the target is the read/serving path.
  - Robustness: this run's absolute 82.3 Hz @ 32 BD is below the 175.3 Hz anchor
    (FFB live-FS variance and/or `-n 200`), but the breakdown is speed-independent
    — at the 175 Hz moment H2D was 11.85 of 182 ms/event = 6.5%, an even smaller
    share. The proportions, not the absolute Hz, are the result.
  - Open question (next): the delivery cost is a latency/serialization limit,
    NOT storage bandwidth (iter 4: 2.78 GB/s achieved = 35% of the 7.9 GB/s
    ceiling). The decisive next measurement is a BD-ranks-per-node concurrency
    sweep (8/16/32/48/64 BD, 1 node/FFB): if achieved read GB/s climbs toward
    7.9 with more ranks it is latency-bound (unlock = concurrency/async
    prefetch); if it plateaus well below, it is a per-rank CPU/serialization
    limit (points at `det.raw.raw` deserialization or the EB->BD path).
- [x] BD-per-node concurrency sweep (GPU path, 1 node/FFB/r47) — ANSWERS the above
  - 2026-07-10 (Ralph iter 5): `bench_calib.py` GPU path at BD = 8/16/32/48/64,
    `mpirun --bind-to none --oversubscribe -n 100`, job 31267701 (sdfampere029).
    Logs `bench_mpi_sweep/ralph_tmp/conc_*_172029_*.log`, summary
    `conc_summary_172029.log`. Read BW = agg Hz × 33.5 MB/event.

    | BD ranks | aggregate Hz | aggregate read GB/s |
    |---:|---:|---:|
    | 8  | 66.8 | 2.24 |
    | 16 | 83.9 | 2.81 |
    | 32 | 84.9 | 2.84 |
    | 48 | OOM  | — |
    | 64 | OOM  | — |
    | 32 (bracket, +13 min) | 39.9 | 1.34 |

  - **Verdict = per-rank SERIALIZATION, not latency-bound.** Aggregate read BW
    rises 8→16 (+26%) then is FLAT 16→32 (+1%), plateauing at ~2.8 GB/s = 35%
    of the 7.9 GB/s per-node storage ceiling. It does NOT climb toward 7.9 as
    ranks rise, so more BD ranks / more in-flight reads is NOT the unlock; the
    wall is per-rank CPU work in the serving/deserialize chain (`det.raw.raw` +
    EB→BD). Confound: end-bracket 32 dropped to 39.9 Hz (FFB window degraded ~2×
    over the sweep) — the plateau *level* is FS-window dependent, but 16 and 32
    were measured back-to-back (both ~84) so the rank-count flatness is robust.
  - **NEW hard ceiling: single-A100 GPU path OOMs at 48 BD ranks.** Each BD rank
    = own CUDA context (~0.3–0.6 GB) + replicated calib constants (peds+gmask ≈
    384 MB) + 201 MB working buffer; 48 ranks exhaust the 40 GB A100
    (`OutOfMemoryError: allocating 201,326,592 bytes`; 48 and 64 both abort).
    So ~32 BD ranks/A100 is the practical single-GPU concurrency ceiling. This
    MEETS the DEFERRED `share_calib_between_gpu_peers` (CUDA-IPC) trigger — but
    per the serialization verdict, IPC sharing buys memory headroom, not
    throughput.
- [x] Record NIC recv bandwidth during GPU run vs CPU run
  - 2026-07-08: sampled /proc/net/dev at 2 s during every sweep config.
    Finding: bulk storage I/O (~7 GB/s at 210 Hz) is INVISIBLE to netdev
    byte counters. Only TCP-side chatter (20-60 MB/s on enp225s0) appears.
  - 2026-07-10 (iter 4) CORRECTION: IB counters are ALSO blind to FFB here.
    FFB is **WekaFS** (`type wekafs`, DPDK userspace client via /opt/weka
    hugepages), NOT Lustre — it bypasses the kernel IB verbs stack entirely.
    Validated: a 2.1 GB `dd iflag=direct` read of an FFB xtc2 → IB
    `port_rcv_data` delta = 0; a full 32-BD run (~236 GB read, 131 samples)
    → delta = 0. So `/sys/class/infiniband` is NOT a usable FFB-traffic probe
    on this facility. The only direct Weka throughput probe (`weka stats
    realtime`) requires cluster login (`weka user login`) we don't have.
    To gauge storage load without a counter: aggregate Hz × 33.5 MB = achieved
    read GB/s, compared to the 7.9 GB/s node ceiling.
  - 2026-07-10 (iter 4): storage BANDWIDTH is not the 32-BD ceiling. Achieved
    read BW at 32 BD = 83.1 Hz × 33.5 MB = **2.78 GB/s = 35% of 7.9 GB/s**
    (link ~65% idle while the pipeline is slow). Iter 3's "delivery = 91% of
    wall" is therefore a latency/serialization limit in psana's serving chain,
    not a facility bandwidth wall. Lever = concurrency/latency-hiding (more BD
    ranks, async prefetch, read batching), not more storage bandwidth.
    Logs: `bench_mpi_sweep/ralph_tmp/prof_ib_32bd_r47_*.log`, `ib_32bd_r47_*.log`.
- [x] Decompose the `read` (`det.raw.raw`) bucket: storage I/O vs in-process CPU
  - 2026-07-10 (iter 6): `--profile-read` mode splits `det.raw.raw` into seg /
    stack (per-seg `.raw` deserialize + copyto memcpy) / copy (reshape + final
    `arr.copy()`), and times a 2nd `det.raw.raw` on the same event. **Verdict:
    the read bucket is CPU-bound, NOT storage I/O** — `read2 ≈ read1` at both
    scales (98% @ 1 BD, 99% @ 32 BD), so the cost repeats every call = pure
    in-process CPU memcpy/deserialize; the dgram bytes are already resident from
    the generator-advance `wait` bucket. This reclassifies iter 3's 31%-of-wall
    read bucket from storage to CPU.
  - Internal split (ms/event): 1 BD read1 9.19 = seg 0.008 + stack 4.18 + **copy
    5.00**; 32 BD read1 154.2 = seg 0.14 + stack 64.5 + **copy 89.5**. The final
    `.copy()` is the single largest component and is **redundant for the GPU
    path** (`cp.asarray(raw)` copies host→device immediately after, so the extra
    33.5 MB host copy is pure waste). `det.raw.raw(evt, copy=False)` (flag from
    commit ecf74b87f) removes it: copy = 58% of read @ 32 BD, ~14% of per-event
    wall → expected ~+14–16% throughput plus DRAM-contention relief. read1's
    16.8× super-linear scaling (9.2→154 ms over 32 ranks) is host-memory-BW
    contention, not the FS (work is pure repeatable CPU). NEXT: land copy=False
    behind the gate. Logs: `bench_mpi_sweep/ralph_tmp/profread_{1,32}bd_174837.log`.
- [x] Land `copy=False` on the GPU path — the first landed throughput win
  - 2026-07-10 (iter 7): promoted `det.raw.raw(evt, copy=False)` to the DEFAULT
    of the GPU bench path (`run_gpu_bench`); `--copy-true` restores the old
    baseline for A/B. Interleaved copy=True vs copy=False back-to-back (2 brackets
    each, to control for FFB minute-to-minute variance):

    | config | copy=True (Hz) | copy=False (Hz) | gain |
    |---|---|---|---|
    | 1 BD  | 39.8, 42.7 → 41.3 | 53.6, 54.6 → 54.1 | **+31%** |
    | 32 BD | 82.6, 86.7 → 84.7 | 107.6, 112.2 → 109.9 | **+30%** |

    The measured **+30% at both scales** exceeds iter 6's naive +14% prediction,
    confirming the DRAM-bandwidth-contention hypothesis: eliminating one of the
    two 33.5 MB host memcpys also speeds the remaining `stack` memcpy + H2D that
    fight for the same host DRAM bus. Correctness: **bit-exact, max_diff 0.0** —
    the standard gate now defaults to copy=False so it guards the real GPU path.
    Logs: `bench_mpi_sweep/ralph_tmp/cf_{1bd,32bd}_copy{True,False}_{a,b}_175942.log`,
    driver `cf_driver_175942.log`. This is the FIRST code change on the branch to
    move the throughput number. NEXT: `stack` (64.5 ms @32BD) is now the largest
    read component — its per-seg deserialize + copyto is the next CPU lever.
- [x] Per-segment H2D — skip the host `stack` memcpy (2nd landed throughput win)
  - 2026-07-10 (iter 8): added `run_gpu_bench_seg_h2d` + `--seg-h2d` to
    `bench_calib.py`. Instead of `det.raw.raw(evt,copy=False)` (host `stack`
    np.copyto into contiguous `_raw_buf`) + one big `cp.asarray` H2D, it copies
    each segment's `.raw` DIRECTLY host→device (`raw_gpu_buf[idx].set(segs[sid].raw)`,
    32× per-seg 1 MB H2Ds) into a pre-allocated device buffer — no host stack.
    Bit-exact via `test_jungfrau_calib.py --seg-h2d` (20/20 OK, max_diff 0.0).
    Bug fixed first: per-seg `.raw` is `(1,512,1024)`, so the buffer must be
    reshaped `(-1,512,1024)` to match `det.raw.raw`'s shape (else 4-D broadcast
    → all-pixel mismatch). Interleaved A/B, job 31267701 (sdfampere029),
    `mpirun --bind-to none --oversubscribe`, r47/FFB:

    | config | baseline copy=False (Hz) | --seg-h2d (Hz) | gain |
    |---|---|---|---|
    | 1 BD  | 52.0, 54.2 → 53.1  | 72.7, 74.0 → 73.4   | **+38%** |
    | 32 BD | 115.9, 113.6 → 114.8 | 141.0, 162.8 → 151.9 | **+32%** |

    Every seg-h2d bracket exceeds both baseline brackets measured seconds earlier
    (not an FFB-window artifact). Mechanism: the host `stack` memcpy is UNTIMED in
    the anchor loop (inside det.raw.raw, before the H→D timer) yet inflates wall;
    seg-h2d folds ingestion into H→D and its combined cost (1 BD 3.50 ms; 32 BD
    ~46 ms) is far below baseline's stack (64.5 ms @32BD) + H2D (~44 ms) ≈ 108 ms.
    After iter 7 killed the first 33.5 MB host memcpy (`.copy()`), this kills the
    second (`stack`) — same DRAM-bandwidth-contention mechanism, no host memcpy
    left on the raw→GPU path. Kept as opt-in `--seg-h2d` (structurally different
    loop). Logs: `bench_mpi_sweep/ralph_tmp/segh2d_{1bd,32bd}_{base,seg}_{a,b}_183300.log`,
    driver `segh2d_driver.sh`. NEXT: promote seg-h2d to the default GPU ingestion
    pattern + public-API docs, then re-`--profile` (read/stack bucket collapsed →
    `wait`/serving is likely the new largest bucket).
- [x] Re-profile the wall after both host memcpys are gone (measurement)
  - 2026-07-10 (iter 9): added `run_gpu_bench_seg_h2d_profile` + `--seg-h2d
    --profile` to `bench_calib.py` — the iter-8 seg-h2d loop with the `wait`
    bucket added (numeric path byte-identical; bit-exact 20/20, max_diff 0.0).
    Buckets wall into wait / h2d(per-seg) / kernel. Job 31267701 (sdfampere029),
    `mpirun --bind-to none --oversubscribe`, r47/FFB. Per-rank ms/event:

    | bucket | 1 BD (72.5 Hz) | 32 BD (141.5 Hz agg) |
    |---|---|---|
    | **wait** (gen advance = read + EB/MPI) | 9.96 ms (72%) | **180.5 ms (79%)** |
    | H->D (per-seg) | 3.49 ms (25%) | 45.4 ms (20%) |
    | kernel | 0.32 ms (2%) | 1.84 ms (1%) |
    | sum / wall | 13.77 / 13.80 | 227.7 / 226.1 |

    Attribution closes <1% at both scales. iter 8's prediction CONFIRMED: with
    both host memcpys gone, `wait` (psana generator advance = bigdata read +
    EB/MPI serving) is the dominant bucket and inflates hardest under contention
    (9.96→180.5 ms, 18x for 32x ranks; H->D 13x, kernel negligible). GPU-side
    work is only 21% of the 32-BD wall. Consistent with iter 3 (~60% wait then)
    and iter 5 (per-rank serialization plateau ~16 BD); iter 4 already ruled out
    raw FFB bandwidth as this ceiling. Log:
    `bench_mpi_sweep/ralph_tmp/segh2d_profile_32bd_185342.log`. NEXT: split `wait`
    into EB-handoff-blocked vs xtc-read-blocked before building any serving-chain
    change (do NOT skip this — the multi-node ceiling was mis-attributed 4x by
    skipping exactly this step).

## Documentation

- [x] Write `DEFERRED.md` with all removed features and benchmark criteria (see PLANNING.md §Future Considerations for the table)
- [x] Add brief usage example to `__init__.py` docstring showing the two-function API

## Completed Work

- 2026-07-10 (iter 12): **async-prefetch GPU overlap is a DEAD END at 32 BD —
  reverted.** Pipelined the seg-h2d per-seg H->D + kernel across CUDA streams (pinned
  staging, per-slot streams, no per-event sync, ring depth 2) so each event's ~21%
  GPU work overlaps the next event's `os.pread`. Palindrome-bracketed at 32 BD on
  FFB/r47: `--seg-h2d` 122.3/120.0 → **121.2 Hz** vs `--async-prefetch`
  116.3/119.4/117.9 → **117.9 Hz** — flat-to-**~3% slower**. Seg baseline stable
  end-to-end (no window drift), so the gap is real. Why it can't win: at 32 BD all
  ranks share one A100 + one PCIe link, so the contention-inflated H->D (~42 ms) has
  nothing uncontended to hide behind, and the required pinned-staging memcpy re-adds
  the 33.5 MB copy iter 8 removed (~3 ms) with no offsetting gain. This closes the
  **GPU-side half** of lever (b); the **read-side half** (background-thread prefetch
  of `bd_read`, the 42% component) is untested and is the real prize — a different
  change (psexp/BD-loop, all MPI stays on the main thread, only GIL-releasing
  `os.pread` moves to the reader thread). Logs `bench_mpi_sweep/ralph_tmp/async_*.log`.
- 2026-07-10 (iter 14): **Rate-capped reader probe in a COLD window — the cap didn't
  bind, the reader saturated FFB storage and still crushed the loop −63%; the probe
  cannot faithfully model prefetch.** Added `PS_READER_PROBE_MBPS` (throttles the
  background reader; benchmark-only, numeric path byte-identical). 32 BD palindrome
  bracket cap=640: baseline mean **121.0 Hz** (COLD — the iter-10/11/12 117–158 band,
  NOT iter-13's warm 585, palindrome tight 118.9→123.1), probe mean **45.0 Hz = −63%**.
  Cap did NOT bind: reader delivered only ~238 MB/s/rank (7.5–7.7 GB/s agg ≈ 7.9 GB/s
  FFB ceiling) because 32×640 = 20 GB/s ≫ ceiling — COLD, the reader hits the STORAGE
  wall (WARM iter-13 hit 35 GB/s memory-bw). **Verdict — probe methodology exhausted
  (2nd confound):** the probe reads NET-ADDITIONAL bytes (different offsets), but a real
  prefetch reads the SAME bytes earlier (zero net I/O), so in a storage-bound cold regime
  the probe double-counts I/O and OVERSTATES prefetch contention; and the 640 cap models
  a 19 Hz/rank warm consumption while the cold loop consumes only 3.8 Hz/rank = ~127
  MB/s/rank (5x lower). With the cold loop already at 4.05 GB/s = 51% of the storage
  ceiling and iter-5's per-rank serialization (not latency) bound, prefetch upside is
  bounded → CAUTION not GO. Logs `bench_mpi_sweep/ralph_tmp/probecap_*_204050.log`.
  Next: pivot to the window-independent ~27% CPU residual (dgram construction, iter 10),
  free of storage/threading/window confounds; build the real double-buffered reader only
  as deliberate flag-gated psexp surgery, not another probe.
- 2026-07-10 (iter 13): **Concurrency-headroom probe for read-side prefetch — a full
  background reader thread HALVES the main GPU-feed loop.** New `--reader-probe` in
  `bench_calib.py` runs the gated seg-h2d path + one background `os.pread` daemon
  thread per BD rank (benchmark-only, zero psexp change, numeric path byte-identical).
  32 BD palindrome bracket (seg,probe,probe,seg): baseline mean **584.6 Hz**, probe
  mean **292.6 Hz = −50%**; bracket held the window (baseline 615→553, ~10%) so the
  halving is the reader's effect. Reader pulled **35+ GB/s aggregate (>7.9 GB/s
  storage ceiling)** → cache/memory-bandwidth bound, so the contention it created is
  CPU/memory-bandwidth, not storage. **Finding 1 (solid, regime-independent):** a
  concurrent reader thread contends materially with the main loop — supports iter 12's
  "already contended" skepticism; a prefetch reader cannot run free. **Caveat (bounds
  the claim):** today's window is cache-WARM — seg-h2d baseline 585 Hz is 4–5x the
  iter-10/11/12 cold 117–158 Hz (per-rank wall 202→52 ms, NO code change → warm-cache
  artifact of re-reading the same ~7,040 events), so bd_read latency is small right
  now and the cold-regime prefetch premise couldn't be tested. Evidence leans
  CAUTION not GO. Logs `bench_mpi_sweep/ralph_tmp/probe_*_202607.log`. Next: re-run
  cold and rate-capped (`PS_READER_PROBE_MBPS≈640`, a faithful prefetch load) before
  building the core-psexp double-buffered reader.
- 2026-07-10 (iter 11): **PS_EB_NODES=1/2/4 re-measured on the seg-h2d fast path —
  EB count is a DEAD END.** At fixed 32 BD, aggregate throughput is flat across
  EB=1/2/4 (116.7 / 117.9 / 117.5 Hz, within 1% and inside the 113.9–121.0 FFB
  run-to-run spread) and eb_wait does NOT shrink with more EB ranks (66.3 → 72.0 →
  78.9 ms, if anything up). Palindrome-bracketed (1,2,4,4,2,1) to control FFB window
  drift. **Resolves iter 10's stale-flag: the iter-0 "flat PS_EB_NODES" result HOLDS
  even after the iter-7/8 memcpy removals** — the 51–79 ms eb_wait is not an
  EB-parallelism bottleneck (2/4 EB ranks serve 32 BD ranks no faster); it is the
  smd0/EB per-batch serial production upstream of the handoff, which more EB *ranks*
  don't relieve. Lever redirects to bd_read (~82–90 ms, largest wait component) via
  async prefetch. Logs `bench_mpi_sweep/ralph_tmp/ebsweep_*_192953.log`.
- 2026-07-10 (iter 15): **dgram-construction directly counted — the "27% CPU residual =
  dgram construction" hypothesis is REFUTED.** Added additive-only `total_dgram_ns`/
  `total_smdparse_ns` counters on the persistent `dm` (`run.bd_node.dm`; EventManagers are
  transient) timing `_get_next_dgrams` (per-event `dgram.Dgram()`) and `_get_offset_and_size`
  (per-batch array build); plumbed into `--wait-split`. Counted with the SAME `bd_events`
  denominator as bd_read/eb_wait (so ratios are normalization-robust): dgram = **1.07 ms
  @1BD → 4.13 ms @32BD (32-rank mean)** = **~5% of counted delivery, ~4% of the ~95 ms
  wall**, and grows only **3.9x** across 32 ranks (nearly flat per-rank = pure CPU,
  minimally contention-sensitive). iter 10's "27% residual = dgram" was a **subtractive
  artifact**: the residual `wait − eb_wait − bd_read` mixes denominators (`wait` over
  measured-L1 `n`; eb_wait/bd_read over node `bd_events > n`), inflating the leftover. The
  super-linear 32-BD delivery cost lives in bd_read (storage/cache contention) + eb_wait
  (single EB / 32 ranks), NOT dgram. **Verdict: the lighter-dgram-accessor lever is dead
  (ceiling ≤~4%).** Bit-exact (20/20 max_diff 0.0), default MPI path verified. Logs
  `bench_mpi_sweep/ralph_tmp/dgram2_1bd_210*.log`, `dgram_32bd_211018.log`. NOTE:
  `event_manager.py` edits require source→`install/` sync (same as node.py).
- 2026-07-10 (iter 10): **`wait`-bucket split — no single culprit.** At 32 BD the
  79%-of-wall delivery `wait` (166.7 ms/event) splits **42% bd_read (69.6 ms,
  os.pread off FFB) / 31% eb_wait (51.5 ms, blocked on the EB batch handoff) / 27%
  CPU residual (~45.5 ms, dgram construction + generator plumbing)**. Per-rank read
  = 481 MB/s ≈ iter-4 single-stream 490 MB/s → read-side is per-rank
  latency/serialization (bandwidth still below ceiling), lever is overlap/prefetch.
  eb_wait went 0.05 ms @1BD → 51.5 ms @32BD: the single EB rank is now a real
  serialization point — flagged the iter-0 "flat PS_EB_NODES" result as possibly
  stale (predating the iter-7/8 memcpy removals); **iter 11 re-measured it and it
  HOLDS — EB count is not the lever (see iter-11 entry above).** Tooling: additive-only
  cumulative counters on `BigDataNode` (`node.py`) + `--wait-split` in
  `bench_calib.py`. Logs `bench_mpi_sweep/ralph_tmp/waitsplit_{1bd,32bd}_190923.log`.
  NOTE: `node.py` edits require syncing source→`install/` (psana imports from
  install, not source) — pure-Python `cp` or `./build_all.sh`.
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

- 2026-07-13 (iters 18/20): **node-local EB placement (`PS_EB_NODE_LOCAL`) is a
  confirmed, compounding multi-node lever.** One EB rank per node instead of a
  single EB (on N0) serving all BD ranks. 2-node (iter-18, job 31543609):
  colocate 328.4 Hz vs 301.0 default mean, +9.1% aggregate. **3-node clean
  in-window bracket (iter-20, job 31574438, `bench_mpi_sweep/enl3v2_*.log`):**
  sn_a=193.0 Hz (32 BD, 1 node) / colocate=444.0 Hz (95 BD) / default=338.3 Hz
  (97 BD) — colocate **+31.3% aggregate, +33.8% per-rank**, and the gap *widens*
  with node count. In-window per-node efficiency: colocate 2.30× on 3 nodes
  (77%) vs default 1.75× (58%); default collapses 83%→58% (2n→3n) while colocate
  holds 90%→77%. Mechanism is **eb_wait**: a single EB serving 96 BDs serializes
  (eb_wait 70.1 ms) vs 26.3 ms with one EB/node; residual follows (136.2 vs
  85.8); **bd_read flat** (95.1 vs 95.4 — storage scales per-node, re-confirmed).
  This directly measures the "one EB rank saturating" loss the iters-16/18
  attribution predicted. Clean 4-node point remains structurally unreachable
  in-loop (parent assoc `lcls:_regular_@ampere` GrpTRES node=4 shared LCLS-wide;
  preemptable branch escapes the cap but preempts multi-node MPI jobs mid-run).
  Correctness: gated when PS_EB_NODE_LOCAL landed (iter-18); no numeric-path
  change since. NEXT: promote to documented opt-in multi-node GPU launch flag
  (NOT a silent psana-wide default — shared EB/BD dispatch, rule 4); residual per-
  node loss at 3n points at intra-node EB serialization (2nd EB/node) as the next
  lever.

- 2026-07-13 (iter 22): **intra-node EB fan-out (`PS_EB_PER_NODE`) is a confirmed
  multi-node lever that COMPOUNDS the node-local win.** New opt-in (default-off,
  gated behind PS_EB_NODE_LOCAL): sub-splits each node's shared comm into
  `PS_EB_PER_NODE` groups, each with its own EB serving ~1/N of the node's BDs
  (node.py); `_ensure_local_eb_nodes` sizes PS_EB_NODES to node_count*eb_per_node
  so the SmdReader C send-buf array (sized from that env var) doesn't OOB-segfault
  smd0. **Clean 2-node in-window bracket (job 31586774, `bench_mpi_sweep/epn2n_*.log`):**
  sn_a=221.1 Hz (32 BD) / **mn2_2eb=504.9 Hz (61 BD)** / mn2_1eb=410.6 Hz (63 BD).
  2 EB/node beats 1 EB/node **+23.0% aggregate, +27.0% per-rank (8.28 vs 6.52 Hz)
  using FEWER BD ranks**, with eb_wait −17.8% (21.16→17.39 ms) — REFUTES iter-17's
  "the residual eb_wait is pure cross-node coordination, not fixable by more
  EBs/node." residual (55.5→24.1) and bd_read (88.8→72.0) also drop: one EB/32-BD
  bunches reads+construction into contended bursts; splitting de-bunches the whole
  chain. Correctness gate PASS bit-exact (numeric path untouched); default MPI path
  (flags off) verified unbroken (32 BD, exit 0). Two starved-rank benchmark bugs
  fixed en route (bench_calib.py snap0 tuple width; per-rank 0-event ZeroDivision).
  NEXT: confirm the single-window number; sweep PS_EB_PER_NODE=1/2/3 for a knee;
  test whether it COMPOUNDS at 3 nodes like node-local placement did.

- 2026-07-13 (iter 23): **PS_EB_PER_NODE knee is at 2 — 2 EB/node is the sweet
  spot at 2 nodes; 3 and 4 EB both REGRESS.** Clean 2-node in-window knee bracket
  (job 31587292, all exit=0, `bench_mpi_sweep/epnknee_{3eb,2eb,4eb}.log`):
  mn2_2eb=**502.9 Hz** (61 BD, per-rank 8.24) / mn2_3eb=414.1 Hz (59 BD, 7.02) /
  mn2_4eb=414.1 Hz (57 BD, 7.27). The in-window 2eb (502.9) reproduces iter-22's
  504.9 Hz to <0.4% — DOUBLES as the confirm bracket (recommended step a) and
  bounds window-to-window noise as small. **Mechanism:** eb_wait falls
  MONOTONICALLY with more EBs (10.06 → 4.29 → 3.64 ms) exactly as the
  BDs-per-EB-serialization model predicts — but aggregate PEAKS at 2 EB and drops
  −17.7% at 3/4. Past 2 EB, eb_wait is no longer the binding term (already ~4 ms);
  further fan-out only sacrifices BD-reader slots (61→59→57) and inflates bd_read
  (77.4 → 102.7 → 113.4 ms), so per-rank rate falls despite the smaller eb_wait.
  So the lever is a genuine optimum, not monotone: **graduate PS_EB_PER_NODE=2 as
  the recommended 2-node GPU default (with PS_EB_NODE_LOCAL=1); do NOT go higher.**
  NEXT: does the =2 optimum COMPOUND at 3 nodes (iter-20 node-local widened
  +9%→+31% from 2n→3n), and is the knee still 2 there or does more node count
  shift it? A 3-node 1-vs-2-vs-3 EB bracket answers both.

- 2026-07-13 (iter 24, CONFIRMED — reversed-order control passed): **the
  PS_EB_PER_NODE knee GROWS with node count — 3 EB/node wins at 3 nodes; the
  2-node =2 optimum is the knee *at 2 nodes*, not a fixed constant.** Two 3-node
  in-window brackets, both all exit=0, order-robust:
  · forward (2→1→3, job 31588126, `epn3n_*.log`): mn3_2eb=411.5 / mn3_1eb=419.5 /
    mn3_3eb=**553.3** Hz (89 BD, 6.22/rank) — 3eb wins as the last (warm) phase.
  · reversed (3→2→1, job 31589010, `epn3nrev_*.log`): mn3_3eb=**802.5** Hz (89
    BD, **9.02**/rank) / mn3_2eb=648.3 (7.28) / mn3_1eb=622.9 (6.56) — 3eb wins as
    the COLD first phase, +23.8% agg / +23.8% per-rank over 2eb, +28.8% over 1eb.
  The ordering `3 > 2 > 1` is identical in both brackets → robust to phase order,
  order-confound discharged. eb_wait monotone in k (reversed: 25.19→13.97→8.94 ms).
  **Mechanism:** one smd0 feeds all node-local EBs, so more nodes = more EBs
  contending for smd batches = larger eb_wait at low k = crossover moves to higher
  k. **Emergent rule: knee ≈ node_count (2@2n, ≥3@3n).** GRADUATED into PLANNING.md
  ("Multi-node launch" → `PS_EB_PER_NODE` subsection: scale k with node count,
  start at k=node_count, do NOT exceed the knee — −18% past it at 2n). iter-23's
  2-node =2 entry stands (it is the knee at 2 nodes). NEXT: k=4 @ 3n to pin whether
  the 3-node knee is exactly 3 or higher; a 4-node k∈{2,3,4} bracket to test the
  knee≈node_count rule as a law.

Remaining (all beyond pure measurement):
1. AsyncD2HJoiner — trigger MET (DEFERRED.md updated); build when a
   production workflow needs calibrated frames back on host.
2. Storage-side levers for the ~10-12 GB/s FFB ceiling — none are psana
   code: detector-data compression at write time (~2x), node-local NVMe
   staging for reprocessing, facility conversation on FFB bandwidth/QoS.
3. Pinned-host H->D staging — secondary, only relevant once per-GPU feed
   can exceed ~270 Hz (i.e., after the storage ceiling moves).
