# psana2 GPU performance summary — Jungfrau, single S3DF node

## Executive summary

This note consolidates the accepted Jungfrau GPU measurements made from the
`codex/psana2-gpu-two-phase-retire` branch on 2026-08-27 through 2026-08-29.
The jobs ran from base commit
`ca2345b75eabd4aaefd9f4c849447ebb03b093d2` plus the then-uncommitted GPU-only
event metadata and MPI transport fixes recorded in their logs. Those exact
fixes are now committed as
`e18cf6bb700152c461c2884bbfa28014d8d6e006` (`Fix GPU-only events and MPI batch
transport`). Use that commit as the reproducible benchmark code snapshot.

The main results are:

- For node-local cold NVMe, one A100 with four BigData (BD) ranks reaches
  **9.197 GB/s (274.1 events/s)**. More GPUs do not improve the approximately
  9.2-GB/s storage/fallback plateau.
- With all input resident in the CPU DRAM page cache, four A100s with eight BDs
  reach **26.153 GB/s (779.4 events/s)**, or 209.2 Gbit/s of Jungfrau XTC
  payload. This is an application-capacity result, not a network benchmark.
- The selected per-BD settings are eight KvikIO workers, 1-MiB KvikIO tasks,
  `batch_size=20`, and `pool_depth=1`.
- The benchmark runs GPU read and Jungfrau calibration, but the user loop only
  reads `ctx.timestamp`. It calls no detector method and performs no D2H copy.

## Hardware, data, and rate definition

The dataset is `mfx101210926` run 387, Jungfrau streams 005–009. It is distinct
from the pixel-exact acceptance dataset `mfx100848724` run 51.

All measurements in this note read staged data under node-local `/lscratch`:

```text
/lscratch -> XFS on /dev/md0
/dev/md0  -> RAID0 over two KIOXIA KCD6XLUL3T84 3.84-TB NVMe SSDs
```

The exact Jungfrau payload is 33,557,176 bytes/event. The tuning sweep used
1,000 events (33,557,176,000 bytes); the sustained scale matrix used 10,000
events (335,571,760,000 bytes). Payload bandwidth is exact payload bytes divided
by event-loop time. DataSource construction, calibration setup, staging, cache
preparation, and cleanup are excluded from the rate.

| Measurement | Node and allocation | Input window |
| --- | --- | --- |
| Batch/pool/1–8-worker tuning, job `36101699` | shared `sdfampere040`, one A100 | 1,000 events, cold |
| Task-size and bottleneck profiles, jobs `36183869` and `36185349` | shared allocation on `sdfampere032`, one A100 | 1,000 events, cold unless marked warm |
| Sustained cold scale, job `36329902` | exclusive `sdfampere032`, four A100-SXM4-40GB GPUs, 112 CPUs | 10,000 events |
| Sustained warm scale, job `36346847` | exclusive `sdfampere032`, four A100-SXM4-40GB GPUs, 112 CPUs | 10,000 events |

KvikIO 24.08.02 reported `is_gds_available=False`; all application results use
compatibility mode. The cold timed path is:

```text
NVMe RAID -> Linux/CPU DRAM -> KvikIO host bounce -> A100 VRAM
          -> Jungfrau raw gather -> fused GPU calibration -> GPU result
```

The warm timed path starts from the page cache:

```text
CPU DRAM page cache -> KvikIO host bounce -> A100 VRAM
                    -> Jungfrau raw gather -> fused GPU calibration -> GPU result
```

Warm input does not eliminate CPU-to-GPU PCIe traffic. It eliminates the timed
SSD read. No result is copied back to CPU because `gpu_d2h_chunk_size=0`.

## Single-BD tuning

These short, mostly single-repetition measurements are intended to select
parameters, not to establish the final sustained capacity. The tuning jobs ran
on shared nodes, so differences of only a few percent should not be treated as
statistically significant.

### KvikIO worker count

This controlled probe holds `batch_size=10`, `pool_depth=2`, and the 4-MiB
KvikIO task size fixed. Every case begins after `sync` plus
`POSIX_FADV_DONTNEED` on all five BigData files.

| KvikIO workers | Loop time (s) | Rate (events/s) | Payload (GB/s) | Relative to 1 worker |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 28.40 | 35.2 | 1.182 | 1.00x |
| 2 | 16.02 | 62.4 | 2.095 | 1.77x |
| 4 | 12.36 | 80.9 | 2.715 | 2.30x |
| 8 | 10.09 | 99.1 | 3.326 | 2.81x |

Compatibility-mode I/O needs concurrency. A follow-up at
`batch_size=20/pool_depth=1` did not show a stable advantage for 16 workers:
8 workers/1 MiB produced 3.60–3.94 GB/s across two runs, while 16 workers/1 MiB
produced 3.73 GB/s. Eight workers were retained to avoid doubling the worker
count per BD for an unproven gain.

### Batch size and pool depth

All points below use eight KvikIO workers and the 4-MiB task size. Each cell is
`events/s (GB/s)` for the same cold 1,000-event payload.

| Batch size | Pool 1 | Pool 2 | Pool 4 |
| ---: | ---: | ---: | ---: |
| 1 | 89.3 (2.996) | 80.9 (2.715) | 84.3 (2.829) |
| 5 | 102.1 (3.428) | 98.1 (3.293) | 96.7 (3.245) |
| 10 | 101.5 (3.407) | 99.1 (3.326) | 101.4 (3.403) |
| 20 | **103.2 (3.463)** | 101.0 (3.390) | 71.5 (2.399) |

Batch sizes 5–20 amortize fixed batch/orchestration work. Extra pool slots do
not help this implementation and multiply reusable raw, gather, and calibrated
output buffers. `batch_size=20`, `pool_depth=1` is the best measured point and
the memory-efficient choice. The pool-4/batch-20 slowdown is only one shared
node observation; it is evidence against choosing that setting, not a precise
estimate of its penalty.

### KvikIO task size

This application-level task-size probe holds eight workers,
`batch_size=20`, and `pool_depth=1` fixed on `sdfampere032`.

| Task size (MiB) | Loop time (s) | Rate (events/s) | Payload (GB/s) |
| ---: | ---: | ---: | ---: |
| **1** | **8.518** | **117.4** | **3.940** |
| 4 | 9.138 | 109.4 | 3.672 |
| 16 | 9.043 | 110.6 | 3.711 |
| 64 | 8.696 | 115.0 | 3.859 |

One MiB was best observed and was used for the 10,000-event matrix. The task
sizes are close enough that the ranking should be confirmed on an exclusive
node before changing a global default. The independent read-only KvikIO control
also favored 1 MiB: 6.252 GB/s versus 5.238, 6.057, and 5.819 GB/s for 4, 16,
and 64 MiB respectively at eight workers.

The later 10,000-event single-BD cold median is 174.5 events/s or 5.857 GB/s,
higher than the short tuning points because the 52–64-second observation
amortizes startup and reaches a steadier read pipeline. Use the short sweep to
choose relative settings and the long matrix for capacity planning.

## Sustained single-node scaling

Both matrices use the selected configuration:

```text
KVIKIO_NTHREADS=8
KVIKIO_TASK_SIZE=1 MiB
batch_size=20
pool_depth=1
gpu_d2h_chunk_size=0
```

Every point is the median of two fresh MPI processes. Each process returns
exactly 10,000 events and unique timestamps and reports exactly
335,571,760,000 bytes read. Physical-GPU occupancy was checked independently
with `nvidia-smi`.

For cold cases, the runner calls `sync` and then
`os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on every complete staged
prefix before the process starts. For warm cases, `mincore` verified zero
resident pages after the initial eviction and exactly 100% residency before
and after every timed case.

| GPUs | BDs | Cold rate (events/s) | Cold GB/s | Warm rate (events/s) | Warm GB/s | Warm/cold |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 174.5 | 5.857 | 339.2 | 11.383 | 1.94x |
| 1 | 2 | 247.5 | 8.307 | 417.3 | 14.002 | 1.69x |
| **1** | **4** | **274.1** | **9.197** | 408.9 | 13.721 | 1.49x |
| 1 | 6 | 272.4 | 9.140 | **445.3** | **14.942** | 1.63x |
| 1 | 8 | 271.6 | 9.115 | 403.1 | 13.528 | 1.48x |
| 2 | 2 | 263.7 | 8.849 | 549.8 | 18.448 | 2.08x |
| **2** | **4** | **276.7** | **9.286** | **635.0** | **21.310** | 2.29x |
| 2 | 6 | 274.2 | 9.202 | 614.8 | 20.630 | 2.24x |
| 4 | 4 | 268.5 | 9.011 | 683.1 | 22.923 | 2.54x |
| **4** | **8** | **269.5** | **9.043** | **779.4** | **26.153** | **2.89x** |
| 4 | 12 | 264.8 | 8.885 | 693.3 | 23.267 | 2.62x |

For cold `/lscratch`, four BDs supply enough parallel fallback reads to reach
the node plateau. The best cold median, 9.286 GB/s, is 79.3% of the measured
11.714-GB/s four-reader `O_DIRECT` control and 74.9% of the two-drive
12.4-GB/s vendor sequential-read sum. Adding GPUs cannot accelerate that
shared source. One GPU/four BDs is the efficient cold configuration; it uses
about 17.4 GiB at peak, while one GPU/eight BDs reaches about 34.5 GiB without
adding throughput.

Warm DRAM input exposes compute/host-pipeline scaling hidden by NVMe. The best
one-, two-, and four-GPU medians are 14.942, 21.310, and 26.153 GB/s. Four
GPUs/eight BDs therefore sustain 104.6% of a 25-GB/s, 200-Gbit/s payload rate.
That does not prove an end-to-end 200-Gbit/s network/filesystem path: real
ingestion would compete for CPU, DRAM bandwidth, PCIe, NUMA links, and KvikIO
workers while filling the page cache.

## Code path and measured bottlenecks

The relevant branch path is:

```text
SMD0 -> EventBuilder -> BD request/response -> GpuEvents
     -> GPUBAT1 descriptor validation and splitting
     -> KvikioGpuReader.issue_batch()
          one CuFile.pread future per event/stream descriptor
     -> wait for futures
     -> GPU raw gather/compaction -> fused Jungfrau calibration
     -> timestamp join -> yield GpuEventContext
```

The branch already sends a look-ahead BD request before processing the current
GPU batch, allowing EventBuilder to prepare the next batch concurrently. It
also rotates among BDs whose requests are waiting. These mechanisms help, but
do not guarantee perfectly balanced totals.

Measured bottlenecks and non-bottlenecks are:

- **Cold storage/fallback path:** one cold BD spends 41.10–52.45 seconds waiting
  for KvikIO I/O in a 52.04–63.72-second loop. Four BDs reduce the loop to
  about 36.5 seconds, after which more BDs or GPUs do not improve bandwidth.
- **Per-descriptor host work:** at `batch_size=20`, five streams create roughly
  100 KvikIO futures per batch—about 5,000 for 1,000 events. In the profiled
  8-worker/1-MiB run, issue-to-completion latency was 4.12 seconds, GPU submit
  calls used 0.87 seconds, and about 3.3 seconds remained in descriptor,
  MPI/batch, timestamp-join, and Python event-yield work.
- **Calibration is not the cold limiter:** slot retirement synchronization was
  below one millisecond in the profile. Additional physical GPUs do not raise
  cold throughput.
- **SMD0/EventBuilder construction is not the present ceiling:** the separate
  BD-bypass control sustained a median 13,064.5 events/s, far above the full
  pipeline's 779.4-events/s warm maximum. It excludes EB-to-BD sends, so it
  does not clear the complete transport path, but it rules out SMD alignment
  and batch construction as the primary measured bottleneck.
- **Warm load balance still varies:** the two four-GPU/eight-BD runs span
  24.782–27.525 GB/s. In the slower four-GPU/four-BD repetition, individual
  BDs received 2,060–3,580 events, a 1.74x max/min imbalance.
- **GPU memory limits per-GPU BD scaling:** ten and twelve BDs on one 40-GB
  A100 failed allocation in the earlier sweep. Slot count is not a complete
  byte budget, and independent BD processes still need aggregate GPU fairness.
- **D2H was intentionally absent:** these rates apply when calibrated data
  remain on the GPU. Historical synchronous D2H of every 64-MiB Jungfrau result
  reduced a one-BD GPU run from 100.3 to 39.6 events/s. CPU-result workflows
  need a separate bounded asynchronous-D2H capacity measurement.

## Further improvements and trade-offs

1. **Keep scaling as the operational baseline.** For the current two-NVMe
   source, one GPU/four BDs obtains essentially the best node rate with no
   invasive I/O refactor. For a DRAM-speed or future fast source, four
   GPUs/eight BDs already pass 200 Gbit/s. This has the highest confidence and
   zero code complexity.
2. **Coalesce reads by stream inside one BD.** `KvikioGpuReader` currently
   creates one future per descriptor. Combining adjacent or acceptably close
   file extents into stream-major reads could reduce futures, validation,
   bounce-buffer setup, and Python work. This is a medium/high-complexity
   efficiency change: descriptor-to-output mapping and buffer lifetime must be
   preserved, and reading gaps can waste bandwidth. It is unlikely to raise
   current cold node throughput beyond the NVMe ceiling, but it could let fewer
   BDs reach that ceiling and help a future faster source.
3. **Refine BD line-up and work-aware dispatch.** Keep the existing eager
   request and waiting-BD rotation, then test an initial all-BD line-up/credit
   barrier or select using the read/process statistics already carried in BD
   requests. This is moderate complexity and is most likely to reduce warm-run
   variance and tail time rather than move the cold storage plateau.
4. **Set a node-wide KvikIO/NUMA budget.** Eight workers per BD means 32 workers
   at the useful cold four-BD point and 64 at the warm four-GPU/eight-BD point.
   Pin BD processes, workers, and host buffers near their NVMe/GPU NUMA domains
   and sweep total workers, not just workers per rank. This is a relatively
   low-code experiment before changing the reader.
5. **Enable and verify true cuFile/GDS.** GDS would bypass the CPU bounce path.
   It cannot exceed the current two-drive device ceiling, so its local benefit
   is CPU/DRAM efficiency and possibly closing the remaining `O_DIRECT` gap.
   Its larger value should be judged with a source capable of more than
   12.4 GB/s. Always verify `DriverProperties().is_gds_available`; do not infer
   GDS from using KvikIO.
6. **Benchmark the branch's bounded D2H path with a production result.** The
   no-D2H matrix is an upper bound for workflows that need CPU-visible output.
   Measure full calibrated D2H and, preferably, a compact downstream GPU result
   using pinned buffers and explicit completion ownership. This may matter more
   than further raw-input tuning for real analysis pipelines.
7. **Treat fused raw gather as lower priority.** A fused strided calibration
   kernel could remove the multi-segment D2D compaction, but calibration/retire
   time is not the measured cold bottleneck. Revisit this only after I/O and
   host orchestration are reduced or when profiling a faster source.
8. **Add repetitions and hardware counters before changing defaults.** The
   sustained matrix has two repetitions, but the tuning matrix mostly has one
   on shared nodes. Repeat promising 8-vs-16-worker and 1-vs-64-MiB points on an
   exclusive node with CPU/NUMA binding, PCIe counters, CPU utilization, and
   KvikIO worker occupancy.

## Short reproduction procedure

Run from the same worktree with its local build installed. Confirm the source
snapshot first:

```bash
cd /sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline
git rev-parse HEAD
git status --short
source setup_env.sh
```

The benchmark implementation commit is:

```text
e18cf6bb700152c461c2884bbfa28014d8d6e006
```

A later documentation-only commit may be at the branch tip; verify that
`e18cf6bb7` is an ancestor when reproducing from that tip.

The runners stage the five SMD files and required BigData prefixes into a
job-specific `/lscratch/monarin` directory and verify the KvikIO/GPU
configuration. The scale runners guard cleanup with an exact path check; the
profile job records `keep_stage=` and its allocation epilog removed the stage
in the accepted run.

Single-BD batch/pool/worker sweep:

```bash
sbatch notes/jungfrau_gpu_sweep_20260827/run_gpu_bs_pd_cold.sh
python3 notes/jungfrau_gpu_sweep_20260827/summarize_gpu_sweep.py \
  notes/jungfrau_gpu_sweep_20260827 --job JOB_ID
```

Single-BD task-size and bottleneck controls:

```bash
sbatch notes/jungfrau_1bd_bottleneck_20260828/run_1bd_profiles.sh
python3 notes/jungfrau_1bd_bottleneck_20260828/summarize.py \
  notes/jungfrau_1bd_bottleneck_20260828/gpu_*_JOB_ID.log \
  notes/jungfrau_1bd_bottleneck_20260828/kvikio_*_JOB_ID.log
```

Exclusive-node cold and warm scale matrices:

```bash
sbatch notes/jungfrau_gpu_scale_20260828/run_gpu_scale_10k.sh
sbatch notes/jungfrau_gpu_scale_20260828/run_gpu_scale_warm_10k.sh

python3 notes/jungfrau_gpu_scale_20260828/summarize_10k.py --complete \
  notes/jungfrau_gpu_scale_20260828/scale10k_COLD_JOB_ID.log
python3 notes/jungfrau_gpu_scale_20260828/summarize_warm_10k.py --complete \
  notes/jungfrau_gpu_scale_20260828/warm10k_WARM_JOB_ID.log
```

Submit cold and warm jobs sequentially if both scripts target the same fixed
node. Check for `22` validated records, zero initial warm resident pages,
100% pre/post warm residency, successful physical-GPU checks, and
`removed_stage=` in each accepted transcript.

## Accepted artifacts

- Single-BD tuning report and job `36101699`:
  `notes/jungfrau_gpu_sweep_20260827/README.md`.
- Single-BD controls and jobs `36183869`/`36185349`:
  `notes/jungfrau_1bd_bottleneck_20260828/README.md`.
- Cold/warm scale report and jobs `36329902`/`36346847`:
  `notes/jungfrau_gpu_scale_20260828/README.md`.
- Strict scale validators: `summarize_10k.py` and
  `summarize_warm_10k.py` in the scale artifact directory.
