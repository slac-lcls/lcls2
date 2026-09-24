# Single-file read concentration on the cold Weka FFB reproducer

Job **39009893**, 2026-09-24, `sdfampere030`, **COMPLETED, 8m40s, exit 0**.
Two rounds, bulk off/on, each with an untraced control and native fallback
trace: eight accepted samples in one allocation. Account `lcls:data@ampere`,
normal QoS, one A100 and one BD plus EB/SMD0.

## Finding

**The single-file concentration seen on local NVMe also appears on Weka FFB.**
Pooling the wall-time numerator and denominator across the two traces gives
**10.19% bulk off versus 94.79% bulk on**. Both modes use all eight KvikIO pool
workers. This demonstrates much less concurrency across files with bulk on;
it does not establish an inode-lock or storage-device cause.

| Bulk | Round | Exactly one file active, s | Any POSIX read active, s | Single-file percentage |
|---|---:|---:|---:|---:|
| Off | 1 | 0.457371 | 4.930019 | **9.28%** |
| Off | 2 | 0.544802 | 4.902616 | **11.11%** |
| On | 1 | 8.665415 | 9.144669 | **94.76%** |
| On | 2 | 8.693884 | 9.168984 | **94.82%** |

The percentage is `100 × exactly-one-file wall time / any-POSIX-read wall
time`, accumulated over each **1,000-event** loop. Native `CLOCK_MONOTONIC`
timestamps bracket each KvikIO-origin `pread64`. Sorting all starts/ends
partitions time by the number of distinct active file descriptors. Overlap
counts once; gaps with no POSIX read are excluded. Several workers reading
the same file still count as one file. These are six stable input handles
within a single-chunk run; per-handle byte totals match the six input extents.

Feespec does not explain this concentration: recomputing from only the five
JF files gives off **9.32% / 11.14%**, on **94.97% / 95.00%**.
The earlier JF-only NVMe values were off 1.56%, on 93.84%; that campaign used
10,000 events and batch 20, so its absolute values are not a controlled
filesystem comparison.

## Performance and native operations

| Metric | Bulk off R1 / R2 | Bulk on R1 / R2 |
|---|---:|---:|
| Control events/s | 131.49 / 133.37 | 78.34 / 80.62 |
| Control loop s | 7.6054 / 7.4978 | 12.7646 / 12.4033 |
| Trace loop s | 7.3962 / 7.4399 | 11.9243 / 11.9551 |
| Trace read request-to-ready s | 4.9377 / 4.9108 | 9.1492 / 9.1740 |
| Mean duration of a full 1 MiB POSIX call, ms | 0.989 / 0.970 | 2.143 / 2.149 |

Control rates from median loop time are **132.42 versus 79.47 events/s**;
bulk on takes **66.6% longer**. Both traces contain 32,000 full 1 MiB reads.
Total POSIX reads, matching H2D calls and existing stream waits are
**38,000 off versus 32,114 on** per sample. API requests are **6,000 versus
114**; bytes are identical at **33,566,911,424**.

The default KvikIO GDS/thread-pool threshold is 1 MiB in this environment.
The small-read shortcut executes deferred reads on the caller: **1,000 off
versus 19 on**. All eight pool workers also perform reads, giving nine observed
thread IDs. The audit distinguishes the caller from the eight pool workers.

Trace medians were 1.8% faster than controls for off and 5.1% faster for on.
This is instrumentation sensitivity plus run variation, not evidence that
tracing improves performance. Use control rates for throughput. POSIX timing
includes kernel, filesystem and scheduling delays; it is not device-only
latency or a count of physical storage commands. No CUDA synchronization was
added by tracing.

## Workload and acceptance checks

Same workload as `jf_feespec_cold_quick_baseline.md`: `mfx101210926` run 387,
JF streams 5–9 plus shared feespec s000, batch 100, depth 1, 8 GiB GPU budget,
D2H chunk 0, frozen E runtime `ac87a93b2`, KvikIO 24.08.02 CPU fallback,
eight pool workers and 1 MiB tasks. Both cases retain JF GPU calibration and
per-event feespec GPU sum; the global bulk switch affects both detectors.

- **Every one of the six measured prefixes was 0% page resident before all
  eight timed loops**, verified with mincore after eviction. Bounds cover the
  first 1,000 events, not the larger staged-file denominator.
- Physical NIC RX was **36.32–37.04 GB** per sample, above the required 98%
  of input bytes. Counters include filesystem overhead and background traffic.
- All twelve data/SMD files had complete Weka SSD coverage and zero
  object/remote backing before and after. Server-side caches were not flushed.
- Each sample checked all 200 warmup feespec arrays and three JF raw/calib
  samples. All measured timestamps and feespec GPU-sum hashes matched.
- All four traces passed complete read/copy/wait triplet, successful return,
  byte, operation-count, interval, six-file and eight-pool-worker audits.
- Frozen build, calibration, script hashes and process placement passed.

## Changes and artifacts

Only benchmark instrumentation and analysis changed: optional trace support
in `feespec_bulk_benchmark/bench.py` and `quick.py`, a trace summary, generic
file-overlap accounting, caller-thread provenance, and README instructions.
Seven trace-accounting tests and seven harness tests passed. Production
runtime scheduling is unchanged.

Scratch campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-feespec-cold1k-trace-20260924`.
`job-39009893/` contains `summary.md`, `control-summary.md`, their JSON,
`results.json`, provenance, manifests, references, per-case logs and GPU CSVs,
four native `.bin`/metadata/`.audit.json` sets, and `access-details.json`.
`analyze_access.py` reproduces the per-file-byte and same-size-read analysis.
`staging.json` records frozen scripts/C/shared-library hashes. The workspace
symlink is `validation/jf-feespec-cold1k-trace-20260924`.

Excluded setup attempts: 39009567 stopped before measurements because a
staging helper was omitted; 39009692 was cancelled to correct the audit's
eight-total-thread assumption for small reads. Neither contributes samples.
