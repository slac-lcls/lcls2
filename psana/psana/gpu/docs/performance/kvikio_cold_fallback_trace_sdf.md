# Cold JF KvikIO fallback trace

Completed in Slurm job **38923994**, two cold rounds in one allocation.
All eight measured samples and both traced CPU/GPU pixel preflights passed.

## Findings

Bulk-on control loop time was **106.79 s**, versus **73.77 s** bulk-off (**44.8% longer**). The traced read request-to-ready windows increased from **64.67 s to 94.61 s**. That increase accounts for **96.9%** of the traced loop-time difference.

The discrepancy is concentrated in the POSIX fallback read stage. Bulk-on makes fewer actual reads and H2D transfers, but its full 1 MiB reads take longer. H2D submission worker time stays small, and summed stream-wait time decreases.

The trace identifies a strong access-order lead: bulk-on concentrates concurrent reads on a single file, while bulk-off interleaves several files. This establishes the changed access pattern and the slower POSIX calls; it does not isolate the kernel/filesystem mechanism causing the latency change.

All times below are accumulated over **10,000 events / 500 reader batches**, then reported as the median of two runs. Counts are per run and identical across rounds.

## Actual operations

| Metric | Bulk off | Bulk on |
|---|---:|---:|
| Psana reader batches | 500 | 500 |
| KvikIO API pread requests | 50,000 | 2,885 |
| Actual POSIX pread64 calls | 370,000 | 322,885 |
| Actual H2D calls | 370,000 | 322,885 |
| Existing fallback stream waits | 370,000 | 322,885 |
| Full 1 MiB POSIX reads | 320,000 | 320,000 |
| Small tail POSIX reads | 50,000 | 2,885 |
| Bytes read and copied | 335,571,760,000 | 335,571,760,000 |

## Read-path timing

The following three rows partition the read request-to-ready interval.

| Read-window block, seconds | Bulk off | Bulk on |
|---|---:|---:|
| Request submission | 0.609 | 0.144 |
| Host work between submission and wait | 0.432 | 0.007 |
| Read completion wait | 63.632 | 94.459 |
| Total request-to-ready | 64.673 | 94.610 |

The next table partitions that same interval by active native operations. Overlap is counted once; these rows can be added.

| Exclusive wall occupancy, seconds | Bulk off | Bulk on |
|---|---:|---:|
| POSIX calls only | 39.200 | 73.247 |
| POSIX and H2D/wait overlap | 25.299 | 21.213 |
| H2D/wait only | 0.050 | 0.034 |
| Neither operation active | 0.124 | 0.116 |

The existing nested BD timers also retain reader preparation, submission, and wait scopes separately from parser/calibration work. Preparation/bookkeeping below is total `read.submit` minus its `read.pread_loop` child and mostly precedes the first read request.

| BD host block, seconds | Bulk off | Bulk on |
|---|---:|---:|
| Reader preparation/bookkeeping | 0.509 | 1.012 |
| KvikIO submission loop | 0.594 | 0.135 |
| Reader wait | 63.610 | 94.439 |

## Worker durations and file access

Worker sums overlap across eight threads and must not be added to the wall-time tables.

| Summed worker host seconds | Bulk off | Bulk on |
|---|---:|---:|
| POSIX read | 470.416 | 717.016 |
| H2D API call | 2.264 | 2.523 |
| Existing stream wait | 33.366 | 26.558 |

| Same-size reads / file concurrency | Bulk off | Bulk on |
|---|---:|---:|
| 1 MiB mean latency, ms | 1.175 | 2.228 |
| 1 MiB median latency, ms | 0.221 | 2.220 |
| 1 MiB p95 latency, ms | 4.479 | 3.326 |
| POSIX-active time with exactly one file, % | 1.56 | 93.84 |

KvikIO `parallel_io` queues the tasks of one API request before returning. Psana submits the next range afterward. Larger coalesced per-file requests therefore produce longer runs of tasks for one file. Both modes still use all eight workers. A focused next experiment is to preserve bulk input ownership while interleaving bounded read ranges across files. This campaign makes no runtime scheduling changes.

## Controls and repeatability

| Variant | Control R1 / R2 (s) | Trace R1 / R2 (s) | Median trace − control |
|---|---:|---:|---:|
| Integrated-off | 73.824 / 73.714 | 75.326 / 78.947 | +3.368 s (+4.57%) |
| Integrated-on | 107.387 / 106.198 | 108.613 / 107.448 | +1.238 s (+1.16%) |

These differences measure instrumentation sensitivity plus run variability, not pure hook overhead. Measured pre-run cache residency was **0% to 0.00000245%** (at most two 4 KiB pages across the staged files). No operation errors, short reads, unmatched triplets, byte mismatches, or trace overflows occurred in accepted samples.

**Excluded attempt:** the first round-2 bulk-off control stalled in MPI before completing. A native stack showed the BD in `MPI_Probe`; KvikIO workers were idle. Its log, resource samples, and process snapshot are preserved under `job-38923994/excluded/`. That incomplete attempt contributes no timing sample. The final control/trace pair was rerun using unchanged launch/audit functions in the same allocation, on the same GPU, CPU mask, and staged files. `continuation.log` records the retry. The allocation is released after the continuation and audits finish. Slurm shows CANCELLED because the held original controller was explicitly released; the continuation step completed with exit code 0.

![Cold read-path wall time and same-size latency](/sdf/scratch/users/m/monarin/gpu-validation/kvikio-cold-trace-20260923/job-38923994/cold-read-trace.png)

## Workload and capture

- JF only, `mfx101210926` run 387, streams 5–9, 10,000 events per sample.
- One A100 and one BD, three MPI ranks; batch 20, depth 1, budget 8 GiB.
- No automatic/user D2H in measured loops. Fresh MPI processes and 100-event
  warmup per sample. CPU-reference raw/calibrated pixel preflights run with
  native tracing enabled for both variants.
- KvikIO 24.08.02, CPU fallback ON, eight workers, 1 MiB task size.
- Two rounds of integrated bulk off/on with tracing enabled/disabled,
  in one allocation on `sdfampere026`. The second round reverses variant
  and tracing order. Each cold sample verifies at most 1% page residency
  on private local NVMe staged files immediately before timing.
- Runtime is the frozen integrated install used in the previous timing
  campaign. Runtime module hashes, dataset timestamps/bytes, GPU UUID,
  affinity, and resource samples are retained in the campaign directory.

The trace captures each KvikIO-origin `pread64`, each corresponding
`cuMemcpyHtoDAsync_v2`, and its existing `cuStreamSynchronize`. A benchmark-only
preload intercepts POSIX calls. Two process-local function pointers in KvikIO's
CUDA shim are replaced after checking their ABI against the installed CUDA
symbols. Installed psana/KvikIO files are unchanged. No CUDA synchronization
or events are added. Records are buffered in 128 MB of preallocated host RAM
and written after the timed loop has drained.

Every read/copy/wait triplet is matched by thread, batch, descriptor, offset,
and size. Audits check successful full reads, successful CUDA calls, byte
conservation, expected task counts, complete batch windows, and trace capacity.
The recorded unit is a POSIX fallback operation; physical NVMe commands and
kernel readahead operations are outside this capture.

## Timing definitions

- **Read request-to-ready:** from `PendingBatch.issued_ns` immediately before
  the request loop through return from the reader wait. It excludes read
  planning/buffer preparation and subsequent parser/gather/calibration work.
  Submission, intervening host work, and completion wait partition this window.
- **Exclusive BD host phases:** the existing nested host recorder subtracts
  child time. Its reader preparation, submission, and wait measurements are
  reported separately from parser, calibration, allocation, and other work.
- **Disjoint read-window occupancy:** POSIX calls only, H2D/wait calls only,
  both active on different workers, or neither. These categories partition
  the read request-to-ready wall time.
- **Worker sums:** total host duration across eight workers. These overlap
  and cannot be added to wall time. H2D API and stream-wait measurements are
  host timings, not CUDA device copy durations.
- **File concurrency:** number of different file descriptors with a POSIX
  read in progress, measured only while at least one POSIX read is active.

## Reproduction and artifacts

Campaign root:
`/sdf/scratch/users/m/monarin/gpu-validation/kvikio-cold-trace-20260923`.

The frozen `run.sbatch`, `run_baseline.py`, and `bench.py` retain the complete
launch/staging/cache protocol. `native-trace-hashes.json` identifies the C
source, shared object, and installed KvikIO headers. `job-38923994/primary/`
contains logs, 64-byte native binary records, request/batch metadata, and
per-trace audits. `summarize_fallback.py` describes the binary NumPy dtype and
generates the audited comparison. `analyze_access.py` provides file concurrency
and equal-size read latency analysis.

Repository additions are diagnostic scripts only:
`kvikio_fallback_trace.c`, `kvikio_fallback_trace.py`,
`summarize_kvikio_fallback.py`, and six CPU trace-accounting tests.
Those tests and the ten existing phase-recorder tests pass.
