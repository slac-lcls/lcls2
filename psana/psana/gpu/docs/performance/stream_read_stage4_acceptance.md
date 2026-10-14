# Stage 4: stream-interleaved read performance acceptance

## Reviewed runtime

Stage 3 is committed as `c8f6b6cdf`. Review fixed partial parser-window setup
cleanup: already-created child windows are explicitly detached before raw
ownership is released; failed cleanup is quarantined for retry. Validation
passed 375 CPU tests and 29 GPU tests (job 39027254). The benchmark harness
also passed 10 tests, including bounded warm preparation and exclusion of
traced rates from the acceptance verdict.

The earlier plan called performance Stage 5. This report uses the user's
requested **Stage 4 performance acceptance** numbering. Broader retained-view,
transition, tight-budget, and long-run/scaling gates remain separate from this
minimum-reproducer comparison; see `../stream_read_refactor_cleanup.md`.

## Comparison and controls

- Previous E: frozen `ac87a93b2`, bulk off/on.
- Current E: `c8f6b6cdf`, bulk off/on, sharing the previous native dependencies.
- Real mfx101210926 run 387: 1,000 events, JF plus feespec, streams 0 and 5–9.
  The benchmark-only exclusive routing override for feespec's shared stream
  is identical across builds.
- Batch 100, depth 1, GPU budget 8 GiB, D2H chunk 0; one BD/EB/SMD0 on one A100,
  eight KvikIO fallback workers and 1 MiB KvikIO tasks.
- Two cold and two warm rounds per build/mode: 16 timing controls.
  Four additional current-build cold traces measure native read concurrency.
- Each fresh MPI run validates 200 warmup feespec arrays and three JF raw/calib
  samples. Measured timestamps, feespec GPU sum hashes, bytes and request counts
  must match. JF calibration runs during timing; full images are not exported.
- Cold: each measured file prefix <=1% resident immediately before timing;
  physical NIC RX >=98% of the 33,566,911,424-byte payload.
- Warm: explicitly read only the measured prefixes, then require each >=99%
  resident both before and after timing. Cold means node-page-cache eviction
  on Weka FFB; server-side caches are not flushed.
- Weka files must remain fully SSD-backed without object/remote backing.
  Placement and frozen code/reference hashes are checked before and after.
- Performance target: current bulk-on median loop time <= current bulk-off
  for both cache states. Traced rates do not enter this verdict. Two rounds
  are a focused comparison, not a statistical confidence interval.

Timing controls retain the lightweight existing read-counter wrapper. It runs
once per completed reader input (20 off, 10 previous on, 5,019 current on),
so the remaining overhead should be profiled with that instrumentation cost
kept explicit. These are untraced controls, not completely hook-free runs.

## Results

Job **39028351**, `sdfampere038`, 48 CPU allocation, 128 GiB host memory.
Completed successfully in **21m37s**. All **16 timing controls and four native
traces** passed validation, followed by the final placement/hash checks.
Every cold prefix was **0% resident**; every warm prefix was **100% resident**
before and after timing. The lowest cold NIC RX / payload ratio was **1.0616**.

Rates below are 1,000 divided by the median of the two loop times.

| Build | Bulk | Cold R1 / R2 events/s | Cold median rate | Warm R1 / R2 events/s | Warm median rate |
|---|---|---:|---:|---:|---:|
| Previous | off | 139.47 / 138.75 | 139.11 | 225.13 / 232.21 | 228.62 |
| Previous | on | 81.24 / 84.01 | 82.60 | 225.84 / 208.70 | 216.93 |
| Current | off | 136.21 / 141.59 | 138.85 | 239.41 / 233.92 | 236.64 |
| Current | on | 121.53 / 120.51 | 121.02 | 181.01 / 181.10 | 181.05 |

The timing target is **not met**: current on/off loop-time ratios are **1.1474
cold** and **1.3070 warm**. Current bulk on improves cold throughput **46.5%**
over previous bulk on, but reduces warm throughput **16.5%**. Current bulk-off
loop time changes +0.19% cold and -3.39% warm versus previous bulk off; these
two rounds provide no evidence of a material bulk-off regression.

### Read counts

These are psana-issued physical ranges / KvikIO API requests for 1,000 events.
Detector attribution follows the six selected streams; the feespec stream
includes its original shared-stream payload.

| Build / mode | Feespec stream | Five JF streams | Total | Reader input completions |
|---|---:|---:|---:|---:|
| Previous or current off | 1,000 | 5,000 | 6,000 | 20 |
| Previous on | 19 | 95 | 114 | 10 |
| Current on | 19 | 5,000 | 5,019 | 5,019 |

Stage 3's multi-buffer parser remains batched across groups; one reader input
completion does not imply one GPU parser launch set.

### Native read concurrency

| Current mode | Round | Single-file wall s | POSIX-active wall s | Single-file % | Native POSIX reads |
|---|---:|---:|---:|---:|---:|
| off | 1 | 0.336034 | 4.726090 | 7.11 | 38,000 |
| off | 2 | 0.311976 | 4.554927 | 6.85 | 38,000 |
| on | 1 | 0.431195 | 4.601872 | 9.37 | 37,019 |
| on | 2 | 0.340673 | 4.754291 | 7.17 | 37,019 |

Pooled percentages use the sums of numerator and denominator across rounds:
**6.98% off**, **8.25% on**. The numerator is wall time with exactly one file
in a POSIX read; the denominator is wall time with any POSIX read active.
Concurrent workers count once in wall time and idle gaps are excluded.

Both modes used eight pool workers. The caller thread also performed feespec
reads: **1,000 off**, **19 on**. JF contributed **37,000 native reads** in each
case; KvikIO splits its 5,000 psana requests into 1 MiB tasks. Thus the small
stream's coalescing really eliminates 981 API and native reads without
combining JF events into file-major bulks.

The previous bulk-on implementation's **94.79%** pooled single-file time came
from the earlier job 39009893, not this allocation; see
[the earlier cold trace baseline](jf_feespec_cold_quick_trace.md).
The new traces demonstrate that the near-exclusive single-file scheduling
pattern is gone. Matching bulk-off throughput still requires more work.

### Where the remaining time lies

Existing `read_wait_s` measures the host's future-get loop. Its sum is lower
with current bulk on, despite the longer total loop:

| Current mode | Cache | Median loop s | Median future-get block s | Loop minus that block s |
|---|---|---:|---:|---:|
| off | cold | 7.2021 | 4.5657 | 2.6364 |
| on | cold | 8.2633 | 3.6413 | 4.6220 |
| off | warm | 4.2259 | 1.6210 | 2.6048 |
| on | warm | 5.5233 | 1.1207 | 4.4026 |

The extra **1.8–2.0 seconds** outside this block is a useful next profiling
target. This subtraction is a residual, not an exclusive CPU timing block:
it includes submission, parsing, ownership work, other GPU waits, and pipeline
coordination. In particular, it does not prove which function is responsible.
Profile grouped slot selection/submission, parser setup/binding, and ownership
poll/retirement, while measuring the read-counter wrapper's contribution.

Request-to-ready **sums** overlap heavily with independent groups and must not
be interpreted as wall time. The revised native audit reports both their sum
and union, and derives file concurrency directly from POSIX start/end intervals.

### Acceptance decision

Correctness, cache, placement, provenance and trace-audit gates passed for this
minimum reproducer. **Performance acceptance failed**: current bulk on takes
14.7% longer cold and 30.7% longer warm than current bulk off. Keep the
stream-interleaved scheduling improvement and use this frozen run as the next
baseline for profiling the remaining group/pipeline overhead. Do not declare
long-run acceptance or retire the legacy correctness coverage from this sweep.

Preliminary job 39027954 was stopped after discovering that the old trace
audit assumed nonoverlapping batches. Its samples are excluded. The revised
audit uses file/range attribution and union wall time for overlapping groups;
11 CPU trace-audit tests passed. The native tracer and runtime are unchanged.

## Artifacts

Scratch campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-acceptance-stage4-20260924-v2/`

- `run.sbatch`, `acceptance.py`, helper scripts, `builds.json`, `python/`:
  frozen launch, harness and runtime.
- `job-39028351.log`: progress and failure output.
- `job-39028351/summary.md`, `summary.json`, `results.json`: updated after every
  validated sample; final `complete` flag distinguishes partial output.
- `job-39028351/provenance.json`: GPU identity, CPU affinity, placement, hashes
  and CPU reference.
- Per-case `.log`, GPU-monitor `.csv`, and cold trace `.bin`/`.json` files.

Maintained harness: `../../scripts/feespec_bulk_benchmark/acceptance.py`.
