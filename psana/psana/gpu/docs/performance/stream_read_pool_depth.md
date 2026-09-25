# Current bulk off/on: execution pool depth 1 versus 2

## Question and controlled configuration

Does increasing `n_gpu_streams` from 1 to 2 improve I/O/compute overlap after
the stream-interleaved read refactor? Both cases use frozen runtime
`c8f6b6cdf`, unchanged from Stage 4. KvikIO still has eight workers and the
controller admits one subbatch's read set ahead of processing. Depth 2 adds
an execution slot/CUDA stream, not a second KvikIO worker pool.

Real mfx101210926 run 387, first 1,000 JF+feespec events; batch 100, budget
8 GiB, D2H chunk 0, one BD/EB/SMD0, CPU fallback with 1 MiB KvikIO tasks.
Use the same benchmark-only feespec shared-stream routing override, private
Weka FFB input, frozen JF constants and CPU references as Stage 4.

Job **39034776**, `sdfampere012`: one A100, 48 CPUs, 128 GiB host memory,
account `lcls:data`, normal QoS. Matrix:

1. Four separate warm pipeline diagnostics: bulk off/on at depths 1/2.
2. Sixteen controls: those four cases, cold/warm, two rounds with reversed
   ordering in round 2.
3. Eight cold native traces: those four cases, two reversed-order rounds.

Controls retain the lightweight read-counter wrapper. Native tracing and
pipeline diagnostics are excluded from throughput comparisons. Each sample
checks 200 warmup feespec arrays and three JF raw/calib samples; the measured
loop checks all timestamps, feespec GPU sums and payload/request totals.
Cold requires <=1% node-page residency per prefix plus physical NIC RX >=98%
of payload. Warm requires >=99% residency before and after timing. Weka tier
and immutable code/reference hashes must pass before/after checks.

## Measurements and interpretation

- Rates are 1,000 / median loop time across the two untraced controls.
- Single-file % is exactly-one-file POSIX-active wall / any-POSIX-active wall.
- No-POSIX-read time is BD loop duration minus the union of native POSIX read
  intervals. It includes H2D, useful computation, setup and tail drain; it is
  not a GPU-idle metric. Report seconds and percentage alongside throughput.
- Separate diagnostics record execution subbatch sizes and selected launch
  counts: walker, locator initialization, field location, JF gather and JF
  calibration. These are not all kernels in the program. JF calibration is
  still per event; parsing and gathering are batched.
- Peak charged bytes track every successful allocation reservation, including
  owned cached backing. They exclude allocator cache, CUDA context and unowned
  user allocations. GPU-monitor CSVs provide separate device-memory samples.
- The same 8 GiB budget can shrink execution subbatches when depth increases,
  increasing launch/setup counts. This effect is part of this comparison.

## Results

Job completed successfully in **34m43s**, exit 0. All **28 samples** and final
placement/code/reference checks passed. Maximum cold prefix residency was
below **0.001%**, minimum cold NIC RX was **107.42%** of payload, and all warm
prefixes were **100% resident** before/after. All 12 input/SMD files passed
the before/after SSD-only Weka tier check.

Rates are events/s; the median-time rate is not the arithmetic mean of the
two rates.

| Bulk | Depth | Cold R1 / R2 | Cold median-time rate | Warm R1 / R2 | Warm median-time rate |
|---|---:|---:|---:|---:|---:|
| off | 1 | 132.29 / 132.32 | 132.31 | 85.02 / 80.01 | 82.44 |
| off | 2 | 135.20 / 137.04 | 136.11 | 218.15 / 227.11 | 222.54 |
| on | 1 | 115.41 / 117.94 | 116.66 | 173.13 / 97.84 | 125.03 |
| on | 2 | 119.29 / 115.98 | 117.61 | 181.76 / 174.26 | 177.93 |

Cold improvement at depth 2 is **2.9% off**, **0.8% on**. The latter is smaller
than the round-to-round variation and is not compelling evidence of a gain.

### Warm behavior needs further profiling

Depth 2 was faster in this allocation, but the size of the warm difference
requires care. Depth-1 bulk off was slow in both rounds, while depth-1 bulk on
varied substantially. No sample was discarded. The earlier Stage 4 allocation
reported 236.64 events/s for depth-1 bulk off; its much faster result prevents
generalizing this allocation's large warm ratio as an established depth effect.

All warm prefixes were 100% resident before/after. In the slow depth-1 off
runs, NIC RX was only 71–75 MB versus the 33.6 GB payload. The slowdown is
outside the existing future-get read-wait block:

| Bulk | Depth | Median warm loop s | Future-get block s | Loop minus block s |
|---|---:|---:|---:|---:|
| off | 1 | 12.1301 | 1.8730 | 10.2571 |
| off | 2 | 4.4936 | 1.8481 | 2.6454 |
| on | 1 | 7.9983 | 0.9161 | 7.0822 |
| on | 2 | 5.6202 | 0.5294 | 5.0908 |

This residual includes setup within the loop, GPU submission/waits, ownership
and CPU coordination. Its cause is not established. Neither cache evidence
nor sampled GPU clocks identifies the responsible block. Do not attribute
this warm difference to improved file-read concurrency from the cold traces.

### Cold native traces

All eight traces passed exact range/byte/triplet attribution and interval
audits. Every case used eight KvikIO pool workers. API requests were 6,000
off and 5,019 on at both depths; native POSIX reads were 38,000 off and 37,019
on. Depth did not change the physical read grouping.

| Bulk | Depth | Single-file % R1 / R2 | Pooled single-file % | Mean POSIX-active s | Mean no-POSIX-read s | Pooled no-POSIX-read % of BD loop |
|---|---:|---:|---:|---:|---:|---:|
| off | 1 | 7.05 / 6.40 | 6.73 | 4.8307 | 2.5773 | 34.79 |
| off | 2 | 7.18 / 6.97 | 7.07 | 4.9060 | 2.5754 | 34.42 |
| on | 1 | 9.03 / 7.04 | 8.03 | 5.0939 | 3.7347 | 42.30 |
| on | 2 | 8.35 / 10.08 | 9.22 | 4.8000 | 3.8252 | 44.35 |

Pooled percentages sum their wall-time numerators and denominators across
rounds. Depth 2 did not improve the measured file-concurrency percentage or
materially reduce time without POSIX reads. Bulk-off gaps are essentially
unchanged; bulk-on gaps increase slightly. Trace rates are excluded from the
control-rate table above, and trace/control variability remains visible in
the raw logs. No warm native traces were collected.

### Separate pipeline diagnostics

All four 1,000-event diagnostics passed. Calibration launched 1,000 times in
every case. Each parser phase (walk/init/locate) and JF gather launched once
per execution subbatch:

| Bulk | Depth | Subbatches | Peak charged MiB | Allocation reservations |
|---|---:|---:|---:|---:|
| off | 1 | 20 | 7,302.275 | 28 |
| off | 2 | 40 | 7,302.277 | 82 |
| on | 1 | 29 | 7,319.674 | 750 |
| on | 2 | 49 | 7,306.007 | 4,470 |

Depth 2 stays within the shared budget but increases execution/setup work,
especially allocation reservations with bulk on. These reservations are not
CUDA allocation calls: CuPy may reuse cached physical memory. This is evidence
of additional owned-buffer setup, not proof of a particular runtime bottleneck.
Diagnostic rates are intentionally excluded from throughput conclusions.

## Conclusion and next step

Depth 2 works correctly within the same byte budget and gives a small observed
cold gain with bulk off. This test does not demonstrate a material cold gain
with bulk on or increased concurrent-file reading. The existing eight-worker
pool already spreads reads across files at depth 1; increasing execution
depth also increases subbatch and buffer-management work.

Warm performance on this allocation favors depth 2, with steadier depth-2
runs, but profiling is needed to explain the slow depth-1 cases and reconcile
them with the earlier Stage 4 result. Preserve all these samples; do not infer
that they establish a portable 2.7x warm bulk-off speedup. Profile setup within
the event loop, producer/consumer GPU waits and group/budget/cache work. The
diagnostic allocation-reservation increase is a useful lead, not a measured
CUDA-allocation bottleneck. Keep the depth-1 baseline and avoid increasing
depth further until those costs are separated. No runtime/default change was
made for this experiment.

## Artifacts

Campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-depth-20260925-v2/`

- `run.sbatch`, frozen harness/helper scripts and `builds.json`.
- `job-39034776.log`: progress/failures.
- `job-39034776/summary.md`, `summary.json`, `results.json`: incremental results.
- `job-39034776/provenance.json`: allocation, GPU, hashes, references and tier.
- Per-sample MPI logs, GPU-monitor CSVs and cold native traces/audits.
- Runtime reused read-only from `stream-read-acceptance-stage4-20260924-v2/python`;
  native dependencies remain the frozen Integrated install from Stage 4.

Preliminary job 39034738 was stopped before accepted results to correct a
diagnostic expectation: calibration launch count is per event, not per
subbatch. Its artifacts remain in the campaign directory without `-v2` and
are excluded. The runtime was not modified.

Maintained entry point: `../../scripts/feespec_bulk_benchmark/acceptance.py
--study depth`; reports in `depth_summary.py`, isolated hooks in `pipeline_stats.py`.
Eleven harness tests passed before submission.
