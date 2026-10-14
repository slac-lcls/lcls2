# Matched warm A/B/B+ comparison on one SDF allocation

Job **38513845** completed with exit **0:0** on **sdfampere033**, elapsed
**1h33m47s**, on September 17, 2026. This rerun addresses the changed B baseline
between the earlier A/B and B/B+ allocations. No production code or installed
build was changed for this rerun; gathering and bulk-read integration remain
outside this change.

## Clean results

Six uninstrumented 10,000-event measurements per variant, covering all six
execution orders. Every accepted sample was **100% file-cache resident before
and after timing**. Rates below are 10,000 divided by the median elapsed time.

| Variant | Median seconds | Events/s from median | Seconds range |
| --- | ---: | ---: | ---: |
| A | 21.850 | 457.7 | 21.177–25.175 |
| B | 33.561 | 298.0 | 32.581–33.932 |
| B + batched location | 28.743 | 347.9 | 28.558–29.916 |

Batched location reduces the B median by **4.818 seconds
(14.4%)**, equivalent to **16.8% higher throughput**.
B+ is faster than B in all six repetitions; within-repetition differences range
from 3.474 to 5.374 seconds. A remains faster: B+ takes
**6.893 seconds (31.5%) longer** than A by these medians.
No samples were dropped, including A's slower first and sixth runs.

| Repetition | Execution order | A seconds | B seconds | B+ seconds |
| --- | --- | ---: | ---: | ---: |
| 1 | A, B, B+ | 25.175 | 33.521 | 29.916 |
| 2 | B, B+, A | 21.457 | 32.622 | 29.148 |
| 3 | B+, A, B | 21.954 | 33.601 | 28.765 |
| 4 | B+, B, A | 21.747 | 33.932 | 28.558 |
| 5 | B, A, B+ | 21.177 | 32.581 | 28.720 |
| 6 | A, B+, B | 23.253 | 33.776 | 28.721 |

## Comparison with prior allocations

| Job / node | Clean samples per variant | A median | B median | B+ median |
| --- | ---: | ---: | ---: | ---: |
| 38478311 / sdfampere003 | 3 | 21.768 s | 33.451 s | — |
| 38504647 / sdfampere004 | 3 | — | 40.238 s | 35.452 s |
| 38513845 / sdfampere033 | 6 | 21.850 s | 33.561 s | 28.743 s |

The new A median differs from the original by **0.38%**;
the unchanged B differs by **0.33%**. This allocation closely
reproduces the original A/B baseline. The 40.238-second B baseline did not recur.
Its earlier 37.380–45.331-second range is also substantially wider than the
32.581–33.932-second B range here. The current B+ comparison is
against the B control measured in this same allocation.

This confirms that the earlier cross-job absolute-time difference did not
require a changed B build or workload. It does not isolate which environmental
factor caused it: CPU scheduling, NUMA memory locality, and shared-node activity
can still vary. A common allocation and balanced orders reduce confounding;
they do not make these six samples a precise causal speedup estimate.

## Separate host-scope measurements

Three CPU/NVTX runs per variant, kept separate from clean throughput. Except
for startup setup, these are medians over 9,980 events / 499 steady subbatches;
the first batch is excluded. Units are elapsed host seconds, not GPU execution.

| Scope | A | B | B+ |
| --- | ---: | ---: | ---: |
| Detector setup (startup) | 0.982 | 0.966 | 0.976 |
| Read submission | 1.057 | 0.929 | 0.919 |
| Read completion waiting | 15.682 | 15.185 | 15.552 |
| Whole slot submission (inclusive) | 1.737 | 14.417 | 9.871 |
| Eager field location | — | 5.232 | 0.553 |
| Field access/dependency/gather submission | 0.853 | 7.151 | 7.241 |
| Detector source selection | 0.224 | 0.488 | 0.486 |
| Calibration submission | 0.243 | 0.277 | 0.272 |
| Producer retirement waiting | 1.077 | 0.004 | 0.004 |

Eager field location falls **89.4%**, a **4.679-second** scope
reduction. Gathering remains nearly unchanged between B and B+ and remains the
largest measured submission difference from A. Read-wait and median startup
setup scopes are similar across all three variants in this allocation.

Instrumented total medians are 22.162 s (A),
33.275 s (B), and 28.918 s (B+).
The first A instrumented sample has 19.068 s of steady read waiting versus
15.682 s for its three-run median, with only 0.975 s in startup setup. These
separate samples cannot identify the exact cause of any individual clean outlier.
Independent scope medians and inclusive parents must not be added as an elapsed
time budget; asynchronous reads and GPU work overlap host submission.

No new Nsight traces were taken. The previous isolated traces remain the device
kernel/count evidence documented in
[calls per execution subbatch](batched_locators_sdf.md#calls-per-execution-subbatch).
For that same 20-event subbatch size and 192 configured handles, B/B+ decoding
launches are **192/1**, initialization kernels **192/1**, and locator memsets
**192/0** per subbatch. Including the unchanged walker, parser kernels drop
from **385 to 3**. These counts come from job 38504647's separate captures;
the table above contains host timings from the current matched A/B/B+ job.

## Controlled settings and recorded placement

- A: frozen `f52e90cc66d4c8c175b7689922c7e441e78b367f`, historical
  `sources/a/install_acceptance`.
- B: frozen `803a70011d18168200927e279cbeaca90568e13f`, historical
  `sources/b/install_psana`.
- B+: the validated uncommitted batched-locator implementation based on B, using
  this worktree's `install_psana`. Installed hashes before/after match for all
  three variants; actual imports and module hashes are retained per sample.
- Workload: `mfx101210926` run 387, 10,000 events, files s005–s009, 32 Jungfrau
  segments, payload 335,571,760,000 bytes. The complete stage manifest and saved
  calibration hash match the original A/B job.
- Topology: one SMD0, one EB, one BD, one GPU; MPI ranks 0/1/2 respectively.
  Batch 20, pool depth 1, 8 GiB device budget, no user D2H; eight KvikIO threads,
  1 MiB tasks, compatibility mode ON, GDS unavailable.
- Hardware: AMD EPYC 7542, A100-SXM4-40GB,
  UUID `GPU-619aab5e-5e5d-760a-9fc8-cc9fc9221aa1`, driver 575.57.08;
  CuPy 13.6.0, CUDA runtime 12.9, KvikIO 24.08.02.
- Slurm: 48 logical CPUs and 450 GiB host memory. Every rank uses the same
  recorded OS affinity: `0-11,40-51,64-75,104-115`, spanning NUMA nodes 0/1/5/6.
  GPU 0 has NUMA affinity 1. CPU cores are not assigned individually by rank,
  and host memory is not explicitly NUMA-bound. Per-process NUMA page totals
  are recorded before/after each loop; they do not establish staged-cache locality.
- All variants use the same explicit CuPy cache directory. Each sample has a
  separate 100-event warmup. The timer includes lazy detector setup and the
  final device synchronization, and excludes staging/cache preparation. Rank
  placement logging occurs outside each rank's timer, with a barrier before
  timing. Node CPU/load/memory/network snapshots and GPU clocks/power samples
  are saved without changing clock or memory policies.

Staging required about 9m49s, including local filesystem flush/verification.
The first cache preparation needed three passes: 88.70%, 96.80%, then 100%.
The allocation was nearly at its 450 GiB memory limit, with about 442 GiB
charged to file cache. Subsequent accepted measurements all retained 100%
residency. Shared source caches were not purged.

## Validation and artifacts

The full audit passed: **27 measurements** (18 clean, nine instrumented),
**six CPU-reference raw/calibrated pixel preflights** (three sampled events each),
correct 10,000 unique timestamps per timed run, correct execution order,
identical affinity and GPU, matching runtime settings, full cache residency,
and unchanged installed hashes. Sampled peak device memory is 3,631 MiB for A
and 3,633 MiB for B/B+. Eight timing-hook tests passed, including AST preservation
for A/B/B+. The summarizer reproduces both earlier result sets and excludes
profiled runs. `git diff --check` passes.

Reproduction: `validation/batched-locators-abo-20260917/README.md`.
Evidence: `validation/batched-locators-abo-20260917/job-38513845-warm-ab/`:

- `results.json`, `summary.json`, `audit.json`: individual samples and validation.
- `provenance.json`, `stage.json`: revisions, runtime environment, installed hashes,
  shared CPU mask, execution orders, input manifest and calibration checksum.
- Per-case `.log`, `-gpu.csv`, and `-host.json`: runtime/placement, cache checks,
  GPU clocks/memory/power, host CPU/memory/load/network counters and timing scopes.
- `slurm-allocation.txt`, `slurm-completion.txt`, `cache-preparation-memory.txt`:
  scheduler assignment, successful completion, and cache accounting.
- Parent `job-38513845.log`: CPU/NUMA/GPU topology, correctness acceptance and
  controller completion markers.
