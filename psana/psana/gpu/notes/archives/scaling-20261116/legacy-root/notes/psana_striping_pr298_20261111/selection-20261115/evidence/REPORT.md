# Sim10 single-node parameter screening report

**November 14, 2026 · current staging workflow · A100 40 GB class**

## Recommendation

For the **current Sim10 staging-only workflow**, carry forward **one GPU,
four BDs, four KvikIO workers per BD, one CUDA stream per BD, batch 200,
1 MiB tasks and 16 MiB bounce buffers**. Retain the 512 KiB bulk target,
SMD batch 1000, 6 GiB input budget per BD, KvikIO compatibility ON and
auto-direct OFF. Use cold-host-cache Weka as the operational baseline; keep
qualified NVMe for diagnostics, not as a demonstrated speed upgrade.

This is the **best-supported practical configuration**, balancing measured
throughput and resource requirements, not a statistically established global
optimum. It has been exercised as a complete configuration, rather than assembled
from unrelated best cells. The worker cohort measured **1,635.45 events/s
(5.1150 GiB/s)** with it. The latest Weka cohort measured **1,529.58 events/s
(4.7839 GiB/s)** with the same parameter choices; these are separate observations,
not a pooled benchmark or a predicted performance range.

For a **one-BD baseline**, retain **eight workers, batch 50**, with the other
controls unchanged. That is the stronger throughput candidate for one BD;
four workers need fewer bound cores but were slower. Neither profile is yet
an optimized expansion/calibration/cube configuration.

| Control | Throughput-oriented staging | One-BD control |
| --- | --- | --- |
| Active GPUs / BDs | 1 / 4 | 1 / 1 |
| Workers per BD / total | 4 / 16 | 8 / 8 |
| CUDA streams per BD | 1 | 1 |
| Batch events | 200 | 50 |
| KvikIO task / bounce | 1 MiB / 16 MiB | 1 MiB / 16 MiB |
| Bulk target / SMD events | 512 KiB / 1000 | 512 KiB / 1000 |
| Input budget | 6 GiB per BD; 24 GiB aggregate | 6 GiB |
| Bound physical-core requirement | 34 | 17 |
| Worker bounce demand estimate | 256 MiB | 128 MiB |
| Data path | Compatibility ON; auto-direct OFF; not GDS | Same |

The 34/17 core counts include helper allowances and are not measured busy cores.
Request enough physical cores with disjoint affinity; Slurm rounding may require
more (the latest six-rank allocation requested 36). Exclusive allocations may
reserve the whole node. Input budgets and bounce estimates do not include all
framework, CUDA, application or host allocations; retain measured headroom gates.

## Evidence covered

The main heatmaps consolidate **seven allocation cohorts**, **48 measured
parameter configurations** and **102 full runs**. There are also **48 validation
pilots**, **six Weka anchors** and two separate CPU slab probes. Pilots, anchors
and probes are not additional repeats in the parameter heatmaps.

| Screen | Job / node suffix | Parameters explored | Full cells × repeats | Result review |
| --- | --- | --- | --- | --- |
| BD / streams | 44232284 / 023 | BDs 1/2/3/4 × streams 1/2, workers 8, batch 100 | 8 × 2 | Passed |
| Initial batch | 44309691 / 036 | BDs 1/4 × batches 100/200/500 | 6 × 2 | Passed |
| Extended batch | 44321906 / 040 | BDs 1/4 × batches 10/50/100/200/500 | 10 × 2 | Passed |
| Workers | 44770441 / 036 | Workers 4/8/16 at selected BD/batch combinations; fixed-total-worker comparison | 10 × 2 | Passed |
| Task size | 44852752 / 017 | Tasks 1/4 MiB for 1-BD and 4-BD profiles | 4 × 2 | Pending |
| Bounce size | 44870463 / 018 | Bounce 16/64 MiB for 1-BD and 4-BD profiles | 4 × 2 | Pending |
| Storage / GPU sharing | 45028778 / 036 | Weka/NVMe × 1/2/4 GPUs, four total BDs | 6 × 3 | Pending |

All seven completed with exit 0:0 and passed their recorded automated audits and
cleanup gates. Prelaunch review is distinct from result review. The first four
cohorts have an independent reviewed consolidation; the last three have automated
acceptance but no completed independent result review. This new synthesis and
its new figures have calculation tests and visual checks, **not an independent
review verdict**.

Every main-table full run partitions **100,000 unique events globally**, reads
**500,000 datagrams / 335,820,000,000 payload bytes**, and uses the pinned PR298
psana runtime `67c327a148a08b2aaf972e67e1e02e367c2b298f` with KvikIO 26.10 CPU
compatibility mode. The task validates descriptors/metadata; it has **no
application expansion, calibration or cube accumulation kernels**, and no
per-event publication D2H. Input reads, host processing, H2D and framework
scheduling remain included. A nominally empty user task is not a zero-cost loop.

## How to read the heatmaps

- Each panel identifies its job and node. **Do not pool different allocations
  or divide a rate by another job's rate**, even if the hostname/GPU matches.
- Cells show arithmetic means of full-run rates, not total events / total time.
  Two repeats support descriptive rankings, not confidence intervals or equivalence.
- Gray cells are **not run**, never zeros or interpolation. In particular,
  four BDs × 16 workers requires 82 physical cores and was not tested here.
- Changes use controls stated in each panel. Storage/GPU changes average
  **same-medium, same-repeat ratios**, not ratios of the overall means.
- Resource-normalized rates use configured resources, not utilization, power,
  reserved GPU-hours, billing or proven savings. All storage/GPU cases allocated
  four GPUs even when only one or two were active.

### Throughput

![Mean payload throughput heatmaps, with event rates in each measured cell](artifacts/throughput.svg)

The common color scale is payload GiB/s; the second line in each cell is global
events/s. Both metrics rank cells identically because bytes/event is fixed.
The timing bracket is the event loop plus final GPU drain/barrier, excluding
staging, setup, cache conditioning and audit. It is not whole-job throughput.

**1,400 events/s = 4.3786 GiB/s = 4.7015 decimal GB/s** because each event
requests 3,358,200 bytes (3.202629 MiB). These are aggregate node input rates,
not per-GPU rates or expanded/calibrated image bandwidth.

### Change versus controls

![Percent changes against explicitly matched controls within each allocation](artifacts/relative_change.svg)

Zero denotes the stated control, not zero throughput. Colors show descriptive
effect size, not statistical significance. The fixed-total-worker panel has a
large positive effect and therefore a wider positive scale than the negative
scale; numeric labels remain the authoritative comparison.

### Repeat spread

![Observed full-run repeat spread for each measured configuration](artifacts/repeat_spread.svg)

Spread is `100 * (maximum - minimum) / mean`. Small spread in two repeats does
not prove repeatability across days/nodes. Weka's latest one-/four-GPU cells show
material spread and its separate anchors range **1,451.83–1,667.02 events/s**.
Those anchors diagnose drift but are not extra full-cell replicates.

### Resource efficiency and memory estimates

![Throughput per bound CPU core, estimated bounce memory, and throughput per active GPU](artifacts/resources.svg)

The CPU panels show aggregate events/s divided by the bound physical-core
requirement. The bounce panel assumes one buffer per worker; it does not measure
CUDA pinned allocations. The GPU panel divides by active GPUs, not all four
allocated devices. A configuration with fewer resources may rank well here
without delivering the highest aggregate throughput.

## Findings by parameter

### BD count and streams

With eight workers/BD and batch 100, increasing one to four BDs raised the
same-allocation one-stream mean from **605.62 to 1,355.01 events/s (2.24×)**.
Total workers and bound cores also increased; this is not GPU-only scaling.
In the worker cohort, **1 BD × 16, 2 BDs × 8 and 4 BDs × 4 workers**, all batch
100, yielded **616.48, 1,087.46 and 1,551.44 events/s**. Four BDs reached **2.52×**
the one-BD rate with fixed total workers, although helper/core requirements
still increased from 25 through 28 to 34. Multiple independent BDs matter for
this path; merely assigning more workers to one BD did not substitute for them.

The historical two-stream means were **6.91%, 3.97%, 2.21% and 1.33% lower**
than their one-stream controls at 1/2/3/4 BDs. They do not prove that one stream
is inherently optimal. Continue the recorded **one-stream policy** until the
#308 concurrency/lifetime acceptance gate is verified, preserving #298/#299
requirements and establishing a fresh control before switching runtimes.

### Batch size

The five-point screen favors **batch 50 for one BD** (647.27 events/s,
**+3.22%** versus batch 100) and **batch 200 for four BDs** (1,585.47 events/s,
**+3.95%** versus batch 100). Batch 10 was too small to perform well in this
screen; batch 500 did not improve on the candidates and was particularly poor
and variable for one BD. The separate three-point screen also ranked batch 200
highest for four BDs, but its repeats must not be merged with the five-point job.

Carry these as candidates, with batch 100 as a confirmation control—not proven
unique optima. Expanded/calibrated batches have a different memory and compute
cost; do not promote staging batch 200 into that workflow.

### KvikIO workers and CPU provisioning

At one BD / batch 50, eight workers reached **652.17 events/s**, versus 579.34
with four and 635.99 with sixteen: **+12.57%** versus four, **+2.54%** versus
sixteen. Eight workers are the practical one-BD throughput choice; four remain
a lower-CPU alternative, not the best aggregate-rate result.

At four BDs / batch 200, four workers/BD reached **1,635.45 events/s**, versus
1,606.41 with eight: a small **+1.81%** difference, while requiring **34 rather
than 50 bound cores (32% fewer)**. The smaller pool is preferable because it
retains the measured throughput with lower provisioning and estimated bounce
demand, not because the 1.81% gain is a proven algorithmic advantage. Actual
charged resources depend on the allocation, especially on an exclusive node.

### Task and bounce sizes

Changing tasks **1 → 4 MiB** increased the two-repeat means by **1.82% at one
BD and 1.15% at four BDs**. Traced POSIX requests remained unsplit and below
1 MiB (524,724 / 629,664 / 734,604 bytes). Native task counts were not directly
measured. There is no established benefit warranting a change from **1 MiB**.

Changing bounce buffers **16 → 64 MiB** changed means by **−2.37% at one BD
and +1.28% at four BDs**. Estimated worker-buffer demand quadruples: 128 → 512
MiB for the one-BD profile and 256 → 1024 MiB for the four-BD profile. Retain
**16 MiB**: there is no consistent demonstrated advantage to the larger buffer.

Do not combine the task-size and bounce-size gains as an expected speedup;
they came from different jobs, and a 4 MiB task / 64 MiB bounce combination was
not measured. Likewise, the absolute fastest small-sample cell in another
cohort does not automatically become the recommended operating point.

### Storage and number of GPUs

With four BDs and 16 workers fixed, the latest means were:

| Storage | 1 GPU | 2 GPUs | 4 GPUs |
| --- | ---: | ---: | ---: |
| Weka payload GiB/s | 4.7839 | 4.6073 | 4.8049 |
| NVMe payload GiB/s | 4.6032 | 4.5973 | 4.6882 |
| Weka mean paired speedup | 1.0000× | 0.9675× | 1.0115× |
| NVMe mean paired speedup | 1.0000× | 0.9988× | 1.0185× |

There is no compelling staging-throughput reason to use more than **one GPU**
for these four BDs. Nor did NVMe provide a clear pipeline throughput uplift.
Its CPU-only slab probe reached **6.1710 GiB/s**, versus **3.7199 GiB/s** on Weka,
but those probes are not matched descriptor/H2D controls or storage ceilings.

The plateau is consistent with a shared input/host/transfer/scheduling limit,
not user-kernel compute. It **does not prove physical storage saturation**.
Physical NIC/disk checks establish the intended path: network-served Weka,
or both local RAID0 NVMe leaves active with near-zero adjusted NIC traffic.
They do not establish exclusive traffic attribution or cold backend devices.

For NVMe diagnostics, the qualified 036 inventory is two **Samsung
MZQL27T6HBLA-00A07 7.68 TB drives**, firmware GDC5602Q, PCIe 4.0 x4, NUMA 5,
RAID0 under `/lscratch`. Requalify model, firmware, topology, negotiated links,
capacity and idle traffic in any new allocation; do not substitute a node just
because it also has local storage.

## Earlier parameters and non-comparable measurements

The heatmaps intentionally isolate the 100k staging series. Earlier application
and correctness work remains relevant but does not fit that metric population:

| Parameter / diagnostic | What was tested | What it supports |
| --- | --- | --- |
| Bulk read target | 8 MiB versus 512 KiB on the earlier 1k application fixture | Retain 512 KiB to avoid observed small-group admission stalls for Sim10; not a universal bulk-size optimum. |
| Compute-bearing streams | 1/2 streams, 1k and 10k serial application cases, batch 7 / 100 bins | Historical application controls, not current MPI staging throughput or permission to bypass #308. |
| Dense cube capacity | 30/100/200 bins on a 1k correctness fixture | Numerical/count and bounded-export feasibility; not a bin-count performance optimum. |
| Input budget / slots | 6 GiB current control; separate 128 MiB/one-slot pilot failed admission sizing in job 44525611 | No accepted full-run budget optimum. The 3,788-byte mismatch remains a correctness follow-up; do not plot the failed case as zero throughput. |
| Compute batches 4/7/14 | Batch 7 is the earlier application control; the proposed full comparison is still next | Do not present 4/14 as completed compute-screen cells. |
| GDS / direct I/O / production output | Not varied in this staging series | No claims of true GDS, direct-I/O benefit, or end-to-end production cube throughput. |

The earlier short warmed 512 KiB application means/medians and 10k warmed-window
results use different workloads/timing windows. Their one-/two-stream advantages
must not be normalized against the current staging curves. For example, the
earlier 10k application warmed-window medians were 257.25/283.58 events/s;
they are **not** evidence that the current 1.5k staging rate applies to full cubes.

## Decision and next measurements

1. **Use the practical staging profile above**, retaining the existing source,
   cache/NIC, CPU-affinity and per-GPU memory gates. Finish independent result
   review for the three newest cohorts before calling their conclusions reviewed.
2. **One bounded bottleneck check, not another broad empty-task sweep:** compare
   matched offsets/sizes/concurrency with CPU-only reads against GPU staging.
   Use existing traces first; fill missing CPU/read-wait/H2D timing separately
   from clean rate runs. The slab probes do not replace this control.
3. **Restore expansion/calibration and then persistent cube** on one BD/one GPU.
   Require independent numerical/count verification. Re-screen bounded batches
   4/7/14 with actual intermediate/cube-bank memory accounting.
4. **Scale the real workload within one node:** one BD/GPU at 1/2/4 GPUs,
   fixed global unique events, then GPU sharing where capacity permits. This
   varies both BDs and GPUs and is end-to-end node scaling, not GPU-only isolation.
5. **Measure bounded D2H, cross-BD merge/reduction and output** separately and
   in the end-to-end total. A 100-bin full FP32 bank is about 6.25 GiB per BD;
   four banks plus four 6 GiB input budgets already imply 49 GiB before other
   arrays. The staging four-BD/one-GPU profile cannot be carried blindly into
   a 40 GB full-cube workload.

## Data, validation and reproduction

- [Full configuration tables](artifacts/tables.md), [all 156 case records](artifacts/runs.csv),
  [48-cell summary](artifacts/summary.csv), [explicit comparisons](artifacts/comparisons.json).
- [Data definitions](SCHEMA.md) and [report validation](validation.json).
- [Four-page heatmap PDF](artifacts/heatmaps.pdf) and the embedded SVGs are compact
  durable artifacts. PNG previews and rendering/test caches stay on shared scratch.
- [Source and figure hashes](artifacts/provenance.json) preserve exact source
  identities. Raw-case hashes are inherited from source CSVs, not newly rehashed.
  Original large cases remain at their recorded scratch locations; hashes are
  not a substitute for those files if a full event-identity re-audit is needed.
- [Earlier reviewed consolidation](../sim10_single_node_scaling_20261113/README.md),
  [task-size results](../sim10_kvikio_tasksize_20261113/RESULTS.md),
  [bounce results](../sim10_kvikio_bounce_20261113/RESULTS.md),
  [storage/GPU results](../sim10_storage_gpu_review_20261114/RESULTS.md).
- [Historical bulk screen](../mfx_expand10_512k_20261108/README.md),
  [cube capacity](../mfx_expand10_capacity_20261106/README.md),
  [10k application baseline](../mfx_expand10_baseline10k_20261108/README.md),
  [current policy and failed budget diagnostic](../mfx_expand10_single_node_baseline_20261108.md).

`build_report.py` reads only existing measurements; it never submits GPU work.
`test_report.py` checks units, phase exclusion, all 48 plotted configurations,
missing cells, cohort isolation, paired GPU ratios and resource metrics.

To reproduce into a **new** shared-scratch directory, use the existing validated
Python environment with `PYTHONDONTWRITEBYTECODE=1` and `MPLCONFIGDIR` on shared
scratch, then run `build_report.py --output <new-directory>`. `--archive` additionally
preserves compact artifacts here and refuses an existing archive directory.
No framework/harness change, benchmark rerun, job submission, branch or commit
is part of this report.
