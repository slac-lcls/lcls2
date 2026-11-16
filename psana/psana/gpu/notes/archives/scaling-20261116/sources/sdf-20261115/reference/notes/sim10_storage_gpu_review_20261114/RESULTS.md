# Single-node Weka and NVMe GPU-sharing results

Recorded November 14, 2026. Job **45028778** completed on **sdfampere036**,
**11:40:08–12:54:08 PST** (1h 14m), exit **0:0**. The final automated audit
passed all **30 cases**; **31 GPU tests passed on each of four devices**
(124 passes, no failures/skips), and both private input stages were removed.
Independent prelaunch review passed in round 2. **Independent result review
is still pending**; the saved automated pass is not that separate gate.

## Measured throughput

The experiment holds **four total BDs** and **100,000 global unique events**
fixed while distributing BDs across one, two or four GPUs. Each table row uses
three full measurements from this allocation. Six traced pilots and six
separate Weka/one-GPU anchors are excluded from these means.

| Storage | GPUs | BDs/GPU | Mean events/s | Observed min–max events/s | Mean GiB/s | Mean paired speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weka | 1 | 4 | 1529.58 | 1443.44–1694.27 | 4.7839 | 1.0000 |
| Weka | 2 | 2 | 1473.11 | 1438.34–1499.84 | 4.6073 | 0.9675 |
| Weka | 4 | 1 | 1536.32 | 1460.02–1663.05 | 4.8049 | 1.0115 |
| NVMe | 1 | 4 | 1471.83 | 1461.07–1486.24 | 4.6032 | 1.0000 |
| NVMe | 2 | 2 | 1469.94 | 1438.55–1497.50 | 4.5973 | 0.9988 |
| NVMe | 4 | 1 | 1498.99 | 1489.76–1513.80 | 4.6882 | 1.0185 |

Speedup is calculated against the **same-medium, same-repeat, one-GPU full
case**, then averaged across three repeats; it is not a ratio of overall means.
Observed extrema are not confidence intervals. Weka's six anchors range from
1451.83 to 1667.02 events/s, so small differences cannot be treated as a stable
ranking. NVMe's four-GPU improvement is small, about 1.85% by paired ratios;
these measurements establish neither a useful application speedup nor equivalence.

### What an event means in bandwidth units

Every full case reads **335,820,000,000 payload bytes in 500,000 requests**:
**3,358,200 bytes/event = 3.202629 MiB/event**, five requests per event.

`payload GiB/s = global events/s * 3,358,200 / 1,073,741,824`

Thus **1,400 events/s = 4.378594 GiB/s = 4.70148 decimal GB/s**. These are
aggregate node payload rates, not per-GPU rates, physical wire traffic or
expanded/calibrated image bandwidth. The timing bracket covers the event loop,
GPU drain and final barrier, excluding input staging, setup, cache conditioning
and subsequent audit. It is not the 74-minute whole-job rate.

## Interpretation

This is a **staging-only task**, with descriptor validation and metadata
recording but no application expansion, calibration or cube kernels and no
per-event publication D2H. Little benefit from adding GPUs is plausible:
four BD processes and their total 16 KvikIO workers remain fixed, while storage,
host processing, bounce-buffer handling and H2D remain in the path.

The supported conclusion is: **spreading four BDs over more GPUs does not
materially increase measured staging throughput in this cohort**. An upstream
data-delivery or host-side bottleneck is a working hypothesis; the empty task
does not prove that the physical storage devices or NIC are saturated.

The separate CPU-only slab probes measured **3.7199 GiB/s on Weka** and
**6.1710 GiB/s on NVMe**. They use bounded, disjoint contiguous preads at
representative request sizes, without descriptor scheduling, KvikIO or H2D.
They are **not matched pipeline controls or storage ceilings**. NVMe's higher
probe rate suggests possible headroom, but its pipeline rate did not improve
over Weka. This screen therefore did not demonstrate removal of the I/O limit.

## Controls and storage evidence

- Four allocated **NVIDIA A100-SXM4-40GB** devices; six MPI ranks (SMD0, EB,
  four BDs), 34 required physical cores within 36 requested, exclusive node,
  192G host allocation. UUID mapping, peer counts and disjoint affinities passed.
- Per BD: four KvikIO workers, one CUDA stream, batch 200, 6 GiB input budget.
  Task 1 MiB, bounce 16 MiB, bulk target 512 KiB, SMD batch 1000.
- KvikIO 26.10 CPU compatibility ON, auto-direct OFF; **not GDS**. Active
  psana runtime is pinned PR298 `67c327a148a08b2aaf972e67e1e02e367c2b298f`.
- Each case passed exact event/timestamp/read/byte and callback checks,
  expected batching and complete drain; no unexpected CPU payload reads or
  pressure drains. All four device-test receipts have zero failures/skips.
- Both media passed zero initial host payload page residency after DataSource
  setup. This does not establish cold Weka servers or SSD internal caches.
- Full Weka cases: idle-adjusted physical NIC RX/payload **1.07264–1.08527**,
  supporting network-served data, not exclusive per-process traffic attribution.
- Full NVMe cases: physical-leaf disk bytes/payload **1.00000037**, both drives
  active; idle-adjusted NIC RX/payload approximately zero
  (**−0.00001719 to +0.00000061**). Negative residuals reflect idle subtraction.
- Live NVMe qualification: `/lscratch` is XFS on `/dev/md0`, RAID0 across
  `nvme0n1` and `nvme3n1`, 512 KiB stripe chunks. Both are Samsung
  `MZQL27T6HBLA-00A07`, 7.68 TB each, firmware `GDC5602Q`, negotiated PCIe
  4.0 x4, NUMA node 5. The compact receipt preserves exact inventory values.

## Recommended next single-node screens

These are proposals, **not submitted jobs**. First complete independent result
review, including anchor drift, GPU placement, storage counters and timing.
Retain the [current one-stream policy](../mfx_expand10_single_node_baseline_20261108.md)
and [bounce-buffer controls](../sim10_kvikio_bounce_20261113/RESULTS.md).

### 1. One bounded input-path attribution screen

Use four BDs and one GPU, batch 200 and the existing task/bounce/worker settings.
For each medium compare the existing staging control against a CPU-only replay
of the **same file offsets, sizes, request ordering and outstanding-work limit**.
Record the unavoidable API/buffering differences; disabling the application
callback alone does not disable H2D. A matched-read replay adapter needs its own
correctness checks and review before it can serve as a control.

Inspect existing pilot traces and telemetry first. Add short diagnostic captures
only for missing read issue/wait, CPU scheduling, H2D and queue evidence; measure
clean rates separately and do not sum overlapping timings. Use cold-host-cache
Weka NIC and NVMe leaf-disk checks. Three paired full repeats per medium/mode
are a compact starting point, not another worker/batch/storage Cartesian grid.

If CPU-only replay is clearly faster but GPU staging plateaus, investigate the
shared host/H2D/scheduling path. Similar rates only suggest a read-path limit;
they do not by themselves prove storage hardware saturation. If attribution
cannot be made cleanly, retain that uncertainty and proceed to application work
rather than repeatedly tuning the empty task.

### 2. Restore compute on one BD and one GPU

Establish actual MPI SMD0 + EB + one-BD controls for staging, then
expansion/calibration, then persistent cube accumulation, with one CUDA stream.
Use a small correctness pilot before clean measurements. Validate expanded
pixels/calibration against an independent CPU reference and cube counts/sums
against the existing numerical acceptance policy, including the empty bin.

Use the existing unique-event fixture and a qualified NVMe node for controlled
local-storage tests, without assuming NVMe is faster. Include a matched Weka
control at the selected application setting rather than doubling every cell.
Choose event count from actual runtime and variance: begin with 1k acceptance
and a 10k timing pilot, extend toward 100k only if needed and budgeted, and use
the same global event count for every final comparison in a cohort.

### 3. Small compute-batch and memory screen

Screen the established bounded compute candidates **4, 7 and 14** on one
BD/one GPU, subject to acceptance and capacity. Do not carry staging batch 200
straight into expanded-image processing. Measure actual subbatch sizes, peaks,
input credits, scratch/output arrays, constants and persistent cube banks.
Use the intended 100-bin workload when it fits; repeat promising settings three
times with a fresh baseline. Avoid forced-split/multi-slot tuning under the
current policy.

### 4. Compute-bearing node scaling, then GPU sharing

First run **one BD per GPU at 1/2/4 GPUs**, globally partitioning the same unique
events, with fixed per-BD settings and adequate physical CPU provisioning.
This changes both BD and GPU counts: label it **end-to-end node scaling**, not
an isolated GPU-count effect. Record common-wall aggregate rates, rank balance,
storage bytes, host/GPU peaks and efficiency relative to the same-node MPI
one-BD control.

Then compare **one versus two BDs per GPU** on a selected fixed GPU count only
if capacity and CPU availability permit. Persistent cube state is per BD and
per bank: a full 100-bin FP32 32-panel bank is about **6.25 GiB**, before input
budgets, constants and temporaries. Four such banks plus four 6 GiB input
budgets already total **49 GiB** on a 40 GB-class GPU. The old fixed-four-BD
one-GPU staging cell cannot be reused blindly for the full cube. Do not lower
bins or memory for just one cell to make it fit; define a separate matched
feasible cohort or report the configuration as infeasible.

### 5. Close the application baseline with bounded output

Include final bin-by-bin D2H, cross-BD merge/reduction, write/fsync and global
count/numerical verification. Report event-loop throughput and end-to-end
throughput separately. Measure output to an explicitly chosen destination;
local-NVMe export alone is not production Weka output qualification. Preserve
bounded host/device memory and all completion/lifetime guarantees.

Multiple CUDA streams, forced-split concurrency and related budget sweeps stay
deferred until the recorded **#308** implementation/acceptance gate is verified
and a new matched one-stream control is established. Do not silently change the
pinned framework runtime between parameter cells. Every changed harness needs
review and CPU/GPU acceptance before an authorized launch.

## Evidence and provenance

- [All 30 cases](results/runs.csv): pilot/full/anchor labels, repeats, raw rates,
  paired speedups for full cases, physical traffic ratios and result-file SHA256.
- [Compact receipts and summary](results/summary.json): exact means/samples,
  probes, GPU tests, live storage inventory, cleanup, accounting and evidence hashes.
- [Current status](status.json) and [submission record](submission.md).
- Raw results remain under
  `/sdf/scratch/users/m/monarin/gpu-validation/sim10-storage-gpu-20261114/attempt02/results/`;
  stdout/stderr are in its parent. Bulky traces and raw results remain on scratch.
- Frozen manifest SHA256:
  `a06e3c9571cdfebde916bf7af2b3f5d83eb36a6ba36909ba2bae79a9c1bdfc98`.
  The compact receipt records the exact checkout HEAD. The execution identity
  is the reviewed manifest, not a claim of a clean checkout.

This recording preserves compact evidence without modifying the frozen attempt,
benchmark harness or pre-existing baseline edits. No new job, branch or commit
is created. The saved final audit was inspected; it was not independently
re-reviewed or rerun during this recording update.
