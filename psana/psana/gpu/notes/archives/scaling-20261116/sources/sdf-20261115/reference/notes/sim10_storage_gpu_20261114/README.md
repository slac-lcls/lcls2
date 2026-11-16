# Paired storage / GPU-sharing diagnostic — 2026-11-14

Status: revised after first independent review; **56 CPU-only tests passed**;
**not submitted**. Review-loop records are in `../sim10_storage_gpu_review_20261114`.
Independent review and allocation-time hardware/GPU acceptance remain gates.
This is descriptor staging, **not calibrated XPP cube throughput**.

## Question and fixed controls

Does spreading the same four BDs over more GPUs help once the storage path
changes, or do CPU workers, host-memory traffic, PCIe, or storage still dominate?
Keep the total offered work constant; do not confuse fewer BDs/GPU with fewer
total readers. Do not assume local NVMe is faster than Weka.

| Item | Fixed value |
| --- | --- |
| Media | Cold-host-cache Weka FFB; cold-host-cache local NVMe |
| GPU count | 1, 2, 4; always four BDs, hence 4, 2, 1 BDs/GPU |
| Events | 100,000 global; 335,820,000,000 payload bytes; 500,000 reads |
| Per BD | Four KvikIO workers, one CUDA stream, batch 200 |
| I/O settings | 1 MiB task, 16 MiB bounce, KvikIO 26.10, CPU fallback ON, auto-direct OFF |
| Framework | 6 GiB input budget/BD, 512 KiB bulk target, SMD batch 1000 |
| Runtime | PR298 `67c327a148a08b2aaf972e67e1e02e367c2b298f`, validated install-B native extensions |
| Work excluded | Calibration, expansion, cube kernels, event-publication D2H, GDS |

`plan.py` emits the exact ordered matrix: six separately traced 4,000-event
pilots, eighteen full runs (three/cell), and six additional Weka/1-GPU drift
anchors. GPU ordering rotates across repetitions; the second repetition reverses
media order. Each repetition starts and ends with an anchor. Anchors are not
silently pooled into the three repeats. Compare each curve with its same-media,
same-repeat 1-GPU baseline from this allocation, never an older node's baseline.

## Node selection and availability

Slurm observation: **2026-11-14 11:12:30 PST**. Availability is not a reservation.

| Node | Observed availability | Historical local NVMe inventory |
| --- | --- | --- |
| `sdfampere036` | IDLE; no CPUs, GPUs or memory allocated — preferred candidate | 2 × 7.68 TB Samsung MZQL27T6HBLA-00A07, firmware GDC5602Q |
| `sdfampere018` | MIXED, but all four GPUs allocated to job 45019598 | 2 × 7.68 TB KCM6XRUL7T68, firmware 0105 |
| `sdfampere017` | IDLE+DRAIN; task-kill failure, not usable | 2 × 7.68 TB KCM6XRUL7T68, firmware 0105 |

Historical receipts (not fresh physical qualification):

- `../sim10_single_node_scaling_20261113/evidence/44770441/hardware.json` — 036.
- `../sim10_kvikio_tasksize_20261113/results/hardware.json` — 017.
- `../sim10_kvikio_bounce_20261113/results/hardware.json` — 018.

Recorded configuration: four A100 40GB GPUs, EPYC 7542, 56 schedulable physical
cores, and two NVMe drives on NUMA node 5, PCIe 4.0 ×4, behind `/lscratch` md0
RAID0. 036 is a different drive-model cohort; obtain fresh Weka controls there.
Read-only SSH to 036 was denied by `pam_slurm_adopt` because there is no active
allocation. Do not bypass this restriction. Free NVMe capacity and current
device firmware/link/topology must be captured inside the authorized allocation.

## Execution and evidence gates

- Request one exclusive node, all four GPUs, six tasks × six physical CPUs,
  192 GiB host memory and three hours. The actual fixed binding consumes 34
  physical cores (`6 + 4 × (4 + 3)`); no clamping or SMT substitution.
- Resolve allocated GPU UUIDs in PCI order once. Use nested subsets: first
  device for 1 GPU, first two for 2 GPUs, all four for 4 GPUs. Contiguous BD
  groups share devices. Mask each rank **before** importing psana/MPI/CUDA.
  Keep CPU affinity and interleaved NUMA policy identical across media/counts.
  This is a controlled mapping, not a claim of optimal storage/GPU locality.
- Verify reviewed source hashes and pinned Python/native/KvikIO files; run
  all 31 inherited GPU acceptance tests separately on every allocated GPU.
- Before copying, qualify both mounts and capacity. Resolve NVMe mount → md
  RAID0 → physical namespaces; record model, firmware, capacity, PCIe links,
  NUMA, GPU topology and physical NIC inventory. Stop if the selected model,
  two-drive RAID0 layout or expected link/NUMA configuration changes.
- Make new private job-owned stages on FFB and `/lscratch/monarin`; SHA/size
  check all five accepted JF BD files and their SMD companions on both media.
  Never copy the input to shared scratch. Only these new stages are removed,
  after owned subprocess groups and copy workers have stopped. Preserve stages
  if process-group termination cannot be confirmed.
- After DataSource setup, use owner-scoped `fsync`, `POSIX_FADV_DONTNEED` and
  full-file `mincore` on both media. No global `drop_caches`. Cold here means
  **host page cache**, not Weka server caches or SSD internal caches.
- Weka needs idle-adjusted **physical NIC RX ≥90%** of payload. NVMe needs
  idle-adjusted physical-leaf disk bytes ≥90%, reads on both drives, and NIC RX
  ≤10%. Never sum md and NVMe bytes. Save per-leaf read latency/queue estimates
  and raw counters. These are whole-device observations; unexpected background
  traffic invalidates clean attribution rather than proving the intended path.
- After the pilots, run one read-only CPU slab probe/media: 16 workers/maximum
  16 outstanding reads, 80,000 reads, 53,731,200,000 bytes. Sizes follow the
  524724/629664/734604-byte payload mix; regions are contiguous and disjoint.
  This is **not event-aligned replay, KvikIO/H2D, or a storage ceiling**. The
  actual-workload pilots separately verify observed POSIX read sizes. Treat a
  faster probe but flat GPU curve as a reason to investigate downstream sharing.
- Require exact timestamp coverage, nominal batch sizes, actual per-GPU peer
  counts and aggregate budget headroom, no CPU payload reads, and clean drains.
  Save one-second per-GPU utilization/memory/power/clocks and per-rank CPU-thread
  counters, RSS/high-water and affinity. Full timing runs are never straced.
- Audit every case and all required receipts before comparing rates. Use the
  repeated rates, Weka anchor drift and probe evidence together. Utilization
  suggests mechanisms but cannot alone prove DRAM/PCIe saturation. If NVMe does
  not improve observed bandwidth, this experiment has not removed the I/O limit.

## Preparation and launch boundary

### Empirical walltime budget and partial-run safety

The three-hour request is retained based on **measured**, not requested,
predecessor time. Job 44870463 completed in 34m43s, including eight full cases,
four traced pilots, acceptance and one full staging/verification. Its two
4-BD/bounce16 full loops took 61.18 and 62.30 seconds for 100,000 events;
the one-BD cases were slower (149.71–159.19 seconds). The manifest now freezes
`../sim10_kvikio_bounce_20261113/results/runs.csv` and `accounting.psv`.

Planning allowances (not performance guarantees): 24 full/anchor cases ×180s
(over twice the slowest matching loop plus 50s startup), six pilots ×120s,
four GPU tests ×300s, two probes ×300s, two copy-plus-full-reread SHA staging
phases ×900s, 600s qualification, 300s cleanup/audit = **9,540s (2h39m)**.
There is 21m additional margin. Staging's 900s planning allowance assumes about
0.75 GB/s effective aggregate copy+verification traffic across four workers;
its hard deadline is 1,800s/media. The historical cohort does not guarantee
036 or NVMe performance; a slow run may remain partial.

The 900s case timeout is an abnormal-case ceiling, not an expected duration;
we do not promise all ceilings can be reached within three hours. Before each
case, test, probe, or staging phase, read the cached Slurm end-time deadline
(environment epoch or bounded `squeue` lookup at startup) and require its
ceiling plus **300s cleanup reserve** to fit. A refusal or other exception
writes `campaign-status.json` as `partial_not_accepted`, keeps case/progress
evidence, performs cleanup and exits nonzero. Never interpret partial results
as a complete matrix. The scheduler's advance TERM is an additional fallback.

Copy and full SHA verification check cancellation/deadline between bounded
8 MiB reads/writes. Workers are joined before removing any stage. A kernel I/O
call stuck beyond Slurm's grace can still prevent orderly cleanup; in that
case inspect and manually remove only this job's private stages after proving
all workers are gone. Each medium gets an independent cleanup success/error
receipt, so one removal failure cannot hide the other or mask the original
campaign error.

### Hardware and import provenance

The manifest now includes the historical 036 `hardware.json` from job44770441.
Its NVMe controllers at `0000:c4:00.0` and `0000:c5:00.0` report model
`SAMSUNG MZQL27T6HBLA-00A07`, firmware `GDC5602Q`; corresponding PCI records
have exact sysfs strings `current_link_width="4"`,
`current_link_speed="16.0 GT/s PCIe"`, `numa_node="5"`. These are the source
of the qualification constants, not a fresh observation. The two smaller
960GB system drives are not the selected RAID leaves.

The inherited `adapter.py` imports the hash-frozen checkout's
`jungfrau_calibration.py` example at module load, not the PR298 version.
StagingTask does not call that example's calibration path; it uses only the
adapter's reader registration. Active psana Python/native runtime remains
pinned PR298. The manifest records this intentional mixed harness provenance.

Unique output root:
`/sdf/scratch/users/m/monarin/gpu-validation/sim10-storage-gpu-20261114/`.
Shared scratch had about **7.9 GiB free of a 100 GiB quota** at preparation;
use it only for the small source snapshot, logs, traces, caches and receipts.
The existing validated runtime is reused in place and must remain available.
Each input stage additionally needs its full fixture size plus 10 GiB margin.

`prepare.py --attempt <new-shared-scratch-attempt> --node sdfampere036` freezes
the explicit source dependency set, including untracked new harness files. It
creates no branch, worktree, commit or job. It prints a candidate `sbatch` command
with an intentionally invalid review placeholder. Review the exact snapshot,
then replace the placeholder with its manifest SHA256 only after independent
approval and authorization to launch. A digest is an integrity guard, not an
automatic attestation that a review happened.

`sbatch --test-only ...` checks scheduler feasibility without submitting or
executing the script. It cannot qualify the disks or replace the GPU pilots.
Refresh availability immediately before any authorized submission. No automatic
fallback to a different node/model or to a different I/O mode is allowed.

The new runner adapts the accepted bounce-buffer runner; baseline source and
results are unchanged. Inherited runtime, ownership tests, trace parsing,
host-memory and CPU-plan helpers are frozen as dependencies rather than edited.
PR #308 remains a gate for multi-stream/forced-split work, not this one-stream
multi-GPU diagnostic. Independent result review will also be needed afterward.
