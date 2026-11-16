# Sim10 expansion and cube comparison plan

## Recommendation

Keep the proposed I/O controls fixed, and measure the incremental cost of
expansion, calibration, and accumulation against fresh staging-only controls.
The staging heatmap does not establish that the application fits the same GPU
topology or that the selected batches are valid compute batches.

**Proposed first cohort: one BD on one GPU, and four BDs on four GPUs, with
one CUDA stream per BD.** This is a topology change from four BDs sharing one
GPU, not a claim that additional GPUs improved staging. It preserves batches
50/200 and the 6 GiB framework budget without introducing an unvalidated tiled
compute path. Both topologies need device-memory qualification; neither is
already accepted for compute. User confirmation of this topology and the
100-bin application target is required before implementation/launch preparation.
This is a plan review, not implementation acceptance or job-submission authority.

## Fixed controls and historical interpretation

| Control | One BD | Four BDs |
| --- | ---: | ---: |
| Proposed active GPUs | 1 | 4, one BD per GPU |
| Nominal EB batch events | 50 | 200 |
| KvikIO workers per BD | 4 | 4 |
| Total KvikIO workers | 4 | 16 |
| CUDA streams per BD | 1 | 1 |
| KvikIO task / worker bounce | 1 MiB / 16 MiB | 1 MiB / 16 MiB |
| Bulk target | 512 KiB | 512 KiB |
| Framework budget per BD | 6 GiB | 6 GiB |
| SMD events per chunk | 1000 | 1000 |
| Bound physical cores including helper allowances | 13 | 34 |

Compatibility mode ON, auto-direct reads OFF: this is not a GDS experiment.
Use the pinned PR298 runtime `67c327a148a08b2aaf972e67e1e02e367c2b298f`,
KvikIO 26.10 and the recorded native-compatibility receipt. Freeze all runtime,
task, calibration and fixture hashes. Do not silently import the working tree's
HEAD as the runtime. Do not adopt #299 or a new runtime between cells.
Multi-stream, forced framework splitting, and slot-budget tuning remain gated
by #308 and its concurrency/lifetime acceptance, preserving #298/#299.

Four workers is an intentional common control, **not the established one-BD
throughput winner**. In job 44770441, one BD/batch50 gave 579.34 events/s with
four workers versus 652.17 with eight: eight was 12.57% faster, using 17 rather
than 13 bound cores. Four BDs/batch200 gave 1635.45 versus 1606.41 events/s
with four/eight workers per BD. Retain four for this comparison as requested;
never divide new one-BD compute rates by the old eight-worker staging rate.
The two-repeat task/bounce screens support conservative controls, not a unique
optimum. Their independent result reviews remain pending.

Source: [reviewed worker summary](../sim10_single_node_scaling_20261113/summary.csv),
[parameter report](../sim10_parameter_report_20261114/REPORT.md).

## What each stage measures

Use four explicit cumulative stages to remove ambiguity about “blow-up.”
The main workflow comparisons are S, EC and ECC; E separates the expansion cost
from calibration rather than calling the combined operation expansion alone.

| Stage | Work included | Question |
| --- | --- | --- |
| S | Current five-stream seed staging, descriptor validation, event delivery | Fresh same-setting framework/read/H2D baseline |
| E | S plus seed expansion to all 32 physical uint16 panels | Cost of descriptor-based assembly and 10x raw expansion |
| EC | E plus unchanged FP32 Jungfrau calibration semantics | Incremental calibration cost and memory demand |
| ECC | EC plus persistent dense FP32 cube accumulation and exact counts | Cost of the application, plus separate final export |

All stages read the same compressed data; expansion happens on the device.
E materializes 32 MiB/event; EC additionally materializes 64 MiB/event. Freeze
the same calibration constants with offsets enabled, status masking disabled,
and no common-mode correction, following the reviewed Stage2 semantics. Record
constant setup time and actual resident bytes; S/E need not allocate unused
calibration constants or cube banks. These extra resources are part of the
application treatment, not a hidden I/O-parameter change.

Assume the prior application target: **100 total bins, 99 active and one empty
control bin**, `global_ordinal % 99`, all 32 panels, FP32 sums and uint64 counts.
This is synthetic bin assignment, not an XPP scientific-coordinate cube.
Keep event selection and full image content identical in all four stages.
No per-event full-image D2H in clean runs; retain buffers until their stream's
consumers finish, and transfer only bounded validation/counter data after drain.
For E/EC/ECC, retain raw and calibrated images via `keepalive` or equivalent
uncopied ownership, **not `publish`**. PR298 copies published groups to the host
even when the consumer never accesses `.on_cpu`; `gpu_d2h_chunk_size=0` does not
disable task publication copies. The historical adapter publishes full images
and cannot be reused unchanged. Allow only explicitly named compact error/accept
data, at most 64 KiB total published bytes/callback at batch200. Set and record
`gpu_d2h_pinned_bytes=64 MiB`; record actual task-publication copy bytes, pinned
high water and ordinary-host fallback count. Fail on image publication, excess
bytes or any ordinary-host task-output fallback in the timed loop. This excludes
the deliberately separate bounded final cube export and acceptance transfers.

## Memory feasibility is an admission gate

For one live full-batch compute set, raw plus calibrated images cost
`batch * (32 + 64) MiB`. One 100-bin bank costs
`100 * (64 MiB + 8 bytes) = 6.25 GiB + 800 bytes` per BD/stream.

| Mapping | Raw plus calibrated images per GPU | Cube banks per GPU | Framework ceilings per GPU | Planning subtotal per GPU |
| --- | ---: | ---: | ---: | ---: |
| 1 BD, batch50, 1 GPU | 4.6875 GiB | 6.25 GiB | 6 GiB | 16.9375 GiB |
| 4 BDs, batch200, 1 GPU | 75 GiB | 25 GiB | 24 GiB | 124 GiB |
| 4 BDs, batch200, 2 GPUs | 37.5 GiB | 12.5 GiB | 12 GiB | 62 GiB |
| 4 BDs, batch200, 4 GPUs | 18.75 GiB | 6.25 GiB | 6 GiB | 31 GiB |

Table omits the tiny count arrays. Subtotals are conservative planning sums,
not measured residency: the 6 GiB setting is a ceiling, not a preallocation,
and includes charged framework metadata/constants, not just compressed payload.
The user task's `cp.empty` images and persistent banks are outside that ceiling.
Inventory charged versus uncharged storage; do not double-count charged
constants, or mistake unused credits for available physical GPU memory.
Even banks plus framework ceilings alone total 49 GiB in the shared-GPU case;
the materialized images make direct batch200 compute untenable there. The
four-GPU 31 GiB subtotal is only a candidate on A100-SXM4-40GB, not a fit proof.

Before admitting work, construct a per-physical-GPU ledger including input
high waters, parser/metadata, constants and their actual sharing, raw/calibrated
owners, accepted/error/bin arrays, retained outputs, CUDA contexts, allocator
reserved capacity, all BDs, and at least 4 GiB device safety headroom. Reconcile
current free bytes against already allocated versus outstanding demand; do not
charge the same resident allocation twice. Check actual CUDA total/free bytes,
not the marketing “40GB.” Fail closed if the complete model does not fit.

Verify that one stream actually bounds live task output sets to one; generator
advancement alone is not a release. If aliases, pool retention or task lifetimes
require another batch's storage, include it and reject the cell if needed.
Sample CUDA free/total and CuPy used/reserved at setup, allocation, compute,
delivery, drain and export; label these sampled peaks, not exact continuous peaks.

The existing [Sample10 task](../mfx_expand10_stage2_20261105/adapter.py) explicitly
rejects callbacks above 16 events, and the capacity runner accepts only 1..16.
The staging runner expects callbacks equal to nominal batches. Therefore this
is **not** a job-script-only extension: implement a separately reviewed task
variant with explicit maximum batch and memory checks before allowing 50/200.
Do not merely remove the guard, edit the frozen historical adapter, or rely on
the framework to budget arbitrary task outputs. Record actual callback sizes;
unexpected splitting or pressure drains invalidate the intended cohort.

If four BDs must remain on one GPU, pause for a separate design choice: smaller
cube state and a different budget and/or explicitly bounded compute tiles.
Keeping 100-bin banks and the 6 GiB ceilings is not a conservative 40GB plan even
with tiles. Tiling needs same-stream reuse and owner-lifetime tests; allocating
each tile and retaining every tile until callback completion does not bound
memory. Such a cohort needs its own matched controls and explicit user approval;
never silently reduce bins, batch, budget, or panels for only ECC.

## Required application and MPI adaptation

Preserve the existing kernels and physical panel ordering; implement the new
stage switches, guarded batch capacities and compact device error counters in
a task-specific harness, leaving the frozen staging and historical compute
packages unchanged. Correctness must precede timing.

The existing Cube is serial: its global ordinal map is also used to demand
complete local coverage at export. Passing the full global timestamp list to
every MPI BD would therefore fail finalization; resetting ordinal numbering
per BD would silently corrupt bin assignment. Separate global ordinal lookup
from local ownership/completion. MPI distributes each input event once; the
coordinator validates the disjoint union against the complete ordered source
manifest before declaring success. Never require equal work per BD or replay
100k events independently on every BD.

Use one stream-owned bank per BD. After successful global drain and coverage
checks, perform a bounded, deterministic rank-order host merge, one bin at a
time, in FP64 from FP32 rank contributions; sum uint64 counts exactly. Transfer
at most one 64 MiB FP32 image per participating BD plus a bounded collector
buffer and one 128 MiB FP64 merge image, not whole cubes through MPI gather.
Record D2H, MPI transfer, merge, and write/fsync separately. Publish success only
after every bin and metadata write completes; failure/partial export is not a
valid result and is not automatically retried. Single-rank `finish_to` is not
already an MPI reducer. Verify the MPI constant-injection path before task setup;
the old RunSerial-only constructor hook cannot be assumed valid here.
Pin `PS_JF_SHARE_DERIVED=0` for every stage in this first cohort: the GPU task uses
the frozen Stage2 arrays, not the optional CPU-derived Jungfrau calibration
cache. Record that setting and verify no unexpected derived-cache artifacts.
Supply the hash-verified frozen arrays through an explicitly reviewed smd0 /
MPI calibration-buffer path, not an implicit live DB lookup. Record the source,
buffer identity, shapes and dtypes, and verify each BD's device `batch.calibconst`
arrays against the frozen hashes during an untimed setup/acceptance preflight.
Use bounded transfers for this check; finish it before input cache conditioning
and the timed start. Report broadcast/upload/check time and resident host/device
bytes. If optional derived sharing is enabled later, it is a new declared
cohort with its own artifact provenance and matched staging controls.

## Measurement and storage sequence

1. After topology confirmation and implementation review, qualify one exclusive
   node with four A100-SXM4-40GB devices, enough physical cores, and sufficient
   RAM and local output space. Record UUID/PCI/NUMA, allocation, firmware and
   NVMe/controller/RAID topology. sdfampere036 is a previously inventoried
   candidate, not a currently available or automatically selected node.
2. Start with cold **host page-cache** Weka and NIC/physical-read checks. This
   does not promise cold server caches. Use the same owned data stage, source
   manifest and global event list as the current staging campaign. Recondition
   after setup, hashing or correctness reads, and before every timed case.
3. Use bounded acceptance fixtures (including partial tails and transitions),
   then 4000-global-event pilots for all eight stage/topology cells at the exact
   nominal batches. No automatic full run after a failed capacity/correctness
   gate. Any reference/cache preparation is outside the performance interval.
4. Proposed full screen: 100,000 global events/cell, eight cells, three repeats
   (24 clean runs). Rotate stage order and alternate topology order; add a fresh
   same-topology S anchor after each four-stage block (six anchors). Every block
   starts with separately prepared process state, not state carried from a prior
   stage. Freeze the case order and identities before launch. No cross-job ratios.
5. NVMe is a second, explicitly selected matched cohort, not a substitute for a
   failed Weka cell. Repeat the same controls/matrix with qualified local NVMe
   if storage sensitivity remains a question. The prior slab probe advantage
   did not yield a pipeline staging speedup; it is not an NVMe speed guarantee.

Time clean event processing from a common post-conditioning MPI start barrier
through all ranks' event iteration, completed kernels and final drain/barrier.
Include lazy work occurring there, and separately report setup/JIT/constants,
cube initialization, export, validation and whole-job time. For ECC also report
completed events divided by processing plus bounded export time. No warmed
subwindow substitutes for the complete event interval.

Record global events/s, payload GiB/s, paired stage/S ratios, repeat ranges,
CPU occupancy/affinity and BD imbalance, GPU utilization/memory, input issue/
wait/pressure metrics and NIC/disk counters. Use separate short instrumented
captures for expansion/calibration/count/sum CUDA timings and overlap analysis;
do not sum overlapping inclusive timers or use their rates as clean throughput.
Full-run input payload is 335,820,000,000 bytes (3,358,200/event), so
`GiB/s = events/s * 3358200 / 2**30`; 1400 events/s is 4.378594 GiB/s.
Expanded/calibrated image production is not storage bandwidth.

Cross-topology rates change BD count, total workers, GPU count, and batch;
label them configuration comparisons, not isolated BD or GPU scaling.
Each stage ratio is against its own same-repeat, same-topology S control.
A plateau with an empty task does not by itself prove storage saturation.

## Correctness and completion gates

- Require exact global unique timestamp coverage, callback/delivery agreement,
  five completed datagram reads/event and exact byte/request-size totals; no CPU
  payload fallback. Permit valid partial final batches in acceptance fixtures,
  but no unplanned splitting in the full 50/200 comparison. Confirm one stream.
- Acceptance fixtures must check all expanded uint16 words and calibrated FP32
  words against independent NumPy expansion and CPU-v3 calibration, including
  nominal sizes 50/200, tails, descriptor rejection, and stream-lifetime reuse.
  Exercise MPI uneven ownership and global ordinal/bin identities explicitly.
- For the 4000-event ECC pilots, verify all pixels of every final merged bin
  against independently accumulated FP64 CPU references. Require exact counts,
  no rejected events on valid input and a zero empty bin. Bound with
  `gamma_n * sum(abs(float32_inputs)) + 1e-5`, `gamma_n=n*u/(1-n*u)`, `u=2^-24`,
  using global per-bin n conservatively and accounting for FP64 merge rounding.
  Freeze the bound and CPU reference implementation before results are seen.
- Full runs require exact identities/counts, finite outputs, and predetermined
  sampled-pixel CPU references spanning every active bin, panel and BD, plus the
  full empty-bin zero check. Fix sample indices/seed in the launch manifest and
  record checked coverage. This is not all-pixel numerical validation of 100k
  events; the exhaustive numerical acceptance is the separate bounded pilot.
- Confirm complete input/read/task/consumer drain, no dangling owners or held
  budget credits, no admission-pressure drains, no unexpected data-path changes,
  and consistent hashes. Audit the compact publication allowlist and 64 KiB cap,
  actual task-output copy/pinned bytes, and zero ordinary-host fallback copies in
  the timed loop. A failed case remains failed, never a throughput datum.

Budget actual pilot runtime, instrumented diagnostics, CPU reference work,
staging, 24 full runs, six anchors, export, and audit before choosing Slurm
walltime. Historical staging-only walltime is not a cube-runtime estimate.
Predeclare a remaining-walltime gate; stop with an incomplete campaign rather
than shortening event counts, skipping gates or calling partial results complete.

Keep logs and compact evidence on shared scratch, with durable plan/review/results
in this checkout. Shared scratch has only about 6.7 GiB free at this review;
refresh availability at preparation. Do not place the 336 GB input or full cubes
there. One merged 100-bin FP64 cube is about 12.5 GiB plus headers; use an owned
node-local output directory with explicit capacity/retention checks. Stream
verification and retain compact hashes/error metrics before any authorized cleanup.
Do not accumulate outputs of every repeat on quota-limited shared scratch.

## Interpretation and next decision

If EC/ECC lower rates materially while S stays stable, inspect compute intervals,
GPU memory bandwidth, allocation/retirement stalls and CPU feeding before calling
the loss a pure kernel cost. Similar rates only show hidden/matched costs under
these conditions, not zero compute cost. Prefer the least-resource configuration
meeting the actual application rate and memory/correctness goals, not the fastest
staging-only cell. No new application optimum has been measured.

The next decision is whether to accept **four BDs on four GPUs for the full
100-bin comparison**, or instead scope a separate single-GPU memory-bounded
design. A plan-review pass does not waive that decision, implementation review,
allocation-time gates, or new job-submission authorization.
