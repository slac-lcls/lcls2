# Simulated-compression cube: progress and single-node baseline plan

Updated November 8, 2026, after the small real-JF concurrency diagnostics.
This decision/progress note links completed measurements and proposes the next
stages; the clean 10k baseline has now completed.
Implementation and fixes continue to use the session's independent Claude
implement/review loop. Existing evidence below has its own review records.

## Decision: keep 512 KiB for sample10

Continue compressed-data tests with `gpu_bulk_target_bytes=524288` (512 KiB),
already the sim10 runner default. Framework default remains 1 MiB. The sampler
retains every tenth uint16 pixel per panel, and the GPU repeats each word ten
times, trimming the panel tail. It is simulated compression, not a real codec.
The five JF files retain physical panel assignments and full calibration shapes.

Keep these two findings distinct:

1. At the earlier 8 MiB benchmark read-group target, small-group credits per
   physical XTC file caused forced drains. This was a real admission restriction,
   with ample byte budget and free raw slots. At 512 KiB, sim10's 524,724–734,604
   byte JF datagrams all classify as large; the measured pressure drains vanish.
2. With 512 KiB, the original sim10 trace still shows no kernel overlap. All 149
   next expansion launch APIs occur after the previous sum ends (minimum 4.11 ms).
   Work is not submitted soon enough to overlap. Reading/preparation dominating
   the short compute window is the leading explanation; the compressed trace
   has not isolated storage read, H2D, parsing, metadata and other host costs.

The subsequent real-JF test with twelve calibration passes proves that this
framework does not universally require one execution stream's entire task to
finish before the other submits or executes work. Seven of eleven steady
handoffs return from the next gather launch before the previous sum finishes;
actual user-kernel overlap also occurs. This is evidence of concurrency being
permitted, not proof of the exact bottleneck in the original compressed run.
Slot reuse still waits for that slot's producer/consumers, and input-read
completion is still a prerequisite for parsing and task submission.

No core small-group policy fix is included. 512 KiB is the selected workload
setting, not a general policy repair or a guarantee of kernel overlap.

## Completed and evidence

| Work | Status / evidence |
|---|---|
| Synthetic schema, panel identities, expansion and calibration | Reviewed pilot and 1k acceptance; original ancillary streams retained. [Stage 2](mfx_expand10_stage2_20261105/README.md). |
| GpuTask expansion + calibration + cube | Working prototype; reviewed count and numerical checks. [Stage 3A](mfx_expand10_stage3a_20261105/README.md). |
| Cube capacity/export | 30/100/200 bins validated at 1k, two streams, one A100 40 GB. 100 bins uses 12.5 GiB persistent banks; 200 uses 25 GiB. Bounded export validated. [Capacity](mfx_expand10_capacity_20261106/README.md). |
| 10k fixture | Generated and fully CPU/DataSource validated: 10,000 unique original events, ten physical streams, one native step, 37.683 GB. [Receipt](mfx_expand10_10k_20261106/README.md). |
| Performance runner | Reviewed serial single-worker runner supports 10k with manifest checks, warmup, exact event/bin checks, optional profile/export and pressure counters. [Preparation](mfx_expand10_performance_20261106/README.md). |
| 512 KiB sim10 smoke | Three 1k clean repeats per stream count: median 387.1 events/s (one stream), 487.4 (two); zero pressure drains. Short warmed measurements, not sustained 10k baseline. [Results](mfx_expand10_512k_20261108/README.md). |
| Real-JF cross-check | Small actual MPI SMD0 + EB + one BD captures. Original kernels do not overlap; extended compute permits early submission and some simultaneous execution. [Original](jf_small_timeline_20261108/README.md), [extended compute](jf_compute_overlap_20261108/README.md). |

10k fixture location:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_10k/`

It retains source experiment/run filenames and requires explicit `dir=`; it is
not registered as a new prjdatpsana run. Use frozen calibrated constants already
validated for this fixture. No new data generation or calibration DB deployment
is needed for the next baseline. The old 400k repetition draft remains deferred.

The extended-compute trace also demonstrates KvikIO worker read calls and raw
H2D copies overlapping user kernels, and XTC walk/gather overlapping compute on
the other execution stream. This observation applies to that real-JF diagnostic,
not automatically to the compressed workload. No pipeline publication D2H is
exercised: observed D2H is after drain for validation. User deferred further D2H
investigation; it is not a gate for a cube accumulation baseline with no per-event
publication. Do not describe these captures as a complete D2H pipeline test.

## Remaining stages (new baseline campaign sequence)

### 1. Establish the clean 10k single-worker/one-GPU baseline — completed

Use the existing serial runner first, accurately labeled single-worker rather
than an actual MPI BD measurement. Keep normal single-pass calibration, expansion
and cube accumulation; artificial twelve-pass compute is diagnostic only.

- 10,000 unique events; requested 1,000-event warmup rounded to a whole callback.
- One A100; 100 bins (99 populated plus one empty control); nominal batch 7;
  framework budget 4 GiB; read target 512 KiB; frozen runtime/calibration.
- One and two execution streams, three fresh-process repeats each, alternating
  on the same node/GPU allocation. Use the existing CPU-fallback configuration.
- No Nsight, per-kernel timing or per-event output D2H in the clean runs. Retain
  count-only pressure monitoring and event/bin checks. No full cube file export
  inside timing. Verify source fixture identity before accepting a case.
- Report whole-iteration completed rate and the explicitly defined warmed-window
  rate, median/range, actual warmup/measured events and actual subbatch sizes.
  The existing callback-boundary window can exclude input preparation for its
  first measured callback; do not call it an exact full-I/O window.
- Record node/GPU, source/runtime hashes, CPU allocation, compatibility mode,
  cache/preflight history, bank memory and pressure drains. No cold-cache claim.
  Historical file hashes plus fresh manifest/size checks do not rehash 37.7 GB;
  optional full verification must be documented as warming input cache.

Acceptance: exact 10k ordered timestamps, all expected accepted/bin counts,
correct stream ownership, no unaccounted events and zero pressure drains for
unconstrained comparisons. Capture failures/drains explicitly rather than using
their rate as a clean scaling result. This stage yields a repeatable baseline;
it does not by itself establish the 35 kHz target.

Stage 1 completed in job 43859846: all six 10k cases passed with zero admission
drains. Warmed-window medians were 257.25 and 283.58 events/s (one/two streams),
with material variation among one-stream repeats; no causal scaling claim.
See the [baseline results](mfx_expand10_baseline10k_20261108/README.md).

### 2. Tune batching and identify the limiting stages — attribution measured; tuning next

Hold 512 KiB, 100 bins and the input/runtime fixed. Use a compact nominal batch
comparison (4,7,14; runner supports 1..16), one/two execution streams, reusing the
stage 1 batch 7 results. Start with bounded screening, then repeat the promising
configuration three times. Measure actual GPU subbatches because nominal batch
size is not GPU execution size; keep memory headroom and pressure counts visible.

Section attribution was measured in two separate 10k diagnostic captures using
the existing runner's profile mode (job 43863315). Slot-reuse waiting falls from
5.95 s with one stream to about 0.024 s with two; input-completion waiting is the
largest measured two-stream section (10.03 s), with substantial CPU preparation.
Kernels occupy about 28% of both steady diagnostic envelopes; calibration is
about two thirds of user-kernel time. The 19–22% uninstrumented host remainder
is explicitly retained. These instrumented cases on another node do not replace
the clean throughput baseline. See [section timings](mfx_expand10_sections_20261108/README.md).

A separate [KvikIO read-only sweep](mfx_kvikio_read_20261108/README.md) completed
33 cases in job 43866709. Repeated datagram reads confirmed 15.62 GiB/s median
with four workers, explicit 4 MiB tasks/16 MiB bounce buffers and 35 outstanding
requests; eight workers and large reads did not exceed that confirmed control.
The first pass was only 4.14 GiB/s, so cache/history must remain explicit. This
is five-file JF input throughput with no psana preparation/compute, not physical
Weka bandwidth or an end-to-end speedup. Keep the current pipeline settings
while investigating input submission and host preparation.

Use these captures before commissioning another profile to
attribute the gap to read issue/completion, worker file reads/H2D, XTC parsing,
expansion/calibration/count/sum and slot retirement. Quantify interval overlaps
without adding inclusive timers together. The sim10 adapter reads seeds through
field descriptors and combines assembly with expansion; it has no separate
contiguous-seed gather. Maintain this distinction from the real-JF dense gather.
Use clean 10k runs for rate claims, short captures for diagnosis. Extend event
count only if measured window duration/variance shows 10k is insufficient.

### 3. Establish actual MPI and scale within one node

Adapt/review the compressed runner for actual SMD0 + EB + one BD on one GPU; use the
small real-JF MPI capture as a topology reference, not as existing compressed
MPI throughput evidence. First verify identities/counts with a small input,
then measure 10k with the selected settings to establish the MPI baseline.

Then compare one/two BDs sharing one GPU, subject to measured memory headroom,
and one BD per GPU across the available GPUs of the same node (e.g. 1,2,4).
Increase one dimension at a time.100-bin/two-stream state costs 12.5 GiB per BD,
before constants, input/output buffers and other caches: do a per-GPU allocation
check before each topology. More BDs cannot be admitted by slot count alone.

Partition the same unique input events through MPI, rather than reading the
whole 10k independently per BD and counting duplicates as scaling. Validate global
exact event identity/counts and worker/bin ownership; report aggregate completed
rate using a common wall interval including drain, worker balance, CPU/GPU memory,
read volume/path and scaling efficiency relative to actual MPI one BD. Separate
profiling/sampling runs if instrumentation materially affects the clean rate.
Do not infer a cross-node rate comparison from separately scheduled nodes.

### 4. Close the single-node baseline and choose the next bottleneck

Save reproducible settings, median/range rates, actual batching, memory per GPU,
pressure counts, clean-vs-profile distinctions and interpretation limits. Retain
only useful short native traces with paths/hashes. Choose any targeted framework
or adapter optimization from measured bottlenecks, with correctness checks and
the implement/review loop. Measure final cube export separately if needed; do
not silently include/exclude it in the accumulation rate.

Multi-node work, 400k generation, real codec qualification and per-batch D2H overlap
remain outside this baseline sequence until the single-node result warrants them.
